"""
Julia — main orchestrator.

Three modes:
  python agent.py                   → Slack huddle mode (production demo)
  python agent.py --input audio.wav → WAV file test mode
  python agent.py --text-script     → built-in text demo (no audio)

The TEST modes (handle_chunk) drive the same async pipeline used by the huddle
mode, just bypassing the Slack-Bolt huddle detector. Both paths post to the
real Slack channel via tools/slack.post_message when triage returns ACT and
mutual agreement has been detected.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import time
import wave
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

import observability
from confirmation import ask_confirmation, register_interaction_handlers
from huddle_detector import HuddleDetector
from meeting_summary import post_end_summary, post_join_notification
from memory import MemoryStore
from middleware import triage
from planner import plan
from schema import TriageAction, ConfirmationAction, TaskProposal, ToolResult
from session import HuddleSession
from sub_agent import execute as execute_slack_post
from transcription import capture_and_transcribe
from agents import voice_agent  # narration only — uses its own (stable) schema

# Initialize Datadog LLMObs at import time. No-op if env vars are unset.
observability.ensure_llmobs_enabled()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("julia.agent")
console = Console()

SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "")
MIN_INTERJECT_GAP_S = 30.0
_last_interject_at = 0.0

STATE_FILE = "/tmp/jarvis_state.txt"
QUESTION_FILE = "/tmp/jarvis_question.txt"
RESULT_FILE = "/tmp/jarvis_result.txt"


def _write_state(state: str) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            f.write(state)
    except Exception:
        pass


def _write_question(question: str) -> None:
    try:
        with open(QUESTION_FILE, "w") as f:
            f.write(f"{time.time():.0f}\t{question}")
    except Exception:
        pass


def _write_result(message: str) -> None:
    try:
        with open(RESULT_FILE, "w") as f:
            f.write(f"{time.time():.0f}\t{message}")
    except Exception:
        pass


# ============================================================================
# Slack huddle mode — async JuliaAgent
# ============================================================================

class JuliaAgent:
    def __init__(self) -> None:
        self._memory = MemoryStore()
        self._session: HuddleSession | None = None
        self._stop_mic = asyncio.Event()
        self._bolt_app = None
        self._bolt_handler = None

    async def on_huddle_start(self, channel_id: str) -> None:
        log.info("Huddle started")
        self._session = HuddleSession(channel_id=channel_id)
        self._stop_mic = asyncio.Event()
        await post_join_notification(channel_id)
        asyncio.create_task(
            capture_and_transcribe(self._on_transcript, self._stop_mic),
            name="mic_capture",
        )

    async def on_huddle_stop(self, channel_id: str) -> None:
        log.info("Huddle ended")
        self._stop_mic.set()
        if self._session:
            self._session.end()
            await post_end_summary(self._session)
            self._session = None

    async def _on_transcript(self, text: str) -> None:
        if not self._session:
            return

        with observability.workflow(name="jarvis_voice_turn",
                                    session_id=self._session.channel_id) as span:
            self._session.add_transcript(text)
            self._memory.write_episodic(text)

            context = self._memory.context_for(text)
            decision = await triage(text, context)
            log.info("TRIAGE: %s — %s", decision.action.value, decision.reason)
            _write_state("listening")

            if decision.action != TriageAction.ACT:
                observability.annotate(span,
                    input_data={"text": text},
                    output_data={"action": decision.action.value, "reason": decision.reason},
                )
                return

            _write_state("thinking")
            try:
                proposal = await plan(text, context)
            except Exception as exc:
                log.error("Planner failed: %s", exc)
                return

            log.info("PLAN content: %r", proposal.content)
            _write_question(proposal.voice_prompt)

            # Optional Slack confirmation card. If SLACK_BOT_TOKEN absent, skip.
            should_post = True
            if SLACK_BOT_TOKEN and SLACK_CHANNEL:
                intent = await ask_confirmation(proposal)
                should_post = (intent.action == ConfirmationAction.YES)

            if not should_post:
                self._memory.write_episodic(f"[rejected] {proposal.voice_prompt}")
                _write_state("listening")
                return

            _write_state("speaking")
            result = await execute_slack_post(proposal, self._memory)
            self._session.add_action(result)

            narration = _safe_narration(proposal, result)
            _write_result(narration)
            _write_state("booked")
            log.info("EXECUTED → %s | narration=%r",
                     "OK" if result.success else "FAIL", narration)

            observability.annotate(span,
                input_data={"text": text, "channel": SLACK_CHANNEL},
                output_data={
                    "action": "ACT",
                    "content": proposal.content,
                    "success": result.success,
                    "message": result.message,
                    "dry_run": result.dry_run,
                    "narration": narration,
                },
                metadata={"reason": decision.reason},
            )

    async def run(self) -> None:
        if SLACK_APP_TOKEN:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
            self._bolt_app = AsyncApp(token=SLACK_BOT_TOKEN)
            register_interaction_handlers(self._bolt_app)
            self._bolt_handler = AsyncSocketModeHandler(self._bolt_app, SLACK_APP_TOKEN)
            asyncio.create_task(self._bolt_handler.start_async(), name="socket_mode")
            log.info("Socket Mode active — button clicks enabled")
        else:
            log.info("No SLACK_APP_TOKEN — using reaction-poll fallback")

        detector = HuddleDetector(
            on_start=self.on_huddle_start,
            on_stop=self.on_huddle_stop,
        )
        log.info("Julia is ready. Waiting for a Slack huddle…")
        try:
            await detector.run()
        except asyncio.CancelledError:
            pass
        finally:
            self._stop_mic.set()
            if self._bolt_handler:
                await self._bolt_handler.close_async()


# ============================================================================
# Sync test path (used by app.py / Gradio + run_text + run_file)
# ============================================================================

_SHARED_MEMORY: MemoryStore | None = None


def _shared_memory() -> MemoryStore:
    global _SHARED_MEMORY
    if _SHARED_MEMORY is None:
        _SHARED_MEMORY = MemoryStore()
    return _SHARED_MEMORY


def _safe_narration(proposal: TaskProposal, result: ToolResult) -> str:
    """Short, past-tense confirmation. Falls back on LLM error."""
    try:
        return voice_agent.compose_narration_for_slack(proposal, result)
    except Exception as exc:
        log.warning("Narration LLM failed (%s) — using fallback", exc)
        if result.success:
            return f"Done. Posted: {proposal.content[:80]}"
        return f"Tried to post but it failed: {result.message[:80]}"


def handle_chunk(text: str, speaker: Optional[str] = None,
                 session_id: Optional[str] = None) -> dict:
    """Process one transcript chunk through the pipeline. Sync wrapper around async."""
    global _last_interject_at
    now = time.time()
    if not text.strip():
        return {"action": "skip_empty"}

    _write_state("listening")
    sess = session_id or os.getenv("JARVIS_SESSION_ID", "live-test")
    return asyncio.run(_handle_chunk_async(text, speaker, now, sess))


async def _handle_chunk_async(text: str, speaker: Optional[str],
                              now: float, session_id: str) -> dict:
    global _last_interject_at
    mem = _shared_memory()

    with observability.workflow(name="jarvis_voice_turn", session_id=session_id) as span:
        # Always record + log the chunk
        mem.write_episodic(f"[{speaker or 'speaker'}] {text}" if speaker else text)
        context = mem.context_for(text)

        decision = await triage(text, context)
        console.log(f"[bold]TRIAGE[/]: {decision.action.value} — {decision.reason}")

        out = {"action": decision.action.value.lower(),
               "decision": {"action": decision.action.value, "reason": decision.reason}}

        if decision.action != TriageAction.ACT:
            observability.annotate(span, input_data={"text": text, "speaker": speaker},
                                   output_data=out)
            return out

        if (now - _last_interject_at) < MIN_INTERJECT_GAP_S:
            console.log("[yellow]ACT suppressed by interject cooldown[/]")
            out["action"] = "suppressed"
            return out

        _write_state("thinking")
        try:
            proposal = await plan(text, context)
        except Exception as exc:
            log.error("Planner failed: %s", exc)
            out["action"] = "plan_failed"
            return out

        console.log(f"[bold magenta]PLAN[/]: {proposal.content!r}")
        _write_question(proposal.voice_prompt)

        # In the test path we trust the triage's mutual-agreement gate and skip
        # the Slack confirmation card. The async huddle path still uses it.
        result = await execute_slack_post(proposal, mem)
        console.log(f"[bold green]EXECUTED[/]: success={result.success} dry_run={result.dry_run} msg={result.message[:80]}")
        _last_interject_at = now

        narration = _safe_narration(proposal, result)
        _write_result(narration)
        _write_state("booked" if result.success else "listening")
        console.log(f"[bold blue]NARRATE[/]: {narration}")

        out.update({
            "proposal": {"content": proposal.content, "voice_prompt": proposal.voice_prompt,
                         "recipients": proposal.recipients},
            "result": {"success": result.success, "message": result.message,
                       "dry_run": result.dry_run},
            "narration": narration,
        })

        observability.annotate(span,
            input_data={"text": text, "speaker": speaker},
            output_data=out,
            metadata={"reason": decision.reason, "channel": SLACK_CHANNEL},
        )
        return out


# ============================================================================
# WAV / text test modes
# ============================================================================

def chunk_wav(path: str, chunk_seconds: float = 4.0):
    with wave.open(path, "rb") as w:
        frame_rate = w.getframerate()
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        frames_per_chunk = int(frame_rate * chunk_seconds)
        idx = 0
        while True:
            frames = w.readframes(frames_per_chunk)
            if not frames:
                break
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(n_channels)
                out.setsampwidth(sample_width)
                out.setframerate(frame_rate)
                out.writeframes(frames)
            yield idx, buf.getvalue()
            idx += 1


def run_file(path: str, fake_speakers: Optional[list[str]] = None) -> None:
    speakers = fake_speakers or ["speaker_1", "speaker_2"]
    # local import to avoid dragging async client init for non-audio runs
    import transcription as _stt
    for idx, chunk in chunk_wav(path):
        console.rule(f"chunk {idx}")
        text = _stt.transcribe_bytes(chunk)
        speaker = speakers[idx % len(speakers)]
        console.log(f"[dim]TRANSCRIPT[/dim] [{speaker}] {text!r}")
        if not text.strip():
            continue
        handle_chunk(text, speaker=speaker)
        time.sleep(0.1)
    console.rule("done")


def run_text(snippets: list[tuple[str, str]]) -> None:
    for i, (sp, txt) in enumerate(snippets):
        console.rule(f"chunk {i}: [{sp}] {txt!r}")
        handle_chunk(txt, speaker=sp)
        time.sleep(0.5)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="WAV file to feed through the pipeline")
    p.add_argument("--text-script", action="store_true", help="Run built-in text demo (no audio)")
    args = p.parse_args()

    if args.input:
        run_file(args.input)
    elif args.text_script:
        # Demonstrates mutual-agreement gating: first chunk should STORE only,
        # second (with explicit confirmation) should ACT → plan → post → narrate.
        run_text([
            ("alex", "Yo, want to grab drinks at 7:30 tonight near Fort Point?"),
            ("sam",  "Yeah, sounds great, let's do it"),
        ])
    else:
        # Default: real Slack huddle mode
        asyncio.run(JuliaAgent().run())


if __name__ == "__main__":
    main()
