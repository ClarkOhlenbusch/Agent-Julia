"""
Julia — main orchestrator.

Two modes:
  python agent.py                   → Slack huddle mode (production demo)
  python agent.py --input audio.wav → WAV file test mode
  python agent.py --text-script     → built-in text demo (no audio)

Slack huddle mode:
  1. HuddleDetector polls for an active Slack huddle on this machine.
  2. Posts "Julia is listening now" to the channel.
  3. Mic capture begins. Each Whisper chunk is triaged.
  4. If ACT: plan a Slack message → post YES/NO card → wait → if YES post it.
  5. Huddle ends → post meeting summary to the channel.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import threading
import time
import wave
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

import memory
import transcription
from agents import middleware as sync_middleware, planner as sync_planner
from agents import voice_agent, sub_agent as sync_sub_agent, fact_extractor
from confirmation import ask_confirmation, register_interaction_handlers
from huddle_detector import HuddleDetector
from meeting_summary import post_end_summary, post_join_notification
from memory import MemoryStore
from middleware import triage
from planner import plan
from schema import TriageAction, ConfirmationAction, TriageRoute
from session import HuddleSession
from sub_agent import execute
from transcription import capture_and_transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log      = logging.getLogger("julia.agent")
console  = Console()

SLACK_APP_TOKEN     = os.environ.get("SLACK_APP_TOKEN", "")
SLACK_BOT_TOKEN     = os.environ.get("SLACK_BOT_TOKEN", "")
MIN_INTERJECT_GAP_S = 30.0
_last_interject_at  = 0.0

STATE_FILE    = "/tmp/jarvis_state.txt"
RESULT_FILE   = "/tmp/jarvis_result.txt"


def _write_state(state: str) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            f.write(state)
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
        self._memory       = MemoryStore()
        self._session: HuddleSession | None = None
        self._stop_mic     = asyncio.Event()
        self._bolt_app     = None
        self._bolt_handler = None

    async def on_huddle_start(self, channel_id: str) -> None:
        log.info("Huddle started")
        self._session  = HuddleSession(channel_id=channel_id)
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

        self._session.add_transcript(text)
        self._memory.write_episodic(text)

        context  = self._memory.context_for(text)
        decision = await triage(text, context)

        if decision.action != TriageAction.ACT:
            return

        try:
            proposal = await plan(text, context)
        except Exception as exc:
            log.error("Planner failed: %s", exc)
            return

        intent = await ask_confirmation(proposal)

        if intent.action == ConfirmationAction.YES:
            result = await execute(proposal, self._memory)
            self._session.add_action(result)
        else:
            self._memory.write_episodic(f"[rejected] {proposal.voice_prompt}")

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
# File / text test modes (sync — for offline testing without a huddle)
# ============================================================================

def handle_chunk(text: str, speaker: Optional[str] = None) -> dict:
    global _last_interject_at
    now = time.time()
    if not text.strip():
        return {"action": "skip_empty"}

    _write_state("listening")
    memory.episodic_write(text, speaker=speaker)

    decision = sync_middleware.decide(text, speaker=speaker)
    console.log(f"[bold]TRIAGE[/]: {decision.route.value} ({decision.confidence:.2f}) — {decision.reason}")

    out = {"action": decision.route.value.lower(), "decision": decision.model_dump()}

    if decision.route == TriageRoute.STORE:
        threading.Thread(target=fact_extractor.extract, kwargs={"force": True}, daemon=True).start()

    elif decision.route == TriageRoute.ACT and (now - _last_interject_at) >= MIN_INTERJECT_GAP_S:
        _write_state("thinking")
        recent   = memory.episodic_recent(10)
        speakers = list({c.get("speaker") for c in recent if c.get("speaker") and c["speaker"] != "agent"})
        proposal = sync_planner.plan(text, attendees_hint=speakers or None)
        console.log(f"[bold magenta]PLAN[/]: {proposal.task_type.value} — {proposal.summary}")
        result   = sync_sub_agent.execute(proposal)
        console.log(f"[bold green]EXECUTED[/]: {result}")
        _last_interject_at = now
        _write_state("booked")
        try:
            narration = voice_agent.compose_narration(proposal, result)
        except Exception as e:
            narration = f"Done. {result.message}"
        _write_result(narration)
        out.update({"proposal": proposal.model_dump(), "result": result.model_dump()})

    return out


def chunk_wav(path: str, chunk_seconds: float = 4.0):
    with wave.open(path, "rb") as w:
        frame_rate     = w.getframerate()
        n_channels     = w.getnchannels()
        sample_width   = w.getsampwidth()
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
    for idx, chunk in chunk_wav(path):
        console.rule(f"chunk {idx}")
        text     = transcription.transcribe_bytes(chunk, filename=f"chunk_{idx}.wav")
        speaker  = speakers[idx % len(speakers)]
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
    p.add_argument("--input",       help="WAV file to feed through the pipeline")
    p.add_argument("--text-script", action="store_true", help="Run built-in text demo (no audio)")
    p.add_argument("--seed",        action="store_true", help="Seed semantic memory before running")
    p.add_argument("--reset",       action="store_true", help="Wipe memory collections before running")
    args = p.parse_args()

    if args.reset:
        memory.reset_all()
        console.log("[yellow]reset memory[/]")
    if args.seed:
        n = memory.seed_from_file(os.path.join(os.path.dirname(__file__), "data/seed_facts.json"))
        console.log(f"[green]seeded {n} facts[/]")

    if args.input:
        run_file(args.input)
    elif args.text_script:
        run_text([
            ("alex", "Yo, want to grab drinks at 7:30 tonight near Fort Point?"),
            ("sam",  "Yeah, sounds great, let's do it"),
        ])
    else:
        # Default: Slack huddle mode
        asyncio.run(JuliaAgent().run())


if __name__ == "__main__":
    main()
