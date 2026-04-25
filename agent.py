"""Orchestrator — wires Whisper → Triage → Plan → Sub-Agent → Voice (narration).

Flow:
  - Triage classifies each transcript chunk: STORE | DISCARD | ACT.
  - STORE: kick fact extractor in a background thread to promote noteworthy
    items into long-term semantic memory.
  - ACT: triage has already gated on mutual agreement, so we plan + execute
    immediately, then narrate (past tense). No "may I?" round-trip.
  - DISCARD: nothing.

Fact extractor runs in a background thread on STORE.
"""
from __future__ import annotations

import argparse
import io
import os
import threading
import time
import wave
from queue import Queue, Empty
from typing import Optional

from rich.console import Console

import memory
import transcription
from agents import middleware, planner, voice_agent, sub_agent, fact_extractor
from schema import TriageRoute, TaskProposal

console = Console()

# Throttle interjections to avoid annoying false positives.
MIN_INTERJECT_GAP_S = 30.0
_last_interject_at = 0.0

# Files consumed by the laptop-side Tivoo relay (visual + audio).
STATE_FILE = "/tmp/jarvis_state.txt"
QUESTION_FILE = "/tmp/jarvis_question.txt"
RESULT_FILE = "/tmp/jarvis_result.txt"
TTS_MUTE_DURATION_S = 8.0  # drop mic input for this long after TTS fires

_tts_started_at = 0.0


def _write_state(state: str) -> None:
    global _tts_started_at
    try:
        with open(STATE_FILE, "w") as f:
            f.write(state)
    except Exception:
        pass
    if state in ("speaking", "booked"):
        _tts_started_at = time.time()


def is_tts_active() -> bool:
    return (time.time() - _tts_started_at) < TTS_MUTE_DURATION_S


def _write_question(question: str) -> None:
    """Stamps a fresh interjection question for the laptop voice relay to TTS."""
    try:
        import time as _t
        with open(QUESTION_FILE, "w") as f:
            f.write(f"{_t.time():.0f}\t{question}")
    except Exception:
        pass


def _write_result(message: str) -> None:
    """Stamps an executed-action confirmation for the laptop voice relay to TTS."""
    try:
        import time as _t
        with open(RESULT_FILE, "w") as f:
            f.write(f"{_t.time():.0f}\t{message}")
    except Exception:
        pass


def handle_chunk(text: str, speaker: Optional[str] = None) -> dict:
    """Process one transcript chunk. Returns a status dict for UI/logging."""
    global _last_interject_at
    now = time.time()
    if not text.strip():
        return {"action": "skip_empty"}

    _write_state("listening")

    # Always buffer in episodic (rolling 10-min in-memory)
    memory.episodic_write(text, speaker=speaker)

    decision = middleware.decide(text, speaker=speaker)
    console.log(f"[bold]TRIAGE[/]: {decision.route.value} ({decision.confidence:.2f}) — {decision.reason}")

    out = {"action": decision.route.value.lower(), "decision": decision.model_dump()}

    if decision.route == TriageRoute.STORE:
        # Promote to long-term semantic memory via fact extraction
        threading.Thread(target=fact_extractor.extract, kwargs={"force": True}, daemon=True).start()
        console.log("[dim]STORE → fact extraction triggered[/]")

    elif decision.route == TriageRoute.ACT and (now - _last_interject_at) >= MIN_INTERJECT_GAP_S:
        # Triage already gated on mutual agreement. Plan + execute + narrate.
        _write_state("thinking")
        recent = memory.episodic_recent(10)
        speakers = list({c.get("speaker") for c in recent if c.get("speaker") and c["speaker"] != "agent"})
        proposal = planner.plan(text, attendees_hint=speakers or None)
        console.log(f"[bold magenta]PLAN[/]: {proposal.task_type.value} — {proposal.summary}")

        # Execute first, then narrate the result
        result = sub_agent.execute(proposal)
        console.log(f"[bold green]EXECUTED[/]: {result}")
        _last_interject_at = now
        _write_state("booked")

        try:
            narration = voice_agent.compose_narration(proposal, result)
        except Exception as e:
            console.log(f"[yellow]narration compose failed[/]: {e}")
            narration = f"Done. {result.message}"
        console.log(f"[bold blue]NARRATE[/]: {narration}")
        _write_state("speaking")
        _write_result(narration)
        out.update({
            "proposal": proposal.model_dump(),
            "result": result.model_dump(),
            "narration": narration,
        })

    # DISCARD: nothing beyond the episodic buffer

    return out


# ============================================================================
# Modes
# ============================================================================

def chunk_wav(path: str, chunk_seconds: float = 4.0):
    """Yield audio chunks (bytes) from a WAV file."""
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
    """Run the agent over a recorded WAV — alternates speakers if specified."""
    speakers = fake_speakers or ["speaker_1", "speaker_2"]
    for idx, chunk in chunk_wav(path):
        console.rule(f"chunk {idx}")
        text = transcription.transcribe_bytes(chunk, filename=f"chunk_{idx}.wav")
        speaker = speakers[idx % len(speakers)]
        console.log(f"[dim]TRANSCRIPT[/dim] [{speaker}] {text!r}")
        if not text.strip():
            continue
        handle_chunk(text, speaker=speaker)
        time.sleep(0.1)
    console.rule("done")
    console.log(f"episodic={memory.episodic_count()} semantic={memory.semantic_count()}")


def run_text(snippets: list[tuple[str, str]]) -> None:
    """Run over a list of (speaker, text) tuples — for offline testing without audio."""
    for i, (sp, txt) in enumerate(snippets):
        console.rule(f"chunk {i}: [{sp}] {txt!r}")
        handle_chunk(txt, speaker=sp)
        time.sleep(0.5)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="WAV file to feed through the pipeline")
    p.add_argument("--text-script", action="store_true",
                   help="Run a built-in text-only demo script (no audio).")
    p.add_argument("--seed", action="store_true",
                   help="Seed semantic memory from data/seed_facts.json before running.")
    p.add_argument("--reset", action="store_true",
                   help="Wipe both memory collections before running.")
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
        # Demonstrates mutual-agreement gating: first chunk should STORE only,
        # second (with confirmation) should ACT → plan → execute → narrate.
        run_text([
            ("alex", "Yo, want to grab drinks at 7:30 tonight near Fort Point?"),
            ("sam",  "Yeah, sounds great, let's do it"),
        ])
    else:
        p.print_help()


if __name__ == "__main__":
    main()
