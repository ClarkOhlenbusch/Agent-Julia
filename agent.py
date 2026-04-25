"""Orchestrator — wires Whisper → Triage → Plan → Voice → Confirm → Sub-Agent.

Two execution modes:
  - file mode: --input data/demo_script.wav  (chunks the WAV, runs through pipeline)
  - server mode: starts a FastAPI shim that accepts audio chunks via POST /chunk
                 and routes through the pipeline; used by app.py / Gradio.

Fact extractor runs every 10 chunks, in a background thread.
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
from schema import TriageRoute, ConfirmIntent, TaskProposal

console = Console()

# Throttle interjections to avoid annoying false positives.
MIN_INTERJECT_GAP_S = 30.0
_last_interject_at = 0.0
_pending_confirmation: Optional[dict] = None

# Files consumed by the laptop-side Tivoo relay (visual + audio).
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
    global _last_interject_at, _pending_confirmation
    now = time.time()
    if not text.strip():
        return {"action": "skip_empty"}

    _write_state("listening")
    # If we're awaiting a yes/no confirmation, parse this chunk against that.
    if _pending_confirmation:
        question = _pending_confirmation["question"]
        proposal: TaskProposal = _pending_confirmation["proposal"]
        ci = voice_agent.parse_response(text, question)
        console.log(f"[bold cyan]CONFIRM PARSE[/]: {ci.intent.value} mod={ci.modifier!r}")
        if ci.intent == ConfirmIntent.YES:
            result = sub_agent.execute(proposal)
            console.log(f"[bold green]EXECUTED[/]: {result}")
            _pending_confirmation = None
            _write_state("booked")
            _write_result(f"Done. {result.message}")
            return {"action": "executed", "result": result.model_dump()}
        if ci.intent == ConfirmIntent.NO:
            sub_agent.execute_rejection(proposal)
            console.log(f"[yellow]REJECTED[/]: {proposal.summary}")
            _pending_confirmation = None
            return {"action": "rejected"}
        if ci.intent == ConfirmIntent.MODIFY:
            sub_agent.execute_rejection(proposal, modifier=ci.modifier)
            _pending_confirmation = None
            # Fall through — re-plan with the modifier as a fresh chunk
            text = f"User wants modification: {ci.modifier}. Originally: {proposal.summary}"
        else:
            return {"action": "unclear"}

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
        _write_state("thinking")
        recent = memory.episodic_recent(10)
        speakers = list({c.get("speaker") for c in recent if c.get("speaker") and c["speaker"] != "agent"})
        proposal = planner.plan(text, attendees_hint=speakers or None)
        console.log(f"[bold magenta]PLAN[/]: {proposal.task_type.value} — {proposal.summary}")
        question = voice_agent.compose_question(proposal)
        console.log(f"[bold blue]ASK[/]: {question}")
        _pending_confirmation = {"proposal": proposal, "question": question, "asked_at": now}
        _last_interject_at = now
        _write_state("speaking")
        _write_question(question)
        out.update({"proposal": proposal.model_dump(), "question": question})

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
    speakers = fake_speakers or ["alex", "sam"]
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
        run_text([
            ("alex", "Yo, we should grab drinks tonight."),
            ("sam",  "Yeah! When are you free? Coffee earlier today wiped me out though."),
            # planner should fire here
            ("sam",  "Yeah that works"),  # confirm
        ])
    else:
        p.print_help()


if __name__ == "__main__":
    main()
