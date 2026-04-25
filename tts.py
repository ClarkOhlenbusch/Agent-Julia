"""Text-to-speech via Piper. Returns WAV bytes for Gradio's audio output widget.

Piper is CPU-runnable on the box; we use the en_US-amy-medium voice by default.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
from typing import Optional

PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-amy-medium")
PIPER_MODELS_DIR = os.getenv("PIPER_MODELS_DIR", os.path.expanduser("~/piper-voices"))


def _voice_path() -> str:
    return os.path.join(PIPER_MODELS_DIR, f"{PIPER_VOICE}.onnx")


def synthesize(text: str) -> bytes:
    """Synthesize WAV audio from text. Returns raw WAV bytes."""
    if not shutil.which("piper"):
        raise RuntimeError("piper binary not on PATH; install piper-tts first.")
    if not os.path.exists(_voice_path()):
        raise RuntimeError(
            f"Piper voice not found at {_voice_path()}. "
            f"Download with: python -m piper.download {PIPER_VOICE} --download-dir {PIPER_MODELS_DIR}"
        )
    proc = subprocess.run(
        ["piper", "--model", _voice_path(), "--output_raw"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    return proc.stdout


def synthesize_to_file(text: str, out_path: str) -> str:
    audio = synthesize(text)
    with open(out_path, "wb") as f:
        f.write(audio)
    return out_path


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Hey, I think you're both free at 7:30 — want me to put it on the calendar?"
    out = synthesize_to_file(text, "/tmp/tts_test.wav")
    print(f"Wrote {out}")
