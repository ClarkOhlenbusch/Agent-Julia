"""Whisper transcription client.

Sends audio chunks to vLLM's Whisper endpoint at :9000 and returns text.
Inside NemoClaw use http://host.openshell.internal:9000/v1.
"""
from __future__ import annotations

import io
import os
from typing import BinaryIO, Optional

import httpx

WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "http://localhost:9000/v1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-turbo")


def transcribe_bytes(audio_bytes: bytes, filename: str = "chunk.wav",
                     content_type: str = "audio/wav") -> str:
    """Transcribe audio bytes via vLLM Whisper. Returns plain text."""
    files = {"file": (filename, io.BytesIO(audio_bytes), content_type)}
    data = {"model": WHISPER_MODEL, "response_format": "text"}
    with httpx.Client(timeout=60) as c:
        r = c.post(f"{WHISPER_ENDPOINT}/audio/transcriptions", files=files, data=data)
    r.raise_for_status()
    text = r.text.strip()
    return text


def transcribe_file(path: str) -> str:
    with open(path, "rb") as f:
        return transcribe_bytes(f.read(), filename=os.path.basename(path))


def transcribe_with_meta(audio_bytes: bytes, filename: str = "chunk.wav") -> dict:
    """Verbose transcription incl. segments + timestamps."""
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data = {"model": WHISPER_MODEL, "response_format": "verbose_json"}
    with httpx.Client(timeout=60) as c:
        r = c.post(f"{WHISPER_ENDPOINT}/audio/transcriptions", files=files, data=data)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 transcription.py <audio.wav>")
        sys.exit(1)
    print(transcribe_file(sys.argv[1]))
