"""
Mic capture + Whisper transcription.

Captures audio from the default mic in CHUNK_SECONDS blocks.
Skips silent chunks (RMS below threshold), sends the rest to vLLM's
Whisper endpoint, and calls the provided async callback with the text.

Runs on the laptop (or wherever the mic is). The Whisper endpoint
lives on the Brev VM, accessed via SSH tunnel.
"""
import asyncio
import io
import logging
import os

import httpx
import numpy as np
try:
    import sounddevice as sd
    import scipy.io.wavfile as wav_io
    _MIC_AVAILABLE = True
except OSError:
    _MIC_AVAILABLE = False

from config import WHISPER_BASE_URL, WHISPER_MODEL

log = logging.getLogger(__name__)

SAMPLE_RATE       = 16_000
CHUNK_SECONDS     = int(os.environ.get("CHUNK_SECONDS", "3"))
SILENCE_THRESHOLD = float(os.environ.get("SILENCE_THRESHOLD", "0.01"))


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    wav_io.write(buf, SAMPLE_RATE, (audio * 32_767).astype(np.int16))
    buf.seek(0)
    return buf.read()


async def transcribe_bytes(wav_bytes: bytes) -> str:
    """Send raw WAV bytes to the vLLM Whisper endpoint, return transcript text."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{WHISPER_BASE_URL}/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": WHISPER_MODEL, "response_format": "text"},
        )
        resp.raise_for_status()
        return resp.text.strip()


async def capture_and_transcribe(callback, stop_event: asyncio.Event) -> None:
    """
    Continuously capture mic audio until stop_event is set.
    For each non-silent chunk, transcribes and calls await callback(text).
    Requires PortAudio (sounddevice) — not available on headless VMs.
    """
    if not _MIC_AVAILABLE:
        log.error("sounddevice/PortAudio not available — mic capture disabled")
        await stop_event.wait()
        return

    loop       = asyncio.get_event_loop()
    audio_queue: asyncio.Queue = asyncio.Queue()

    def _sd_callback(indata, frames, t, status):
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    blocksize = int(SAMPLE_RATE * CHUNK_SECONDS)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=_sd_callback,
    ):
        log.info("Mic capture started (chunk=%.1fs, silence_threshold=%.3f)",
                 CHUNK_SECONDS, SILENCE_THRESHOLD)
        while not stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < SILENCE_THRESHOLD:
                log.debug("Silence chunk (rms=%.4f), skipping", rms)
                continue

            try:
                wav_bytes = _to_wav_bytes(chunk.flatten())
                text      = await transcribe_bytes(wav_bytes)
                if text:
                    log.debug("Whisper: %r", text)
                    await callback(text)
            except Exception as exc:
                log.warning("Transcription error: %s", exc)

    log.info("Mic capture stopped.")
