"""Voice Relay — runs on the laptop, polls the Brev box for fresh agent
questions and confirmations, synthesizes them with Kokoro-TTS, plays through
whatever audio output is selected (Tivoo-2-Audio in our setup).

Kokoro voice af_jessica is high-quality and natural. The pipeline loads once
at startup (~3s) and every subsequent sentence synthesizes in ~1s.

Run with:
    BREV_INSTANCE=jarvis-track5 \
    KOKORO_VOICE=af_jessica \
    python3 ~/vllm-hackathon/jarvis-scheduler/scripts/laptop/voice_relay.py

Stop with Ctrl+C.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Quiet the noisy Kokoro / torch / weight-norm warnings on import.
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf

BREV_INSTANCE = os.getenv("BREV_INSTANCE", "jarvis-track5")
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "0.6"))
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_jessica")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")  # 'a' = American English
QUESTION_FILE = "/tmp/jarvis_question.txt"
RESULT_FILE = "/tmp/jarvis_result.txt"

_pipeline = None


def _ensure_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=KOKORO_LANG, repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def fetch_remote(path: str) -> str:
    """SSH-poll a file on the box. Empty string = file missing or error."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
             BREV_INSTANCE, f"cat {path} 2>/dev/null"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def synthesize(text: str, out_path: str) -> bool:
    pipe = _ensure_pipeline()
    try:
        gen = pipe(text, voice=KOKORO_VOICE)
        chunks = [r.audio for r in gen]
        if not chunks:
            return False
        audio = np.concatenate(chunks)
        sf.write(out_path, audio, 24000)
        return True
    except Exception as e:
        print(f"  ! kokoro error: {e}")
        return False


def play(wav_path: str) -> None:
    subprocess.run(["afplay", wav_path], check=False, timeout=30)


def parse_stamped(blob: str) -> tuple[str, str]:
    if not blob or "\t" not in blob:
        return "", ""
    ts, text = blob.split("\t", 1)
    return ts.strip(), text.strip()


def main():
    print(f"[voice_relay] polling {BREV_INSTANCE} every {POLL_INTERVAL_S:.1f}s")
    print(f"[voice_relay] voice: Kokoro {KOKORO_VOICE} (lang={KOKORO_LANG})")
    print(f"[voice_relay] loading Kokoro pipeline (~3s) ...")
    _ensure_pipeline()
    print(f"[voice_relay] ready. Speaking through current audio output.")

    last_q_ts = ""
    last_r_ts = ""

    while True:
        try:
            ts, text = parse_stamped(fetch_remote(QUESTION_FILE))
            if ts and ts != last_q_ts and text:
                print(f"\n[Q  {ts}] {text}")
                wav = "/tmp/_voice_q.wav"
                if synthesize(text, wav):
                    play(wav)
                last_q_ts = ts

            ts, text = parse_stamped(fetch_remote(RESULT_FILE))
            if ts and ts != last_r_ts and text:
                print(f"[R  {ts}] {text}")
                wav = "/tmp/_voice_r.wav"
                if synthesize(text, wav):
                    play(wav)
                last_r_ts = ts

            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            print("\n[voice_relay] stopping.")
            break
        except Exception as e:
            print(f"[voice_relay] tick error: {e}")
            time.sleep(POLL_INTERVAL_S * 2)


if __name__ == "__main__":
    main()
