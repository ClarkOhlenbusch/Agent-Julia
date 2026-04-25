"""Voice Relay — runs on the laptop, polls the Brev box for fresh agent questions
and confirmations, synthesizes them with Piper, plays through whatever audio
output is selected (Tivoo-2-Audio in our setup).

Replaces the earlier tivoo_relay.py (pixel-art) with audio output, which is
what the Tivoo-2-Audio is actually good at.

Run with:
    PIPER_VOICE=en_US-lessac-high \
    BREV_INSTANCE=jarvis-track5 \
    python3 ~/vllm-hackathon/jarvis-scheduler/scripts/laptop/voice_relay.py

Stop with Ctrl+C.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BREV_INSTANCE = os.getenv("BREV_INSTANCE", "jarvis-track5")
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "0.6"))
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-high")
PIPER_VOICES_DIR = os.path.expanduser(os.getenv("PIPER_VOICES_DIR", "~/piper-voices"))
PIPER_PYTHON = os.getenv("PIPER_PYTHON",
                         os.path.expanduser("~/vllm-hackathon/.venv/bin/python3"))
QUESTION_FILE = "/tmp/jarvis_question.txt"
RESULT_FILE = "/tmp/jarvis_result.txt"


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
    """Run Piper to generate a WAV. Returns True on success."""
    voice = os.path.join(PIPER_VOICES_DIR, f"{PIPER_VOICE}.onnx")
    if not os.path.exists(voice):
        print(f"  ! voice not found: {voice}")
        return False
    try:
        r = subprocess.run(
            [PIPER_PYTHON, "-m", "piper", "-m", voice, "-f", out_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=15,
        )
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception as e:
        print(f"  ! piper error: {e}")
        return False


def play(wav_path: str) -> None:
    subprocess.run(["afplay", wav_path], check=False, timeout=30)


def parse_stamped(blob: str) -> tuple[str, str]:
    """Parse '<unix_ts>\\t<text>' format. Returns (ts, text) or ('', '')."""
    if not blob or "\t" not in blob:
        return "", ""
    ts, text = blob.split("\t", 1)
    return ts.strip(), text.strip()


def main():
    print(f"[voice_relay] polling {BREV_INSTANCE} every {POLL_INTERVAL_S:.1f}s")
    print(f"[voice_relay] Piper voice: {PIPER_VOICE}  ->  current audio output")
    print(f"[voice_relay] question file: {QUESTION_FILE}")
    print(f"[voice_relay] result   file: {RESULT_FILE}")
    print(f"[voice_relay] (output to whatever 'SwitchAudioSource -c' says it is)")

    last_q_ts = ""
    last_r_ts = ""

    while True:
        try:
            # Question
            ts, text = parse_stamped(fetch_remote(QUESTION_FILE))
            if ts and ts != last_q_ts and text:
                print(f"\n[Q  {ts}] {text}")
                wav = "/tmp/_voice_q.wav"
                if synthesize(text, wav):
                    play(wav)
                last_q_ts = ts

            # Result (post-execution confirmation)
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
