"""Tivoo Relay — runs on the laptop, polls the Brev box's agent state file
(/tmp/jarvis_state.txt) over SSH, and forwards state changes to the Tivoo-2.

The agent (running on the box) writes one of these states to that file:
  listening | thinking | speaking | booked | rejected | idle

Run on the laptop with:
    TIVOO_MAC=B1:21:81:09:57:BE BREV_INSTANCE=jarvis-track5 \
    python3 ~/vllm-hackathon/jarvis-scheduler/scripts/laptop/tivoo_relay.py

Stop with Ctrl+C.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import divoom

BREV_INSTANCE = os.getenv("BREV_INSTANCE", "jarvis-track5")
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "1.0"))
STATE_FILE = "/tmp/jarvis_state.txt"


def fetch_state() -> str:
    """SSH-poll the agent state file. Returns 'idle' on any failure."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
             BREV_INSTANCE, f"cat {STATE_FILE} 2>/dev/null || echo idle"],
            capture_output=True, text=True, timeout=5,
        )
        s = r.stdout.strip()
        return s if s else "idle"
    except subprocess.TimeoutExpired:
        return ""  # signal "couldn't poll"
    except Exception:
        return ""


def main():
    print(f"[tivoo_relay] polling {BREV_INSTANCE}:{STATE_FILE} every {POLL_INTERVAL_S:.1f}s")
    print(f"[tivoo_relay] forwarding to Tivoo @ {divoom.TIVOO_MAC}")
    last = ""
    last_change = 0.0
    # On boot, set Tivoo to idle for clarity
    divoom.set_state("idle")
    while True:
        try:
            state = fetch_state()
            if state and state != last:
                # Debounce — don't fire if we just changed (Tivoo BT calls take ~1-2s)
                if time.time() - last_change > 2.0:
                    print(f"  → tivoo: {state}")
                    ok, out = divoom.set_state(state)
                    if not ok:
                        print(f"  ! tivoo error: {out}")
                    last = state
                    last_change = time.time()
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            print("\n[tivoo_relay] stopping, restoring clock")
            divoom.show_clock()
            break
        except Exception as e:
            print(f"[tivoo_relay] tick error: {e}")
            time.sleep(POLL_INTERVAL_S * 2)


if __name__ == "__main__":
    main()
