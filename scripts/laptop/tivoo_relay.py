"""Tivoo Relay — runs on the laptop, polls the Brev box's Gradio UI for state,
forwards state changes to the Tivoo-2 over the local hotspot.

Brev box can't reach the Tivoo directly (Tivoo is on phone hotspot, behind NAT).
This relay bridges them: laptop polls box → laptop sends to Tivoo on local LAN.

Run on the laptop with:
    TIVOO_IP=192.168.x.x BREV_GRADIO=http://localhost:7860 python3 tivoo_relay.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# import the local divoom wrapper
sys.path.insert(0, str(Path(__file__).parent))
import divoom

BREV_GRADIO = os.getenv("BREV_GRADIO", "http://localhost:7860")
POLL_S = float(os.getenv("POLL_S", "1.5"))

# Map agent action → Tivoo state
ACTION_STATE = {
    "act":       "thinking",   # Triage routed ACT, agent is planning
    "store":     "listening",  # noteworthy but not acting
    "discard":   "listening",  # default listening posture
    "executed":  "booked",
    "rejected":  "idle",
    "skip_empty": "idle",
}


def fetch_state() -> str | None:
    """Get the latest action verb from the Gradio UI's live log endpoint.

    Cheap heuristic: scrape the visible log via the Gradio ws/api isn't worth it.
    Instead we expose a tiny /state JSON endpoint from app.py (TODO if needed).
    For now, poll a sentinel file written by agent.py.
    """
    # MVP: read the state from a file on the box via a small HTTP fetch.
    # If the agent.py writes /tmp/jarvis_state.txt with one of {idle, listening,
    # thinking, speaking, booked}, we just GET that.
    try:
        import httpx
        with httpx.Client(timeout=2) as c:
            r = c.get(f"{BREV_GRADIO}/file=/tmp/jarvis_state.txt")
            if r.status_code == 200:
                return r.text.strip()
    except Exception:
        pass
    return None


def main():
    print(f"[tivoo_relay] polling {BREV_GRADIO} every {POLL_S}s, forwarding to Tivoo @ {divoom.TIVOO_IP}")
    last_state = ""
    while True:
        state = fetch_state()
        if state and state != last_state:
            try:
                divoom.set_state(state)
                print(f"  → tivoo: {state}")
                last_state = state
            except Exception as e:
                print(f"  ! tivoo error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
