"""Divoom Tivoo-2 HTTP API wrapper.

Tivoo-2 (model TIIV00-2) supports the Divoom HTTP API over Wi-Fi.
Both the laptop and the Tivoo must be on the same network (phone hotspot for the demo).

Find the device IP via the Divoom phone app: My Devices → tap the device → "Local IP".
Set TIVOO_IP env var or pass to functions.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import httpx

TIVOO_IP = os.getenv("TIVOO_IP", "")
DIVOOM_PORT = 80


def _post(payload: dict, ip: Optional[str] = None) -> dict:
    target = ip or TIVOO_IP
    if not target:
        raise RuntimeError("TIVOO_IP not set. Set env var or pass ip=.")
    url = f"http://{target}:{DIVOOM_PORT}/post"
    with httpx.Client(timeout=8) as c:
        r = c.post(url, json=payload)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}


# ============================================================================
# State display — emoji-style icons via the device's built-in channels
# ============================================================================

# State → channel/clock face index (rough mapping; tune to whatever looks best on device)
_STATE_TO_CHANNEL = {
    "idle":      {"Command": "Channel/SetIndex", "SelectIndex": 0},   # clock
    "listening": {"Command": "Channel/SetIndex", "SelectIndex": 1},   # cloud (substitute for ear)
    "thinking":  {"Command": "Channel/SetIndex", "SelectIndex": 2},   # planet / radar
    "speaking":  {"Command": "Channel/SetIndex", "SelectIndex": 3},   # animation
    "booked":    {"Command": "Channel/SetIndex", "SelectIndex": 0},   # back to clock = success
}


def set_state(state: str, ip: Optional[str] = None) -> dict:
    """Show one of: idle | listening | thinking | speaking | booked."""
    payload = _STATE_TO_CHANNEL.get(state, _STATE_TO_CHANNEL["idle"])
    return _post(payload, ip=ip)


def show_text(text: str, ip: Optional[str] = None, color: str = "FFFFFF") -> dict:
    """Scroll text across the display."""
    return _post({
        "Command": "Draw/SendHttpText",
        "TextId": 1,
        "x": 0,
        "y": 0,
        "dir": 0,         # 0 = scroll right-to-left
        "font": 4,
        "TextWidth": 64,
        "speed": 30,
        "TextString": text,
        "color": color,
        "align": 1,
    }, ip=ip)


def beep(ip: Optional[str] = None, ms: int = 200) -> dict:
    """Short beep — for interjection alert."""
    return _post({"Command": "Device/PlayBuzzer",
                  "ActiveTimeInCycle": ms, "OffTimeInCycle": 0, "PlayTotalTime": ms}, ip=ip)


def set_brightness(level: int, ip: Optional[str] = None) -> dict:
    return _post({"Command": "Channel/SetBrightness", "Brightness": max(0, min(100, level))}, ip=ip)


def discover() -> list[str]:
    """Try to find Divoom devices on the LAN via the public discovery endpoint."""
    try:
        with httpx.Client(timeout=5) as c:
            r = c.post("https://app.divoom-gz.com/Device/ReturnSameLANDevice", json={})
        r.raise_for_status()
        return [d.get("DevicePrivateIP") for d in r.json().get("DeviceList", []) if d.get("DevicePrivateIP")]
    except Exception:
        return []


if __name__ == "__main__":
    import sys
    if not TIVOO_IP:
        print("Set TIVOO_IP first. Try discovery:")
        for ip in discover():
            print(f"  found: {ip}")
        print("Or open the Divoom app and check 'Local IP' for the device.")
        sys.exit(1)
    print(f"Pinging Tivoo at {TIVOO_IP} ...")
    set_state("idle")
    time.sleep(1)
    show_text("JARVIS")
    time.sleep(2)
    for s in ("listening", "thinking", "speaking", "booked"):
        set_state(s)
        print(f"  → {s}")
        time.sleep(1.5)
    print("Done.")
