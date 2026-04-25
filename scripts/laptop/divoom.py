"""Divoom Tivoo-2 control via Bluetooth (macOS).

Wraps solar2ain/tivoo-control's CLI (clone to ~/tivoo-control, compile bridge
once with: clang -framework Foundation -framework IOBluetooth -o tivoo_cmd
tivoo_cmd.m -fobjc-arc).

Required env:
  TIVOO_MAC=AA:BB:CC:DD:EE:FF   (find via System Preferences → Bluetooth)
  TIVOO_REPO=~/tivoo-control     (defaults to ~/tivoo-control)
  TIVOO_PYTHON=...               (path to a python with click + Pillow)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Optional

TIVOO_MAC = os.getenv("TIVOO_MAC", "B1:21:81:09:57:BE")
TIVOO_REPO = os.path.expanduser(os.getenv("TIVOO_REPO", "~/tivoo-control"))
TIVOO_PYTHON = os.getenv("TIVOO_PYTHON",
                         os.path.expanduser("~/vllm-hackathon/.venv/bin/python3"))

# Agent state → CLI sub-command mapping. We use presets where we can since
# they're snappy and self-contained.
STATE_TO_CMD: dict[str, list[str]] = {
    "idle":      ["preset", "moon", "--duration", "5"],
    "listening": ["preset", "searching", "--duration", "8"],
    "thinking":  ["preset", "loading", "--loop", "0", "--duration", "6"],
    "speaking":  ["preset", "bell", "--duration", "5"],
    "booked":    ["preset", "success", "--duration", "6"],
    "rejected":  ["preset", "cross", "--duration", "4"],
    "error":     ["preset", "error", "--duration", "4"],
}


def _run(args: list[str], timeout: float = 15.0) -> tuple[bool, str]:
    """Run tivoo_macos.py with given args. Returns (success, stdout/err)."""
    cmd = [TIVOO_PYTHON, "tivoo_macos.py"] + args
    env = {**os.environ, "TIVOO_MAC": TIVOO_MAC}
    try:
        r = subprocess.run(cmd, cwd=TIVOO_REPO, env=env, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError as e:
        return False, f"not found: {e}"


def set_state(state: str) -> tuple[bool, str]:
    """Display the Tivoo preset for the given agent state."""
    args = STATE_TO_CMD.get(state) or STATE_TO_CMD["idle"]
    return _run(args)


def show_text(text: str, color: str = "cyan") -> tuple[bool, str]:
    return _run(["text", text, "--color", color])


def show_clock() -> tuple[bool, str]:
    return _run(["clock"])


def brightness(level: int) -> tuple[bool, str]:
    return _run(["brightness", str(max(0, min(100, level)))])


def status() -> tuple[bool, str]:
    return _run(["status"])


def off() -> tuple[bool, str]:
    return _run(["off"])


if __name__ == "__main__":
    import sys, time
    print(f"Tivoo @ {TIVOO_MAC} via {TIVOO_REPO}")
    if len(sys.argv) > 1:
        # Pass-through mode: python3 divoom.py text "Hello"
        ok, out = _run(sys.argv[1:])
        print(out)
        sys.exit(0 if ok else 1)
    # Demo: cycle through every state
    for st in ("listening", "thinking", "speaking", "booked", "idle"):
        print(f"  → {st}")
        ok, out = set_state(st)
        if not ok:
            print(f"    FAIL: {out}")
        time.sleep(3)
    print("Done.")
