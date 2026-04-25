"""
macOS Slack huddle detector.

Polls every POLL_INTERVAL seconds to check whether a Slack huddle is active
on the local machine. When the state transitions, it fires async callbacks:

  on_start()  — huddle just became active  → begin mic capture
  on_stop()   — huddle just ended          → stop mic capture

Detection strategy (tried in order):
  1. AppleScript — looks for Slack's floating huddle window by title/role
  2. lsof        — checks whether the Slack process has an audio device open
  3. CoreAudio   — checks which PIDs have audio input streams open (macOS only)

Usage:
  detector = HuddleDetector(on_start=..., on_stop=...)
  await detector.run()   # blocks; runs until cancelled
"""

import asyncio
import logging
import os
import re
import subprocess
from typing import Callable, Coroutine

log = logging.getLogger(__name__)

POLL_INTERVAL = float(os.environ.get("HUDDLE_POLL_INTERVAL", "2.0"))

StartCallback = Callable[[], Coroutine]
StopCallback  = Callable[[], Coroutine]

# AppleScript: returns "true" if Slack has a huddle-related window open.
# Slack's huddle UI shows a floating toolbar whose title contains "Huddle"
# or an accessibility description containing "huddle". We check both.
_APPLESCRIPT = """
tell application "System Events"
    if not (exists process "Slack") then return "false"
    tell process "Slack"
        set allWindows to name of every window
        repeat with wName in allWindows
            if wName contains "Huddle" or wName contains "huddle" then
                return "true"
            end if
        end repeat
        -- Also check UI elements for the floating huddle bar
        try
            set uiElems to description of every UI element of window 1
            repeat with d in uiElems
                if d contains "huddle" then return "true"
            end repeat
        end try
    end tell
    return "false"
end tell
"""


async def _check_applescript() -> bool:
    """Return True if Slack shows a huddle window (macOS only)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", _APPLESCRIPT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode().strip() == "true"
    except (asyncio.TimeoutError, FileNotFoundError):
        return False


async def _slack_pid() -> int | None:
    """Return the PID of the main Slack process, or None if not running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "pgrep", "-x", "Slack",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        pids = stdout.decode().split()
        return int(pids[0]) if pids else None
    except (ValueError, FileNotFoundError):
        return None


async def _check_lsof() -> bool:
    """
    Return True if the Slack process has an audio device file open.
    Matches /dev/audio*, CoreAudio virtual devices, or 'audio' in the path.
    """
    pid = await _slack_pid()
    if pid is None:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "lsof", "-p", str(pid), "-n", "-P",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        output = stdout.decode()
        # CoreAudio on macOS shows up as character device files like /dev/audiopipe*
        # or as named pipes used by coreaudiod
        return bool(re.search(r"(audio|coreaudio|snd)", output, re.IGNORECASE))
    except (asyncio.TimeoutError, FileNotFoundError):
        return False


async def _check_coreaudio_mic() -> bool:
    """
    Use ioreg to check if any audio engine reports an active input client
    belonging to Slack. Less reliable but a useful third signal on macOS.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ioreg", "-l", "-n", "AppleHDAEngineInput",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return b"Slack" in stdout
    except (asyncio.TimeoutError, FileNotFoundError):
        return False


async def is_huddle_active() -> bool:
    """
    Aggregate check: returns True if any detection strategy says a huddle
    is active. Tries strategies in order and short-circuits on first True.
    """
    strategies = [_check_applescript, _check_lsof, _check_coreaudio_mic]
    for strategy in strategies:
        try:
            if await strategy():
                return True
        except Exception as exc:
            log.debug("Detection strategy %s failed: %s", strategy.__name__, exc)
    return False


class HuddleDetector:
    def __init__(self, on_start: StartCallback, on_stop: StopCallback):
        self.on_start     = on_start
        self.on_stop      = on_stop
        self._was_active  = False
        self._running     = False

    async def run(self) -> None:
        """Poll until cancelled. Fires on_start / on_stop on state transitions."""
        self._running = True
        log.info(
            "Huddle detector polling every %.1fs (AppleScript → lsof → CoreAudio)",
            POLL_INTERVAL,
        )
        while self._running:
            try:
                active = await is_huddle_active()

                if active and not self._was_active:
                    log.info("Slack huddle detected — starting Jarvis")
                    self._was_active = True
                    try:
                        await self.on_start()
                    except Exception as exc:
                        log.error("on_start callback failed: %s", exc)

                elif not active and self._was_active:
                    log.info("Slack huddle ended — stopping Jarvis")
                    self._was_active = False
                    try:
                        await self.on_stop()
                    except Exception as exc:
                        log.error("on_stop callback failed: %s", exc)

            except Exception as exc:
                log.warning("Huddle detection error (will retry): %s", exc)

            await asyncio.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False


def make_callbacks(channel_id: str):
    """
    Build the on_start / on_stop callbacks that wire the huddle detector
    into the Juliah pipeline.

    on_start:
      - Posts "Juliah joined" notification to Slack
      - Returns a fresh HuddleSession (caller should store it)

    on_stop:
      - Ends the session
      - Generates + posts meeting summary to Slack
    """
    from meeting_summary import post_join_notification, post_end_summary
    from session import HuddleSession

    # Mutable container so the closure can share state
    state: dict = {"session": None}

    async def on_start() -> "HuddleSession":
        session = HuddleSession(channel_id=channel_id)
        state["session"] = session
        await post_join_notification(channel_id)
        return session

    async def on_stop() -> None:
        session: HuddleSession | None = state.get("session")
        if session is None:
            return
        session.end()
        await post_end_summary(session)
        state["session"] = None

    return on_start, on_stop


# ------------------------------------------------------------------
# Standalone smoke-test: python huddle_detector.py
# ------------------------------------------------------------------

async def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def on_start() -> None:
        print(">>> HUDDLE STARTED — would post join notification + begin listening")

    async def on_stop() -> None:
        print(">>> HUDDLE ENDED — would generate summary + post to Slack")

    print("Running huddle detector. Start/stop a Slack huddle to test. Ctrl+C to quit.")
    detector = HuddleDetector(on_start=on_start, on_stop=on_stop)
    try:
        await detector.run()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(_smoke_test())
