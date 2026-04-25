"""
macOS Slack huddle detector.

Polls every POLL_INTERVAL seconds. When a Slack huddle becomes active on
this machine, fires on_start(channel_id) / on_stop(channel_id).

Channel is fixed via SLACK_CHANNEL env var — no auto-detection needed.

Required env vars:
  SLACK_CHANNEL  — the Slack channel ID to post into (e.g. C0XXXXXXX)
"""
import asyncio
import logging
import os
import re

log = logging.getLogger(__name__)

POLL_INTERVAL = float(os.environ.get("HUDDLE_POLL_INTERVAL", "2.0"))
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "")

# AppleScript: returns "true" if Slack has an active huddle window.
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
    end tell
    return "false"
end tell
"""


async def _huddle_active() -> bool:
    """Return True if Slack is currently in a huddle on this machine."""
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


class HuddleDetector:
    def __init__(self, on_start, on_stop):
        self.on_start       = on_start
        self.on_stop        = on_stop
        self._was_active    = False
        self._running       = False

    async def run(self) -> None:
        if not SLACK_CHANNEL:
            raise EnvironmentError("SLACK_CHANNEL env var is not set.")

        self._running = True
        log.info("Huddle detector polling every %.1fs → channel %s", POLL_INTERVAL, SLACK_CHANNEL)

        while self._running:
            try:
                active = await _huddle_active()

                if active and not self._was_active:
                    self._was_active = True
                    log.info("Huddle started")
                    await self.on_start(SLACK_CHANNEL)

                elif not active and self._was_active:
                    self._was_active = False
                    log.info("Huddle ended")
                    await self.on_stop(SLACK_CHANNEL)

            except Exception as exc:
                log.warning("Detection error (will retry): %s", exc)

            await asyncio.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False


def make_callbacks():
    from meeting_summary import post_join_notification, post_end_summary
    from session import HuddleSession

    state: dict = {"session": None}

    async def on_start(channel_id: str) -> None:
        session = HuddleSession(channel_id=channel_id)
        state["session"] = session
        await post_join_notification(channel_id)

    async def on_stop(channel_id: str) -> None:
        session = state.get("session")
        if session is None:
            return
        session.end()
        await post_end_summary(session)
        state["session"] = None

    return on_start, on_stop, state


# ------------------------------------------------------------------
# Smoke-test: python huddle_detector.py
# ------------------------------------------------------------------
async def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def on_start(ch): print(f">>> STARTED → {ch}")
    async def on_stop(ch):  print(f">>> STOPPED → {ch}")

    print("Start/stop a Slack huddle to test. Ctrl+C to quit.")
    try:
        await HuddleDetector(on_start, on_stop).run()
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(_smoke_test())
