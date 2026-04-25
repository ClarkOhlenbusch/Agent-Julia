"""
macOS Slack huddle detector.

Polls every POLL_INTERVAL seconds. When a Slack huddle becomes active,
reads the channel name from Slack's window title, resolves it to a channel ID
via the Slack API, then fires callbacks — no manual channel config needed.

  on_start(channel_id)  — huddle just became active
  on_stop(channel_id)   — huddle just ended
"""

import asyncio
import logging
import os
import re
from typing import Callable, Coroutine

import httpx

log = logging.getLogger(__name__)

POLL_INTERVAL   = float(os.environ.get("HUDDLE_POLL_INTERVAL", "2.0"))
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

StartCallback = Callable[[str], Coroutine]
StopCallback  = Callable[[str], Coroutine]

# AppleScript: returns the Slack window title that contains "Huddle", or "".
# Slack's window title during a huddle looks like: "# channel-name | Workspace"
_APPLESCRIPT_HUDDLE_WINDOW = """
tell application "System Events"
    if not (exists process "Slack") then return ""
    tell process "Slack"
        set allWindows to name of every window
        repeat with wName in allWindows
            if wName contains "Huddle" or wName contains "huddle" then
                return wName as string
            end if
        end repeat
    end tell
    return ""
end tell
"""


async def _get_huddle_window_title() -> str:
    """Return the Slack window title if a huddle is active, else empty string."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", _APPLESCRIPT_HUDDLE_WINDOW,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode().strip()
    except (asyncio.TimeoutError, FileNotFoundError):
        return ""


def _parse_channel_name(window_title: str) -> str | None:
    """
    Extract a channel name from a Slack window title.
    Examples:
      "Huddle | # team-standup | Acme Workspace"  → "team-standup"
      "# general – Huddle"                         → "general"
    """
    # Match "# channel-name" anywhere in the title
    m = re.search(r"#\s*([\w\-]+)", window_title)
    return m.group(1) if m else None


async def _resolve_channel_id(channel_name: str) -> str | None:
    """
    Call conversations.list to find the channel ID for a given name.
    Returns None if the bot isn't in the channel or the token is missing.
    """
    if not SLACK_BOT_TOKEN:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://slack.com/api/conversations.list",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                params={"types": "public_channel,private_channel", "limit": 200},
            )
            data = resp.json()
            if not data.get("ok"):
                log.warning("conversations.list error: %s", data.get("error"))
                return None
            for ch in data.get("channels", []):
                if ch.get("name") == channel_name:
                    return ch["id"]
    except Exception as exc:
        log.warning("Could not resolve channel name %r: %s", channel_name, exc)
    return None


async def _detect_active_channel() -> str | None:
    """
    Full detection pipeline:
      1. AppleScript → window title
      2. Parse channel name from title
      3. Resolve name → channel ID via Slack API
    Returns the channel ID string, or None if no huddle detected.
    """
    title = await _get_huddle_window_title()
    if not title:
        return None

    channel_name = _parse_channel_name(title)
    if not channel_name:
        log.debug("Huddle detected but could not parse channel name from: %r", title)
        # Fall back to posting to first channel we find with an active huddle
        return await _fallback_find_huddle_channel()

    channel_id = await _resolve_channel_id(channel_name)
    if channel_id:
        log.debug("Resolved #%s → %s", channel_name, channel_id)
    return channel_id


async def _fallback_find_huddle_channel() -> str | None:
    """
    If window title parsing fails, scan conversations.list for a channel
    that has an active huddle (Slack marks these with has_ongoing_call=True).
    """
    if not SLACK_BOT_TOKEN:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://slack.com/api/conversations.list",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                params={"types": "public_channel,private_channel", "limit": 200},
            )
            data = resp.json()
            for ch in data.get("channels", []):
                if ch.get("has_ongoing_call") or ch.get("is_open"):
                    return ch["id"]
    except Exception as exc:
        log.warning("Fallback huddle channel search failed: %s", exc)
    return None


class HuddleDetector:
    def __init__(self, on_start: StartCallback, on_stop: StopCallback):
        self.on_start        = on_start
        self.on_stop         = on_stop
        self._active_channel: str | None = None
        self._running        = False

    async def run(self) -> None:
        self._running = True
        log.info("Huddle detector polling every %.1fs", POLL_INTERVAL)

        while self._running:
            try:
                channel_id = await _detect_active_channel()

                if channel_id and not self._active_channel:
                    log.info("Huddle started in %s — activating Juliah", channel_id)
                    self._active_channel = channel_id
                    try:
                        await self.on_start(channel_id)
                    except Exception as exc:
                        log.error("on_start failed: %s", exc)

                elif not channel_id and self._active_channel:
                    log.info("Huddle ended in %s — stopping Juliah", self._active_channel)
                    stopped_channel      = self._active_channel
                    self._active_channel = None
                    try:
                        await self.on_stop(stopped_channel)
                    except Exception as exc:
                        log.error("on_stop failed: %s", exc)

            except Exception as exc:
                log.warning("Detection error (will retry): %s", exc)

            await asyncio.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False


def make_callbacks():
    """
    Build on_start / on_stop callbacks that wire the detector into Juliah.
    Channel ID is passed in automatically — no config needed.
    """
    from meeting_summary import post_join_notification, post_end_summary
    from session import HuddleSession

    state: dict = {"session": None}

    async def on_start(channel_id: str) -> None:
        session = HuddleSession(channel_id=channel_id)
        state["session"] = session
        await post_join_notification(channel_id)

    async def on_stop(channel_id: str) -> None:
        session: HuddleSession | None = state.get("session")
        if session is None:
            return
        session.end()
        await post_end_summary(session)
        state["session"] = None

    return on_start, on_stop, state


# ------------------------------------------------------------------
# Standalone smoke-test
# ------------------------------------------------------------------

async def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def on_start(channel_id: str) -> None:
        print(f">>> STARTED in channel {channel_id}")

    async def on_stop(channel_id: str) -> None:
        print(f">>> STOPPED in channel {channel_id}")

    print("Start/stop a Slack huddle to test. Ctrl+C to quit.")
    detector = HuddleDetector(on_start=on_start, on_stop=on_stop)
    try:
        await detector.run()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(_smoke_test())
