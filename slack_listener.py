"""
Slack Socket Mode listener.

Starts the Jarvis transcription pipeline automatically when a Slack huddle
begins in a monitored channel. Also responds to the /jarvis slash command
as a manual fallback trigger.

Required env vars:
  SLACK_BOT_TOKEN   — xoxb-...  (bot token, chat:write + channels:read scopes)
  SLACK_APP_TOKEN   — xapp-...  (app-level token, connections:write scope)
  SLACK_WATCH_CHANNEL — channel ID to monitor (e.g. C0XXXXXXX); if blank, watches all

Slack app setup checklist:
  1. Enable Socket Mode in your app settings
  2. Add slash command /jarvis pointing to your app
  3. Subscribe to events: huddle_started, huddle_ended (under Event Subscriptions)
  4. Bot token scopes: channels:read, chat:write, commands
  5. App token scope: connections:write
"""
import asyncio
import logging
import os
from typing import Callable, Coroutine

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

log = logging.getLogger(__name__)

SLACK_BOT_TOKEN    = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN    = os.environ.get("SLACK_APP_TOKEN", "")
SLACK_WATCH_CHANNEL = os.environ.get("SLACK_WATCH_CHANNEL", "")

# Callback type: async fn(channel_id, trigger_reason) → None
StartCallback = Callable[[str, str], Coroutine]
StopCallback  = Callable[[str, str], Coroutine]


class SlackHuddleListener:
    """
    Connects to Slack via Socket Mode and fires callbacks when a huddle
    starts or ends in the watched channel.
    """

    def __init__(
        self,
        on_start: StartCallback,
        on_stop:  StopCallback,
    ):
        self.on_start = on_start
        self.on_stop  = on_stop
        self._app     = AsyncApp(token=SLACK_BOT_TOKEN)
        self._handler: AsyncSocketModeHandler | None = None
        self._active_channel: str | None = None

        self._register_handlers()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        app = self._app

        # ── Automatic trigger: huddle_started event ────────────────────
        @app.event("huddle_started")
        async def handle_huddle_started(event: dict, say) -> None:
            channel = event.get("channel_id") or event.get("channel", "")
            if not self._should_watch(channel):
                return
            log.info("Huddle started in %s — activating Jarvis", channel)
            self._active_channel = channel
            await self.on_start(channel, "huddle_started")

        # ── Automatic trigger: huddle_ended event ──────────────────────
        @app.event("huddle_ended")
        async def handle_huddle_ended(event: dict) -> None:
            channel = event.get("channel_id") or event.get("channel", "")
            if not self._should_watch(channel):
                return
            log.info("Huddle ended in %s — stopping Jarvis", channel)
            self._active_channel = None
            await self.on_stop(channel, "huddle_ended")

        # ── Manual fallback: /jarvis slash command ─────────────────────
        @app.command("/jarvis")
        async def handle_slash_jarvis(ack, body, say) -> None:
            await ack()
            channel  = body.get("channel_id", "")
            sub_cmd  = (body.get("text") or "").strip().lower()

            if sub_cmd in ("stop", "off", "quit"):
                log.info("/jarvis stop in %s", channel)
                self._active_channel = None
                await say("Jarvis stopped listening.")
                await self.on_stop(channel, "slash_command_stop")
            else:
                log.info("/jarvis start in %s", channel)
                self._active_channel = channel
                await say(
                    ":ear: Jarvis is now listening. I'll suggest actions when I "
                    "sense scheduling intent. Say `/jarvis stop` to pause."
                )
                await self.on_start(channel, "slash_command_start")

        # ── Catch-all for unhandled events (keeps Socket Mode happy) ───
        @app.event("message")
        async def handle_message(event: dict) -> None:
            pass  # transcription happens outside Slack; we just need the trigger

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_watch(self, channel: str) -> bool:
        if not SLACK_WATCH_CHANNEL:
            return True  # watch every channel the bot is in
        return channel == SLACK_WATCH_CHANNEL

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not SLACK_APP_TOKEN:
            raise EnvironmentError(
                "SLACK_APP_TOKEN (xapp-...) is required for Socket Mode. "
                "Enable Socket Mode in your Slack app settings and generate an app token."
            )
        self._handler = AsyncSocketModeHandler(self._app, SLACK_APP_TOKEN)
        log.info("Connecting to Slack via Socket Mode…")
        await self._handler.start_async()

    async def stop(self) -> None:
        if self._handler:
            await self._handler.close_async()
            log.info("Slack Socket Mode handler closed.")

    @property
    def active_channel(self) -> str | None:
        return self._active_channel


# ------------------------------------------------------------------
# Standalone smoke-test
# ------------------------------------------------------------------

async def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO)

    async def on_start(channel: str, reason: str) -> None:
        print(f"[START] channel={channel} reason={reason}")

    async def on_stop(channel: str, reason: str) -> None:
        print(f"[STOP]  channel={channel} reason={reason}")

    listener = SlackHuddleListener(on_start=on_start, on_stop=on_stop)
    print("Listening for huddle events and /jarvis commands. Ctrl+C to quit.")
    await listener.start()  # blocks until interrupted


if __name__ == "__main__":
    asyncio.run(_smoke_test())
