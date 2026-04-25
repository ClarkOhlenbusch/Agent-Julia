"""
Julia — main orchestrator (Slack-only).

Lifecycle:
  1. HuddleDetector sees a Slack huddle start on this machine.
  2. Posts "Julia is listening now" to the channel.
  3. Mic capture begins. Each Whisper chunk is triaged.
  4. If ACT: plan a Slack message → post YES/NO card → wait → if YES post it.
  5. Huddle ends → post meeting summary to the channel.

Run:
  python agent.py
"""
import asyncio
import logging
import os

from confirmation import ask_confirmation, register_interaction_handlers
from huddle_detector import HuddleDetector
from meeting_summary import post_end_summary, post_join_notification
from memory import MemoryStore
from middleware import triage
from planner import plan
from schema import TriageAction, ConfirmationAction
from session import HuddleSession
from sub_agent import execute
from transcription import capture_and_transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("julia.agent")

SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")


class JuliaAgent:
    def __init__(self) -> None:
        self._memory      = MemoryStore()
        self._session: HuddleSession | None = None
        self._stop_mic    = asyncio.Event()
        self._bolt_app    = None
        self._bolt_handler = None

    async def on_huddle_start(self, channel_id: str) -> None:
        log.info("Huddle started")
        self._session  = HuddleSession(channel_id=channel_id)
        self._stop_mic = asyncio.Event()
        await post_join_notification(channel_id)
        asyncio.create_task(
            capture_and_transcribe(self._on_transcript, self._stop_mic),
            name="mic_capture",
        )

    async def on_huddle_stop(self, channel_id: str) -> None:
        log.info("Huddle ended")
        self._stop_mic.set()
        if self._session:
            self._session.end()
            await post_end_summary(self._session)
            self._session = None

    async def _on_transcript(self, text: str) -> None:
        if not self._session:
            return

        self._session.add_transcript(text)
        self._memory.write_episodic(text)

        context  = self._memory.context_for(text)
        decision = await triage(text, context)

        if decision.action != TriageAction.ACT:
            return

        try:
            proposal = await plan(text, context)
        except Exception as exc:
            log.error("Planner failed: %s", exc)
            return

        intent = await ask_confirmation(proposal)

        if intent.action == ConfirmationAction.YES:
            result = await execute(proposal, self._memory)
            self._session.add_action(result)
        else:
            self._memory.write_episodic(f"[rejected] {proposal.voice_prompt}")

    async def run(self) -> None:
        if SLACK_APP_TOKEN:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
            self._bolt_app = AsyncApp(token=SLACK_BOT_TOKEN)
            register_interaction_handlers(self._bolt_app)
            self._bolt_handler = AsyncSocketModeHandler(self._bolt_app, SLACK_APP_TOKEN)
            asyncio.create_task(self._bolt_handler.start_async(), name="socket_mode")
            log.info("Socket Mode active — button clicks enabled")
        else:
            log.info("No SLACK_APP_TOKEN — using reaction-poll fallback")

        detector = HuddleDetector(
            on_start=self.on_huddle_start,
            on_stop=self.on_huddle_stop,
        )

        log.info("Julia is ready. Waiting for a Slack huddle…")
        try:
            await detector.run()
        except asyncio.CancelledError:
            pass
        finally:
            self._stop_mic.set()
            if self._bolt_handler:
                await self._bolt_handler.close_async()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(JuliaAgent().run())
