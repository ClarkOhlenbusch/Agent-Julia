"""
Slack YES / NO confirmation flow.

Posts a Block Kit message with YES / NO buttons to the huddle channel.
Waits for the user to click.

Two modes:
  - Socket Mode (SLACK_APP_TOKEN set): instant button response via bolt
  - Reaction fallback (no app token): user adds ✅ or ❌ to the message
"""
import asyncio
import logging
import os

import httpx

from schema import TaskProposal, ConfirmationIntent, ConfirmationAction

log = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")
SLACK_CHANNEL   = os.environ.get("SLACK_CHANNEL", "")
CONFIRM_TIMEOUT = int(os.environ.get("CONFIRM_TIMEOUT_SECONDS", "30"))

_pending: dict[str, asyncio.Queue] = {}


def _build_blocks(proposal: TaskProposal) -> list:
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f":thought_balloon: *Julia:* {proposal.voice_prompt}"},
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅  Yes", "emoji": True},
                    "style": "primary",
                    "action_id": "juliah_confirm_yes",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌  No", "emoji": True},
                    "style": "danger",
                    "action_id": "juliah_confirm_no",
                },
            ],
        },
    ]


async def _post_card(proposal: TaskProposal) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={
                "channel": SLACK_CHANNEL,
                "text":    proposal.voice_prompt,
                "blocks":  _build_blocks(proposal),
            },
        )
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error"))
        return data["ts"]


async def _update_card(ts: str, chosen: str) -> None:
    icon = "✅" if chosen == "YES" else "❌"
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            "https://slack.com/api/chat.update",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": SLACK_CHANNEL, "ts": ts, "text": f"{icon} Got it.", "blocks": []},
        )


async def _poll_reactions(ts: str, timeout: int) -> str:
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=10) as client:
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(2)
            try:
                resp = await client.get(
                    "https://slack.com/api/reactions.get",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    params={"channel": SLACK_CHANNEL, "timestamp": ts},
                )
                reactions = {r["name"] for r in resp.json().get("message", {}).get("reactions", [])}
                if "white_check_mark" in reactions:
                    return "YES"
                if "x" in reactions:
                    return "NO"
            except Exception:
                pass
    return "NO"


def register_interaction_handlers(bolt_app) -> None:
    @bolt_app.action("juliah_confirm_yes")
    async def handle_yes(ack, body) -> None:
        await ack()
        ts = body["message"]["ts"]
        if ts in _pending:
            _pending[ts].put_nowait("YES")
        await _update_card(ts, "YES")

    @bolt_app.action("juliah_confirm_no")
    async def handle_no(ack, body) -> None:
        await ack()
        ts = body["message"]["ts"]
        if ts in _pending:
            _pending[ts].put_nowait("NO")
        await _update_card(ts, "NO")


async def ask_confirmation(proposal: TaskProposal) -> ConfirmationIntent:
    ts = await _post_card(proposal)

    if SLACK_APP_TOKEN:
        queue: asyncio.Queue = asyncio.Queue()
        _pending[ts] = queue
        try:
            answer = await asyncio.wait_for(queue.get(), timeout=CONFIRM_TIMEOUT)
        except asyncio.TimeoutError:
            answer = "NO"
        finally:
            _pending.pop(ts, None)
    else:
        answer = await _poll_reactions(ts, CONFIRM_TIMEOUT)

    await _update_card(ts, answer)
    return ConfirmationIntent(action=ConfirmationAction.YES if answer == "YES" else ConfirmationAction.NO)
