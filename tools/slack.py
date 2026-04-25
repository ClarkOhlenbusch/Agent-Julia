"""
Post a message to Slack via the Web API.

Used by agents/sub_agent.py (team pipeline) and our sub_agent.py (huddle pipeline).

Required env vars:
  SLACK_BOT_TOKEN  — xoxb-... bot token
  SLACK_CHANNEL    — default channel ID
"""
import asyncio
import logging
import os

import httpx

log = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL   = os.environ.get("SLACK_CHANNEL", "")
DRY_RUN         = os.environ.get("DRY_RUN", "false").lower() == "true"


async def post_message(channel: str, text: str) -> str:
    """Post a message, return the message ts."""
    if DRY_RUN:
        log.info("[DRY RUN] Slack → %s: %s", channel, text)
        return "dry_run"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": channel, "text": text},
        )
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "Slack error"))
        return data["ts"]


def post_slack(channel: str, message: str, mentions: list[str] | None = None) -> dict:
    """Sync wrapper kept for compatibility with agents/sub_agent.py."""
    full_msg = message
    if mentions:
        full_msg = " ".join(f"<@{m}>" for m in mentions) + " " + message
    channel = channel or SLACK_CHANNEL
    if DRY_RUN:
        log.info("[DRY RUN] Slack → %s: %s", channel, full_msg)
        return {"status": "posted", "message": full_msg}
    try:
        ts = asyncio.run(post_message(channel, full_msg))
        return {"status": "posted", "message": full_msg, "ts": ts}
    except Exception as exc:
        log.error("Slack post failed: %s", exc)
        return {"status": "failed", "message": str(exc)}
