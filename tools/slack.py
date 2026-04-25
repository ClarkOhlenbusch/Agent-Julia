"""
Post a message to Slack via the Web API.

Required env vars:
  SLACK_BOT_TOKEN   — xoxb-... bot token (must have chat:write scope)
  SLACK_CHANNEL     — default channel ID or name (e.g. #general or C0XXXXXXX)
                      Overridden per-proposal if recipients contains a channel ID.

Usage:
  result = await post_slack(proposal)
"""
import logging
import os

import httpx

from schema import TaskProposal, ToolResult, TaskType

log = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL   = os.environ.get("SLACK_CHANNEL", "#general")
DRY_RUN         = os.environ.get("DRY_RUN", "false").lower() == "true"

_SLACK_API = "https://slack.com/api/chat.postMessage"


async def _post(channel: str, text: str) -> dict:
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type":  "application/json; charset=utf-8",
    }
    payload = {"channel": channel, "text": text}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(_SLACK_API, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


async def post_slack(proposal: TaskProposal) -> ToolResult:
    if not SLACK_BOT_TOKEN:
        return ToolResult(
            success=False,
            tool=TaskType.post_slack,
            message="SLACK_BOT_TOKEN env var not set.",
        )

    # Use first recipient as channel if provided, else fall back to default
    channel = proposal.recipients[0] if proposal.recipients else SLACK_CHANNEL
    text    = proposal.content

    if DRY_RUN:
        log.info("[DRY RUN] Would post to %s: %s", channel, text)
        return ToolResult(
            success=True,
            tool=TaskType.post_slack,
            message=f"[dry-run] Slack → {channel}: \"{text[:80]}\"",
            dry_run=True,
        )

    try:
        data = await _post(channel, text)
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "unknown Slack error"))
        ts = data.get("ts", "")
        log.info("Slack message posted channel=%s ts=%s", channel, ts)
        return ToolResult(
            success=True,
            tool=TaskType.post_slack,
            message=f"Slack message posted to {channel} (ts: {ts})",
        )
    except Exception as exc:
        log.error("Slack API error: %s", exc)
        return ToolResult(
            success=False,
            tool=TaskType.post_slack,
            message=f"Slack error: {exc}",
        )
