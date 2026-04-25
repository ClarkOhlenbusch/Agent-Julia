"""
Sub-agent: posts the confirmed Slack message and logs the result to session memory.
"""
import logging
import os
from typing import TYPE_CHECKING

import httpx

from schema import TaskProposal, ToolResult

if TYPE_CHECKING:
    from memory import MemoryStore

log = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL   = os.environ.get("SLACK_CHANNEL", "")
DRY_RUN         = os.environ.get("DRY_RUN", "false").lower() == "true"


async def _post(channel: str, text: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": channel, "text": text},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "Slack error"))
        return data["ts"]


async def execute(proposal: TaskProposal, memory: "MemoryStore") -> ToolResult:
    channel = SLACK_CHANNEL

    if DRY_RUN:
        log.info("[DRY RUN] Would post to %s: %s", channel, proposal.content)
        result = ToolResult(success=True, message=f"[dry-run] {proposal.content}", dry_run=True)
    else:
        try:
            ts = await _post(channel, proposal.content)
            log.info("Posted to %s (ts=%s)", channel, ts)
            result = ToolResult(success=True, message=proposal.content)
        except Exception as exc:
            log.error("Post failed: %s", exc)
            result = ToolResult(success=False, message=str(exc))

    status = "posted" if result.success else "failed to post"
    memory.write_episodic(f"[Julia] {status}: {proposal.content}")
    return result
