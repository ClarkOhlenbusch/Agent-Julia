"""
Meeting summary generator.

Called when the Slack huddle ends. It:
  1. Calls the agent LLM to generate a concise summary from the session transcript
  2. Formats the summary + action log into a Slack Block Kit message
  3. Posts it to the huddle channel

The summary is intentionally short — 3-5 bullets, not an essay.
"""
import logging
import os
from typing import List

import httpx
from openai import AsyncOpenAI

from config import AGENT_BASE_URL, AGENT_MODEL
from schema import ToolResult
from session import HuddleSession

log = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
DRY_RUN         = os.environ.get("DRY_RUN", "false").lower() == "true"
# Channel is detected automatically from the active huddle — no manual config needed

_client = AsyncOpenAI(base_url=AGENT_BASE_URL, api_key="vllm")

_SUMMARY_SYSTEM = """You are Juliah, a proactive scheduling assistant.
You listened to a team huddle and now need to write a brief meeting summary.
Keep it to 3-5 bullet points. Focus on decisions made, topics discussed, and anything actionable.
Be concise — this goes straight into Slack."""

_SUMMARY_USER = """Here is the full transcript of the huddle:

{transcript}

Write a 3-5 bullet summary of the key points discussed. Use plain text bullets (• )."""


async def _generate_summary(transcript: str) -> str:
    if not transcript.strip():
        return "• No transcript captured."
    try:
        resp = await _client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user",   "content": _SUMMARY_USER.format(transcript=transcript)},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.error("Summary LLM call failed: %s", exc)
        return "• (Summary generation failed — see transcript for details.)"


def _format_actions(actions: List[ToolResult]) -> str:
    if not actions:
        return "_No actions taken._"
    lines = []
    for r in actions:
        status = "✅" if r.success else "❌"
        lines.append(f"{status} 💬 {r.message}")
    return "\n".join(lines)


def _build_blocks(summary: str, actions: List[ToolResult], duration: str) -> list:
    """Build Slack Block Kit payload for the end-of-huddle post."""
    action_text = _format_actions(actions)
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "👋 Julia left the huddle", "emoji": True},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*📝 Meeting Summary*\n{summary}",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*⚡ Actions Taken*\n{action_text}",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Huddle duration: *{duration}* · Powered by Juliah on vLLM + Red Hat AI",
                }
            ],
        },
    ]


async def _post_to_slack(channel: str, blocks: list, fallback_text: str) -> None:
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "channel": channel,
        "text":    fallback_text,
        "blocks":  blocks,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "unknown Slack error"))


async def post_join_notification(channel: str) -> None:
    """Post 'Juliah joined' when the huddle starts."""
    if DRY_RUN:
        log.info("[DRY RUN] Would post Juliah joined to %s", channel)
        return
    if not SLACK_BOT_TOKEN:
        log.warning("SLACK_BOT_TOKEN not set — skipping join notification")
        return
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "👂 *Juliah joined the huddle.* I'm listening and will suggest actions when I sense scheduling intent.",
            },
        }
    ]
    try:
        await _post_to_slack(channel, blocks, "Juliah joined the huddle.")
        log.info("Join notification posted to %s", channel)
    except Exception as exc:
        log.error("Failed to post join notification: %s", exc)


async def post_end_summary(session: HuddleSession) -> None:
    """Generate and post the meeting summary when the huddle ends."""
    log.info("Generating meeting summary for session in %s", session.channel_id)

    summary = await _generate_summary(session.full_transcript)
    blocks  = _build_blocks(summary, session.actions_taken, session.duration_str)
    fallback = f"Juliah left the huddle after {session.duration_str}. Summary posted."

    if DRY_RUN:
        log.info("[DRY RUN] Would post summary to %s:\n%s", session.channel_id, summary)
        for r in session.actions_taken:
            log.info("  action: %s", r.message)
        return

    if not SLACK_BOT_TOKEN:
        log.warning("SLACK_BOT_TOKEN not set — skipping summary post")
        return

    try:
        await _post_to_slack(session.channel_id, blocks, fallback)
        log.info("Meeting summary posted to %s", session.channel_id)
    except Exception as exc:
        log.error("Failed to post meeting summary: %s", exc)
