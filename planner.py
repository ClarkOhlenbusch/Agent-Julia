"""
Planner — turns a detected scheduling intent into a Slack message proposal.

Everything stays in Slack. The planner crafts the message Julia will post
to the channel after the user confirms.
"""
import json
import logging

from openai import AsyncOpenAI

from config import AGENT_BASE_URL, AGENT_MODEL
from schema import TaskProposal

log = logging.getLogger(__name__)

_client = AsyncOpenAI(base_url=AGENT_BASE_URL, api_key="vllm")

_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "recipients":   {"type": "array", "items": {"type": "string"}},
        "content":      {"type": "string"},
        "voice_prompt": {"type": "string"},
    },
    "required": ["recipients", "content", "voice_prompt"],
    "additionalProperties": False,
}

_SYSTEM = """You are Julia's planning agent. Scheduling intent has been detected in a Slack huddle.
Your job is to draft a Slack message that captures the key information and posts it to the channel.

content: the Slack message Julia will post (clear, friendly, includes the key details — who, what, when).
voice_prompt: a short question Julia asks before posting, e.g. "Should I post a message about tonight's drinks?"
recipients: leave as an empty list — Julia will post to the huddle channel automatically."""

_USER_TMPL = """{context}

Transcript with scheduling intent:
{transcript}

Draft the Slack message."""


async def plan(transcript: str, context: str = "") -> TaskProposal:
    prompt = _USER_TMPL.format(context=context, transcript=transcript).strip()

    try:
        resp = await _client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            extra_body={"guided_json": _PROPOSAL_SCHEMA},
            max_tokens=300,
            temperature=0.2,
        )
        proposal = TaskProposal(**json.loads(resp.choices[0].message.content))
        log.info("Plan ready: %r", proposal.voice_prompt)
        return proposal
    except Exception as exc:
        log.error("Planner error: %s", exc)
        raise
