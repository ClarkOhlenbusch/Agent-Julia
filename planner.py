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

_SYSTEM = """You are Julia's planning agent. Scheduling intent has been detected in a Slack huddle.
Draft a Slack message that captures the key information.

Respond ONLY with valid JSON:
{"recipients": [], "content": "<the Slack message Julia will post — clear, friendly, includes who/what/when>", "voice_prompt": "<short question Julia asks before posting, e.g. Should I post a message about tonight drinks?>"}"""

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
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.2,
        )
        proposal = TaskProposal(**json.loads(resp.choices[0].message.content))
        log.info("Plan ready: %r", proposal.voice_prompt)
        return proposal
    except Exception as exc:
        log.error("Planner error: %s", exc)
        raise
