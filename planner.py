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

    # Two-strategy approach: prefer guided_json, fall back to json_object.
    last_err: Exception | None = None
    for params in (
        {"extra_body": {"guided_json": _PROPOSAL_SCHEMA}},
        {"response_format": {"type": "json_object"}},
    ):
        try:
            resp = await _client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": prompt + '\n\nReturn JSON only matching: {"recipients": [], "content": str, "voice_prompt": str}'},
                ],
                max_tokens=400,
                temperature=0.2,
                **params,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                raise ValueError("planner returned empty content")
            # Strip code fences if the model wrapped JSON in ```json ... ```
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].lstrip()
            proposal = TaskProposal(**json.loads(raw))
            log.info("Plan ready: %r", proposal.voice_prompt)
            return proposal
        except Exception as exc:
            last_err = exc
            log.warning("Planner attempt failed (%s); trying fallback…", exc)

    log.error("Planner exhausted retries: %s", last_err)
    raise last_err  # type: ignore[misc]
