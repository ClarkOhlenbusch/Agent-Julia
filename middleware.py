"""
Triage — decides whether Julia should act on a transcript chunk.

Calls the Triage LLM (Llama 3.1 8B FP8) with grammar-guided JSON output
so the model literally cannot return anything except a valid TriageDecision.

Output: STORE | DISCARD | ACT
"""
import json
import logging
import time

from openai import AsyncOpenAI

from config import TRIAGE_BASE_URL, TRIAGE_MODEL, TRIAGE_COOLDOWN_SECONDS
from schema import TriageDecision, TriageAction

log = logging.getLogger(__name__)

_client = AsyncOpenAI(base_url=TRIAGE_BASE_URL, api_key="vllm")

_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["STORE", "DISCARD", "ACT"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}

_SYSTEM = """You are the triage agent for Julia, a proactive scheduling assistant listening to a Slack huddle.

Given a transcript chunk and conversation context, decide:

ACT   — clear scheduling intent: someone wants to book a meeting, send a message, create an invite, or follow up via email.
STORE — relevant context (names, times, preferences) but no immediate action needed.
DISCARD — small talk, filler words, or nothing actionable.

Rules:
- Only choose ACT when intent is explicit and actionable right now.
- One ACT per 30 seconds maximum — do not spam.
- If unsure, choose STORE."""

_last_act_time: float = 0.0


async def triage(transcript: str, context: str = "") -> TriageDecision:
    global _last_act_time

    prompt = transcript
    if context:
        prompt = f"Context:\n{context}\n\nNew transcript:\n{transcript}"

    try:
        resp = await _client.chat.completions.create(
            model=TRIAGE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            extra_body={"guided_json": _TRIAGE_SCHEMA},
            max_tokens=120,
            temperature=0.1,
        )
        decision = TriageDecision(**json.loads(resp.choices[0].message.content))
    except Exception as exc:
        log.warning("Triage LLM error, defaulting to STORE: %s", exc)
        return TriageDecision(action=TriageAction.STORE, reason="triage error")

    # Enforce cooldown — prevent back-to-back ACTs
    if decision.action == TriageAction.ACT:
        now = time.monotonic()
        if now - _last_act_time < TRIAGE_COOLDOWN_SECONDS:
            log.debug("ACT suppressed by cooldown (%.0fs remaining)",
                      TRIAGE_COOLDOWN_SECONDS - (now - _last_act_time))
            return TriageDecision(action=TriageAction.STORE, reason="cooldown active")
        _last_act_time = now

    log.info("Triage → %s | %s", decision.action, decision.reason)
    return decision
