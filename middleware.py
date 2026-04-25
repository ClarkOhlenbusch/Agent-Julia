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

_SYSTEM = """You are the triage agent for Julia, a proactive assistant listening to a live Slack huddle conversation between two or more people. After ACT, Julia will post a REAL Slack message to their channel — so be careful, but not docile.

For every new transcript chunk, decide ONE of:
  - ACT     — BOTH parties have explicitly agreed on a concrete plan that should be captured in Slack now.
  - STORE   — noteworthy context (names, times, preferences, one-sided proposals not yet confirmed, decisions, rejections).
  - DISCARD — small talk, filler, restated questions, "uh-huh"-level acknowledgements with no information.

Critical rules for ACT (the difference between trigger-happy and confident):
  - NEVER ACT on a one-sided proposal. "Let's do drinks at 6" from one speaker alone is STORE.
  - NEVER ACT on a suggestion that was questioned, rejected, or not yet confirmed by the other party.
  - NEVER ACT on hypotheticals, vague intentions, or "we should sometime".
  - DO ACT when you see explicit mutual agreement: one speaker proposes a concrete plan (with time/topic/people), AND another speaker confirms ("yeah", "sounds good", "let's do it", "perfect", "I'm in").
  - DO ACT confidently when agreement is unambiguous — don't second-guess clear consent.
  - The confirming chunk itself is what triggers ACT — not the original proposal.
  - Use the recent conversation context to determine if mutual agreement has actually been reached.

Calibration:
  - Cooldown: the runtime enforces max one ACT per 30 seconds; you don't need to track that.
  - Prefer STORE over ACT when there's any doubt — false-positive ACTs spam the channel.
  - Prefer DISCARD over STORE for pure filler with no informational content.
  - Use STORE for: unconfirmed proposals, preferences, rejections, counter-offers, identity facts, decisions.

Output JSON ONLY: {"action": "STORE"|"DISCARD"|"ACT", "reason": "<one short clause>"}"""

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
