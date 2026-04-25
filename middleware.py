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

_SYSTEM = """You are the triage agent for Julia, a proactive assistant listening to a live conversation. After ACT, Julia will post a REAL Slack message and speak aloud, so the call matters — but the bigger failure mode right now is being too cautious. If a clear agreement just happened, you must ACT.

For every new transcript chunk, decide ONE of:

  - ACT     — The current chunk contains explicit agreement language ("sounds good", "let's do it", "perfect", "yeah let's", "alright", "I'm in", "let's go", "deal", "works for me") AND a concrete plan was mentioned in recent context (time, place, or activity). This means a confirmation just happened and Julia should capture it.
  - STORE   — A proposal, time, name, preference, or decision was mentioned but no confirmation in this chunk. Things worth remembering.
  - DISCARD — Pure filler with no scheduling content: greetings, off-topic chatter, partial mid-sentence cuts, "uh-huh", "right".

When the same speaker label keeps appearing (the input is a single mic stream), do not assume it's all one person — multiple people can be speaking through the same channel. Trust the WORDS, not the speaker tag.

Critical rules for ACT (be confident, not docile):
  - DO ACT when the current chunk contains clear agreement language after any proposal in recent context. Even if the proposal came from the same speaker tag.
  - DO ACT on phrases like: "yeah let's do it", "sounds good", "perfect, do it", "let's go", "all right, let's", "alright bet", "deal", "works for me", "I'm down", "down for that".
  - The confirming chunk itself triggers ACT — not the original proposal.
  - DO NOT delay ACT waiting for further confirmation if the speakers already said yes.

Critical rules for STORE / DISCARD:
  - NEVER use "repeated", "restated", "already confirmed", or "already happened" as your reason. Deduplication is the runtime's job (a 30-second cooldown is enforced after every ACT). Always classify based on the chunk's own content.
  - STORE goes to: unconfirmed proposals, preferences, names, times mentioned, plans being negotiated.
  - DISCARD goes to: pure filler with no scheduling-relevant content.

When in doubt:
  - Doubt between STORE and ACT, with explicit agreement language present → choose ACT.
  - Doubt between STORE and DISCARD → choose STORE.

Output JSON ONLY: {"action": "STORE"|"DISCARD"|"ACT", "reason": "<one short clause that does NOT use the word 'repeated' or 'restated'>"}"""

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
