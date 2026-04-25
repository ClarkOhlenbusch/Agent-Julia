"""Triage / Middleware agent.

Per transcript chunk:  STORE | DISCARD | ACT
Backed by Llama 3.1 8B FP8 at :9001.
Augmented with episodic + semantic memory context.
Uses JSON mode to enforce schema.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

import memory
import observability
from schema import TriageDecision, TriageRoute

TRIAGE_ENDPOINT = os.getenv("TRIAGE_ENDPOINT", "http://localhost:9001/v1")
TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "triage")

SYSTEM_PROMPT = """You are the Triage layer of a proactive scheduling assistant that listens to live conversations between people.

For every new transcript chunk you receive, you must decide ONE of:
  - STORE: noteworthy enough to remember (preferences, decisions, identity), but no immediate action.
  - DISCARD: small talk / chitchat with no lasting value.
  - ACT: BOTH parties have agreed on a concrete plan — time, activity, and mutual confirmation are all present.

Critical rules for ACT:
  - NEVER ACT on a one-sided proposal. "Let's do drinks at 6" from one person is STORE, not ACT.
  - NEVER ACT on a suggestion that was questioned, rejected, or not yet confirmed by the other party.
  - ACT ONLY when you see mutual agreement: one person proposes, AND the other confirms ("yeah that works", "sounds good", "let's do it").
  - The confirming chunk itself is what triggers ACT — not the original proposal.
  - Look at the recent conversation context to determine if agreement has been reached.

Calibration:
  - Prefer STORE over ACT when in doubt — false-positive ACTs interrupt the conversation.
  - Prefer DISCARD over STORE for pure filler ("yeah", "okay", "hmm") with no informational content.
  - Use STORE for: proposals not yet confirmed, preferences, rejections, counter-offers, identity info.
  - confidence reflects how certain you are.

Output JSON ONLY matching:
{ "route": "STORE"|"DISCARD"|"ACT", "confidence": 0.0-1.0, "reason": "short string" }
"""


@observability.task(name="triage")
def decide(transcript_chunk: str, speaker: Optional[str] = None) -> TriageDecision:
    """Run triage on one transcript chunk. Pulls memory context, returns decision."""
    # Pull memory context
    episodic = memory.episodic_search(transcript_chunk, k=3)
    semantic = memory.semantic_search(transcript_chunk, k=5)

    context = []
    if episodic:
        context.append("Recent conversation context:")
        for h in episodic:
            context.append(f"  - [{h.get('speaker', '?')}] ({h.get('age_s', 0)}s ago) {h['text']}")
    if semantic:
        context.append("\nRelevant facts I know:")
        for f in semantic:
            context.append(f"  - {f['text']} (confidence {f['confidence']:.2f})")

    user_msg = (
        ("\n".join(context) + "\n\n" if context else "")
        + f"New transcript chunk from [{speaker or 'unknown'}]: {transcript_chunk!r}\n\n"
        + "Decide STORE | DISCARD | ACT and respond with JSON only."
    )

    payload = {
        "model": TRIAGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=20) as c:
        r = c.post(f"{TRIAGE_ENDPOINT}/chat/completions", json=payload)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]

    try:
        data = json.loads(raw)
        return TriageDecision(**data)
    except Exception as e:
        # Fall back to a safe default — never crash the loop.
        return TriageDecision(
            route=TriageRoute.DISCARD,
            confidence=0.0,
            reason=f"parse_error: {e}; raw={raw[:100]}",
        )


if __name__ == "__main__":
    import sys
    chunk = " ".join(sys.argv[1:]) or "Yo, want to grab drinks at 6 tonight?"
    d = decide(chunk, speaker="alex")
    print(f"route={d.route.value} confidence={d.confidence:.2f} reason={d.reason!r}")
