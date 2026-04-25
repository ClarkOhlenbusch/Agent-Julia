"""Voice Agent.

Two responsibilities:
1. Compose a confirmation question from a TaskProposal (returns natural language string).
2. Parse the user's next utterance to detect YES / NO / MODIFY.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

from schema import ConfirmationIntent, ConfirmIntent, TaskProposal

VOICE_ENDPOINT = os.getenv("VOICE_ENDPOINT", os.getenv("PLANNER_ENDPOINT", "http://localhost:9002/v1"))
VOICE_MODEL = os.getenv("VOICE_MODEL", os.getenv("PLANNER_MODEL", "planner"))

COMPOSE_SYSTEM = """You are the voice of a proactive assistant. You're going to interject into a conversation.

You'll receive a TaskProposal. Compose a short, natural-sounding question (1-2 sentences,
under 25 words) asking the user to confirm. Tone: warm, conversational, NOT corporate.
Reference the rationale lightly if it adds context. Don't mention internal mechanics
("triage", "memory", "tool"). Just sound like a helpful friend.

Examples:
  - "I see you're both free at 7:30 — want me to put it on the calendar?"
  - "Last time 6pm didn't work for Sam — should I try 7:30 again?"
  - "Want me to send Julie a quick email to loop her in?"

Output ONLY the question text — no JSON, no preamble.
"""

PARSE_SYSTEM = """You are parsing a user's spoken response to a yes/no question from an assistant.

Decide ONE of:
  - YES — clear acceptance ("yeah", "sure", "do it", "perfect", "let's go")
  - NO — clear rejection ("no", "skip it", "nevermind", "let's not")
  - MODIFY — wants a different version ("how about 8 instead?", "make it Friday", "smaller group")
  - UNCLEAR — can't tell or response was off-topic

If MODIFY, extract the modifier as a short string ("8pm instead", "Friday").

Output JSON ONLY: {"intent": "YES"|"NO"|"MODIFY"|"UNCLEAR", "modifier": null|"string"}
"""


def compose_question(proposal: TaskProposal) -> str:
    user_msg = (
        f"TaskProposal:\n"
        f"  task_type: {proposal.task_type.value}\n"
        f"  summary:   {proposal.summary}\n"
        f"  rationale: {proposal.rationale}\n"
        f"  params:    {json.dumps(proposal.parameters)}\n\n"
        f"Compose the question:"
    )
    payload = {
        "model": VOICE_MODEL,
        "messages": [
            {"role": "system", "content": COMPOSE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": 80,
    }
    with httpx.Client(timeout=15) as c:
        r = c.post(f"{VOICE_ENDPOINT}/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip().strip('"')


def parse_response(user_utterance: str, original_question: str) -> ConfirmationIntent:
    user_msg = (
        f"Question I asked: {original_question!r}\n"
        f"User's response:  {user_utterance!r}\n\n"
        f"Classify:"
    )
    payload = {
        "model": VOICE_MODEL,
        "messages": [
            {"role": "system", "content": PARSE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=10) as c:
        r = c.post(f"{VOICE_ENDPOINT}/chat/completions", json=payload)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(raw)
        return ConfirmationIntent(
            intent=ConfirmIntent(data["intent"]),
            modifier=data.get("modifier"),
        )
    except Exception:
        return ConfirmationIntent(intent=ConfirmIntent.UNCLEAR, modifier=None)


if __name__ == "__main__":
    from schema import TaskProposal, TaskType
    p = TaskProposal(
        task_type=TaskType.CREATE_CALENDAR_EVENT,
        summary="Drinks tonight at 7:30 with Alex and Sam at The Black Rose",
        parameters={"title": "Drinks", "start": "2026-04-25T19:30:00-04:00",
                    "end": "2026-04-25T20:30:00-04:00", "attendees": ["alex", "sam"]},
        rationale="Both free at 7:30; Sam's 6pm rejection from earlier in this session is in semantic memory.",
    )
    q = compose_question(p)
    print(f"Q: {q}")
    for utt in ["yeah, go for it", "no thanks", "how about 8 instead?", "what?"]:
        i = parse_response(utt, q)
        print(f"  '{utt}' -> {i.intent.value}  modifier={i.modifier!r}")
