"""Voice Agent.

Composes the natural-language line the assistant speaks aloud.

Two modes:
- compose_narration(proposal, result) — used after the agent has already acted on
  detected mutual agreement. Past-tense confirmation. The default in the new flow.
- compose_question(proposal) — used only when there's true ambiguity (multiple
  recipients, unclear time). Kept around for future routing.

(parse_response is retained for code paths that still ask follow-up questions.)
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

from schema import ConfirmationIntent, ConfirmIntent, TaskProposal, TaskResult

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

NARRATION_SYSTEM = """You are the voice of a proactive assistant. The two parties in the
conversation just reached mutual agreement on a plan, and you've already executed it.
Speak a SHORT, friendly past-tense confirmation (1 sentence, under 20 words).

Tone: warm, helpful, not corporate. Sound like a friend who just took care of something.
Don't ask for permission — they already agreed; you've done it.
Don't mention internal mechanics ("triage", "tool", "scheduled it for both calendars").
Spell out times in words when natural ("seven thirty" reads better than "7:30").

Examples:
  - "Got it — drinks at seven thirty are on both your calendars."
  - "Done. I sent that email to Julie."
  - "Booked. Calendar invite's out."
  - "All set — meeting's on the books."

Output ONLY the narration text — no JSON, no preamble.
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


def compose_narration(proposal: TaskProposal, result: Optional[TaskResult] = None) -> str:
    """Past-tense confirmation, used after the agent has already acted."""
    result_block = ""
    if result is not None:
        result_block = f"  result_status: {result.status}\n  result_message: {result.message}\n"
    user_msg = (
        f"TaskProposal:\n"
        f"  task_type: {proposal.task_type.value}\n"
        f"  summary:   {proposal.summary}\n"
        f"  rationale: {proposal.rationale}\n"
        f"  params:    {json.dumps(proposal.parameters)}\n"
        f"{result_block}"
        f"\nThe action has already been executed. Compose the narration:"
    )
    payload = {
        "model": VOICE_MODEL,
        "messages": [
            {"role": "system", "content": NARRATION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.6,
        "max_tokens": 60,
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
    from schema import TaskProposal, TaskType, TaskResult
    p = TaskProposal(
        task_type=TaskType.CREATE_CALENDAR_EVENT,
        summary="Drinks tonight at 7:30 with Alex and Sam at The Black Rose",
        parameters={"title": "Drinks", "start": "2026-04-25T19:30:00-04:00",
                    "end": "2026-04-25T20:30:00-04:00", "attendees": ["alex", "sam"]},
        rationale="Both confirmed 7:30 works after agreeing in the conversation.",
    )
    r = TaskResult(status="booked", message="Drinks with Alex and Sam at 7:30 PM",
                   artifact_id="cal_demo123")
    print(f"NARRATION: {compose_narration(p, r)!r}")
    print(f"QUESTION:  {compose_question(p)!r}")
