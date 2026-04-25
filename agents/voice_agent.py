"""Voice Agent.

Composes the natural-language line the assistant speaks aloud.

The new pipeline uses Slack-shaped TaskProposal (recipients, content,
voice_prompt) and ToolResult (success, message, dry_run). The narrator turns
those into a short past-tense confirmation suitable for TTS.

Older legacy schema (TaskProposal with task_type/parameters/summary/rationale)
is no longer present in schema.py — the helpers below detect either shape.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

import observability
from schema import TaskProposal, ToolResult

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


@observability.agent(name="voice_question")
def compose_question(proposal: TaskProposal) -> str:
    user_msg = (
        f"TaskProposal:\n"
        f"  content:      {proposal.content}\n"
        f"  voice_prompt: {proposal.voice_prompt}\n"
        f"  recipients:   {proposal.recipients}\n\n"
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


@observability.agent(name="voice_narration")
def compose_narration_for_slack(proposal: TaskProposal,
                                result: Optional[ToolResult] = None) -> str:
    """Past-tense confirmation about a Slack post that just happened."""
    result_block = ""
    if result is not None:
        status_word = "posted" if result.success else "failed"
        result_block = (
            f"  result_status: {status_word}\n"
            f"  result_message: {result.message}\n"
            f"  dry_run: {result.dry_run}\n"
        )
    user_msg = (
        f"Slack action just executed:\n"
        f"  posted_content: {proposal.content}\n"
        f"  voice_prompt:   {proposal.voice_prompt}\n"
        f"  recipients:     {proposal.recipients}\n"
        f"{result_block}"
        f"\nCompose the narration:"
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


if __name__ == "__main__":
    from schema import TaskProposal, ToolResult
    p = TaskProposal(
        recipients=[],
        content="Drinks at The Black Rose, 7:30 PM tonight with Alex and Sam.",
        voice_prompt="Should I post tonight's drinks plan to the channel?",
    )
    r = ToolResult(success=True, message="posted to #huddle", dry_run=False)
    print("NARRATION:", compose_narration_for_slack(p, r))
    print("QUESTION: ", compose_question(p))
