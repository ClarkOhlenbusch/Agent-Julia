"""Planner agent.

Given a snippet of conversation flagged ACT, produce a TaskProposal.
Backed by Mistral Small 3.2 24B FP8 at :9002.
Uses semantic memory for personalization.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

import memory
import observability
from schema import TaskProposal, TaskType
from tools import calendar as cal_tool

PLANNER_ENDPOINT = os.getenv("PLANNER_ENDPOINT", "http://localhost:9002/v1")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "planner")

SYSTEM_PROMPT = """You are the Planner of a proactive assistant.

You will receive recent conversation transcript chunks and learned facts about
the participants. Decide what action would help most and produce a TaskProposal.

Choose ONE task_type:
  - send_email: clear ask for email contact / reminder.
  - post_slack: workplace-style ask, channel mentioned.
  - create_calendar_event: meeting/drinks/lunch — pick a concrete time.

For create_calendar_event, USE the calendar tool's find_overlap() (you'll see free slots in the context).
Bias proposals using the learned facts:
  - If a fact says "X prefers afternoons" → don't propose 9am.
  - If a fact says "Y rejected 6pm earlier" → propose 7pm or later.
  - If a fact says "Z is vegetarian" → don't suggest a steakhouse.

Output JSON ONLY matching this schema:
{
  "task_type": "send_email" | "post_slack" | "create_calendar_event",
  "summary": "one sentence the assistant will say aloud",
  "parameters": { task-specific args },
  "rationale": "short reason — reference learned facts when relevant"
}

Calendar parameters MUST include: title, start (ISO), end (ISO), attendees (list of names).
Email parameters MUST include: to, subject, body.
Slack parameters MUST include: channel, message.
"""


@observability.task(name="plan")
def plan(recent_transcript: str, attendees_hint: Optional[list[str]] = None) -> TaskProposal:
    semantic = memory.semantic_search(recent_transcript, k=8)
    episodic = memory.episodic_search(recent_transcript, k=5)

    candidates = []
    if attendees_hint:
        candidates = cal_tool.find_overlap(attendees_hint, duration_min=60)

    context_lines = []
    if episodic:
        context_lines.append("Recent conversation:")
        for h in episodic:
            context_lines.append(f"  - [{h.get('speaker', '?')}] {h['text']}")
    if semantic:
        context_lines.append("\nLearned facts:")
        for f in semantic:
            context_lines.append(f"  - {f['text']} (conf {f['confidence']:.2f})")
    if candidates:
        context_lines.append("\nCandidate calendar slots (ALL attendees free):")
        for slot in candidates[:5]:
            context_lines.append(f"  - {slot['start']} → {slot['end']} ({slot['duration_min']}m)")

    user_msg = (
        "\n".join(context_lines)
        + f"\n\nLatest snippet to act on: {recent_transcript!r}\n\n"
        + "Produce a TaskProposal as JSON."
    )

    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.4,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=30) as c:
        r = c.post(f"{PLANNER_ENDPOINT}/chat/completions", json=payload)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]

    try:
        data = json.loads(raw)
        # Coerce task_type to enum if needed
        if isinstance(data.get("task_type"), str):
            data["task_type"] = TaskType(data["task_type"])
        return TaskProposal(**data)
    except Exception as e:
        # Safe fallback — propose nothing if parsing fails.
        return TaskProposal(
            task_type=TaskType.CREATE_CALENDAR_EVENT,
            summary="(parse error) please rephrase",
            parameters={},
            rationale=f"parse_error: {e}; raw={raw[:120]}",
        )


if __name__ == "__main__":
    import sys
    snippet = " ".join(sys.argv[1:]) or "Yo, drinks tonight? When are you free?"
    p = plan(snippet, attendees_hint=["alex", "sam"])
    print(f"task={p.task_type.value} summary={p.summary!r}")
    print(f"params={p.parameters}")
    print(f"rationale={p.rationale!r}")
