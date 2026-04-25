"""Sub-Agent — executes the chosen task tool.

Dispatches a confirmed TaskProposal to the matching tool function.
Logs the outcome back to episodic memory so the agent remembers what it did.
"""
from __future__ import annotations

import json

import memory
import observability
from schema import TaskProposal, TaskType, TaskResult, Fact, FactType
from tools import calendar as cal_tool
from tools import email as email_tool
from tools import slack as slack_tool


@observability.agent(name="sub_agent")
def execute(proposal: TaskProposal) -> TaskResult:
    """Dispatch and run."""
    params = proposal.parameters
    try:
        if proposal.task_type == TaskType.CREATE_CALENDAR_EVENT:
            result = cal_tool.book_meeting(
                title=params.get("title", "(untitled)"),
                start=params["start"],
                end=params["end"],
                attendees=params.get("attendees", []),
                location=params.get("location"),
                description=params.get("description"),
            )
        elif proposal.task_type == TaskType.SEND_EMAIL:
            result = email_tool.send_email(
                to=params["to"],
                subject=params.get("subject", "(no subject)"),
                body=params.get("body", ""),
                cc=params.get("cc"),
            )
        elif proposal.task_type == TaskType.POST_SLACK:
            result = slack_tool.post_slack(
                channel=params["channel"],
                message=params["message"],
                mentions=params.get("mentions"),
            )
        else:
            result = TaskResult(status="failed",
                                message=f"unknown task_type: {proposal.task_type}")
    except KeyError as e:
        result = TaskResult(status="failed", message=f"missing required param: {e}")
    except Exception as e:
        result = TaskResult(status="failed", message=f"tool error: {e}")

    # Log decision to episodic memory + drop a learned-fact about the decision
    log_text = (
        f"[ACTED] {proposal.task_type.value} → {result.status}: "
        f"{proposal.summary} (artifact={result.artifact_id})"
    )
    memory.episodic_write(log_text, speaker="agent")

    if result.status in ("booked", "sent", "posted"):
        # Distill a small fact: this combo of attendees did this kind of task today
        memory.semantic_write([
            Fact(
                subject=", ".join(params.get("attendees", [params.get("to", "user")])),
                type=FactType.DECISION,
                fact=f"agreed to {proposal.task_type.value}: {proposal.summary[:140]}",
                confidence=0.85,
            )
        ])
    return result


def execute_rejection(proposal: TaskProposal, modifier: str | None = None) -> None:
    """User said NO. Log so we don't propose the same thing again immediately."""
    log = (
        f"[REJECTED] {proposal.task_type.value}: {proposal.summary}"
        + (f" (user wanted: {modifier})" if modifier else "")
    )
    memory.episodic_write(log, speaker="agent")
    memory.semantic_write([
        Fact(
            subject=", ".join(proposal.parameters.get("attendees", ["user"])),
            type=FactType.DECISION,
            fact=f"rejected proposal: {proposal.summary[:120]}"
                 + (f"; preferred: {modifier}" if modifier else ""),
            confidence=0.75,
        )
    ])


if __name__ == "__main__":
    p = TaskProposal(
        task_type=TaskType.CREATE_CALENDAR_EVENT,
        summary="Drinks at 7:30 with Alex and Sam",
        parameters={"title": "Drinks", "start": "2026-04-25T19:30:00-04:00",
                    "end": "2026-04-25T20:30:00-04:00",
                    "attendees": ["alex", "sam"], "location": "The Black Rose"},
        rationale="Both free at 7:30",
    )
    print(execute(p))
