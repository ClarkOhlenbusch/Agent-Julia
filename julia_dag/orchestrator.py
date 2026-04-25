from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from typing import Any, Callable

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import agent, task

from .config import settings
from .schemas import ChannelName
from .schemas import ChannelResult
from .schemas import InvokeRequest
from .schemas import InvokeResponse
from .schemas import PlannedAction


EMAIL_DIRECT_HINTS = ("email", "mail", "inbox")
EMAIL_ACTION_PHRASES = ("send email", "send an email", "reply to email", "reply by email")
SLACK_DIRECT_HINTS = ("slack", "dm", "channel")
SLACK_ACTION_PHRASES = ("slack message", "post in slack", "reply in slack", "notify in slack", "ping in slack")
CALENDAR_DIRECT_HINTS = ("calendar",)
CALENDAR_ACTION_HINTS = ("schedule", "reschedule", "invite")
CALENDAR_OBJECT_HINTS = ("meeting", "event", "calendar invite", "invite")


def ensure_llmobs_enabled() -> None:
    if LLMObs.enabled or not settings.dd_llmobs_enabled:
        return

    enable_kwargs: dict[str, Any] = {"ml_app": settings.dd_ml_app}
    if settings.dd_api_key:
        enable_kwargs["agentless_enabled"] = settings.dd_agentless_enabled or True
    if settings.dd_api_key:
        enable_kwargs["api_key"] = settings.dd_api_key
    if settings.dd_app_key:
        enable_kwargs["app_key"] = settings.dd_app_key
    if settings.dd_site:
        enable_kwargs["site"] = settings.dd_site
    if settings.dd_env:
        enable_kwargs["env"] = settings.dd_env
    if settings.dd_service:
        enable_kwargs["service"] = settings.dd_service
    LLMObs.enable(**enable_kwargs)


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase in text


def _contains_token(text: str, token: str) -> bool:
    pattern = rf"\b{re.escape(token)}\b"
    return re.search(pattern, text) is not None


def _matches_direct_hint(text: str, hints: tuple[str, ...]) -> bool:
    return any(_contains_token(text, hint) for hint in hints)


def _is_email_instruction(text: str) -> bool:
    if _matches_direct_hint(text, EMAIL_DIRECT_HINTS):
        return True
    return any(_contains_phrase(text, phrase) for phrase in EMAIL_ACTION_PHRASES)


def _is_slack_instruction(text: str) -> bool:
    if _matches_direct_hint(text, SLACK_DIRECT_HINTS):
        return True
    return any(_contains_phrase(text, phrase) for phrase in SLACK_ACTION_PHRASES)


def _is_calendar_instruction(text: str) -> bool:
    if _matches_direct_hint(text, CALENDAR_DIRECT_HINTS):
        return True
    if any(_contains_token(text, verb) for verb in CALENDAR_ACTION_HINTS):
        return True
    return (
        any(_contains_token(text, obj) for obj in CALENDAR_OBJECT_HINTS)
        and any(_contains_token(text, verb) for verb in ("schedule", "reschedule", "create", "move", "update"))
    )


@dataclass
class Node:
    name: str
    dependencies: tuple[str, ...]
    func: Callable[[dict[str, Any]], Any]


class SimpleDAG:
    def __init__(self, nodes: list[Node]) -> None:
        self._nodes = {node.name: node for node in nodes}

    def run(self, initial_context: dict[str, Any]) -> dict[str, Any]:
        context = dict(initial_context)
        indegree: dict[str, int] = {}
        graph: dict[str, list[str]] = defaultdict(list)

        for node in self._nodes.values():
            indegree[node.name] = len(node.dependencies)
            for dependency in node.dependencies:
                graph[dependency].append(node.name)

        queue = deque(sorted(name for name, degree in indegree.items() if degree == 0))
        execution_order: list[str] = []

        while queue:
            node_name = queue.popleft()
            execution_order.append(node_name)
            node = self._nodes[node_name]
            context[node_name] = node.func(context)

            for downstream in graph[node_name]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    queue.append(downstream)

        if len(execution_order) != len(self._nodes):
            raise ValueError("DAG contains a cycle and cannot be executed.")

        context["trace_steps"] = execution_order
        return context


@task(name="normalize_instruction")
def normalize_instruction(request: InvokeRequest) -> str:
    return " ".join(request.instruction.strip().lower().split())


@task(name="plan_channels")
def plan_channels(normalized_instruction: str) -> list[PlannedAction]:
    plans: list[PlannedAction] = []

    channel_checks: list[tuple[ChannelName, Callable[[str], bool], str]] = [
        ("email", _is_email_instruction, "email-specific language detected"),
        ("slack", _is_slack_instruction, "slack or chat-specific language detected"),
        ("calendar", _is_calendar_instruction, "calendar or scheduling language detected"),
    ]

    for channel, matcher, reason in channel_checks:
        matched = matcher(normalized_instruction)
        plans.append(
            PlannedAction(
                channel=channel,
                should_run=matched,
                confidence=0.92 if matched else 0.18,
                reason=reason if matched else f"no strong {channel} keyword match found",
            )
        )

    if any(plan.should_run for plan in plans):
        return plans

    return [
        PlannedAction(
            channel="slack",
            should_run=True,
            confidence=0.34,
            reason="defaulted to slack because the instruction looked conversational but not domain-specific",
        ),
        PlannedAction(
            channel="email",
            should_run=False,
            confidence=0.2,
            reason="no strong email keyword match found",
        ),
        PlannedAction(
            channel="calendar",
            should_run=False,
            confidence=0.2,
            reason="no strong calendar keyword match found",
        ),
    ]


@task(name="select_channels")
def select_channels(plan: list[PlannedAction]) -> list[ChannelName]:
    return [planned.channel for planned in plan if planned.should_run]


def _planned_action(plan: list[PlannedAction], channel: ChannelName) -> PlannedAction:
    for planned in plan:
        if planned.channel == channel:
            return planned
    raise KeyError(f"Missing planned action for {channel}")


@agent(name="email_specialist")
def run_email_agent(normalized_instruction: str, plan: list[PlannedAction]) -> ChannelResult:
    planned = _planned_action(plan, "email")
    if not planned.should_run:
        return ChannelResult(
            channel="email",
            action="noop",
            status="skipped",
            detail="Email model not selected for this instruction.",
            model_stub="email-agent-stub",
        )

    return ChannelResult(
        channel="email",
        action="draft_email",
        status="planned",
        detail=f"Email model should prepare a draft or reply based on: {normalized_instruction}",
        model_stub="email-agent-stub",
    )


@agent(name="slack_specialist")
def run_slack_agent(normalized_instruction: str, plan: list[PlannedAction]) -> ChannelResult:
    planned = _planned_action(plan, "slack")
    if not planned.should_run:
        return ChannelResult(
            channel="slack",
            action="noop",
            status="skipped",
            detail="Slack model not selected for this instruction.",
            model_stub="slack-agent-stub",
        )

    return ChannelResult(
        channel="slack",
        action="compose_slack_message",
        status="planned",
        detail=f"Slack model should compose a channel post or DM for: {normalized_instruction}",
        model_stub="slack-agent-stub",
    )


@agent(name="calendar_specialist")
def run_calendar_agent(normalized_instruction: str, plan: list[PlannedAction]) -> ChannelResult:
    planned = _planned_action(plan, "calendar")
    if not planned.should_run:
        return ChannelResult(
            channel="calendar",
            action="noop",
            status="skipped",
            detail="Calendar model not selected for this instruction.",
            model_stub="calendar-agent-stub",
        )

    return ChannelResult(
        channel="calendar",
        action="schedule_calendar_change",
        status="planned",
        detail=f"Calendar model should create or update an event for: {normalized_instruction}",
        model_stub="calendar-agent-stub",
    )


def build_dag(request: InvokeRequest) -> SimpleDAG:
    return SimpleDAG(
        nodes=[
            Node("normalized_instruction", tuple(), lambda _: normalize_instruction(request)),
            Node(
                "plan",
                ("normalized_instruction",),
                lambda context: plan_channels(context["normalized_instruction"]),
            ),
            Node("selected_channels", ("plan",), lambda context: select_channels(context["plan"])),
            Node(
                "email_result",
                ("normalized_instruction", "plan"),
                lambda context: run_email_agent(context["normalized_instruction"], context["plan"]),
            ),
            Node(
                "slack_result",
                ("normalized_instruction", "plan"),
                lambda context: run_slack_agent(context["normalized_instruction"], context["plan"]),
            ),
            Node(
                "calendar_result",
                ("normalized_instruction", "plan"),
                lambda context: run_calendar_agent(context["normalized_instruction"], context["plan"]),
            ),
        ]
    )


def _build_response(request: InvokeRequest) -> InvokeResponse:
    dag = build_dag(request)
    context = dag.run({"request": request})
    return InvokeResponse(
        session_id=request.session_id,
        instruction=request.instruction,
        normalized_instruction=context["normalized_instruction"],
        selected_channels=context["selected_channels"],
        plan=context["plan"],
        results=[
            context["email_result"],
            context["slack_result"],
            context["calendar_result"],
        ],
        trace_steps=context["trace_steps"],
    )


def handle_request(request: InvokeRequest) -> InvokeResponse:
    ensure_llmobs_enabled()
    if not LLMObs.enabled:
        return _build_response(request)

    with LLMObs.workflow(name="julia_orchestration_dag", session_id=request.session_id) as span:
        response = _build_response(request)
        LLMObs.annotate(
            span=span,
            input_data=request.model_dump(),
            output_data=response.model_dump(),
            metadata={"source": request.source, "user_id": request.user_id},
        )
        return response
