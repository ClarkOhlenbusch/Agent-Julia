from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ChannelName = Literal["email", "slack", "calendar"]


class InvokeRequest(BaseModel):
    session_id: str = Field(..., description="Conversation or device session identifier.")
    instruction: str = Field(..., min_length=1, description="Natural language instruction from the user.")
    source: str = Field(default="proxy", description="Where the instruction came from.")
    user_id: str | None = Field(default=None, description="Upstream user identifier.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannedAction(BaseModel):
    channel: ChannelName
    should_run: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ChannelResult(BaseModel):
    channel: ChannelName
    action: str
    status: Literal["planned", "skipped"]
    detail: str
    model_stub: str


class InvokeResponse(BaseModel):
    session_id: str
    instruction: str
    normalized_instruction: str
    selected_channels: list[ChannelName]
    plan: list[PlannedAction]
    results: list[ChannelResult]
    trace_steps: list[str]
