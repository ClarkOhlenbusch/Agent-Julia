"""Pydantic schemas for Jarvis agent message types.

These are the structured contracts between agent layers — Triage's output
is enforced JSON, Planner's TaskProposal is enforced JSON, etc. xgrammar /
guided decoding in vLLM consumes these schemas at decode time.
"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


# ============================================================================
# Triage / Middleware
# ============================================================================

class TriageRoute(str, Enum):
    STORE = "STORE"      # noteworthy but not actionable; embed into episodic memory
    DISCARD = "DISCARD"  # irrelevant chatter; drop
    ACT = "ACT"          # scheduling intent or clear task; trigger plan-confirm-execute


class TriageDecision(BaseModel):
    route: TriageRoute
    confidence: float = Field(ge=0, le=1, description="0-1 calibrated confidence")
    reason: str = Field(max_length=240, description="One short sentence rationale")


# ============================================================================
# Planner
# ============================================================================

class TaskType(str, Enum):
    SEND_EMAIL = "send_email"
    POST_SLACK = "post_slack"
    CREATE_CALENDAR_EVENT = "create_calendar_event"


class TaskProposal(BaseModel):
    task_type: TaskType
    summary: str = Field(description="One-sentence summary the voice agent reads aloud")
    parameters: dict[str, Any] = Field(description="Tool-specific arguments")
    rationale: str = Field(max_length=300, description="Why this proposal — references memory if relevant")


# ============================================================================
# Voice Agent — confirmation parsing
# ============================================================================

class ConfirmIntent(str, Enum):
    YES = "YES"
    NO = "NO"
    MODIFY = "MODIFY"
    UNCLEAR = "UNCLEAR"


class ConfirmationIntent(BaseModel):
    intent: ConfirmIntent
    modifier: Optional[str] = Field(None, description="If MODIFY, the user's requested change")


# ============================================================================
# Fact Extractor
# ============================================================================

class FactType(str, Enum):
    PREFERENCE = "preference"          # likes/dislikes ("Sam doesn't drink coffee")
    RELATIONSHIP = "relationship"      # who-knows-whom ("Alex and Sam grab drinks weekly")
    DECISION = "decision"              # outcomes ("Sam picked 7:30 over 6pm")
    SOCIAL_PATTERN = "social_pattern"  # recurring behavior
    IDENTITY = "identity"              # roles/affiliations ("Julie is the manager")


class Fact(BaseModel):
    subject: str = Field(description="Person, group, or topic the fact is about")
    type: FactType
    fact: str = Field(description="The distilled statement")
    confidence: float = Field(ge=0, le=1)


class FactList(BaseModel):
    """Wrapper so the LLM can return zero-or-more facts in one call."""
    facts: list[Fact] = Field(default_factory=list)


# ============================================================================
# Misc
# ============================================================================

class Observation(BaseModel):
    text: str
    speaker: Optional[str] = None
    timestamp: float


class TaskResult(BaseModel):
    status: str  # "sent" | "posted" | "booked" | "failed"
    message: str
    artifact_id: Optional[str] = None
