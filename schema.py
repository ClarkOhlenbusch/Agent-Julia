from enum import Enum
from typing import Optional, List
from pydantic import BaseModel


class TriageAction(str, Enum):
    STORE   = "STORE"
    DISCARD = "DISCARD"
    ACT     = "ACT"


class TriageDecision(BaseModel):
    action: TriageAction
    reason: str


class TaskType(str, Enum):
    send_email            = "send_email"
    post_slack            = "post_slack"
    create_calendar_event = "create_calendar_event"


class TaskProposal(BaseModel):
    task_type:    TaskType
    recipients:   List[str]           # email addresses or Slack user/channel IDs
    subject:      Optional[str] = None
    content:      str
    datetime_str: Optional[str] = None  # ISO-8601 or human string for calendar
    duration_min: Optional[int] = 60    # calendar event length in minutes
    location:     Optional[str] = None
    voice_prompt: str                   # what Jarvis says when asking user to confirm


class ConfirmationAction(str, Enum):
    YES    = "YES"
    NO     = "NO"
    MODIFY = "MODIFY"


class ConfirmationIntent(BaseModel):
    action:       ConfirmationAction
    modification: Optional[str] = None  # free-text if user wants to tweak something


class FactType(str, Enum):
    preference     = "preference"
    relationship   = "relationship"
    decision       = "decision"
    social_pattern = "social_pattern"
    identity       = "identity"


class Fact(BaseModel):
    subject:    str
    type:       FactType
    fact:       str
    confidence: float


class ToolResult(BaseModel):
    success: bool
    tool:    TaskType
    message: str          # human-readable summary of what happened
    dry_run: bool = False
