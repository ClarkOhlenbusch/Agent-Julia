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


class TaskProposal(BaseModel):
    recipients:   List[str]           # Slack channel IDs or user IDs to notify
    content:      str                 # The Slack message Julia will post
    voice_prompt: str                 # What Julia asks the user before posting


class ConfirmationAction(str, Enum):
    YES = "YES"
    NO  = "NO"


class ConfirmationIntent(BaseModel):
    action: ConfirmationAction


class ToolResult(BaseModel):
    success: bool
    message: str
    dry_run: bool = False


class FactType(str, Enum):
    preference     = "preference"
    relationship   = "relationship"
    decision       = "decision"
    social_pattern = "social_pattern"


class Fact(BaseModel):
    subject:    str
    type:       FactType
    fact:       str
    confidence: float


# ----------------------------------------------------------------------------
# Backward-compat aliases — older modules (tools/calendar.py, tools/email.py,
# agents/*) imported these names. Keep aliases so module-level imports don't
# crash even if those modules aren't on the new code path.
# ----------------------------------------------------------------------------
TaskResult = ToolResult
TriageRoute = TriageAction


class TaskType(str, Enum):
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    SEND_EMAIL = "send_email"
    POST_SLACK = "post_slack"


class ConfirmIntent(str, Enum):
    YES = "YES"
    NO = "NO"
    MODIFY = "MODIFY"
    UNCLEAR = "UNCLEAR"
