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
