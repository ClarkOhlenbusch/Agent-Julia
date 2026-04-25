"""
Send email via the Gmail API.

Required:
  - Google OAuth credentials (see gauth.py)
  - GMAIL_SENDER env var (the "From" address — must match the authed account)

Usage:
  result = await send_email(proposal)
"""
import asyncio
import base64
import logging
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from schema import TaskProposal, ToolResult, TaskType
from tools.gauth import get_credentials

log = logging.getLogger(__name__)

GMAIL_SENDER = os.environ.get("GMAIL_SENDER", "")
DRY_RUN      = os.environ.get("DRY_RUN", "false").lower() == "true"


def _build_message(sender: str, recipients: List[str], subject: str, body: str) -> dict:
    msg = MIMEMultipart("alternative")
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return {"raw": raw}


def _send_sync(sender: str, recipients: List[str], subject: str, body: str) -> str:
    creds   = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    message = _build_message(sender, recipients, subject, body)
    sent    = service.users().messages().send(userId="me", body=message).execute()
    return sent["id"]


async def send_email(proposal: TaskProposal) -> ToolResult:
    sender  = GMAIL_SENDER
    subject = proposal.subject or "Jarvis — follow-up"
    body    = proposal.content
    to      = proposal.recipients

    if not sender:
        return ToolResult(
            success=False,
            tool=TaskType.send_email,
            message="GMAIL_SENDER env var not set.",
        )

    if DRY_RUN:
        log.info("[DRY RUN] Would email %s: %s", to, subject)
        return ToolResult(
            success=True,
            tool=TaskType.send_email,
            message=f"[dry-run] Email to {', '.join(to)}: \"{subject}\"",
            dry_run=True,
        )

    try:
        msg_id = await asyncio.to_thread(_send_sync, sender, to, subject, body)
        log.info("Email sent id=%s to=%s", msg_id, to)
        return ToolResult(
            success=True,
            tool=TaskType.send_email,
            message=f"Email sent to {', '.join(to)} (id: {msg_id})",
        )
    except HttpError as exc:
        log.error("Gmail API error: %s", exc)
        return ToolResult(
            success=False,
            tool=TaskType.send_email,
            message=f"Gmail API error: {exc}",
        )
