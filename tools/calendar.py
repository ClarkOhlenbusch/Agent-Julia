"""
Create a Google Calendar event via the Calendar API.

Required:
  - Google OAuth credentials (see gauth.py)
  - GCAL_CALENDAR_ID env var (defaults to "primary")

The proposal's datetime_str is parsed with dateparser so natural strings
like "7:30pm tonight" or "2026-04-25T19:30:00" both work.

Usage:
  result = await create_calendar_event(proposal)
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import dateparser
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from schema import TaskProposal, ToolResult, TaskType
from tools.gauth import get_credentials

log = logging.getLogger(__name__)

GCAL_CALENDAR_ID = os.environ.get("GCAL_CALENDAR_ID", "primary")
GCAL_TIMEZONE    = os.environ.get("GCAL_TIMEZONE", "America/New_York")
DRY_RUN          = os.environ.get("DRY_RUN", "false").lower() == "true"


def _parse_datetime(dt_str: str) -> datetime:
    """Parse a natural-language or ISO datetime string into an aware datetime."""
    parsed = dateparser.parse(
        dt_str,
        settings={
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": GCAL_TIMEZONE,
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )
    if parsed is None:
        raise ValueError(f"Could not parse datetime: {dt_str!r}")
    return parsed


def _create_sync(
    summary: str,
    description: str,
    start: datetime,
    end: datetime,
    attendees: list[str],
    location: str | None,
) -> str:
    creds   = get_credentials()
    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary":     summary,
        "description": description,
        "start": {
            "dateTime": start.isoformat(),
            "timeZone": GCAL_TIMEZONE,
        },
        "end": {
            "dateTime": end.isoformat(),
            "timeZone": GCAL_TIMEZONE,
        },
        "attendees":    [{"email": e} for e in attendees],
        "reminders":    {"useDefault": True},
        "conferenceData": None,
    }
    if location:
        event["location"] = location

    created = (
        service.events()
        .insert(
            calendarId=GCAL_CALENDAR_ID,
            body=event,
            sendUpdates="all",   # emails invites to attendees
        )
        .execute()
    )
    return created.get("htmlLink", created.get("id", ""))


async def create_calendar_event(proposal: TaskProposal) -> ToolResult:
    summary     = proposal.subject or "Jarvis — scheduled event"
    description = proposal.content
    attendees   = proposal.recipients
    location    = proposal.location
    duration    = proposal.duration_min or 60

    if not proposal.datetime_str:
        return ToolResult(
            success=False,
            tool=TaskType.create_calendar_event,
            message="No datetime provided in the proposal.",
        )

    try:
        start = _parse_datetime(proposal.datetime_str)
    except ValueError as exc:
        return ToolResult(
            success=False,
            tool=TaskType.create_calendar_event,
            message=str(exc),
        )

    end = start + timedelta(minutes=duration)

    if DRY_RUN:
        log.info("[DRY RUN] Would create event '%s' at %s with %s", summary, start, attendees)
        return ToolResult(
            success=True,
            tool=TaskType.create_calendar_event,
            message=(
                f"[dry-run] Event \"{summary}\" at {start.strftime('%I:%M %p')} "
                f"with {', '.join(attendees)}"
            ),
            dry_run=True,
        )

    try:
        link = await asyncio.to_thread(
            _create_sync, summary, description, start, end, attendees, location
        )
        log.info("Calendar event created: %s", link)
        return ToolResult(
            success=True,
            tool=TaskType.create_calendar_event,
            message=f"Calendar event created: {link}",
        )
    except HttpError as exc:
        log.error("Calendar API error: %s", exc)
        return ToolResult(
            success=False,
            tool=TaskType.create_calendar_event,
            message=f"Calendar API error: {exc}",
        )
