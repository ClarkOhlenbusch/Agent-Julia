"""Mock Google Calendar tool with realistic seed data.

Three personas:
  - alex  — late riser; lots of afternoon meetings
  - sam   — mornings free; standing 6pm gym Mon/Wed
  - julie — afternoon-only person; vegetarian; manager
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from schema import TaskResult

# All times America/New_York for the demo
TZ = timezone(timedelta(hours=-4))


def _today_at(hour: int, minute: int = 0) -> datetime:
    now = datetime.now(TZ)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


# busy_slots[user] = [(start, end), ...]
_BUSY: dict[str, list[tuple[datetime, datetime]]] = {
    "alex": [
        (_today_at(10), _today_at(11)),       # 10-11 standup
        (_today_at(13), _today_at(14, 30)),   # 1-2:30 client call
        (_today_at(18), _today_at(19)),       # 6-7 dinner with parents
    ],
    "sam": [
        (_today_at(9), _today_at(10, 30)),    # 9-10:30 deep work
        (_today_at(15), _today_at(16)),       # 3-4 1:1
        (_today_at(18, 30), _today_at(19, 30)),  # 6:30-7:30 gym
    ],
    "julie": [
        (_today_at(11), _today_at(12)),       # 11-12 review
        (_today_at(14, 30), _today_at(15, 30)),
        (_today_at(19), _today_at(20)),       # 7-8 dinner with family
    ],
}

_BOOKED_EVENTS: list[dict] = []


def _normalize(user: str) -> str:
    u = user.lower().strip()
    return u.split("@")[0] if "@" in u else u


def get_freebusy(user: str, start: Optional[str] = None, end: Optional[str] = None) -> dict:
    """Return list of busy windows for a user within an optional [start, end] range.

    start/end are ISO-8601 strings; default to today 09:00–22:00 local.
    """
    u = _normalize(user)
    busy = _BUSY.get(u, [])
    win_start = datetime.fromisoformat(start) if start else _today_at(9)
    win_end = datetime.fromisoformat(end) if end else _today_at(22)
    overlapping = [
        {"start": s.isoformat(), "end": e.isoformat()}
        for s, e in busy
        if e > win_start and s < win_end
    ]
    return {"user": u, "busy": overlapping, "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat()}


def find_overlap(users: Iterable[str], duration_min: int = 30,
                 earliest_hour: int = 9, latest_hour: int = 22) -> list[dict]:
    """Return up to 5 candidate slots when ALL users are free for `duration_min`.
    Slots are chronological. 30-min granularity.
    """
    users_n = [_normalize(u) for u in users]
    all_busy: list[tuple[datetime, datetime]] = []
    for u in users_n:
        all_busy.extend(_BUSY.get(u, []))
    # Also block already-booked events
    for ev in _BOOKED_EVENTS:
        if any(_normalize(a) in users_n for a in ev["attendees"]):
            all_busy.append((datetime.fromisoformat(ev["start"]), datetime.fromisoformat(ev["end"])))

    candidates = []
    cursor = _today_at(earliest_hour)
    end_window = _today_at(latest_hour)
    step = timedelta(minutes=30)
    duration = timedelta(minutes=duration_min)

    while cursor + duration <= end_window and len(candidates) < 5:
        window_start, window_end = cursor, cursor + duration
        conflict = any(b_start < window_end and b_end > window_start
                       for b_start, b_end in all_busy)
        if not conflict:
            candidates.append({
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "duration_min": duration_min,
            })
        cursor += step
    return candidates


def book_meeting(title: str, start: str, end: str, attendees: list[str],
                 location: Optional[str] = None, description: Optional[str] = None) -> TaskResult:
    """Create a calendar event. Persists to in-memory store for the session."""
    event_id = f"cal_{uuid.uuid4().hex[:10]}"
    ev = {
        "id": event_id,
        "title": title,
        "start": start,
        "end": end,
        "attendees": [_normalize(a) for a in attendees],
        "location": location,
        "description": description,
    }
    _BOOKED_EVENTS.append(ev)
    # Also add to busy slots so subsequent find_overlap sees it
    s_dt, e_dt = datetime.fromisoformat(start), datetime.fromisoformat(end)
    for a in ev["attendees"]:
        _BUSY.setdefault(a, []).append((s_dt, e_dt))
    print(f"[calendar] booked {event_id} {title!r} {start} → {end} for {ev['attendees']}")
    return TaskResult(
        status="booked",
        message=f"{title} on {start} for {len(attendees)} attendees",
        artifact_id=event_id,
    )


def list_booked() -> list[dict]:
    """For the UI to show what's been booked this session."""
    return list(_BOOKED_EVENTS)


def reset_session() -> None:
    """Clear all booked events. Restores _BUSY to seeded state."""
    global _BOOKED_EVENTS, _BUSY
    _BOOKED_EVENTS = []
    _BUSY = {
        "alex": [(_today_at(10), _today_at(11)),
                 (_today_at(13), _today_at(14, 30)),
                 (_today_at(18), _today_at(19))],
        "sam": [(_today_at(9), _today_at(10, 30)),
                (_today_at(15), _today_at(16)),
                (_today_at(18, 30), _today_at(19, 30))],
        "julie": [(_today_at(11), _today_at(12)),
                  (_today_at(14, 30), _today_at(15, 30)),
                  (_today_at(19), _today_at(20))],
    }


if __name__ == "__main__":
    import json
    print("freebusy alex:", json.dumps(get_freebusy("alex"), indent=2, default=str))
    print("\noverlap alex+sam (30 min):")
    for s in find_overlap(["alex", "sam"], 30):
        print("  ", s)
    print("\nbook a meeting at first overlap:")
    slots = find_overlap(["alex", "sam"], 30)
    if slots:
        r = book_meeting("Drinks", slots[0]["start"], slots[0]["end"],
                         ["alex", "sam"], location="The Black Rose")
        print("  ", r)
