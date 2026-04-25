"""Email tool — STUB for hackathon demo.

Returns a fake successful send. In production, wire to Gmail / SendGrid.
"""
import time
import uuid
from schema import TaskResult


def send_email(to: str, subject: str, body: str, cc: list[str] | None = None) -> TaskResult:
    msg_id = f"msg_{uuid.uuid4().hex[:10]}"
    print(f"[email-stub] to={to} subj={subject!r} body={body[:60]!r}")
    return TaskResult(
        status="sent",
        message=f"Email queued for delivery to {to}",
        artifact_id=msg_id,
    )
