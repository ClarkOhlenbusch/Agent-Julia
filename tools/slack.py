"""Slack tool — STUB for hackathon demo.

Returns a fake successful post. In production, wire to Slack Web API.
"""
import uuid
from schema import TaskResult


def post_slack(channel: str, message: str, mentions: list[str] | None = None) -> TaskResult:
    msg_id = f"slack_{uuid.uuid4().hex[:10]}"
    print(f"[slack-stub] #{channel}: {message[:80]!r}")
    return TaskResult(
        status="posted",
        message=f"Posted to #{channel}",
        artifact_id=msg_id,
    )
