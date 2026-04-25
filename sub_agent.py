"""
Sub-agent: receives a confirmed TaskProposal and dispatches to the right tool.

After execution it logs the outcome back to episodic memory so the agent
doesn't re-propose the same action.
"""
import logging
from typing import TYPE_CHECKING

from schema import TaskProposal, TaskType, ToolResult

if TYPE_CHECKING:
    from memory import MemoryStore

log = logging.getLogger(__name__)


async def execute(proposal: TaskProposal, memory: "MemoryStore") -> ToolResult:
    """Dispatch proposal to the correct tool and record the outcome."""

    # Import here to avoid circular imports at module load time
    from tools.email    import send_email
    from tools.slack    import post_slack
    from tools.calendar import create_calendar_event

    dispatch = {
        TaskType.send_email:            send_email,
        TaskType.post_slack:            post_slack,
        TaskType.create_calendar_event: create_calendar_event,
    }

    handler = dispatch.get(proposal.task_type)
    if handler is None:
        result = ToolResult(
            success=False,
            tool=proposal.task_type,
            message=f"Unknown task type: {proposal.task_type}",
        )
    else:
        log.info("Sub-agent executing %s for %s", proposal.task_type, proposal.recipients)
        result = await handler(proposal)

    # Log the outcome to episodic memory so the agent remembers what it did
    status = "completed" if result.success else "failed"
    memory_entry = (
        f"[sub-agent] {status}: {proposal.task_type} — {result.message}"
    )
    memory.write_episodic(memory_entry)

    if result.success:
        log.info("Sub-agent success: %s", result.message)
    else:
        log.warning("Sub-agent failure: %s", result.message)

    return result
