"""
Session tracker for a single Slack huddle.

Accumulates transcript chunks and tool results while the huddle is live.
Passed into every pipeline stage so each piece can append to it.
"""
import time
from dataclasses import dataclass, field
from typing import List

from schema import ToolResult


@dataclass
class HuddleSession:
    channel_id:    str
    started_at:    float         = field(default_factory=time.time)
    ended_at:      float | None  = None

    # Raw transcript — one entry per Whisper chunk
    transcript:    List[str]     = field(default_factory=list)

    # Every action Juliah took (success or failure)
    actions_taken: List[ToolResult] = field(default_factory=list)

    def add_transcript(self, text: str) -> None:
        if text.strip():
            self.transcript.append(text.strip())

    def add_action(self, result: ToolResult) -> None:
        self.actions_taken.append(result)

    def end(self) -> None:
        self.ended_at = time.time()

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at

    @property
    def duration_str(self) -> str:
        secs = int(self.duration_seconds)
        m, s = divmod(secs, 60)
        return f"{m}m {s}s" if m else f"{s}s"

    @property
    def full_transcript(self) -> str:
        return "\n".join(self.transcript)
