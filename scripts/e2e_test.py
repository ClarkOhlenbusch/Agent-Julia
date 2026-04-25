#!/usr/bin/env python3
"""End-to-end smoke test: ACT → plan → ask → confirm YES → execute → booked."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import memory
from tools import calendar as cal_tool
import agent

memory.reset_all()
cal_tool.reset_session()
memory.seed_from_file(str(Path(__file__).parent.parent / "data" / "seed_facts.json"))
print(f"Seeded {memory.semantic_count()} semantic facts.\n")

agent.run_text([
    ("alex", "Yo, want to grab drinks at 7:30 tonight near Fort Point?"),
    ("sam",  "Yeah, sounds great, let's do it"),
])

print("\n=== Final calendar state ===")
for ev in cal_tool.list_booked():
    title = ev['title']
    start = ev['start']
    attendees = ev['attendees']
    print(f"  - {title} @ {start} for {attendees}")

print(f"\nepisodic={memory.episodic_count()} semantic={memory.semantic_count()}")

print("\n=== Semantic memory dump ===")
for f in memory.semantic_all():
    print(f"  - [{f['type']}] {f['subject']}: {f['text']}  (conf {f['confidence']:.2f})")
