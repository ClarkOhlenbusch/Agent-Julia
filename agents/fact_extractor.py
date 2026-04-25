"""Fact Extractor — distills noteworthy items from recent transcripts → semantic memory.

Periodic: runs every N=10 chunks OR after 30s idle, with hard rate limit of
one extraction per 30 seconds. Reuses the Triage model (Llama 3.1 8B) to keep
the heavy 24B Mistral free for user-facing latency.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import httpx

import memory
import observability
from schema import Fact, FactType, FactList

EXTRACT_ENDPOINT = os.getenv("TRIAGE_ENDPOINT", "http://localhost:9001/v1")
EXTRACT_MODEL = os.getenv("TRIAGE_MODEL", "triage")

MIN_INTERVAL_S = 30.0
_last_run = 0.0

SYSTEM_PROMPT = """You are a memory curator for a personal assistant.

Given recent conversation transcripts, extract structured FACTS that would help
a future assistant act more personally. Categories:
  - preference  — what someone likes/dislikes
  - relationship — how people relate (frequency, context)
  - decision    — outcomes someone agreed/rejected
  - social_pattern — recurring behaviors
  - identity    — roles, affiliations

Rules:
  - Only extract durable facts. Skip ephemeral stuff ("they said hi").
  - Confidence 0.5-0.95. Use 0.9+ only if the speaker stated it directly.
  - Empty list is fine if nothing noteworthy.
  - Subject must be a person, group, or topic name — never "user" or "they".

Output JSON ONLY: {"facts": [{"subject": "...", "type": "...", "fact": "...", "confidence": 0.X}, ...]}
"""


@observability.task(name="fact_extractor")
def extract(force: bool = False) -> list[Fact]:
    """Run extraction over the most recent ~20 episodic chunks. Respects rate limit."""
    global _last_run
    now = time.time()
    if not force and (now - _last_run) < MIN_INTERVAL_S:
        return []

    recent = memory.episodic_recent(20)
    if len(recent) < 3:
        return []

    transcript = "\n".join(
        f"[{c.get('speaker', '?')}] {c['text']}" for c in recent
    )
    user_msg = f"Recent transcript:\n{transcript}\n\nExtract facts as JSON."

    payload = {
        "model": EXTRACT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }

    try:
        with httpx.Client(timeout=30) as c:
            r = c.post(f"{EXTRACT_ENDPOINT}/chat/completions", json=payload)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        data = json.loads(raw)
        # Coerce types
        facts = []
        for f in data.get("facts", []):
            try:
                if isinstance(f.get("type"), str):
                    f["type"] = FactType(f["type"])
                facts.append(Fact(**f))
            except Exception:
                continue
        if facts:
            memory.semantic_write(facts)
        _last_run = now
        return facts
    except Exception as e:
        print(f"[fact_extractor] error: {e}")
        return []


if __name__ == "__main__":
    facts = extract(force=True)
    print(f"Extracted {len(facts)} facts:")
    for f in facts:
        print(f"  - [{f.type.value}] {f.subject}: {f.fact} ({f.confidence:.2f})")
