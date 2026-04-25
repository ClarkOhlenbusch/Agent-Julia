"""Latency benchmark for the agent stack.

Adapted from demo/nemoclaw-agent/benchmarks/latency-test.py.
Measures end-to-end interjection latency: Triage → Planner → Voice question.
Outputs a JSON file + prints summary stats.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import memory
from agents import middleware, planner, voice_agent

# Mix of difficulty: clear ACT, ambiguous STORE, definite DISCARD
TEST_CHUNKS = [
    ("act_clear",     "Yo, want to grab drinks at 7 tonight?"),
    ("act_clear",     "Hey, can we set up a meeting next Tuesday at 2pm?"),
    ("act_implicit",  "Should we do dinner with Julie this week?"),
    ("store",         "I really don't like working from home on Fridays."),
    ("store",         "Sam mentioned his daughter just started school."),
    ("discard",       "Yeah, the Celtics played terribly last night."),
    ("discard",       "Did you see that thing on Twitter?"),
    ("act_clear",     "Email Julie that the report's done."),
    ("act_implicit",  "We should grab coffee soon — feels like it's been forever."),
    ("discard",       "What time is it?"),
]


def bench_chunk(label: str, text: str) -> dict:
    out = {"label": label, "text": text, "ok": True}
    try:
        # Triage
        t0 = time.time()
        decision = middleware.decide(text, speaker="alex")
        out["triage_ms"] = round((time.time() - t0) * 1000, 1)
        out["triage_route"] = decision.route.value
        out["triage_conf"] = round(decision.confidence, 2)

        # Plan + Voice (only if ACT)
        if decision.route.value == "ACT":
            t1 = time.time()
            proposal = planner.plan(text, attendees_hint=["alex", "sam"])
            out["plan_ms"] = round((time.time() - t1) * 1000, 1)
            out["task_type"] = proposal.task_type.value

            t2 = time.time()
            question = voice_agent.compose_question(proposal)
            out["voice_ms"] = round((time.time() - t2) * 1000, 1)
            out["question"] = question

            out["e2e_ms"] = round((time.time() - t0) * 1000, 1)
        else:
            out["e2e_ms"] = out["triage_ms"]
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
    return out


def main():
    # Pre-warm semantic memory so latency includes realistic context retrieval
    seed_path = Path(__file__).parent.parent / "data" / "seed_facts.json"
    if seed_path.exists():
        memory.seed_from_file(str(seed_path))
        print(f"Seeded {memory.semantic_count()} facts.")

    results = []
    for label, text in TEST_CHUNKS:
        r = bench_chunk(label, text)
        print(f"  [{label:14}] e2e={r.get('e2e_ms', '?'):>6}ms  {r.get('triage_route', '?'):8} {text!r}")
        results.append(r)

    # Aggregate
    by_route = {}
    for r in results:
        if not r.get("ok"):
            continue
        route = r.get("triage_route", "?")
        by_route.setdefault(route, []).append(r["e2e_ms"])

    print("\n=== Summary ===")
    for route, latencies in by_route.items():
        if not latencies:
            continue
        print(f"  {route:8} n={len(latencies):2} "
              f"mean={statistics.mean(latencies):>7.1f}ms "
              f"p50={statistics.median(latencies):>7.1f}ms "
              f"max={max(latencies):>7.1f}ms")

    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
