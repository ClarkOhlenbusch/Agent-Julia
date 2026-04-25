# Julia Scheduling Agent — Track 5 Plan

> **TOA vLLM/LLM-D Hackathon · April 25, 2026 · Boston**
> **Track 5 — Efficient Agentic Edge powered by NemoClaw** 

<p align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/76dc176b-9396-4366-839c-bff6fb057218" />
</p>

---

## TL;DR

Coordination is messy, and most agents only act after someone gives an explicit command. Julia is the assistant you wish you had: a continuous-listening, plan-confirm-execute scheduling agent that sits in a live Slack conversation, detects when both parties agree on a plan, confirms with the user, then dispatches a sub-agent to execute (email, Slack, or calendar) and narrates the result aloud via TTS. A proactive voice agent that **listens** to a live conversation, **decides** when to interject, **confirms** with the user before acting, **executes** the chosen task (send email, post to Slack, create calendar invite), and **gets smarter over time** through a two-tier memory system that distills noteworthy moments into long-term semantic memory. Built on Red Hat AI quantized models served by vLLM, sandboxed by NemoClaw, running on a single A100 80GB Brev instance.

Most agent demos are tool-callers in a wrapper. We built a continuous-listening, plan-confirm-execute pipeline with a two-tier memory system — episodic for immediate recall, semantic for accumulated social context — that actually feels like a participant who's been in the room with you. Every model in the stack — STT, triage, planner/executor, fact extractor — runs on Red Hat AI's quantized weights through vLLM, on a single A100. NemoClaw sandboxes the agent so even prompt-injected calendar events can't exfil data.




---

## Current DAG Service In This Branch

The code on branch `dag_orchestration` includes a working FastAPI service that exposes a small orchestration DAG for three task domains:

- `email`
- `slack`
- `calendar`

This implementation is intentionally narrower than the full hackathon plan above. It is the current runnable service in this repo and lives in:

- `main.py`
- `julia_dag/config.py`
- `julia_dag/schemas.py`
- `julia_dag/orchestrator.py`

### API surface

The service exposes two endpoints:

- `GET /health`
  - Returns a simple readiness payload: `{ "status": "ok", "app": "<app_name>" }`
- `POST /invoke`
  - Accepts an orchestration request and returns the DAG plan, selected channels, per-channel results, and execution trace

Example request:

```json
{
  "session_id": "demo-session-1",
  "instruction": "Email the client, notify the team in Slack, and create a calendar invite.",
  "source": "proxy",
  "user_id": "user-123",
  "metadata": {}
}
```

Example response shape:

```json
{
  "session_id": "demo-session-1",
  "instruction": "Email the client, notify the team in Slack, and create a calendar invite.",
  "normalized_instruction": "email the client, notify the team in slack, and create a calendar invite.",
  "selected_channels": ["email", "slack", "calendar"],
  "plan": [
    {
      "channel": "email",
      "should_run": true,
      "confidence": 0.92,
      "reason": "email-specific language detected"
    }
  ],
  "results": [
    {
      "channel": "email",
      "action": "draft_email",
      "status": "planned",
      "detail": "Email model should prepare a draft or reply based on: ...",
      "model_stub": "email-agent-stub"
    }
  ],
  "trace_steps": [
    "normalized_instruction",
    "plan",
    "selected_channels",
    "email_result",
    "slack_result",
    "calendar_result"
  ]
}
```

### DAG nodes and execution order

The current orchestration DAG is built in `julia_dag/orchestrator.py` and always runs the same six-node graph:

1. `normalized_instruction`
2. `plan`
3. `selected_channels`
4. `email_result`
5. `slack_result`
6. `calendar_result`

The dependencies are:

```text
normalized_instruction
  -> plan
    -> selected_channels
normalized_instruction + plan
  -> email_result
  -> slack_result
  -> calendar_result
```

At runtime:

1. `normalize_instruction()` trims whitespace, lowercases text, and compresses repeated spaces.
2. `plan_channels()` decides which domains should run.
3. `select_channels()` extracts only the channels whose `should_run` flag is `true`.
4. Each specialist node returns either a `planned` action for its domain or a `skipped` noop result.
5. The DAG executor records the final `trace_steps` list so downstream systems can see the exact order of execution.

### Channel selection logic

Routing is keyword- and phrase-based today. The planner does not call an LLM yet; it uses deterministic matching so behavior is predictable and easy to smoke test.

Current matching rules:

- Email is selected for explicit email language such as `email`, `mail`, `inbox`, `send email`, or `reply to email`
- Slack is selected for explicit Slack/chat language such as `slack`, `dm`, `channel`, `slack message`, `post in slack`, `reply in slack`, or `ping in slack`
- Calendar is selected for explicit scheduling language such as `calendar`, `schedule`, `reschedule`, `invite`, or calendar-specific event creation phrasing

Important behavior:

- Multi-intent instructions can select multiple channels in one request
- If no strong domain match is found, the DAG falls back to `slack`
- The implementation intentionally avoids broad verbs like `send` or nouns like `meeting` by themselves because they caused false positives during testing

Examples:

- `"Send an email to the customer"` -> `["email"]`
- `"Ping the team in Slack about the deploy"` -> `["slack"]`
- `"Schedule a meeting for tomorrow afternoon"` -> `["calendar"]`
- `"Email the client, notify the team in Slack, and create a calendar invite"` -> `["email", "slack", "calendar"]`
- `"Please follow up with the team about this"` -> `["slack"]` fallback

### Per-channel outputs

The current specialists are stubs that represent future domain agents:

- Email specialist -> `draft_email`
- Slack specialist -> `compose_slack_message`
- Calendar specialist -> `schedule_calendar_change`

Each returns a `ChannelResult` with:

- `channel`
- `action`
- `status`
- `detail`
- `model_stub`

This gives the service a stable response contract now while leaving room for real downstream tool execution later.

### Datadog LLM Observability integration

The branch also includes Datadog LLM Observability instrumentation around each orchestration request.

Configuration is loaded from `.env` by `julia_dag/config.py`:

- `DD_LLMOBS_ENABLED`
- `DD_LLMOBS_ML_APP`
- `DD_SITE`
- `DD_API_KEY`
- `DD_APP_KEY` optional
- `DD_ENV` optional
- `DD_SERVICE` optional

When `handle_request()` runs:

1. `ensure_llmobs_enabled()` enables `ddtrace.llmobs.LLMObs`
2. If an API key is present, the code enables agentless submission to Datadog LLM Observability
3. The request is wrapped in `LLMObs.workflow(name="julia_orchestration_dag", session_id=request.session_id)`
4. The service annotates the span with:
   - `input_data`: the full request payload
   - `output_data`: the full orchestration response payload
   - `metadata`: request `source` and `user_id`

That means a single `/invoke` request can be inspected in Datadog with:

- the original instruction
- the selected channels
- the returned plan and channel results
- the exact DAG trace order
- the session identifier used to correlate related requests

In practice, this makes Datadog useful for both:

- correctness debugging: "Did the DAG pick the right channels?"
- observability: "What orchestration decisions happened for this session?"

### What is and is not instrumented today

Instrumented today:

- the top-level orchestration workflow span
- task and agent decorators on DAG helper functions in `julia_dag/orchestrator.py`
- request/response payload annotation for `/invoke`

Not implemented yet:

- real LLM planner calls
- real email/Slack/calendar API execution
- NemoClaw runtime integration in this local FastAPI service
- memory retrieval or Chroma-backed context injection in the runnable code path

So the current branch should be understood as:

- a working orchestration skeleton
- a deterministic DAG router for three task domains
- Datadog-instrumented request tracing around that DAG

### Local smoke-test behavior

The current service has been smoke tested for:

- `GET /health`
- valid multi-channel `/invoke` requests
- valid single-channel `/invoke` requests
- fallback `/invoke` requests with no explicit domain
- invalid whitespace-only instructions returning `400`

Datadog validation performed on this branch:

- confirmed the service can enable `LLMObs`
- confirmed live agentless submission to Datadog intake succeeds once a valid `DD_API_KEY` is present
- observed separate local APM trace delivery attempts to `localhost:8126`, which are not required for LLM Observability itself

---

## Track 5 Context

| | |
|---|---|
| **Prize** | 1× NVIDIA RTX 5090 |
| **Sponsor** | NVIDIA |
| **Required tech** | NemoClaw (mandatory), vLLM, an open-source LLM |
| **Skill lane** | **Builder/Deep Tech** — multi-agent architecture, plan-execute pattern, Red Hat AI integration |
| **Venue** | Red Hat office, 300 A St, Boston |
| **Date** | Saturday April 25, 2026 |
| **Demo length** | 3 minutes per team |

**Judging dimensions** (from the brief):
1. Meaningful NemoClaw integration (sandbox actually does something)
2. vLLM used well (multiple models, structured outputs, batching)
3. Agent quality (multi-turn, robust to edge cases)
4. Steering / safety (guardrails visible in demo)
5. Demo polish + technical ambition

---

## Architecture

### Topology (where each thing runs)

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAPTOP                                                              │
│  ─ Live mic input  ─────────────────────────────────────┐            │
│  ─ Gradio UI (transcript + agent state)                  │ audio     │
│  ─ Piper TTS playback (laptop speaker)                   │ stream    │
│  ─ Tivoo-2 sidekick (over phone hotspot, status emoji)   │           │
└──────────────────────────────────────────────────────────┼───────────┘
                                                           │
                       phone hotspot / SSH tunnel          │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  BREV VM — A100 80GB ($1.49/hr — massedcompute_A100_sxm4_80G_DGX)    │
│                                                                      │
│  ╔═ HOST processes (use the GPU directly) ════════════════════════╗  │
│  ║                                                                ║  │
│  ║  vLLM :9000  ─ Whisper turbo            (transcription)        ║  │
│  ║  vLLM :9001  ─ Triage / Middleware      (interject decision)   ║  │
│  ║  vLLM :9002  ─ Multi-role agent         (planner/voice/sub)    ║  │
│  ║                                                                ║  │
│  ╚════════════════════════════════════════════════════════════════╝  │
│                       ▲                                              │
│                       │ HTTP via host.openshell.internal             │
│                       │                                              │
│  ┌─ NemoClaw OpenShell sandbox (Docker, no GPU) ──────────────────┐  │
│  │                                                                │  │
│  │  agent.py  ── orchestrator                                     │  │
│  │     │                                                          │  │
│  │     ├─ middleware.py   "should we interject this transcript?"  │  │
│  │     │     ├─ NO  → store in vector DB, loop                    │  │
│  │     │     └─ YES ↓                                             │  │
│  │     ├─ planner.py      "what task? email/slack/calendar?"      │  │
│  │     │     ↓                                                    │  │
│  │     ├─ voice_agent.py  ask user "want me to do X?"             │  │
│  │     │     ├─ user NO  → store, loop                            │  │
│  │     │     └─ user YES ↓                                        │  │
│  │     ├─ sub_agent.py    actually execute the task               │  │
│  │     │     ├─ tools/email.py                                    │  │
│  │     │     ├─ tools/slack.py                                    │  │
│  │     │     └─ tools/calendar.py                                 │  │
│  │     │                                                          │  │
│  │     └─ fact_extractor.py  periodic distillation → semantic mem │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ChromaDB :8001                                                      │
│   ├─ episodic_memory   rolling 10-min transcript chunks              │
│   └─ semantic_memory   distilled facts (people / prefs / decisions)  │
└──────────────────────────────────────────────────────────────────────┘
```

### Why each boundary

- **vLLM on the host**: needs raw GPU, can't be in a sandbox without killing throughput.
- **Agent in NemoClaw**: enforces network/filesystem policy. If an LLM gets prompt-injected (e.g., a calendar event with hidden malicious text), the sandbox blocks egress.
- **Brev**: just the hardware provider. Gives us the VM. After the hackathon the instance dies; code lives in our GitHub repo.

---

## Memory Architecture (the agent learns over time)

The agent doesn't fine-tune model weights — that's infeasible in one day and overkill for a 3-min demo. Instead, it builds **growing memory** that influences behavior. To the user, it looks identical: the agent gets smarter as the conversation progresses and as it observes more sessions.

Two-tier system, both backed by ChromaDB:

### Tier 1 — Episodic Memory (short-term, raw)

| | |
|---|---|
| **Collection** | `episodic_memory` |
| **Content** | Rolling 10-min transcript chunks, raw text |
| **Embedded with** | `bge-small-en-v1.5` |
| **Write trigger** | Every successful Whisper transcription chunk |
| **Read trigger** | Every Triage call (top-3 nearest chunks) |
| **TTL** | 10 minutes (oldest chunks evicted) |
| **Used for** | Immediate context — "what was just said," dedup detection ("did we already propose this?") |

### Tier 2 — Semantic Memory (long-term, distilled)

| | |
|---|---|
| **Collection** | `semantic_memory` |
| **Content** | Structured facts about people, preferences, decisions, social patterns |
| **Embedded with** | `bge-small-en-v1.5` |
| **Write trigger** | `fact_extractor.py` runs every N transcripts (or on conversation idle) |
| **Read trigger** | Every Triage + Planner call (top-K relevant facts injected into context) |
| **TTL** | None within the session (persists for the demo's life) |
| **Used for** | Smarter triage, personalized proposals, "remembering" past choices |

### FactExtractor — how distillation works

Every N transcript chunks (default: 10), or when the conversation idles for 30 seconds, the agent calls Llama 3.1 8B with a structured prompt:

```
Given the recent conversation, extract noteworthy items.
Format: list of { subject, type, fact, confidence } objects.
Types: preference | relationship | decision | social_pattern | identity
Only include items that would help an assistant make better suggestions later.
```

Example output:

```json
[
  {"subject": "Julie", "type": "preference",
   "fact": "doesn't drink coffee after 3pm", "confidence": 0.85},
  {"subject": "Sam-Alex", "type": "relationship",
   "fact": "tend to grab drinks weekly on Fridays", "confidence": 0.7},
  {"subject": "Sam", "type": "decision",
   "fact": "rejected 6pm because of standing commitment, accepted 7:30",
   "confidence": 0.95}
]
```

Each fact is embedded and stored. Triage and Planner pull top-K relevant facts on every call, biasing the model's behavior toward what it has "learned."

### What this gives us

- ✅ The agent's behavior changes based on accumulated context
- ✅ It remembers people, preferences, decisions
- ✅ Proposals get personalized over time
- ✅ It can dedupe ("we already talked about this") via episodic memory
- ❌ Model weights are NOT updated — be honest if a judge asks. Same UX outcome, no GPU time spent on training, more reliable than fine-tuning under demo conditions.

### Cost

**Zero new VRAM. Zero new processes.** FactExtractor reuses the Triage model. Memory reads are vector queries (CPU). Memory writes are embeddings (CPU, sub-millisecond with bge-small).

---

## Data Flow (per conversation turn)

1. **Audio capture** — laptop mic streams audio chunks to the Brev VM (over SSH tunnel or a small WebSocket bridge).
2. **Transcription** — Whisper (vLLM `:9000`) returns text per ~2-second chunk + silero-vad gates silence.
3. **Episodic write (always)** — every Whisper chunk is embedded and written to `episodic_memory`. The 10-min sliding window evicts oldest chunks.
4. **Triage call** — for each new transcript chunk, the agent (in NemoClaw) calls Triage (`:9001`) with: *"Given the rolling 10-min context AND the relevant facts we know, should I interject? Output: STORE / DISCARD / ACT + reason"*. Triage's prompt is augmented with **top-3 episodic hits** AND **top-K semantic facts** retrieved from Chroma. Grammar-guided JSON via xgrammar.
5. **Branch:**
   - **STORE** → mark this chunk as noteworthy in episodic memory. Loop.
   - **DISCARD** → drop. Loop.
   - **ACT** → continue.
6. **Plan** — agent calls Planner persona (`:9002`) with conversation snippet + episodic hits + semantic facts. Planner picks one of `{send_email, post_slack, create_calendar_event}` and fills out a structured `TaskProposal` (recipients, content, time, etc.) — schema-enforced. Memory shapes proposals: e.g., if semantic memory says *"Sam rejected 6pm, accepted 7:30 last time,"* the planner biases toward 7:30.
7. **Voice Agent confirms** — TTS speaks: *"Hey, I can send that email to Julie about Friday — want me to?"* (Piper TTS through laptop speaker; Tivoo-2 shows speaking emoji.)
8. **Wait for user response** — Whisper continues; the next utterance is parsed by a small intent check (yes / no / modify).
9. **If YES** → Sub-Agent calls the chosen tool. Real execution (or mocked for demo). Tivoo shows `✅`. **Decision is logged to episodic memory** for future recall. End of action flow.
10. **If NO** → store the rejection (with reason if user gave one) in episodic memory so we don't propose the same thing again. Loop.

**In parallel, on a periodic timer:**

11. **FactExtractor** — every N=10 chunks (or on 30 sec idle), extracts structured facts from recent transcripts and writes them to `semantic_memory`. This is the "learning over time" mechanism.

The system **never stops listening**. Whisper runs continuously; new transcripts trigger new Triage calls in parallel with any in-flight task or extraction.

---

## Models — Red Hat AI Lineup

All models served via vLLM, all from Red Hat's HF org. Total weights ~35 GB on an A100 80 GB; ample headroom for KV caches.

| Role | Model | Quant | VRAM | Why |
|---|---|---|---|---|
| **Transcription** | `openai/whisper-large-v3-turbo` | FP16 | ~3 GB | Red Hat doesn't publish Whisper; OpenAI's turbo is the fast SOTA. Served via `vllm serve --task transcription`. |
| **Triage / Middleware** | `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` | FP8 | ~8 GB | Fast, accurate routing. Grammar-guided JSON output (`STORE` / `DISCARD` / `ACT`). |
| **Planner + Voice Agent + Sub-Agent** | `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8` | FP8 | ~24 GB | One model, three personas via system prompts. Strong tool-calling, robust 24B reasoning, FP8-quantized by Red Hat for vLLM. |

**Total weights: ~35 GB. KV caches + Whisper context: ~20 GB. Headroom: ~25 GB.**

> **Pending**: confirm exact Red Hat model IDs exist + are licensed. Will validate with `huggingface-cli` once SSH is up.

### Why one model for Planner/Voice/Sub-Agent

- **Simpler deployment** — one vLLM process to manage instead of three.
- **Lower VRAM** — 24 GB once vs 24×3 GB.
- **vLLM continuous batching** handles the parallel calls efficiently.
- **Personas via system prompts** — same model, different "hat" per role.

### Why NOT a bigger Executor (e.g., 70B INT4)

Tempting, but 47 GB weights + 25 GB KV = 72 GB / 80 GB. Tight under load. If we have time and stable system by mid-afternoon, we can swap up. Default keeps headroom.

---

## Demo Flow (3-minute pitch)

### Setup before pitch (off-stage)
- Laptop on phone hotspot
- Tivoo-2 on the same hotspot, IP confirmed
- SSH tunnel to Brev open (`brev port-forward jarvis-track5 --port 7860`)
- Backup WAV recording loaded as fallback
- Two team members rehearsed the live script

### The 3-minute live demo

**Minute 1 — The setup + first interjection:**
- "Two of us are about to have a normal conversation. Jarvis will be listening AND learning."
- Tivoo-2 lights up `👂`
- Person A: *"Yo, we should grab drinks tonight."*
- Person B: *"Yeah! When are you free? Coffee earlier today wiped me out though."* *(seeds a fact: Person B doesn't want caffeine)*
- Tivoo-2 flashes `🤔`
- Voice Agent (TTS): *"You're both free at 6pm tonight at The Black Rose — want me on the calendar and pinging Julie?"*
- Person A: *"Wait, I have something at 6 — make it 7?"*
- Voice Agent: *"Julie's busy at 7. 7:30?"*
- Person B: *"Yeah, that works."*
- Voice Agent: *"Done. Calendar invite sent."* Tivoo-2: `✅`

**Minute 2 — The learning moment (this is the key differentiator):**
- A pause. Person A: *"Actually, want to grab coffee tomorrow morning too?"*
- Tivoo-2 flashes `🤔`
- Voice Agent (TTS): *"How about a smoothie spot instead? Earlier you mentioned coffee wipes you out — I figured I'd suggest something easier on the system."*
- Person B: *"Oh nice, yeah let's do it."*
- *(judges visibly react — the agent learned mid-conversation and applied it)*
- Voice Agent: *"9am at Juice Bar tomorrow — booking now."* Tivoo-2: `✅`

**Minute 3 — The technical pitch:**
- Brief screen-share of the architecture diagram + a memory inspector showing 3-4 distilled facts
- Pitch (15 sec each):
  1. **Architecture** — *"Continuous Whisper, per-sentence triage, plan-confirm-execute. Three Red Hat AI models on one A100, all served by vLLM."*
  2. **Memory** — *"Two-tier memory: episodic for recall, semantic for accumulated facts. The agent doesn't fine-tune — but it learns from every conversation in context. That's how it remembered the caffeine comment."*
  3. **NemoClaw moment** — show a calendar event with a planted prompt injection. Sandbox blocks the egress live. *"In production, agents read untrusted text. We sandboxed ours."*
  4. **Numbers** — *"Interjection latency under 1.5 sec. FP8 weights from Red Hat. $1.49/hr of compute."*

### Backup plan
If live mic fails: swap to pre-recorded WAV, run the same pipeline against it. Demo continues uninterrupted.

---

## Pitch (60-second judge version)

> *"Jarvis is a proactive scheduling agent that listens, **learns**, and offers to act when it senses intent. Three things make it interesting:*
>
> *One — every inference call runs on Red Hat AI's quantized models through vLLM. One A100 80GB, three models, ~35GB of weights, continuous batching across all of them. Red Hat FP8 weights are what make always-on listening cost-viable.*
>
> *Two — a two-tier memory system. Episodic memory for "what was just said" recall, semantic memory for distilled facts the agent learns over time. We don't update model weights — we extract structured facts and inject them at inference. Same UX outcome, no fine-tuning risk. You saw it remember the caffeine comment.*
>
> *Three — the agent runs inside NemoClaw's OpenShell sandbox. Every triage decision is grammar-guided. When it reads calendar events with prompt injection attempts [demo this live], NemoClaw blocks the egress. This is the difference between an agent demo and a production agent.*
>
> *Pattern is plan-confirm-execute: it proposes, asks, then dispatches a sub-agent. Interjection latency under 1.5 seconds. Sub-second TTS response. Cost: $1.49/hr."*

---

## Hardware & Software Stack

### Cloud / Compute
- **Brev** instance: `jarvis-track5`
- **GPU**: 1× A100 80GB (`massedcompute_A100_sxm4_80G_DGX`)
- **Cost**: $1.49/hr (~$12 for 8 hrs of hacking)
- **Disk**: 1TB
- **Boot time**: 2m30s
- **Credits available**: ~$30 personal coupon (more than enough)

### Software stack on Brev VM
| Layer | Tool |
|---|---|
| OS | Ubuntu 22.04 |
| Container runtime | Docker (required by NemoClaw) |
| Inference engine | vLLM 0.11+ |
| Agent runtime | NemoClaw + OpenShell sandbox |
| Vector DB | ChromaDB |
| Embeddings | `bge-small-en-v1.5` (pre-installed) |
| Node | Node.js 20 (NemoClaw dep) |

### Software stack on laptop
| Layer | Tool |
|---|---|
| TTS | `piper-tts` |
| Mic capture | sounddevice / pyaudio |
| UI | Gradio |
| SSH/tunnel | `brev shell`, `brev port-forward` |
| Tivoo control | Divoom HTTP API (Python httpx) |

### Demo room logistics
- **Audio**: Live mic (recommended: directional/headset USB mic per speaker; laptop omnidirectional is risky)
- **Network**: Phone hotspot connecting laptop ↔ Tivoo-2; laptop reaches Brev over the public internet
- **Backup**: Pre-recorded WAV of the script for live-demo fallback

---

## Implementation Plan

### Phase 0 — Provisioning *(in progress as of writing)*
- [x] Brev CLI installed
- [x] Coupon redeemed
- [x] A100 80GB instance `jarvis-track5` provisioned
- [ ] SSH ready (polling)
- [ ] HF token pushed to instance
- [ ] Repo cloned on instance
- [ ] `tier4-nemoclaw/setup.sh` executed (installs Node 20, Docker, NemoClaw, vLLM deps)

### Phase 1 — Stack up and validated (~30 min after SSH ready)
- [ ] Pull Red Hat AI models (`Llama-3.1-8B-Instruct-FP8`, `Mistral-Small-3-Instruct-24B-FP8`)
- [ ] Write 4-process `start_vllm.sh` for ports 9000–9002
- [ ] Validate each endpoint with `curl`/openai SDK smoke test
- [ ] Spin up ChromaDB on `:8001`
- [ ] Smoke test: `huggingface-cli` → `vllm serve` → completion
- [ ] Connect Tivoo-2 to phone hotspot + grab IP

### Phase 2 — Core agent code (~2.5 hours)
- [ ] `schema.py` — Pydantic: `TriageDecision`, `TaskProposal`, `ConfirmationIntent`, `Fact`
- [ ] `memory.py` — Chroma client with two collections (`episodic_memory`, `semantic_memory`); write/query helpers
- [ ] `fact_extractor.py` — periodic LLM call to distill facts from recent transcripts → semantic memory
- [ ] `middleware.py` — Triage with grammar-guided JSON; injects episodic + semantic context
- [ ] `planner.py` — picks task type, fills proposal; uses semantic memory for personalization
- [ ] `voice_agent.py` — natural language interjection + Piper TTS
- [ ] `sub_agent.py` — dispatches to one of 3 tools; logs decisions back to episodic memory
- [ ] `tools/email.py`, `tools/slack.py`, `tools/calendar.py` — mocked APIs
- [ ] `transcription.py` — Whisper via vLLM client + silero-vad
- [ ] `tts.py` — Piper wrapper

### Phase 3 — UI + Tivoo + glue (~1 hour)
- [ ] `app.py` — Gradio: live transcript, state panel, vector DB peek, manual override
- [ ] `divoom.py` — Tivoo-2 HTTP wrapper (state emoji, clock face, beep)
- [ ] State machine connecting all the pieces

### Phase 4 — NemoClaw policy + injection demo (~30 min)
- [ ] `nemoclaw_policy.yaml` — sandbox rules, network whitelist
- [ ] `data/poisoned_event.json` — calendar event with prompt-injection payload
- [ ] Wire agent into NemoClaw via `nemoclaw agentic-edge connect`
- [ ] Live test: poisoned event → sandbox blocks egress → agent narrates the block

### Phase 5 — Demo polish (~1 hour)
- [ ] Record backup WAV of the script
- [ ] Run full dress rehearsal end-to-end
- [ ] Prep slides: architecture diagram, latency numbers, NemoClaw screenshot
- [ ] Practice the 3-minute pitch

### Phase 6 — Submission
- [ ] Push to public GitHub repo
- [ ] Submit form (link, team, track)
- [ ] Demo at the venue

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Live mic fails in noisy room** | High | Pre-recorded WAV backup. Headset mics. |
| **Tivoo-2 can't reach laptop on hotspot** | Medium | Gradio UI panel as fallback (same emoji, on screen). Test connection 1 hour before demo. |
| **vLLM Whisper API has bugs in 0.11** | Medium | `faster-whisper` is a 10-min drop-in fallback. Same OpenAI-compatible shape. |
| **Llama 70B Executor OOM** | Avoided | Going with 24B Mistral by default. 70B only as stretch goal. |
| **NemoClaw setup.sh fails on a step** | Medium | Manual recovery: each step in setup.sh is idempotent and logged. SSH + re-run. |
| **A100 unavailable when we need to restart** | Low | `brev create` auto-falls-back to next-cheapest (massedcompute → hyperstack → denvr). |
| **HF gating on Llama 3.1** | Low (already accepted) | License accepted. Token validated as `MeLoClark`. |
| **False-positive interjections in demo** | Medium | Tune Triage threshold conservatively. Practice run before judges. Hard rate limit: 1 interject / 30 sec. |
| **Time pressure (one-day event)** | High | Prioritize end-to-end working demo over feature completeness. Phase 1–3 are MUST-HAVE; Phase 4 is the differentiator; Phase 5 is polish. |

---

## Open Questions

1. **Final model lineup** — confirm Red Hat AI model IDs (`Llama-3.1-8B-Instruct-FP8` + `Mistral-Small-3-Instruct-24B-FP8`) exist and are accessible.
2. **Tools to enable** — building all three (email/Slack/calendar) is ambitious. Demo only needs ONE working end-to-end. Pick the showcase: **calendar** (most natural for the demo conversation).
3. **Live mic vs WAV primary** — Clark voted live mic for wow factor; team to confirm and prep the WAV backup just in case.
4. **Tivoo-2 IP** — needs to be set up on phone hotspot before integration code is finalized.
5. **GitHub repo** — does the team already have one, or do we make a new one? (For submission.)
6. **FactExtractor cadence** — every N=10 chunks vs idle-triggered vs both? Default: both, with hard rate limit (1 extraction / 30 sec) to keep GPU free for triage.
7. **Memory persistence** — within session only (Chroma in-memory) vs persisted to disk on the Brev box? In-memory is simpler and the box dies tonight anyway. Default: in-memory.
8. **Demo rehearsal seeding** — do we pre-warm semantic memory with a few "previous session" facts so the agent has context to work with from minute 1, or start cold? Default: pre-warm with 3-5 facts (e.g., "Julie prefers afternoon meetings") so the demo flows smoothly.

---

## Glossary

- **vLLM** — Open-source LLM inference engine. Serves models with continuous batching, prefix caching, FP8/INT4 quantization support. The "kitchen."
- **NemoClaw** — NVIDIA's open-source agent runtime. Sandboxes agent code, enforces tool policy, network whitelist. The "playpen."
- **OpenShell** — NemoClaw's sandboxed shell environment (a Docker container with policy rules).
- **Brev** — NVIDIA's GPU cloud (the hardware provider).
- **Whisper** — OpenAI's speech-to-text model. Served as a vLLM endpoint.
- **Piper** — Lightweight neural TTS. Runs on CPU.
- **silero-vad** — Voice activity detection. Tells Whisper when to start/stop.
- **xgrammar** — vLLM's grammar-guided decoding. Enforces structured outputs at decode time.
- **ChromaDB** — Open-source vector database for semantic search.
- **Red Hat AI** — Red Hat's HF org (`huggingface.co/RedHatAI`). Publishes pre-quantized open-source models (FP8, INT4, W8A8).
- **FP8 / W4A16** — Quantization schemes. FP8 = 8-bit floats. W4A16 = 4-bit weights, 16-bit activations.
- **MoE** — Mixture of Experts. Model architecture where only a subset of params activates per token.
- **Tivoo-2** — Divoom's pixel-art Bluetooth/Wi-Fi speaker. We use it as a status display sidekick.

---

## Team & Roles (TBD)

| Role | Owner | Responsibility |
|---|---|---|
| Captain / Infra | Clark | Brev box, vLLM processes, demo coordination |
| Agent code | TBD | `middleware.py`, `planner.py`, `sub_agent.py` |
| Demo polish | TBD | Gradio UI, Tivoo-2, mic setup, dress rehearsal |
| Pitch lead | TBD | Slides, narrative, judge Q&A |

---

## Pre-event State (snapshot)

- ✅ Code of Conduct read
- ✅ Luma registration
- ✅ Discord joined
- ✅ Team formed (3 members)
- ✅ Track 5 chosen
- ✅ Brev account, coupons redeemed (~$30 personal)
- ✅ HF account + read token (`MeLoClark`)
- ✅ Llama 3.1 8B license accepted on HF
- ✅ Local dev env: vLLM 0.11.0 venv, Docker, Cursor, kubectl, git, hf CLI, Brev CLI
- ✅ Starter repo cloned: `~/vllm-hackathon/vLLM-hackathon/`
- ✅ Brev CLI authed as `clark.ohlenbusch-0`
- ✅ A100 80GB instance `jarvis-track5` provisioned (BUILDING → SSH ready soon)
- ⏳ SSH unlock + setup.sh execution
- ⏳ Model downloads (Llama 8B FP8, Mistral 24B FP8)
- ⏳ Coding begins
