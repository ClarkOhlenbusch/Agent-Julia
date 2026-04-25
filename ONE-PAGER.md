# Jarvis — A Proactive Scheduling Agent That Listens and Learns

**Track 5: Agentic Edge powered by NemoClaw** · The Open Accelerator Hackathon · April 25, 2026

---

## The Problem

Knowledge workers lose hours each week to the cognitive overhead of scheduling — coordinating calendars, sending follow-up emails, pinging Slack channels, and remembering everyone's preferences. These are tasks that emerge naturally from conversation ("let's grab drinks tonight") but require manual effort to execute. Existing assistants are reactive: you have to stop your conversation, open an app, and tell it what to do. We wanted an agent that *participates*.

## Our Solution

**Jarvis** is a continuous-listening, proactive scheduling agent. It sits in on a live conversation, detects when both parties agree on a plan, and immediately acts — booking calendar events, drafting emails, or posting to Slack — then confirms what it did in natural speech. It gets smarter over time through a two-tier memory system that distills facts from conversation ("Sam doesn't like coffee after 3pm") and uses them to personalize future proposals.

The key differentiator: **Jarvis doesn't ask permission for things you already agreed on.** When triage detects mutual confirmation, it plans, executes, and narrates — no interruption loop.

## Architecture

```
LAPTOP (mic + TTS + Gradio UI)
    │ audio stream via SSH tunnel
    ▼
BREV VM — A100 80GB
    ├─ vLLM :9000  Whisper turbo         (continuous transcription)
    ├─ vLLM :9001  Llama 3.1 8B FP8     (triage: STORE / DISCARD / ACT)
    ├─ vLLM :9002  Mistral Small 24B FP8 (planner + voice + sub-agent)
    ├─ ChromaDB    episodic + semantic memory
    └─ NemoClaw sandbox (agent code runs here, network-isolated)
```

**Data flow per turn:** Whisper transcribes → episodic buffer stores → triage classifies (grammar-guided JSON) → if ACT: planner proposes → sub-agent executes tool → voice agent narrates result via TTS. In parallel, a fact extractor periodically distills structured facts into long-term semantic memory.

## NemoClaw Integration

NemoClaw is not bolted on — it's the execution boundary. All agent code runs inside an OpenShell sandbox with:

- **Network whitelist:** egress locked to vLLM ports + ChromaDB only. No internet access.
- **Filesystem isolation:** agent can read code but cannot modify itself. Writes restricted to `/sandbox` and `/tmp`.
- **Tool policy:** `max_tool_calls_per_turn = 1`, structured output enforced, schema validation on every tool call.
- **Live demo:** We feed the agent a calendar event containing a prompt injection payload that attempts to exfiltrate data to `attacker.com`. NemoClaw blocks the egress. The agent reports the block. This is the difference between a demo agent and a production agent.

## Tech Stack

| Layer | Technology |
|---|---|
| Inference engine | **vLLM** — 3 model endpoints, continuous batching, FP8 quantization, grammar-guided decoding via xgrammar |
| Models | **RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8** (triage), **RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8** (planner/voice/executor), **openai/whisper-large-v3-turbo** (STT) |
| Agent sandbox | **NemoClaw** OpenShell — policy-enforced Docker container |
| Memory | **ChromaDB** (episodic rolling buffer + semantic fact store), **bge-small-en-v1.5** embeddings |
| TTS | **Piper** (CPU, laptop-side) |
| UI | **Gradio** (live transcript, agent state, memory inspector) |
| Compute | 1× A100 80GB on Brev · ~35GB weights · $1.49/hr |

## Two-Tier Memory — How the Agent Learns

- **Episodic memory:** Rolling 10-minute buffer of raw transcript chunks. Provides immediate conversational context to triage and planner. Enables deduplication ("we already discussed this").
- **Semantic memory:** Structured facts extracted by an LLM (subject, type, fact, confidence) — preferences, relationships, decisions, social patterns. Persists across the session. Injected into every triage and planning call to personalize proposals.

No model weights are updated. The agent learns by accumulating context and retrieving it at inference time.

## Demo Highlights (3 minutes)

1. **Live conversation → automatic booking:** Two people agree on drinks at 7:30. Jarvis detects mutual agreement, books the calendar event, and confirms aloud: *"Got it — drinks at seven thirty are on both your calendars."*
2. **Memory in action:** One person mentions coffee wipes them out. Minutes later, when someone suggests coffee, Jarvis proposes a smoothie spot instead — it remembered the preference mid-conversation.
3. **NemoClaw security demo:** A poisoned calendar event with a hidden prompt injection tries to exfiltrate data. The sandbox blocks it live. *"In production, agents read untrusted text. We sandboxed ours."*

---

**Team:** Clark Ohlenbusch (captain/infra) + 2 · **Repo:** [github.com/jarvis-scheduler](https://github.com) · **Skill Lane:** Builder / Deep Tech
