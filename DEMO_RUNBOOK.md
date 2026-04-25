# Julia — Demo Runbook

> Live demo: continuous-listening agent that detects mutual scheduling agreement,
> posts a real Slack message to `#julia-demos`, and narrates via the Tivoo speaker.

---

## 60-second elevator pitch

> *"Julia is a proactive scheduling agent. She listens to a live conversation,
> detects when both parties have agreed on a plan, drafts a Slack message,
> sends it to the channel, and confirms in voice — all without anyone having
> to stop and tell her what to do.*
>
> *Under the hood: continuous Whisper transcription, a triage LLM that gates
> on mutual agreement, a planner that drafts the Slack message, a sub-agent
> that actually posts via the Slack Web API, and Kokoro TTS that speaks the
> result through the Tivoo. Every step is traced in Datadog LLM Observability.
> All four LLMs run on a single A100 80GB Brev box on Red Hat AI's quantized
> models."*

---

## Architecture (one slide)

```
LAPTOP                                          BREV A100 80GB
─────────                                       ─────────
mic ─────────► Gradio @ :7860 ◄─── port-forward ─── Gradio app.py
                                                     │
                                                     ├─► vLLM :9000 Whisper
                                                     ├─► vLLM :9001 Triage   (Llama 3.1 8B FP8)
                                                     ├─► vLLM :9002 Planner  (Mistral 24B FP8)
                                                     ├─► ChromaDB (memory)
                                                     ├─► Slack Web API ────► #julia-demos
                                                     └─► /tmp/jarvis_result.txt
                                                              │
laptop voice_relay ◄─── SSH polls ──────────────────────────┘
   │ Kokoro TTS (Jessica)
   ▼
Tivoo speaker
```

Spans for every turn ship to Datadog (`ml_app: julia`).

---

## Before the demo (the morning of)

Done already, but verify:

- [x] Brev box `jarvis-track5` running, vLLM × 3 + Chroma up
- [x] `app.py` running with DD + Slack env vars
- [x] Port-forward laptop:7860 → box:7860
- [x] Voice relay running on laptop, polling box state
- [x] Tivoo paired as Bluetooth audio
- [x] System audio output set to **Divoom Tiivoo 2-Audio**
- [x] Slack bot invited to channel `C0AVDAT0UJY`
- [x] DRY_RUN=false (real posts will happen)

Run `bash scripts/laptop/preflight.sh` to verify all of the above.

---

## Right before going on stage (T-30 sec)

```bash
bash ~/vllm-hackathon/jarvis-scheduler/scripts/laptop/demo_reset.sh
```

This:
- Wipes Chroma collections (clean memory)
- Truncates state files (`/tmp/jarvis_*.txt`)
- Clears `_BOOKED_EVENTS` calendar list (it's session-local; restart drops it)

Open Chrome at **http://localhost:7860**.

---

## The 3-minute demo

### Minute 1 — Setup (15 sec)

> *"This is Julia. She's listening to me right now. Let me have a normal
> conversation with my friend and see if she picks up on a plan."*

Click **Record** in the mic widget (top-left).

### Minute 1.5 — Plant the proposal (10 sec)

**Person A (you):** *"Hey, want to grab drinks at 7:30 tonight near Fort Point?"*

Watch the **Live log** tab on the right:
- `[user] Hey, want to grab drinks at 7:30 tonight near Fort Point?`
- `TRIAGE: STORE — proposal noted, no agreement yet`

She's noting the proposal but **NOT acting** because nobody has agreed yet.

### Minute 2 — Confirm + watch the magic (60 sec)

**Person B:** *"Yeah, sounds good — let's do it!"*

Within ~5 seconds:
- Live log: `TRIAGE: ACT — explicit agreement language after recent proposal`
- Live log: `PLAN: "Hey everyone! We're meeting up for drinks at 7:30 PM tonight near Fort Point."`
- Live log: `EXECUTED: success=True dry_run=False`
- **Slack channel** receives the real message
- **Tivoo speaker** says: *"Got it — drinks at seven thirty are all set!"*

### Minute 2.5 — The narration moment (10 sec)

> *"Notice she didn't ask 'should I post this?' — both of you already agreed.
> Asking again would be deferential and slow. The agent says 'done', not
> 'may I'."*

### Minute 3 — Show the trace (60 sec)

Switch to a Datadog tab (https://us5.datadoghq.com/llm/sessions, filter `ml_app: julia`).

Click into the most recent `jarvis_voice_turn` workflow span. Walk through:
- `triage` task span (input transcript, output route + reason)
- `plan` task span (output TaskProposal: recipients + content + voice_prompt)
- `sub_agent` agent span (tool result: success, channel, Slack ts)
- `voice_narration` agent span (the spoken text)

> *"Every decision the agent makes is traceable. We can see what triggered the
> interjection, what the planner drafted, when Slack accepted the post, and
> what was spoken back."*

---

## Repeat the demo cleanly

Between rehearsals or back-to-back shows:

```bash
bash ~/vllm-hackathon/jarvis-scheduler/scripts/laptop/demo_reset.sh
```

---

## Troubleshooting (in priority order)

| Symptom | Fix |
|---|---|
| Mic widget shows recording but nothing happens | Click **stop** before clicking Process. Or use **Text inject** as fallback. |
| Live log says `TRIAGE: DISCARD — repeated…` on a real confirmation | Reset memory: `bash scripts/laptop/demo_reset.sh`. Stale chunks confused the model. |
| No Slack message arrives | Check `brev exec jarvis-track5 'tail -20 ~/app.log \| grep -i slack'`. If `not_in_channel`: `/invite @julia` in `#julia-demos`. |
| Tivoo silent | `SwitchAudioSource -t output -s "Divoom Tiivoo 2-Audio"`. Or test: `say "test"`. |
| Chrome can't reach 7860 | `pkill -f "brev port-forward"; brev port-forward jarvis-track5 -p 7860:7860 &` |
| Box ssh times out | Wait 15s and retry. Massedcompute SSH is intermittently flaky. |

---

## Backup plan (if live mic dies on stage)

The **Text inject** field below the mic widget lets you skip the mic entirely:

1. Type the proposal: `Yo, want to grab drinks at 7:30 tonight near Fort Point?` · speaker `speaker_1` · **Inject**
2. Type the confirmation: `Yeah let's do it, sounds good` · speaker `speaker_2` · **Inject**

Identical pipeline, no mic dependency. Slack post + Tivoo narration still happen.

---

## Submission

Repo: https://github.com/ClarkOhlenbusch/Agent-Julia
Submission form: (paste the URL the hackathon Discord posts)

Include in the submission:
- Repo URL
- This runbook
- Datadog dashboard link with `ml_app: julia` filter applied
- One-pager (`ONE-PAGER.md`)
