# Quickstart — Running the Stack

> Read [README.md](README.md) first for the architecture and demo flow.
> This doc is the boot sequence + cheatsheet for the team during the hackathon.

## Box info
- **Brev instance**: `jarvis-track5` (massedcompute_A100_sxm4_80G_DGX, $1.49/hr)
- **User**: `shadeform`
- **Home**: `/home/shadeform`
- **Project on box**: `/home/shadeform/jarvis-scheduler`
- **Models on box**: `/home/shadeform/models/`

## Connect to the box

```bash
# From your laptop
brev shell jarvis-track5            # interactive SSH
brev exec jarvis-track5 "<cmd>"     # run a one-shot command
brev copy <local> jarvis-track5:<remote>   # push a file
brev port-forward jarvis-track5 --port 7860   # expose Gradio to laptop
brev ls                              # status
brev stop jarvis-track5              # stop (saves credits)
```

## Boot sequence (in tmux panes — easier to monitor)

```bash
# 1. Make sure setup.sh has finished — check ~/setup.log on the box.
brev exec jarvis-track5 "tail -10 ~/setup.log"
# Look for "PHASE_1_DONE"

# 2. Start ChromaDB (port 8001)
brev exec jarvis-track5 "cd ~/jarvis-scheduler && bash scripts/start_chroma.sh"

# 3. Start the 3 vLLM services in tmux (ports 9000/9001/9002)
brev exec jarvis-track5 "cd ~/jarvis-scheduler && bash scripts/start_vllm_services.sh"

# 4. Wait ~3-5 min for all 3 to load. Monitor any pane:
brev shell jarvis-track5
# inside SSH:
tmux attach -t jarvis-vllm
# Ctrl+b then arrow keys to switch panes
# Ctrl+b then d to detach (panes keep running)

# 5. Smoke-test endpoints from inside the box
curl http://localhost:9000/v1/models | jq .data[0].id     # whisper-turbo
curl http://localhost:9001/v1/models | jq .data[0].id     # triage
curl http://localhost:9002/v1/models | jq .data[0].id     # planner
curl http://localhost:8001/api/v1/heartbeat               # chroma

# 6. Smoke-test the agent (memory + middleware) — text-only, no audio needed
cd ~/jarvis-scheduler
python3 -m memory                  # writes a chunk, queries back
python3 -m agents.middleware "Yo, want to grab drinks at 6 tonight?"
python3 -m agents.planner "Drinks tonight at 7?"

# 7. Run a full text-script demo
python3 agent.py --reset --seed --text-script
```

## Run the Gradio UI

```bash
# On the box
cd ~/jarvis-scheduler
python3 app.py
# Listens on 0.0.0.0:7860

# On your laptop, in another terminal — port-forward back
brev port-forward jarvis-track5 --port 7860
# Then open http://localhost:7860 in Chrome
```

The Gradio UI has:
- **Audio in** — click the mic, speak a chunk, click "Process audio"
- **Text inject** — type a chunk if mic is broken
- **Live log / Episodic / Semantic / Calendar** tabs
- **Reset memory / Reload seed facts** controls
- 🔊 Speak it — TTS the agent's pending question

## Run inside NemoClaw sandbox (for the demo)

```bash
# On the box, after ChromaDB and vLLM are up
bash ~/jarvis-scheduler/scripts/onboard_nemoclaw.sh
# Then enter the sandbox
nemoclaw jarvis-scheduler connect
# Inside the sandbox:
cd ~/jarvis-scheduler
python3 agent.py --seed --text-script
# Or: python3 app.py
```

## Demo controls

- **Pre-warm** before pitching: `python3 -c "import memory; memory.seed_from_file('data/seed_facts.json')"`
- **Reset** between rehearsals: in Gradio click "🗑 Reset memory" or run `python3 -c "import memory, tools.calendar; memory.reset_all(); tools.calendar.reset_session()"`
- **Plant the poisoned event** for the NemoClaw demo: load `data/poisoned_event.json`, watch network log for blocked attacker.com egress.

## Common issues

| Symptom | Fix |
|---|---|
| `vllm: command not found` | `export PATH="$HOME/.local/bin:$PATH"` (already in `~/.bashrc`) |
| `hf: command not found` | Same — PATH issue |
| `vLLM serve` crashes with OOM | Reduce `--gpu-memory-utilization` in `start_vllm_services.sh` |
| Triage returns empty / parse_error | Check Llama 8B is loaded; `curl :9001/v1/models` |
| Whisper transcription returns empty | Audio chunk too short; aim for 2-4 sec chunks |
| Chroma "collection not found" | First run? Call `memory._ensure()` once or write a chunk |
| `SSH connection failed` | Box's external networking layer is flaky on massedcompute. Wait 10-30s and retry. |

## Files modified during the day — what to commit

```bash
cd ~/vllm-hackathon/jarvis-scheduler
git status
git add <files>
git commit -m "..."
git push
# Repo: https://github.com/ClarkOhlenbusch/Agent-Julia
```

## Stopping the box (don't forget!)

```bash
brev stop jarvis-track5
# Or fully delete:
brev delete jarvis-track5
```
