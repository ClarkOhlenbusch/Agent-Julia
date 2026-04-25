#!/bin/bash
# Pre-demo health check. Run this from the laptop to verify everything is wired.
# Exits 0 only if every component is green.
set -u

cd "$(dirname "$0")/../.." || exit 1
fail=0
pass=0

ok()    { echo "  ✓ $*";              pass=$((pass+1)); }
bad()   { echo "  ✗ $*";              fail=$((fail+1)); }
warn()  { echo "  ! $*"; }
sec()   { echo ""; echo "── $* ──"; }

# ----------------------------------------------------------------------------
sec "Laptop"
SwitchAudioSource -c | grep -q "Divoom" && ok "audio output is Tivoo" || bad "audio output is NOT Tivoo (run: SwitchAudioSource -t output -s 'Divoom Tiivoo 2-Audio')"

if pgrep -f tivoo-control/tivoo_macos.py >/dev/null; then
    warn "tivoo BT bridge process active"
fi

if ps -ef | grep -q "[v]oice_relay"; then
    ok "voice_relay is running ($(ps -ef | grep '[v]oice_relay' | head -1 | awk '{print "PID="$2}'))"
else
    bad "voice_relay NOT running — start: ~/vllm-hackathon/.venv/bin/python3 -u scripts/laptop/voice_relay.py &"
fi

if curl -sf -o /dev/null --max-time 3 http://localhost:7860; then
    ok "Gradio reachable at http://localhost:7860"
else
    bad "Gradio NOT reachable — restart port-forward: brev port-forward jarvis-track5 -p 7860:7860 &"
fi

# ----------------------------------------------------------------------------
sec "Brev VM"
HOSTNAME_OUT=$(brev exec jarvis-track5 'echo READY' 2>&1 | tail -1)
if [ "$HOSTNAME_OUT" = "READY" ]; then
    ok "SSH reachable to box"
else
    bad "SSH to box failing"
    echo "Aborting further box checks."
    echo ""
    echo "── Result: $pass passed, $fail failed ──"
    exit 1
fi

BOX_OUT=$(brev exec jarvis-track5 '
for p in 9000 9001 9002; do
    if curl -sf -o /dev/null --max-time 2 http://localhost:$p/v1/models; then echo "vllm:$p:OK"; else echo "vllm:$p:DOWN"; fi
done
if curl -sf -o /dev/null --max-time 2 http://localhost:8001/api/v2/heartbeat; then echo "chroma:OK"; else echo "chroma:DOWN"; fi
if ss -tlnp 2>/dev/null | grep -q ":7860"; then echo "gradio:OK"; else echo "gradio:DOWN"; fi
PID=$(pgrep -f "python3.*app.py" | head -1)
if [ -n "$PID" ]; then
    DRY=$(cat /proc/$PID/environ | tr "\0" "\n" | grep "^DRY_RUN=" | cut -d= -f2)
    SLACK=$(cat /proc/$PID/environ | tr "\0" "\n" | grep "^SLACK_CHANNEL=" | cut -d= -f2)
    DDE=$(cat /proc/$PID/environ | tr "\0" "\n" | grep "^DD_LLMOBS_ENABLED=" | cut -d= -f2)
    echo "env:dry_run:${DRY:-unset}"
    echo "env:slack_channel:${SLACK:-unset}"
    echo "env:dd_llmobs:${DDE:-unset}"
fi
' 2>&1)

echo "$BOX_OUT" | while read -r line; do
    case "$line" in
        vllm:*:OK)        ok "vLLM ${line%%:OK}";;
        vllm:*:DOWN)      bad "vLLM ${line%%:DOWN} not responding";;
        chroma:OK)        ok "ChromaDB heartbeat";;
        chroma:DOWN)      bad "ChromaDB DOWN";;
        gradio:OK)        ok "Gradio listening on :7860";;
        gradio:DOWN)      bad "Gradio NOT listening on :7860";;
        env:dry_run:false) ok "DRY_RUN=false (real Slack posts ENABLED)";;
        env:dry_run:true)  warn "DRY_RUN=true (Slack posts will be simulated)";;
        env:dry_run:unset) bad "DRY_RUN env var missing";;
        env:slack_channel:C*) ok "SLACK_CHANNEL=${line#env:slack_channel:}";;
        env:slack_channel:*)  bad "SLACK_CHANNEL not set";;
        env:dd_llmobs:1)   ok "Datadog LLMObs enabled";;
        env:dd_llmobs:*)   warn "Datadog LLMObs not enabled (spans won't ship)";;
    esac
done

echo ""
echo "── Result: $pass passed, $fail failed ──"
[ "$fail" -eq 0 ]
