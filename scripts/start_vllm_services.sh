#!/bin/bash
# Start three vLLM services on different ports, each in its own tmux pane.
# Run on the Brev box after phase1_setup.sh completes.

set -euo pipefail

WHISPER=~/models/whisper-large-v3-turbo
TRIAGE=~/models/llama-3.1-8b-fp8
PLANNER=~/models/mistral-small-24b-fp8

# Sanity check models exist
for m in "$WHISPER" "$TRIAGE" "$PLANNER"; do
    if [ ! -d "$m" ]; then
        echo "FATAL: model not found at $m"
        echo "Run ~/phase1_setup.sh first."
        exit 1
    fi
done

SESSION=jarvis-vllm
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n whisper

# Pane 0: Whisper transcription (vLLM auto-detects model type from config)
tmux send-keys -t $SESSION:whisper "
vllm serve $WHISPER \
    --port 9000 \
    --gpu-memory-utilization 0.12 \
    --max-num-seqs 4 \
    --served-model-name whisper-turbo \
    2>&1 | tee ~/vllm-whisper.log
" C-m

# Pane 1: Triage (Llama 3.1 8B FP8)
tmux new-window -t $SESSION -n triage
tmux send-keys -t $SESSION:triage "
sleep 30  # let Whisper start first to grab its allocation cleanly
vllm serve $TRIAGE \
    --port 9001 \
    --gpu-memory-utilization 0.18 \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name triage \
    2>&1 | tee ~/vllm-triage.log
" C-m

# Pane 2: Planner / Voice / Sub-Agent (Mistral Small 24B FP8)
tmux new-window -t $SESSION -n planner
tmux send-keys -t $SESSION:planner "
sleep 60  # stagger so allocation is sequential
vllm serve $PLANNER \
    --port 9002 \
    --gpu-memory-utilization 0.50 \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --served-model-name planner \
    2>&1 | tee ~/vllm-planner.log
" C-m

echo "tmux session '$SESSION' started with 3 panes."
echo "Attach with: tmux attach -t $SESSION"
echo "Logs: ~/vllm-whisper.log ~/vllm-triage.log ~/vllm-planner.log"
echo ""
echo "Wait ~2-4 min for all three to initialize, then check:"
echo "  curl http://localhost:9000/v1/models"
echo "  curl http://localhost:9001/v1/models"
echo "  curl http://localhost:9002/v1/models"
