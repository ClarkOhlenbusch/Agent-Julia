#!/bin/bash
# Start ChromaDB server on port 8001. Run on the Brev box.
mkdir -p ~/chroma-data
SESSION=jarvis-chroma
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "chroma run --host 0.0.0.0 --port 8001 --path ~/chroma-data 2>&1 | tee ~/chroma.log" C-m
echo "ChromaDB started in tmux session '$SESSION' on port 8001."
echo "Heartbeat: curl http://localhost:8001/api/v1/heartbeat"
