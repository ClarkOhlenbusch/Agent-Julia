#!/bin/bash
# Restart the Gradio app on the box with full env (DD + Slack + DRY_RUN=false).
# Loads the laptop's .env to source secrets, then injects them into the box's
# tmux session as process env vars (no .env on shared disk).
set -eu

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

if [ ! -f .env ]; then
    echo ".env not found at $PROJ_DIR/.env"; exit 1
fi
set -a; . ./.env; set +a

echo "── Restarting Julia app on jarvis-track5 (DRY_RUN=false) ──"
brev exec jarvis-track5 "tmux kill-session -t jarvis-app 2>/dev/null; tmux new-session -d -s jarvis-app \
  -e DD_LLMOBS_ENABLED=1 -e DD_LLMOBS_ML_APP=julia -e DD_SITE=us5.datadoghq.com \
  -e DD_LLMOBS_AGENTLESS_ENABLED=1 -e DD_API_KEY=$DD_API_KEY \
  -e DD_ENV=hackathon -e DD_SERVICE=julia-voice -e DD_TRACE_ENABLED=0 \
  -e SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN -e SLACK_APP_TOKEN=$SLACK_APP_TOKEN \
  -e SLACK_CHANNEL=$SLACK_CHANNEL -e DRY_RUN=false \
  'cd ~/jarvis-scheduler && export PATH=\$HOME/.local/bin:\$PATH && python3 -u app.py 2>&1 | tee ~/app.log'"

echo "  waiting for Gradio to bind :7860 ..."
until brev exec jarvis-track5 "ss -tlnp 2>/dev/null | grep -q ':7860'" >/dev/null 2>&1; do
    sleep 2
done
echo "  app listening"

# Reset port-forward
pkill -f "brev port-forward" 2>/dev/null || true
sleep 1
brev port-forward jarvis-track5 -p 7860:7860 >/dev/null 2>&1 &
sleep 4

if curl -sf -o /dev/null --max-time 3 http://localhost:7860; then
    echo "  ✓ http://localhost:7860 reachable"
else
    echo "  ! Gradio not yet reachable — wait a few seconds and refresh Chrome"
fi
