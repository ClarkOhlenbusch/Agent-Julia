#!/bin/bash
# One-shot NemoClaw onboarding. Adapted from launchable-configs/tier4-nemoclaw/setup.sh.
# Points NemoClaw at our Mistral 24B (planner) endpoint.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

NEMOCLAW_PROVIDER=custom \
NEMOCLAW_ENDPOINT_URL=http://localhost:9002/v1 \
NEMOCLAW_MODEL=planner \
COMPATIBLE_API_KEY=dummy \
NEMOCLAW_PREFERRED_API=openai-completions \
nemoclaw onboard --non-interactive --name jarvis-scheduler

echo "Onboarded. To enter the sandbox:"
echo "    nemoclaw jarvis-scheduler connect"
echo "Then inside the sandbox:"
echo "    cd ~/jarvis-scheduler && python3 agent.py --seed --text-script"
