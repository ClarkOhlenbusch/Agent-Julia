#!/bin/bash
# Phase 1 — Box install: pip deps, model downloads, NemoClaw, vLLM launcher
# Run on the Brev box (jarvis-track5). Writes progress to ~/setup.log.

set -euo pipefail
exec > >(tee -a ~/setup.log) 2>&1

START=$(date +%s)
echo "============================================="
echo "  Phase 1 setup begin — $(date)"
echo "============================================="

# --- 1/6: System packages ---
echo "[1/6] System packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl wget jq htop tmux tree ffmpeg

# --- 2/6: Python deps ---
echo "[2/6] Python deps via pip..."
pip3 install --upgrade -q pip
pip3 install -q hf_transfer
pip3 install -q \
    vllm \
    httpx \
    pydantic \
    'huggingface_hub[cli]' \
    chromadb \
    sentence-transformers \
    openai \
    rich \
    fastapi \
    uvicorn

# --- 3/6: Set up HF auth + transfer accelerator ---
echo "[3/6] HF auth + accelerator..."
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo 'export HF_TOKEN=$(cat ~/.cache/huggingface/token)' >> ~/.bashrc
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

# --- 4/6: Download three models ---
echo "[4/6] Downloading models (sequential, accelerated)..."
mkdir -p ~/models

echo "  ... whisper-large-v3-turbo (~3 GB)"
hf download openai/whisper-large-v3-turbo \
    --local-dir ~/models/whisper-large-v3-turbo \
    2>&1 | tail -3

echo "  ... Meta-Llama-3.1-8B-Instruct-FP8 (~8 GB)"
hf download RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
    --local-dir ~/models/llama-3.1-8b-fp8 \
    2>&1 | tail -3

echo "  ... Mistral-Small-3.2-24B-Instruct-2506-FP8 (~24 GB)"
hf download RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8 \
    --local-dir ~/models/mistral-small-24b-fp8 \
    2>&1 | tail -3

# --- 5/6: NemoClaw install ---
echo "[5/6] Installing NemoClaw..."
curl -fsSL https://nvidia.com/nemoclaw.sh | bash || echo "  WARN: NemoClaw install returned non-zero; continuing."

# --- 6/6: Validation ---
echo "[6/6] Validation..."
python3 -c "import vllm; print('  vllm', vllm.__version__)"
python3 -c "import torch; print('  torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
python3 -c "import chromadb; print('  chromadb', chromadb.__version__)"
ls -la ~/models/
command -v nemoclaw && echo "  nemoclaw OK" || echo "  WARN: nemoclaw not on PATH"

ELAPSED=$(($(date +%s) - START))
echo "============================================="
echo "  PHASE_1_DONE — ${ELAPSED}s"
echo "============================================="
