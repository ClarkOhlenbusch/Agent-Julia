import os

# vLLM host: set VLLM_HOST env var to point at the Brev VM IP (e.g. 154.54.100.126)
IN_NEMOCLAW = os.environ.get("NEMOCLAW_SANDBOX", "false").lower() == "true"
_VLLM_HOST = "host.openshell.internal" if IN_NEMOCLAW else os.environ.get("VLLM_HOST", "localhost")

WHISPER_BASE_URL = f"http://{_VLLM_HOST}:9000/v1"
TRIAGE_BASE_URL  = f"http://{_VLLM_HOST}:9001/v1"
AGENT_BASE_URL   = f"http://{_VLLM_HOST}:9002/v1"

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))

WHISPER_MODEL = "whisper-turbo"
TRIAGE_MODEL  = "triage"
AGENT_MODEL   = "planner"

TIVOO_IP   = os.environ.get("TIVOO_IP", "192.168.1.100")
TIVOO_PORT = 80

GRADIO_PORT = 7860
GRADIO_HOST = "0.0.0.0"

# Memory / extraction tuning
FACT_EXTRACT_EVERY_N       = 10   # chunks between automatic fact extraction
FACT_EXTRACT_IDLE_SECONDS  = 30   # idle silence before extraction fires
EPISODIC_TTL_SECONDS       = 600  # 10-minute rolling window
EPISODIC_TOP_K             = 3
SEMANTIC_TOP_K             = 5

# Safety guard-rails
TRIAGE_COOLDOWN_SECONDS = 30  # min gap between consecutive ACT decisions

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
PIPER_BINARY    = os.environ.get("PIPER_BINARY", "piper")
PIPER_MODEL     = os.environ.get("PIPER_MODEL", "en_US-lessac-medium")
