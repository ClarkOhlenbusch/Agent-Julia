from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "julia")
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    dd_site: str = os.getenv("DD_SITE", "us5.datadoghq.com")
    dd_llmobs_enabled: bool = os.getenv("DD_LLMOBS_ENABLED", "1").lower() in {"1", "true", "yes"}
    dd_ml_app: str = os.getenv("DD_LLMOBS_ML_APP", "julia")
    dd_api_key: str = os.getenv("DD_API_KEY", "")
    dd_app_key: str = os.getenv("DD_APP_KEY", "")
    dd_agentless_enabled: bool = os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "").lower() in {"1", "true", "yes"}
    dd_env: str = os.getenv("DD_ENV", "")
    dd_service: str = os.getenv("DD_SERVICE", "")


settings = Settings()
