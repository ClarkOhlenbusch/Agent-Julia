"""LLMObs observability for the voice pipeline.

Shares the same env-var conventions as julia_dag/config.py so the voice
pipeline and the DAG service appear in the same Datadog dashboard
(filter by ml_app: julia).

Safe to import even when DD_LLMOBS_ENABLED is unset — decorators become no-ops
and the workflow context manager falls back to a no-op span.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from ddtrace.llmobs import LLMObs
    from ddtrace.llmobs.decorators import agent, task
    _DDTRACE_AVAILABLE = True
except Exception:
    _DDTRACE_AVAILABLE = False

    def task(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]
        def deco(fn): return fn
        return deco

    def agent(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]
        def deco(fn): return fn
        return deco


_LLMOBS_ENABLED = os.getenv("DD_LLMOBS_ENABLED", "").lower() in {"1", "true", "yes"}
_DD_API_KEY = os.getenv("DD_API_KEY", "")
_DD_APP_KEY = os.getenv("DD_APP_KEY", "")
_DD_SITE = os.getenv("DD_SITE", "")
_DD_ENV = os.getenv("DD_ENV", "")
_DD_SERVICE = os.getenv("DD_SERVICE", "")
_DD_ML_APP = os.getenv("DD_LLMOBS_ML_APP", "julia")
_DD_AGENTLESS = os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "").lower() in {"1", "true", "yes"}

_initialized = False


def ensure_llmobs_enabled() -> bool:
    """Idempotent. Returns True iff LLMObs is now active."""
    global _initialized
    if not _DDTRACE_AVAILABLE:
        return False
    if _initialized:
        return LLMObs.enabled
    if not _LLMOBS_ENABLED:
        _initialized = True
        return False

    enable_kwargs: dict[str, Any] = {"ml_app": _DD_ML_APP}
    if _DD_API_KEY:
        enable_kwargs["agentless_enabled"] = _DD_AGENTLESS or True
        enable_kwargs["api_key"] = _DD_API_KEY
    if _DD_APP_KEY:
        enable_kwargs["app_key"] = _DD_APP_KEY
    if _DD_SITE:
        enable_kwargs["site"] = _DD_SITE
    if _DD_ENV:
        enable_kwargs["env"] = _DD_ENV
    if _DD_SERVICE:
        enable_kwargs["service"] = _DD_SERVICE
    try:
        LLMObs.enable(**enable_kwargs)
    except Exception:
        pass
    _initialized = True
    return LLMObs.enabled


@contextmanager
def workflow(name: str, session_id: Optional[str] = None) -> Iterator[Any]:
    """Wraps a unit of agent work as an LLMObs workflow span (or no-op)."""
    if not _DDTRACE_AVAILABLE or not ensure_llmobs_enabled():
        yield None
        return
    with LLMObs.workflow(name=name, session_id=session_id) as span:
        yield span


def annotate(span: Any, *, input_data: Any = None, output_data: Any = None,
             metadata: Optional[dict] = None) -> None:
    if not _DDTRACE_AVAILABLE or span is None:
        return
    try:
        kwargs: dict[str, Any] = {"span": span}
        if input_data is not None:
            kwargs["input_data"] = input_data
        if output_data is not None:
            kwargs["output_data"] = output_data
        if metadata:
            kwargs["metadata"] = metadata
        LLMObs.annotate(**kwargs)
    except Exception:
        pass


__all__ = ["task", "agent", "workflow", "annotate", "ensure_llmobs_enabled"]
