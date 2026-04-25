from __future__ import annotations

from fastapi import FastAPI
from fastapi import HTTPException

from julia_dag.config import settings
from julia_dag.orchestrator import handle_request
from julia_dag.schemas import InvokeRequest
from julia_dag.schemas import InvokeResponse


app = FastAPI(
    title="Julia DAG Service",
    version="0.1.0",
    description="DAG orchestrator for email, Slack, and calendar agent workflows.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/invoke", response_model=InvokeResponse)
def invoke(request: InvokeRequest) -> InvokeResponse:
    if not request.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction cannot be empty")
    return handle_request(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=False)
