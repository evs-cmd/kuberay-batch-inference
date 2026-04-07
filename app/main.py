"""
KubeRay Batch Inference API.

Submit batch prompts → Ray distributes across GPU workers → poll for results.

Usage:
    ray start --head --num-gpus=1
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import ray
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.models import (
    BatchRequest,
    BatchResponse,
    BatchStatusResponse,
    JobStatus,
    PromptResult,
)
from app.inference import BatchOrchestrator
from app.config import API_KEY, MODEL_NAME, RESULTS_DIR

MODEL_ALIASES = {
    "Qwen/Qwen2.5-0.5B-Instruct": MODEL_NAME,
}

logger = logging.getLogger("batch-api")
logging.basicConfig(level=logging.INFO)

metrics = {
    "jobs_submitted": 0,
    "jobs_completed": 0,
    "jobs_failed": 0,
    "prompts_processed": 0,
    "total_tokens_generated": 0,
}

jobs: dict[str, dict] = {}
orchestrator: BatchOrchestrator | None = None


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    logger.info(f"Ray resources: {ray.cluster_resources()}")

    orchestrator = BatchOrchestrator(model_name=MODEL_NAME)
    logger.info("API ready.")

    yield

    if orchestrator:
        orchestrator.shutdown()
    ray.shutdown()


app = FastAPI(title="KubeRay Batch Inference API", version="1.0.0", lifespan=lifespan)


# Auth
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in {"/health", "/metrics", "/docs", "/openapi.json"}:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(status_code=401, content={"error": "Missing X-API-Key header"})
    if api_key != API_KEY:
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    return await call_next(request)


# Routes
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "ray_connected": ray.is_initialized(),
        "cluster_resources": ray.cluster_resources() if ray.is_initialized() else {},
    }


@app.post("/v1/batches", response_model=BatchResponse)
async def submit_batch(request: BatchRequest):
    """Submit a batch inference job. Returns immediately with a job_id."""
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    resolved_model = MODEL_ALIASES.get(request.model, MODEL_NAME)
    prompts = [p.prompt for p in request.input]

    jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "model": resolved_model,
        "prompts": prompts,
        "max_tokens": request.max_tokens,
        "total_prompts": len(prompts),
        "completed_prompts": 0,
        "created_at": now,
        "completed_at": None,
        "error": None,
    }

    metrics["jobs_submitted"] += 1
    asyncio.create_task(_process_batch(job_id))

    return BatchResponse(
        job_id=job_id, status=JobStatus.PENDING,
        model=resolved_model, total_prompts=len(prompts), created_at=now,
    )


@app.get("/v1/batches/{job_id}", response_model=BatchStatusResponse)
async def get_batch_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": f"Job {job_id} not found"})

    job = jobs[job_id]
    results = None
    if job["status"] == JobStatus.COMPLETED:
        path = RESULTS_DIR / f"{job_id}.json"
        if path.exists():
            with open(path) as f:
                results = [PromptResult(**r) for r in json.load(f)]

    return BatchStatusResponse(
        job_id=job["job_id"], status=job["status"], model=job["model"],
        total_prompts=job["total_prompts"], completed_prompts=job["completed_prompts"],
        created_at=job["created_at"], completed_at=job["completed_at"],
        results=results, error=job["error"],
    )


@app.get("/v1/batches")
async def list_batches():
    return {
        "jobs": [
            {k: j[k] for k in ("job_id", "status", "model", "total_prompts", "completed_prompts", "created_at")}
            for j in jobs.values()
        ]
    }


@app.get("/metrics")
async def get_metrics():
    lines = [f"kuberay_batch_{k} {v}" for k, v in metrics.items()]
    for status in JobStatus:
        count = sum(1 for j in jobs.values() if j["status"] == status)
        lines.append(f'kuberay_batch_jobs{{status="{status.value}"}} {count}')
    return "\n".join(lines) + "\n"


# Job processor
async def _process_batch(job_id: str):
    job = jobs[job_id]
    job["status"] = JobStatus.RUNNING

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, orchestrator.run_batch, job["prompts"], job["max_tokens"],
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / f"{job_id}.json", "w") as f:
            json.dump(results, f, indent=2)

        job["status"] = JobStatus.COMPLETED
        job["completed_prompts"] = len(results)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        metrics["jobs_completed"] += 1
        metrics["prompts_processed"] += len(results)
        metrics["total_tokens_generated"] += sum(r.get("tokens_generated", 0) for r in results)
        logger.info(f"Job {job_id} completed: {len(results)} prompts")

    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)
        metrics["jobs_failed"] += 1
        logger.error(f"Job {job_id} failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
