"""Pydantic models for the batch inference API."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from app.config import MODEL_NAME


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PromptInput(BaseModel):
    prompt: str


class BatchRequest(BaseModel):
    model: str = MODEL_NAME
    input: list[PromptInput]
    max_tokens: int = Field(default=50, ge=1, le=2048)


class PromptResult(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int
    finish_reason: str


class BatchResponse(BaseModel):
    job_id: str
    status: JobStatus
    model: str
    total_prompts: int
    created_at: str


class BatchStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    model: str
    total_prompts: int
    completed_prompts: int
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[list[PromptResult]] = None
    error: Optional[str] = None
