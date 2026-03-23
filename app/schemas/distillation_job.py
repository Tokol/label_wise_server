from datetime import datetime

from pydantic import BaseModel, Field


class DistillationJobCreateRequest(BaseModel):
    batch_id: str = Field(min_length=1, max_length=120)
    base_model: str = Field(min_length=1, max_length=120)
    task_type: str = Field(min_length=1, max_length=64)
    dataset_mode: str = Field(default="single_batch", max_length=64)


class DistillationJobSummary(BaseModel):
    id: int
    batch_id: str
    base_model: str
    task_type: str
    dataset_mode: str
    status: str
    progress_stage: str
    train_record_count: int | None = None
    validation_record_count: int | None = None
    metrics_json: dict | None = None
    error_message: str | None = None
    progress_percent: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class DistillationJobListResponse(BaseModel):
    jobs: list[DistillationJobSummary]
    total_count: int
    skip: int
    limit: int
    has_more: bool
