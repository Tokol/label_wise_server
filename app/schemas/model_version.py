from datetime import datetime

from pydantic import BaseModel


class ModelVersionSummary(BaseModel):
    id: int
    job_id: int
    batch_id: str
    model_name: str
    base_model: str
    task_type: str
    status: str
    metrics_json: dict | None = None
    artifact_uri: str | None = None
    created_at: datetime
    activated_at: datetime | None = None
    archived_at: datetime | None = None


class ModelVersionListResponse(BaseModel):
    versions: list[ModelVersionSummary]
    total_count: int
    skip: int
    limit: int
    has_more: bool
