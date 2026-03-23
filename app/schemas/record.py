from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalysisRecordCreateRequest(BaseModel):
    payload: dict[str, Any]


class AnalysisRecordCreateResponse(BaseModel):
    id: int
    installation_id: str
    created_at: datetime


class AnalysisRecordSummary(BaseModel):
    id: int
    installation_id: str
    schema_version: int | None
    route_type: str | None
    platform: str | None
    overall_status: str | None
    usable_for_training: bool
    created_at: datetime
    payload: dict[str, Any] | None = Field(default=None)


class AnalysisRecordTrainingUpdateRequest(BaseModel):
    usable_for_training: bool


class AnalysisRecordTrainingUpdateResponse(BaseModel):
    id: int
    usable_for_training: bool


class AnalysisRecordBulkTrainingUpdateRequest(BaseModel):
    record_ids: list[int] = Field(min_length=1)
    usable_for_training: bool


class AnalysisRecordBulkTrainingUpdateResponse(BaseModel):
    updated_count: int
    usable_for_training: bool


class AnalysisRecordsPaginatedResponse(BaseModel):
    records: list[AnalysisRecordSummary]
    total_count: int
    skip: int
    limit: int
    has_more: bool
