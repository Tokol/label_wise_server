from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

DistillationStatus = str


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
    distillation_status: DistillationStatus
    distillation_batch_id: str | None = None
    reviewed_at: datetime | None = None
    exported_at: datetime | None = None
    used_in_training_at: datetime | None = None
    excluded_reason: str | None = None
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


class AnalysisRecordDistillationUpdateRequest(BaseModel):
    distillation_status: DistillationStatus
    excluded_reason: str | None = Field(default=None, max_length=200)
    distillation_batch_id: str | None = Field(default=None, max_length=120)


class AnalysisRecordDistillationUpdateResponse(BaseModel):
    id: int
    distillation_status: DistillationStatus
    distillation_batch_id: str | None = None
    reviewed_at: datetime | None = None
    exported_at: datetime | None = None
    used_in_training_at: datetime | None = None
    excluded_reason: str | None = None


class AnalysisRecordBulkDistillationUpdateRequest(BaseModel):
    record_ids: list[int] = Field(min_length=1)
    distillation_status: DistillationStatus
    excluded_reason: str | None = Field(default=None, max_length=200)
    distillation_batch_id: str | None = Field(default=None, max_length=120)


class AnalysisRecordBulkDistillationUpdateResponse(BaseModel):
    updated_count: int
    distillation_status: DistillationStatus
    distillation_batch_id: str | None = None


class AnalysisRecordExportRequest(BaseModel):
    record_ids: list[int] | None = None
    batch_id: str | None = Field(default=None, max_length=120)
    include_payload: bool = True


class DistillationExportRecord(BaseModel):
    id: int
    installation_id: str
    overall_status: str | None
    usable_for_training: bool
    distillation_status: DistillationStatus
    distillation_batch_id: str
    created_at: datetime
    payload: dict[str, Any] | None = None


class AnalysisRecordExportResponse(BaseModel):
    batch_id: str
    exported_count: int
    records: list[DistillationExportRecord]


class AnalysisRecordsPaginatedResponse(BaseModel):
    records: list[AnalysisRecordSummary]
    total_count: int
    skip: int
    limit: int
    has_more: bool
