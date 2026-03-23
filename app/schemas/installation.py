from datetime import datetime

from pydantic import BaseModel, Field


class InstallationRegisterRequest(BaseModel):
    installation_id: str = Field(min_length=8, max_length=120)
    platform: str | None = Field(default=None, max_length=32)
    app_version: str | None = Field(default=None, max_length=32)


class InstallationRegisterResponse(BaseModel):
    installation_id: str
    client_token: str
    created_at: datetime


class InstallationSummary(BaseModel):
    installation_id: str
    platform: str | None
    app_version: str | None
    created_at: datetime
    last_seen_at: datetime


class InstallationsPaginatedResponse(BaseModel):
    installations: list[InstallationSummary]
    total_count: int
    skip: int
    limit: int
    has_more: bool
