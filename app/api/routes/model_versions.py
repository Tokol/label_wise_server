from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.model_version import ModelVersion
from app.schemas.model_version import (
    ModelVersionListResponse,
    ModelVersionStatusUpdateRequest,
    ModelVersionSummary,
)

router = APIRouter(prefix="/model-versions", tags=["model-versions"])


def _summary(version: ModelVersion) -> ModelVersionSummary:
    return ModelVersionSummary(
        id=version.id,
        job_id=version.job_id,
        batch_id=version.batch_id,
        model_name=version.model_name,
        base_model=version.base_model,
        task_type=version.task_type,
        status=version.status,
        metrics_json=version.metrics_json,
        artifact_uri=version.artifact_uri,
        created_at=version.created_at,
        activated_at=version.activated_at,
        archived_at=version.archived_at,
    )


@router.get("", response_model=ModelVersionListResponse)
def list_model_versions(
    skip: int = 0,
    limit: int = 10,
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    stmt = select(ModelVersion)
    if status:
        stmt = stmt.where(ModelVersion.status == status)

    total_count = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(ModelVersion.created_at.desc()).offset(skip).limit(limit)).all()

    return ModelVersionListResponse(
        versions=[_summary(row) for row in rows],
        total_count=total_count,
        skip=skip,
        limit=limit,
        has_more=(skip + limit) < total_count,
    )


@router.patch("/{version_id}/status", response_model=ModelVersionSummary)
def update_model_version_status(
    version_id: int,
    payload: ModelVersionStatusUpdateRequest,
    db: Session = Depends(get_db),
):
    version = db.get(ModelVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="model version not found")

    previous_status = version.status
    if payload.status == "active_test":
        active_versions = db.scalars(select(ModelVersion).where(ModelVersion.status == "active_test")).all()
        for active in active_versions:
            if active.id == version.id:
                continue
            active.status = "ready_for_test"
            active.activated_at = None
            db.add(active)
        version.status = "active_test"
        version.activated_at = datetime.utcnow()
        version.archived_at = None
    elif payload.status == "archived":
        version.status = "archived"
        version.archived_at = datetime.utcnow()
        if previous_status == "active_test":
            version.activated_at = None
    elif payload.status == "ready_for_test":
        version.status = "ready_for_test"
        version.activated_at = None
        version.archived_at = None
    else:
        raise HTTPException(status_code=422, detail="invalid model version status")

    db.add(version)
    db.commit()
    db.refresh(version)
    return _summary(version)
