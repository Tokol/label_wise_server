from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis_record import AnalysisRecord
from app.models.distillation_job import DistillationJob
from app.schemas.distillation_job import (
    DistillationJobCreateRequest,
    DistillationJobListResponse,
    DistillationJobSummary,
)

router = APIRouter(prefix="/distillation-jobs", tags=["distillation-jobs"])


def _summary(job: DistillationJob) -> DistillationJobSummary:
    return DistillationJobSummary(
        id=job.id,
        batch_id=job.batch_id,
        base_model=job.base_model,
        task_type=job.task_type,
        dataset_mode=job.dataset_mode,
        status=job.status,
        progress_stage=job.progress_stage,
        train_record_count=job.train_record_count,
        validation_record_count=job.validation_record_count,
        metrics_json=job.metrics_json,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("", response_model=DistillationJobListResponse)
def list_distillation_jobs(
    skip: int = 0,
    limit: int = 10,
    status_filter: str | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
):
    stmt = select(DistillationJob)
    if status_filter:
        stmt = stmt.where(DistillationJob.status == status_filter)

    total_count = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(DistillationJob.created_at.desc()).offset(skip).limit(limit)).all()

    return DistillationJobListResponse(
        jobs=[_summary(row) for row in rows],
        total_count=total_count,
        skip=skip,
        limit=limit,
        has_more=(skip + limit) < total_count,
    )


@router.post("", response_model=DistillationJobSummary, status_code=status.HTTP_201_CREATED)
def create_distillation_job(
    payload: DistillationJobCreateRequest,
    db: Session = Depends(get_db),
):
    record_count = db.scalar(
        select(func.count()).select_from(AnalysisRecord).where(AnalysisRecord.distillation_batch_id == payload.batch_id)
    ) or 0

    if record_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export batch not found")

    existing_active_job = db.scalar(
        select(DistillationJob).where(
            DistillationJob.batch_id == payload.batch_id,
            DistillationJob.status.in_(["queued", "preparing_dataset", "training", "evaluating"]),
        )
    )
    if existing_active_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="an active distillation job already exists for this batch",
        )

    validation_count = max(1, int(record_count * 0.15)) if record_count > 1 else 0
    train_count = max(0, record_count - validation_count)

    job = DistillationJob(
        batch_id=payload.batch_id,
        base_model=payload.base_model,
        task_type=payload.task_type,
        dataset_mode=payload.dataset_mode,
        status="queued",
        progress_stage="Queued for dataset preparation",
        train_record_count=train_count,
        validation_record_count=validation_count,
        metrics_json={
            "planned_split": {
                "train": train_count,
                "validation": validation_count,
            }
        },
        created_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return _summary(job)
