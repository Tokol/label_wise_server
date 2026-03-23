from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis_record import AnalysisRecord
from app.models.distillation_job import DistillationJob
from app.models.model_version import ModelVersion
from app.schemas.distillation_job import (
    DistillationJobCreateRequest,
    DistillationJobListResponse,
    DistillationJobSummary,
)

router = APIRouter(prefix="/distillation-jobs", tags=["distillation-jobs"])


def _progress_percent_for_status(status: str) -> int:
    if status == "queued":
        return 5
    if status == "preparing_dataset":
        return 20
    if status == "training":
        return 58
    if status == "evaluating":
        return 86
    if status == "completed":
        return 100
    if status == "failed":
        return 100
    return 0


def _reconcile_job(job: DistillationJob) -> bool:
    if job.status in {"completed", "failed"}:
        return False

    now = datetime.utcnow()
    elapsed = (now - job.created_at).total_seconds()
    changed = False

    if job.started_at is None:
        job.started_at = job.created_at
        changed = True

    if elapsed >= 18:
        if job.status != "completed":
            job.status = "completed"
            job.progress_stage = "Completed simulated evaluation"
            job.finished_at = now
            job.metrics_json = {
                **(job.metrics_json or {}),
                "simulation": True,
                "evaluation": {
                    "status_accuracy": 0.87,
                    "macro_f1": 0.84,
                },
            }
            changed = True
    elif elapsed >= 12:
        if job.status != "evaluating":
            job.status = "evaluating"
            job.progress_stage = "Running validation and scoring"
            changed = True
    elif elapsed >= 6:
        if job.status != "training":
            job.status = "training"
            job.progress_stage = "Fine-tuning the hosted 3B student"
            changed = True
    elif elapsed >= 2:
        if job.status != "preparing_dataset":
            job.status = "preparing_dataset"
            job.progress_stage = "Preparing distillation dataset"
            changed = True

    return changed


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
        progress_percent=_progress_percent_for_status(job.status),
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _ensure_model_version_for_job(job: DistillationJob, db: Session) -> bool:
    existing = db.scalar(select(ModelVersion).where(ModelVersion.job_id == job.id))
    if existing is not None or job.status != "completed":
        return False

    version = ModelVersion(
        job_id=job.id,
        batch_id=job.batch_id,
        model_name=f"slm_{job.id}",
        base_model=job.base_model,
        task_type=job.task_type,
        status="ready_for_test",
        metrics_json=job.metrics_json,
        artifact_uri=f"simulated://model_versions/slm_{job.id}",
        created_at=job.finished_at or datetime.utcnow(),
    )
    db.add(version)
    return True


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
    changed = False
    created_versions = False
    for row in rows:
        if _reconcile_job(row):
            db.add(row)
            changed = True
        if _ensure_model_version_for_job(row, db):
            created_versions = True
    if changed or created_versions:
        db.commit()

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
