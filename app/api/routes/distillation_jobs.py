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
    DistillationJobWorkerClaimRequest,
    DistillationJobWorkerCompleteRequest,
    DistillationJobWorkerFailRequest,
    DistillationJobWorkerProgressRequest,
    DistillationJobListResponse,
    DistillationJobSummary,
)

router = APIRouter(prefix="/distillation-jobs", tags=["distillation-jobs"])

ACTIVE_JOB_STATUSES = {"queued", "preparing_dataset", "training", "evaluating"}
WORKER_PROGRESS_STATUSES = {"preparing_dataset", "training", "evaluating"}


def _progress_percent_for_status(status: str) -> int:
    if status == "queued":
        return 5
    if status == "preparing_dataset":
        return 24
    if status == "training":
        return 68
    if status == "evaluating":
        return 90
    if status == "completed":
        return 100
    if status == "failed":
        return 100
    return 0


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
        logs_json=job.logs_json,
        artifact_uri=job.artifact_uri,
        error_message=job.error_message,
        progress_percent=_progress_percent_for_status(job.status),
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _append_log(job: DistillationJob, message: str) -> None:
    logs = list(job.logs_json or [])
    logs.append({"timestamp": datetime.utcnow().isoformat(), "message": message})
    job.logs_json = logs[-20:]


def _ensure_model_version_for_job(job: DistillationJob, db: Session, *, model_name: str | None = None) -> None:
    version = db.scalar(select(ModelVersion).where(ModelVersion.job_id == job.id))
    if job.status != "completed":
        return

    if version is None:
        version = ModelVersion(
            job_id=job.id,
            batch_id=job.batch_id,
            model_name=model_name or f"slm_{job.id}",
            base_model=job.base_model,
            task_type=job.task_type,
            status="ready_for_test",
            metrics_json=job.metrics_json,
            artifact_uri=job.artifact_uri,
            created_at=job.finished_at or datetime.utcnow(),
        )
        db.add(version)
        return

    version.model_name = model_name or version.model_name
    version.metrics_json = job.metrics_json
    version.artifact_uri = job.artifact_uri
    db.add(version)


def _mark_batch_used_in_training(batch_id: str, db: Session) -> None:
    rows = db.scalars(
        select(AnalysisRecord).where(AnalysisRecord.distillation_batch_id == batch_id)
    ).all()
    now = datetime.utcnow()
    for row in rows:
        row.distillation_status = "used_in_training"
        row.used_in_training_at = now
        metadata = dict((row.payload or {}).get("metadata") or {})
        metadata["distillation_status"] = "used_in_training"
        metadata["used_in_training_at"] = now.isoformat()
        metadata["distillation_batch_id"] = batch_id
        payload = dict(row.payload or {})
        payload["metadata"] = metadata
        row.payload = payload
        db.add(row)


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


@router.get("/{job_id}", response_model=DistillationJobSummary)
def get_distillation_job(
    job_id: int,
    db: Session = Depends(get_db),
):
    job = db.get(DistillationJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="distillation job not found")
    return _summary(job)


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
            DistillationJob.status.in_(list(ACTIVE_JOB_STATUSES)),
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
        progress_stage="Queued for worker pickup",
        train_record_count=train_count,
        validation_record_count=validation_count,
        metrics_json={
            "planned_split": {
                "train": train_count,
                "validation": validation_count,
            }
        },
        logs_json=[{"timestamp": datetime.utcnow().isoformat(), "message": "Job created and queued."}],
        created_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return _summary(job)


@router.post("/{job_id}/claim", response_model=DistillationJobSummary)
def claim_distillation_job(
    job_id: int,
    payload: DistillationJobWorkerClaimRequest,
    db: Session = Depends(get_db),
):
    job = db.get(DistillationJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="distillation job not found")
    if job.status != "queued":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job is not available for claim")

    batch_count = db.scalar(
        select(func.count()).select_from(AnalysisRecord).where(AnalysisRecord.distillation_batch_id == job.batch_id)
    ) or 0
    if batch_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export batch not found")

    now = datetime.utcnow()
    job.status = "preparing_dataset"
    job.progress_stage = "Worker claimed job and is preparing dataset"
    job.started_at = job.started_at or now
    _append_log(job, f"Worker {payload.worker_id} claimed the job.")
    db.add(job)
    db.commit()
    db.refresh(job)
    return _summary(job)


@router.patch("/{job_id}/progress", response_model=DistillationJobSummary)
def update_distillation_job_progress(
    job_id: int,
    payload: DistillationJobWorkerProgressRequest,
    db: Session = Depends(get_db),
):
    job = db.get(DistillationJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="distillation job not found")
    if job.status in {"completed", "failed"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job is already finished")
    if payload.status not in WORKER_PROGRESS_STATUSES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid worker progress status")

    job.status = payload.status
    job.progress_stage = payload.progress_stage
    job.started_at = job.started_at or datetime.utcnow()
    if payload.log_message:
        _append_log(job, payload.log_message)
    if payload.artifact_uri:
        job.artifact_uri = payload.artifact_uri
    if payload.metrics_json is not None:
        job.metrics_json = payload.metrics_json
    db.add(job)
    db.commit()
    db.refresh(job)
    return _summary(job)


@router.post("/{job_id}/complete", response_model=DistillationJobSummary)
def complete_distillation_job(
    job_id: int,
    payload: DistillationJobWorkerCompleteRequest,
    db: Session = Depends(get_db),
):
    job = db.get(DistillationJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="distillation job not found")
    if job.status == "completed":
        return _summary(job)
    if job.status == "failed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="failed jobs cannot be completed")

    job.status = "completed"
    job.progress_stage = "Worker completed training and evaluation"
    job.finished_at = datetime.utcnow()
    job.started_at = job.started_at or job.finished_at
    if payload.artifact_uri:
        job.artifact_uri = payload.artifact_uri
    elif not job.artifact_uri:
        job.artifact_uri = f"artifact://distillation_jobs/{job.id}/model_bundle"
    if payload.metrics_json is not None:
        job.metrics_json = payload.metrics_json
    if payload.log_message:
        _append_log(job, payload.log_message)
    else:
        _append_log(job, "Worker reported successful completion.")

    _mark_batch_used_in_training(job.batch_id, db)
    _ensure_model_version_for_job(job, db, model_name=payload.model_name)
    db.add(job)
    db.commit()
    db.refresh(job)
    return _summary(job)


@router.post("/{job_id}/fail", response_model=DistillationJobSummary)
def fail_distillation_job(
    job_id: int,
    payload: DistillationJobWorkerFailRequest,
    db: Session = Depends(get_db),
):
    job = db.get(DistillationJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="distillation job not found")
    if job.status == "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="completed jobs cannot be failed")

    job.status = "failed"
    job.progress_stage = "Worker execution failed"
    job.error_message = payload.error_message
    job.finished_at = datetime.utcnow()
    job.started_at = job.started_at or job.finished_at
    _append_log(job, payload.log_message or payload.error_message)
    db.add(job)
    db.commit()
    db.refresh(job)
    return _summary(job)
