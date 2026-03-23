from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis_record import AnalysisRecord
from app.models.installation import Installation
from app.schemas.record import (
    AnalysisRecordBulkTrainingUpdateRequest,
    AnalysisRecordBulkTrainingUpdateResponse,
    AnalysisRecordBulkDistillationUpdateRequest,
    AnalysisRecordBulkDistillationUpdateResponse,
    AnalysisRecordCreateRequest,
    AnalysisRecordCreateResponse,
    AnalysisRecordDistillationUpdateRequest,
    AnalysisRecordDistillationUpdateResponse,
    AnalysisRecordExportRequest,
    AnalysisRecordExportResponse,
    AnalysisRecordsPaginatedResponse,
    AnalysisRecordSummary,
    DistillationExportRecord,
    AnalysisRecordTrainingUpdateRequest,
    AnalysisRecordTrainingUpdateResponse,
)
from app.services.security import verify_token

router = APIRouter(prefix="/records", tags=["records"])

VALID_DISTILLATION_STATUSES = {
    "pending_review",
    "approved_for_distillation",
    "excluded",
    "exported",
    "used_in_training",
    "archived",
}


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid Authorization header")
    return token


def _require_installation(
    installation_id: str | None,
    authorization: str | None,
    db: Session,
) -> Installation:
    if not installation_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing X-Installation-Id header")

    installation = db.get(Installation, installation_id)
    if installation is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="installation_id is not registered")

    token = _extract_bearer_token(authorization)
    if not verify_token(token, installation.token_hash):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid client token")

    installation.last_seen_at = datetime.utcnow()
    db.add(installation)
    db.flush()
    return installation


def _set_training_eligibility(record: AnalysisRecord, usable_for_training: bool) -> None:
    record.usable_for_training = usable_for_training
    record.distillation_status = "approved_for_distillation" if usable_for_training else "excluded"
    record.reviewed_at = datetime.utcnow()
    if not usable_for_training:
        record.excluded_reason = record.excluded_reason or "Excluded from dashboard curation"
    else:
        record.excluded_reason = None
    metadata = dict(record.payload.get("metadata") or {})
    metadata["usable_for_training"] = usable_for_training
    metadata["distillation_status"] = record.distillation_status
    metadata["reviewed_at"] = record.reviewed_at.isoformat()
    if not usable_for_training:
        metadata.setdefault("excluded_reason", "Excluded from dashboard curation")
    else:
        metadata.pop("excluded_reason", None)
    payload_json = dict(record.payload)
    payload_json["metadata"] = metadata
    record.payload = payload_json


def _record_metadata(record: AnalysisRecord) -> dict:
    return dict(record.payload.get("metadata") or {})


def _record_distillation_status(record: AnalysisRecord) -> str:
    if record.distillation_status in VALID_DISTILLATION_STATUSES:
        return record.distillation_status
    metadata = _record_metadata(record)
    stored = metadata.get("distillation_status")
    if isinstance(stored, str) and stored in VALID_DISTILLATION_STATUSES:
        return stored
    return "pending_review" if record.usable_for_training else "excluded"


def _record_datetime_field(record: AnalysisRecord, key: str) -> datetime | None:
    direct_value = getattr(record, key, None)
    if isinstance(direct_value, datetime):
        return direct_value
    metadata = _record_metadata(record)
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _record_string_field(record: AnalysisRecord, key: str) -> str | None:
    direct_value = getattr(record, key, None)
    if isinstance(direct_value, str) and direct_value:
        return direct_value
    metadata = _record_metadata(record)
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _sync_distillation_metadata(
    record: AnalysisRecord,
    distillation_status: str,
    *,
    excluded_reason: str | None = None,
    distillation_batch_id: str | None = None,
) -> None:
    if distillation_status not in VALID_DISTILLATION_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid distillation_status: {distillation_status}",
        )

    metadata = _record_metadata(record)
    now = datetime.utcnow()
    now_iso = now.isoformat()
    record.distillation_status = distillation_status
    metadata["distillation_status"] = distillation_status

    if distillation_status in {"approved_for_distillation", "exported", "used_in_training", "archived"}:
        record.usable_for_training = True
        metadata["usable_for_training"] = True
    elif distillation_status == "excluded":
        record.usable_for_training = False
        metadata["usable_for_training"] = False

    if distillation_status in {"approved_for_distillation", "excluded"}:
        record.reviewed_at = now
        metadata["reviewed_at"] = now_iso
    if distillation_status == "exported":
        record.reviewed_at = record.reviewed_at or now
        record.exported_at = now
        record.distillation_batch_id = distillation_batch_id or record.distillation_batch_id
        metadata["reviewed_at"] = metadata.get("reviewed_at") or now_iso
        metadata["exported_at"] = now_iso
        metadata["distillation_batch_id"] = distillation_batch_id or metadata.get("distillation_batch_id")
    elif distillation_status == "used_in_training":
        record.reviewed_at = record.reviewed_at or now
        record.exported_at = record.exported_at or now
        record.used_in_training_at = now
        record.distillation_batch_id = distillation_batch_id or record.distillation_batch_id
        metadata["reviewed_at"] = metadata.get("reviewed_at") or now_iso
        metadata["exported_at"] = metadata.get("exported_at") or now_iso
        metadata["used_in_training_at"] = now_iso
        metadata["distillation_batch_id"] = distillation_batch_id or metadata.get("distillation_batch_id")
    elif distillation_status == "archived":
        record.distillation_batch_id = distillation_batch_id or record.distillation_batch_id
        metadata["distillation_batch_id"] = distillation_batch_id or metadata.get("distillation_batch_id")

    if excluded_reason:
        record.excluded_reason = excluded_reason
        metadata["excluded_reason"] = excluded_reason
    elif distillation_status != "excluded":
        record.excluded_reason = None
        metadata.pop("excluded_reason", None)

    if distillation_batch_id and distillation_status in {"exported", "used_in_training", "archived"}:
        record.distillation_batch_id = distillation_batch_id
        metadata["distillation_batch_id"] = distillation_batch_id

    payload_json = dict(record.payload)
    payload_json["metadata"] = metadata
    record.payload = payload_json


def _summary_from_row(row: AnalysisRecord, include_payload: bool) -> AnalysisRecordSummary:
    return AnalysisRecordSummary(
        id=row.id,
        installation_id=row.installation_id,
        schema_version=row.schema_version,
        route_type=row.route_type,
        platform=row.platform,
        overall_status=row.overall_status,
        usable_for_training=row.usable_for_training,
        distillation_status=_record_distillation_status(row),
        distillation_batch_id=_record_string_field(row, "distillation_batch_id"),
        reviewed_at=_record_datetime_field(row, "reviewed_at"),
        exported_at=_record_datetime_field(row, "exported_at"),
        used_in_training_at=_record_datetime_field(row, "used_in_training_at"),
        excluded_reason=_record_string_field(row, "excluded_reason"),
        created_at=row.created_at,
        payload=row.payload if include_payload else None,
    )


@router.post("", response_model=AnalysisRecordCreateResponse, status_code=status.HTTP_201_CREATED)
def create_record(
    payload: AnalysisRecordCreateRequest,
    x_installation_id: str | None = Header(default=None, alias="X-Installation-Id"),
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
):
    installation = _require_installation(x_installation_id, authorization, db)

    raw_payload = payload.payload
    client = raw_payload.get("client") or {}
    teacher = raw_payload.get("teacher_result") or {}
    metadata = raw_payload.get("metadata") or {}
    initial_distillation_status = metadata.get("distillation_status") or "pending_review"
    metadata["distillation_status"] = initial_distillation_status
    raw_payload["metadata"] = metadata

    record = AnalysisRecord(
        installation_id=installation.installation_id,
        schema_version=raw_payload.get("schema_version"),
        route_type=client.get("route_type"),
        platform=client.get("platform"),
        overall_status=teacher.get("overall_status"),
        usable_for_training=bool(metadata.get("usable_for_training", True)),
        distillation_status=initial_distillation_status,
        payload=raw_payload,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return AnalysisRecordCreateResponse(
        id=record.id,
        installation_id=record.installation_id,
        created_at=record.created_at,
    )


@router.get("", response_model=AnalysisRecordsPaginatedResponse)
def list_records(
    skip: int = 0,
    limit: int = 25,
    include_payload: bool = False,
    query: str | None = Query(default=None),
    route_type: str | None = Query(default=None),
    overall_status: str | None = Query(default=None),
    usable_for_training: bool | None = Query(default=None),
    distillation_status: str | None = Query(default=None),
    platform: str | None = Query(default=None),
    category: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    stmt = select(AnalysisRecord)

    if route_type:
        stmt = stmt.where(AnalysisRecord.route_type == route_type)
    if overall_status:
        if overall_status == "unsafe":
            stmt = stmt.where(AnalysisRecord.overall_status.in_(["unsafe", "violation"]))
        elif overall_status == "cannot_assess":
            stmt = stmt.where(AnalysisRecord.overall_status.in_(["cannot_assess", "cannot assess"]))
        elif overall_status == "unknown":
            stmt = stmt.where(
                or_(
                    AnalysisRecord.overall_status.is_(None),
                    ~AnalysisRecord.overall_status.in_(["safe", "warning", "unsafe", "violation", "cannot_assess", "cannot assess"]),
                )
            )
        else:
            stmt = stmt.where(AnalysisRecord.overall_status == overall_status)
    if usable_for_training is not None:
        stmt = stmt.where(AnalysisRecord.usable_for_training == usable_for_training)
    if distillation_status:
        stmt = stmt.where(AnalysisRecord.distillation_status == distillation_status)
    if platform:
        stmt = stmt.where(AnalysisRecord.platform == platform)
    if category:
        stmt = stmt.where(cast(AnalysisRecord.payload, String).ilike(f'%{category}%'))
    if query:
        term = f"%{query.strip()}%"
        stmt = stmt.where(
            or_(
                cast(AnalysisRecord.id, String).ilike(term),
                AnalysisRecord.installation_id.ilike(term),
                func.coalesce(AnalysisRecord.route_type, "").ilike(term),
                func.coalesce(AnalysisRecord.platform, "").ilike(term),
                func.coalesce(AnalysisRecord.overall_status, "").ilike(term),
                cast(AnalysisRecord.payload, String).ilike(term),
            )
        )

    total_count = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    rows = db.scalars(
        stmt.order_by(AnalysisRecord.created_at.desc()).offset(skip).limit(limit)
    ).all()
    
    records = [_summary_from_row(row, include_payload) for row in rows]
    
    return AnalysisRecordsPaginatedResponse(
        records=records,
        total_count=total_count,
        skip=skip,
        limit=limit,
        has_more=(skip + limit) < total_count,
    )


@router.patch("/distillation-status/bulk", response_model=AnalysisRecordBulkDistillationUpdateResponse)
def bulk_update_distillation_status(
    payload: AnalysisRecordBulkDistillationUpdateRequest,
    db: Session = Depends(get_db),
):
    rows = db.scalars(select(AnalysisRecord).where(AnalysisRecord.id.in_(payload.record_ids))).all()

    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no matching records found")

    for row in rows:
        _sync_distillation_metadata(
            row,
            payload.distillation_status,
            excluded_reason=payload.excluded_reason,
            distillation_batch_id=payload.distillation_batch_id,
        )
        db.add(row)

    db.commit()

    return AnalysisRecordBulkDistillationUpdateResponse(
        updated_count=len(rows),
        distillation_status=payload.distillation_status,
        distillation_batch_id=payload.distillation_batch_id,
    )


@router.post("/export", response_model=AnalysisRecordExportResponse)
def export_distillation_records(
    payload: AnalysisRecordExportRequest,
    db: Session = Depends(get_db),
):
    stmt = select(AnalysisRecord)
    if payload.record_ids:
        stmt = stmt.where(AnalysisRecord.id.in_(payload.record_ids))
    rows = db.scalars(stmt.order_by(AnalysisRecord.created_at.desc())).all()

    eligible_rows = [row for row in rows if _record_distillation_status(row) == "approved_for_distillation"]
    if not eligible_rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no approved records available for export")

    batch_id = payload.batch_id or f"distill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    exported_records = []
    for row in eligible_rows:
        _sync_distillation_metadata(row, "exported", distillation_batch_id=batch_id)
        db.add(row)
        exported_records.append(
            DistillationExportRecord(
                id=row.id,
                installation_id=row.installation_id,
                overall_status=row.overall_status,
                usable_for_training=row.usable_for_training,
                distillation_status="exported",
                distillation_batch_id=batch_id,
                created_at=row.created_at,
                payload=row.payload if payload.include_payload else None,
            )
        )

    db.commit()

    return AnalysisRecordExportResponse(
        batch_id=batch_id,
        exported_count=len(exported_records),
        records=exported_records,
    )


@router.patch("/training-eligibility/bulk", response_model=AnalysisRecordBulkTrainingUpdateResponse)
def bulk_update_training_eligibility(
    payload: AnalysisRecordBulkTrainingUpdateRequest,
    db: Session = Depends(get_db),
):
    rows = db.scalars(
        select(AnalysisRecord).where(AnalysisRecord.id.in_(payload.record_ids))
    ).all()

    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no matching records found")

    for row in rows:
        _set_training_eligibility(row, payload.usable_for_training)
        db.add(row)

    db.commit()

    return AnalysisRecordBulkTrainingUpdateResponse(
        updated_count=len(rows),
        usable_for_training=payload.usable_for_training,
    )


@router.patch("/{record_id}/training-eligibility", response_model=AnalysisRecordTrainingUpdateResponse)
def update_training_eligibility(
    record_id: int,
    payload: AnalysisRecordTrainingUpdateRequest,
    db: Session = Depends(get_db),
):
    record = db.get(AnalysisRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="record not found")

    _set_training_eligibility(record, payload.usable_for_training)

    db.add(record)
    db.commit()
    db.refresh(record)

    return AnalysisRecordTrainingUpdateResponse(
        id=record.id,
        usable_for_training=record.usable_for_training,
    )


@router.patch("/{record_id}/distillation-status", response_model=AnalysisRecordDistillationUpdateResponse)
def update_distillation_status(
    record_id: int,
    payload: AnalysisRecordDistillationUpdateRequest,
    db: Session = Depends(get_db),
):
    record = db.get(AnalysisRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="record not found")

    _sync_distillation_metadata(
        record,
        payload.distillation_status,
        excluded_reason=payload.excluded_reason,
        distillation_batch_id=payload.distillation_batch_id,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return AnalysisRecordDistillationUpdateResponse(
        id=record.id,
        distillation_status=_record_distillation_status(record),
        distillation_batch_id=_record_string_field(record, "distillation_batch_id"),
        reviewed_at=_record_datetime_field(record, "reviewed_at"),
        exported_at=_record_datetime_field(record, "exported_at"),
        used_in_training_at=_record_datetime_field(record, "used_in_training_at"),
        excluded_reason=_record_string_field(record, "excluded_reason"),
    )


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_record(
    record_id: int,
    db: Session = Depends(get_db),
):
    record = db.get(AnalysisRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="record not found")

    db.delete(record)
    db.commit()
