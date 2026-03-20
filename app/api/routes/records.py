from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis_record import AnalysisRecord
from app.models.installation import Installation
from app.schemas.record import (
    AnalysisRecordCreateRequest,
    AnalysisRecordCreateResponse,
    AnalysisRecordSummary,
    AnalysisRecordTrainingUpdateRequest,
    AnalysisRecordTrainingUpdateResponse,
)
from app.services.security import verify_token

router = APIRouter(prefix="/records", tags=["records"])


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

    record = AnalysisRecord(
        installation_id=installation.installation_id,
        schema_version=raw_payload.get("schema_version"),
        route_type=client.get("route_type"),
        platform=client.get("platform"),
        overall_status=teacher.get("overall_status"),
        usable_for_training=bool(metadata.get("usable_for_training", True)),
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


@router.get("", response_model=list[AnalysisRecordSummary])
def list_records(include_payload: bool = False, db: Session = Depends(get_db)):
    rows = db.scalars(select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())).all()
    return [
        AnalysisRecordSummary(
            id=row.id,
            installation_id=row.installation_id,
            schema_version=row.schema_version,
            route_type=row.route_type,
            platform=row.platform,
            overall_status=row.overall_status,
            usable_for_training=row.usable_for_training,
            created_at=row.created_at,
            payload=row.payload if include_payload else None,
        )
        for row in rows
    ]


@router.patch("/{record_id}/training-eligibility", response_model=AnalysisRecordTrainingUpdateResponse)
def update_training_eligibility(
    record_id: int,
    payload: AnalysisRecordTrainingUpdateRequest,
    db: Session = Depends(get_db),
):
    record = db.get(AnalysisRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="record not found")

    record.usable_for_training = payload.usable_for_training
    metadata = dict(record.payload.get("metadata") or {})
    metadata["usable_for_training"] = payload.usable_for_training
    payload_json = dict(record.payload)
    payload_json["metadata"] = metadata
    record.payload = payload_json

    db.add(record)
    db.commit()
    db.refresh(record)

    return AnalysisRecordTrainingUpdateResponse(
        id=record.id,
        usable_for_training=record.usable_for_training,
    )
