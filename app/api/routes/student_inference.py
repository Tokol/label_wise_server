from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.model_version import ModelVersion
from app.schemas.student_inference import (
    ActiveStudentModelResponse,
    StudentInferenceRequest,
    StudentInferenceResponse,
)

router = APIRouter(prefix="/student-inference", tags=["student-inference"])


def _get_active_version(db: Session) -> ModelVersion:
    active_version = db.scalar(select(ModelVersion).where(ModelVersion.status == "active_test"))
    if active_version is None:
        raise HTTPException(status_code=404, detail="no active test model configured")
    return active_version


@router.get("/active-model", response_model=ActiveStudentModelResponse)
def get_active_student_model(db: Session = Depends(get_db)):
    active_version = _get_active_version(db)
    return ActiveStudentModelResponse(
        model_version_id=active_version.id,
        model_name=active_version.model_name,
        base_model=active_version.base_model,
        artifact_uri=active_version.artifact_uri,
        metrics_json=active_version.metrics_json,
    )


@router.post("/predict", response_model=StudentInferenceResponse)
def predict_with_student_model(
    payload: StudentInferenceRequest,
    db: Session = Depends(get_db),
):
    active_version = _get_active_version(db)

    input_block = payload.input or {}
    ingredients = [str(item).lower() for item in (input_block.get("ingredients") or [])]
    preferences = payload.preferences or {}

    overall_status = "safe"
    decision_line = "No conflicts found in the simulated student inference checks."
    confidence = 0.91

    if any("pork" in item or "gelatin" in item for item in ingredients):
        overall_status = "unsafe"
        decision_line = "Potential conflict detected from ingredient tokens in the simulated student model."
        confidence = 0.82
    elif preferences:
        overall_status = "warning"
        decision_line = "Preferences are present, so the simulated student model is flagging this for closer review."
        confidence = 0.74

    return StudentInferenceResponse(
        model_version_id=active_version.id,
        model_name=active_version.model_name,
        base_model=active_version.base_model,
        artifact_uri=active_version.artifact_uri,
        overall_status=overall_status,
        decision_line=decision_line,
        confidence=confidence,
    )
