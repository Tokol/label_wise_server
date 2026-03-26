from pydantic import BaseModel


class StudentInferenceRequest(BaseModel):
    input: dict
    preferences: dict | None = None


class StudentInferenceResponse(BaseModel):
    model_version_id: int
    model_name: str
    base_model: str | None = None
    artifact_uri: str | None = None
    overall_status: str
    decision_line: str
    confidence: float


class ActiveStudentModelResponse(BaseModel):
    model_version_id: int
    model_name: str
    base_model: str
    artifact_uri: str | None = None
    metrics_json: dict | None = None
