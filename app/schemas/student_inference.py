from pydantic import BaseModel


class StudentInferenceRequest(BaseModel):
    input: dict
    preferences: dict | None = None


class StudentInferenceResponse(BaseModel):
    model_version_id: int
    model_name: str
    overall_status: str
    decision_line: str
    confidence: float
