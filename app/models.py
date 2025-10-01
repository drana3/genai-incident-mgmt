from pydantic import BaseModel, validator
from typing import Dict


class Alert(BaseModel):
    incident_id: str | None = None
    description: str
    severity: str
    metrics: Dict = {}

    @validator("description")
    def desc_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Description cannot be empty")
        return v

    @validator("severity")
    def severity_valid(cls, v):
        if v not in ["low", "medium", "high"]:
            raise ValueError("Severity must be low, medium, or high")
        return v


class ApprovalRequest(BaseModel):
    approved: bool
    tweaks: Dict = {}
