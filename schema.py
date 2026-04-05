# schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PatientRecord(BaseModel):
    patient_id: str
    raw_notes: str
    diagnoses: List[str] = []
    prescriptions: List[str] = []
    last_updated: Optional[str] = None

class Observation(BaseModel):
    current_view: str
    record_data: Optional[PatientRecord]
    error_message: Optional[str] = None

class Action(BaseModel):
    command: str = Field(..., description="SEARCH, ADD_DIAGNOSIS, ADD_MED, or FINISH")
    payload: str = Field(..., description="The data for the command (e.g., 'ICD-10 code' or 'Patient Name')")

class Reward(BaseModel):
    value: float
    reason: str