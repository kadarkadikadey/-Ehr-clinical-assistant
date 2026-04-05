from typing import Dict, Any, List
from schema import PatientRecord

def get_field(record: Any, field_name: str) -> List[str]:
    """
    Utility Helper: Safety wrapper to extract data.
    Ensures the grader can read 'diagnoses' or 'prescriptions' regardless of 
    whether the input is a Pydantic object (from local tests) or a 
    raw Dictionary (from the FastAPI/Docker environment).
    """
    if hasattr(record, field_name):
        return getattr(record, field_name)
    if isinstance(record, dict):
        return record.get(field_name, [])
    return []

# --- GRADER FUNCTIONS ---
# Each function evaluates the final state of a patient record and returns a score from 0.0 to 1.0.

def grade_easy_coding(record: Any) -> float:
    """
    TASK: Basic Medical Coding.
    The agent must extract 'Hypertension' from notes and log the ICD-10 code 'I10'.
    """
    diagnoses = get_field(record, "diagnoses")
    
    # Simple binary scoring: 1.0 if the correct code is found, else 0.0.
    if "I10" in diagnoses:
        return 1.0
    return 0.0

def grade_medium_triage(record: Any) -> float:
    """
    TASK: Diagnosis + Treatment.
    The agent must identify Type 2 Diabetes (E11) AND prescribe Metformin.
    """
    diagnoses = get_field(record, "diagnoses")
    prescriptions = get_field(record, "prescriptions")
    
    score = 0.0
    # Award 0.5 points for identifying the correct ICD-10 code.
    if any(code in diagnoses for code in ["E11", "E11.9"]):
        score += 0.5
    
    # Award 0.5 points for prescribing the standard first-line medication.
    if any("metformin" in med.lower() for med in prescriptions):
        score += 0.5
        
    return score

def grade_hard_reconcile(record: Any) -> float:
    """
    TASK: Clinical Audit & Safety (Medication Reconciliation).
    This is the most complex task. The agent must:
    1. Identify Hypertension (I10) - 0.3 pts
    2. Identify Diabetes (E11) - 0.3 pts
    3. Prescribe a SAFE medication (Lisinopril) - 0.4 pts
    
    CRITICAL SAFETY CHECK: Patient P003 has Asthma. 
    Beta-blockers (Propranolol, etc.) can cause life-threatening bronchospasms 
    in asthmatics. If the agent prescribes these, the score is forced to 0.0.
    """
    diagnoses = get_field(record, "diagnoses")
    prescriptions = get_field(record, "prescriptions")
    
    score = 0.0
    
    # Scoring diagnosis accuracy
    if "I10" in diagnoses: score += 0.3
    if "E11" in diagnoses: score += 0.3
    
    # SAFETY GATEKEEPER logic
    dangerous_meds = ["propranolol", "atenolol", "metoprolol"]
    has_danger = any(
        m in med.lower() 
        for med in prescriptions 
        for m in dangerous_meds
    )
    
    # If the agent fails the safety check, it fails the entire task immediately.
    if has_danger:
        return 0.0 
    
    # Award points for choosing a safe antihypertensive alternative (ACE Inhibitor).
    if any("lisinopril" in med.lower() for med in prescriptions):
        score += 0.4
        
    return min(score, 1.0)

# --- TASK REGISTRY ---
# A centralized map that the environment (app.py) uses to switch between patients and goals.
TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy_coding": {
        "patient_id": "P001",
        "description": "Extract hypertension diagnosis from patient P001 notes.",
        "grader": grade_easy_coding,
        "difficulty": "easy"
    },
    "medium_triage": {
        "patient_id": "P002",
        "description": "Diagnose diabetes and suggest standard first-line treatment for P002.",
        "grader": grade_medium_triage,
        "difficulty": "medium"
    },
    "hard_reconcile": {
        "patient_id": "P003",
        "description": "Perform full audit for P003. Beware of medication contraindications.",
        "grader": grade_hard_reconcile,
        "difficulty": "hard"
    }
}