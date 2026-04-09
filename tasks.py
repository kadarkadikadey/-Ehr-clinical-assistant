from typing import Dict, Any, List
import logging

# Setup basic logging for debugging grader issues
logger = logging.getLogger(__name__)

def get_field(record: Any, field_name: str) -> List[str]:
    """
    Robust safety wrapper. Handles Pydantic objects, standard objects, 
    and dictionaries (common in FastAPI/Docker proxy environments).
    """
    if record is None:
        return []
        
    if hasattr(record, field_name):
        value = getattr(record, field_name)
        return value if isinstance(value, list) else []
    
    if isinstance(record, dict):
        value = record.get(field_name, [])
        return value if isinstance(value, list) else []
        
    return []

# --- GRADER FUNCTIONS ---

def grade_easy_coding(record: Any) -> float:
    """
    Goal: Identify Hypertension (I10).
    Constraint: Strictly 0 < score < 1.
    """
    diagnoses = get_field(record, "diagnoses")
    
    # Baseline 0.05 for attempt, capped at 0.95 for completion
    if any(code in diagnoses for code in ["I10", "I10.0"]):
        return 0.95
    return 0.05

def grade_medium_triage(record: Any) -> float:
    """
    Goal: Identify Diabetes (E11) and Metformin.
    Constraint: Strictly 0 < score < 1.
    """
    diagnoses = get_field(record, "diagnoses")
    prescriptions = get_field(record, "prescriptions")
    
    # Start with a baseline to avoid 0.0
    score = 0.05
    
    if any(code in diagnoses for code in ["E11", "E11.9"]):
        score += 0.45
    if any("metformin" in str(m).lower() for m in prescriptions):
        score += 0.45
        
    # Cap at 0.98 to avoid 1.0
    return min(score, 0.98)

def grade_hard_reconcile(record: Any) -> float:
    """
    Goal: Safe BP management for Asthmatic patient.
    Constraint: Strictly 0 < score < 1.
    """
    diagnoses = get_field(record, "diagnoses")
    prescriptions = get_field(record, "prescriptions")
    
    # 1. Baseline baseline effort points
    score = 0.05 
    
    # 2. Safety Gate: Beta-blockers force a "fail" floor score
    dangerous_meds = ["propranolol", "atenolol", "metoprolol"]
    has_danger = any(any(dm in str(med).lower() for dm in dangerous_meds) for med in prescriptions)
    
    if has_danger:
        return 0.02 # Failure but strictly > 0
    
    # 3. Component Scoring
    if "I10" in diagnoses: score += 0.25
    if any(code in diagnoses for code in ["J45", "J45.909"]): score += 0.25
    if any("lisinopril" in str(med).lower() for med in prescriptions): score += 0.40
    
    # 4. Cap at 0.99 to avoid 1.0
    return min(score, 0.99)

# --- TASK REGISTRY ---

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy_coding": {
        "patient_id": "P001",
        "raw_notes": "Patient presents with High BP and chest pain. History of Hypertension.",
        "grader": grade_easy_coding,
        "difficulty": "easy"
    },
    "medium_triage": {
        "patient_id": "P002",
        "raw_notes": "Routine checkup. Blood sugar is 250mg/dL. Requires Diabetes management.",
        "grader": grade_medium_triage,
        "difficulty": "medium"
    },
    "hard_reconcile": {
        "patient_id": "P003",
        "raw_notes": "Complex case. Patient has Chronic Asthma and newly diagnosed Hypertension. Need safe BP med.",
        "grader": grade_hard_reconcile,
        "difficulty": "hard"
    }
}
