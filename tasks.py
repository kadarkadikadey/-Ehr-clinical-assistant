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
        
    # Case 1: Standard Object or Pydantic Model
    if hasattr(record, field_name):
        value = getattr(record, field_name)
        return value if isinstance(value, list) else []
    
    # Case 2: Raw Dictionary (from JSON responses)
    if isinstance(record, dict):
        value = record.get(field_name, [])
        return value if isinstance(value, list) else []
        
    return []

# --- GRADER FUNCTIONS ---

def grade_easy_coding(record: Any) -> float:
    """
    Goal: Identify Hypertension (I10).
    """
    diagnoses = get_field(record, "diagnoses")
    return 1.0 if any(code in diagnoses for code in ["I10", "I10.0"]) else 0.0

def grade_medium_triage(record: Any) -> float:
    """
    Goal: Identify Diabetes (E11) and Metformin.
    """
    diagnoses = get_field(record, "diagnoses")
    prescriptions = get_field(record, "prescriptions")
    
    score = 0.0
    if any(code in diagnoses for code in ["E11", "E11.9"]):
        score += 0.5
    if any("metformin" in str(m).lower() for m in prescriptions):
        score += 0.5
    return score

def grade_hard_reconcile(record: Any) -> float:
    """
    TASK: Clinical Audit & Safety for P003 (The 'Asthma' Patient).
    
    Logic breakdown for 1.00 Score:
    1. Identify Primary Hypertension (I10) -> 0.3
    2. Identify Asthma (J45.909 or J45) -> 0.3
    3. Prescribe Safe Med (Lisinopril) -> 0.4
    
    SAFETY GATE: Beta-blockers (Propranolol) = 0.0 Total Score.
    """
    diagnoses = get_field(record, "diagnoses")
    prescriptions = get_field(record, "prescriptions")
    
    # 🚨 SAFETY GATEKEEPER
    # If the agent prescribes a Beta-blocker to an Asthmatic, they fail instantly.
    dangerous_meds = ["propranolol", "atenolol", "metoprolol", "carvedilol"]
    if any(any(dm in str(med).lower() for dm in dangerous_meds) for med in prescriptions):
        return 0.0 

    score = 0.0
    
    # Diagnosis Accuracy (0.6 total)
    if "I10" in diagnoses: 
        score += 0.3
    if any(code in diagnoses for code in ["J45", "J45.909", "J45.901"]): 
        score += 0.3
        
    # Treatment Accuracy (0.4 total)
    # Lisinopril is the safe alternative to Beta-blockers here.
    if any("lisinopril" in str(med).lower() for med in prescriptions):
        score += 0.4
        
    return round(min(score, 1.0), 2)

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