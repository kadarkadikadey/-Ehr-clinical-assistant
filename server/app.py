import uvicorn
from fastapi import FastAPI, HTTPException, Request
from typing import Optional, List, Dict, Any
from schema import Action, Observation
from tasks import TASK_REGISTRY

# Initialize FastAPI app
app = FastAPI(title="EHR Clinical Assistant")

class EHR_Environment:
    def __init__(self):
        # Mock Database
        self.db = {
            "P001": {"patient_id": "P001", "raw_notes": "High BP, chest pain. History: Hypertension.", "diagnoses": [], "prescriptions": []},
            "P002": {"patient_id": "P002", "raw_notes": "Glucose 240mg/dL. Suggests T2 Diabetes.", "diagnoses": [], "prescriptions": []},
            "P003": {"patient_id": "P003", "raw_notes": "Asthmatic. Presenting with BP 160/100.", "diagnoses": [], "prescriptions": []}
        }
        # Initialize with None to prevent KeyErrors before reset() is called
        self.state_data = {"task_id": None, "steps": 0, "patient_id": None}

    def reset(self, task_id: str = "easy_coding"):
        """Prepares a clean state for the agent."""
        if task_id not in TASK_REGISTRY:
            task_id = "easy_coding"
            
        self.state_data = {
            "task_id": task_id, 
            "steps": 0, 
            "patient_id": TASK_REGISTRY[task_id]["patient_id"]
        }
        
        p_id = self.state_data["patient_id"]
        # Reset the 'database' record for this patient
        self.db[p_id]["diagnoses"] = []
        self.db[p_id]["prescriptions"] = []
        
        return self.state()

    def state(self) -> Observation:
        """Returns what the AI sees. Handles uninitialized states gracefully."""
        p_id = self.state_data.get("patient_id")
        
        if not p_id or p_id not in self.db:
            return Observation(
                current_view="INITIAL",
                record_data={"patient_id": "N/A", "raw_notes": "Environment initialized. Call /reset to start.", "diagnoses": [], "prescriptions": []}
            )
            
        return Observation(
            current_view=f"STEP_{self.state_data['steps']}", 
            record_data=self.db[p_id]
        )

    def step(self, action: Action):
        """Processes the action and returns (obs, reward, done, info)."""
        self.state_data["steps"] += 1
        p_id = self.state_data["patient_id"]
        
        # Guard: Ensure env is reset
        if not p_id:
            return self.state(), 0.0, True, {"error": "Environment not reset"}

        # 1. Action: ADD_DIAGNOSIS
        if action.command == "ADD_DIAGNOSIS":
            if action.payload not in self.db[p_id]["diagnoses"]:
                self.db[p_id]["diagnoses"].append(action.payload)
                return self.state(), 0.1, False, {}
            return self.state(), 0.0, False, {"error": "Duplicate diagnosis"}

        # 2. Action: ADD_MED
        if action.command == "ADD_MED":
            if action.payload not in self.db[p_id]["prescriptions"]:
                self.db[p_id]["prescriptions"].append(action.payload)
                return self.state(), 0.1, False, {}
            return self.state(), 0.0, False, {"error": "Duplicate medication"}
        
        # 3. Action: FINISH
        if action.command == "FINISH":
            score = TASK_REGISTRY[self.state_data["task_id"]]["grader"](self.db[p_id])
            return self.state(), score, True, {}
            
        # 4. Default / Timeout
        done = self.state_data["steps"] >= 10
        return self.state(), 0.0, done, {}

# Global instance
env = EHR_Environment()

# --- FastAPI Endpoints ---

@app.post("/reset")
async def reset_endpoint(request: Request):
    # Flexible parsing: handles raw string or JSON body {"task_id": "..."}
    data = await request.json() if await request.body() else {}
    task_id = data.get("task_id", "easy_coding") if isinstance(data, dict) else "easy_coding"
    return env.reset(task_id)

@app.post("/step")
async def step_endpoint(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
async def get_state_endpoint():
    return env.state()

def main():
    """Main entry point for the OpenEnv validator and CLI."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()