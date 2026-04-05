import asyncio
import os
import json
import textwrap
from typing import List, Optional

import torch  # Moved to top for performance
from openai import OpenAI

# Core environment imports
from server.app import EHR_Environment 
from schema import Action
from reward_model import RewardPredictor, extract_features

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "ehr_clinical_assistant"

MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.8

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical data entry bot. You MUST respond in a strict JSON format.
    
    REQUIRED COMMANDS:
    1. {"command": "ADD_DIAGNOSIS", "payload": "ICD_CODE"}  <-- payload MUST be a string (e.g. "I10")
    2. {"command": "ADD_MED", "payload": "DRUG_NAME"}      <-- payload MUST be a string (e.g. "Metformin")
    3. {"command": "FINISH", "payload": ""}                <-- use when chart is complete
    
    STRICT RULES:
    - DO NOT use 'ADD'. Use 'ADD_DIAGNOSIS' or 'ADD_MED'.
    - DO NOT send a dictionary in the payload. Send ONLY the code or name string.
    - DO NOT repeat codes already in the Chart.
    - If no new info remains, you MUST call FINISH.
    """
).strip()

# --- STANDARDIZED LOGGING ---

def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    done_val = str(done).lower()
    action_str = action.replace("\n", " ").strip()
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# --- INFERENCE ENGINE ---

async def run_task(env: EHR_Environment, predictor: RewardPredictor, client: OpenAI, task_id: str):
    """Executes the agent loop for a single specific task."""
    log_start(task=task_id)
    
    obs = env.reset(task_id=task_id)
    rewards = []
    steps_taken = 0
    done = False

    try:
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            
            # 1. Feature Extraction & Value Prediction
            state_tensor = extract_features(obs)
            with torch.no_grad():
                _val = predictor(state_tensor).item()

            # 2. LLM Planning
            # Accessing record_data safely depending on if obs is a dict or object
            # ✅ NEW WAY (Uses Dot Notation for Objects)
            data = obs.record_data
            user_prompt = f"Notes: {data.raw_notes}\nChart: {data.diagnoses}\nMeds: {data.prescriptions}"
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"{SYSTEM_PROMPT} Respond only in JSON."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1 
            )
            
            raw_content = completion.choices[0].message.content.strip()
            
            # 3. Robust JSON Parsing
            try:
                start_idx = raw_content.find("{")
                end_idx = raw_content.rfind("}") + 1
                clean_json = raw_content[start_idx:end_idx] if start_idx != -1 else raw_content
                
                action_data = json.loads(clean_json)
                action_obj = Action(**action_data)
            except Exception as e:
                log_step(step=step, action=raw_content, reward=0.0, done=False, error="JSON_PARSE_ERROR")
                continue

            # 4. Environment Step
            obs, reward, done, info = env.step(action_obj)
            rewards.append(reward)
            log_step(step=step, action=json.dumps(action_data), reward=reward, done=done, error=info.get("error"))

            if done:
                break

        # Final Scoring
        score = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}")

async def main():
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN environment variable is not set.")
        return

    # Global initializations
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EHR_Environment() 
    predictor = RewardPredictor(input_dim=5)
    predictor.eval()

    # Run through the full task registry
    tasks = ["easy_coding", "medium_triage", "hard_reconcile"]
    for task_id in tasks:
        await run_task(env, predictor, client, task_id)

if __name__ == "__main__":
    asyncio.run(main())