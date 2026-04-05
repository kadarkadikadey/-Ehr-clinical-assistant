import os
import json
import asyncio
from openai import OpenAI
from server.app import EHR_Environment # Updated for your folder structure
from schema import Action

# --- CONFIGURATION ---
# Use the router URL for the hackathon
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = "gpt-4o" 

def run_baseline(task_id):
    client = OpenAI(base_url=API_BASE_URL, api_key=os.getenv("OPENAI_API_KEY"))
    env = EHR_Environment()
    
    # 1. Mandatory [START] Log
    print(f"[START] task={task_id} env=ehr_clinical_assistant model={MODEL_NAME}")
    
    obs = env.reset(task_id=task_id)
    rewards = []
    done = False
    step_count = 0
    
    # 2. Agent Loop
    while not done and step_count < 5:
        step_count += 1
        
        # We ask for JSON specifically to avoid parsing errors
        prompt = (
            f"Notes: {obs.record_data['raw_notes']}\n"
            f"Current Chart: {obs.record_data['diagnoses']}\n"
            "Action: Respond ONLY with a JSON object like: "
            '{"command": "ADD_DIAGNOSIS", "payload": "CODE"}'
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0 # Crucial for reproducibility
        )
        
        # 3. Parse and Step
        try:
            action_data = json.loads(response.choices[0].message.content)
            action_obj = Action(**action_data)
            obs, reward, done, info = env.step(action_obj)
            
            rewards.append(reward)
            
            # 4. Mandatory [STEP] Log
            print(f"[STEP] step={step_count} action={json.dumps(action_data)} "
                  f"reward={reward:.2f} done={str(done).lower()} error=null")
                  
        except Exception as e:
            print(f"[STEP] step={step_count} action=ERROR reward=0.00 done=true error={str(e)}")
            break

    # 5. Mandatory [END] Log
    final_score = rewards[-1] if rewards else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(final_score >= 0.8).lower()} steps={step_count} "
          f"score={final_score:.2f} rewards={rewards_str}")
    
    return final_score

if __name__ == "__main__":
    # You can now loop through all tasks automatically
    for task in ["easy_coding", "medium_triage", "hard_reconcile"]:
        run_baseline(task)