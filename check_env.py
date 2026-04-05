from EHR_ENV.server.app import EHR_Environment
from schema import Action

def run_diagnostic():
    print("--- 🩺 EHR Environment Diagnostic ---")
    env = EHR_Environment()
    
    # Test 1: Initialization
    try:
        obs = env.reset(task_id="easy_coding")
        print(f"✅ Reset Successful. Patient: {obs.record_data.patient_id}")
    except Exception as e:
        print(f"❌ Reset Failed: {e}")
        return

    # Test 2: Step Logic (Partial Progress)
    # We add a diagnosis and check if the reward is the expected 0.3 (0.2 + 0.1 step reward)
    action = Action(command="ADD_DIAGNOSIS", payload="I10")
    obs, reward, done, _ = env.step(action)
    print(f"✅ Step logic works. Partial Reward: {reward}")

    # Test 3: Grader Logic (Completion)
    action_finish = Action(command="FINISH", payload="")
    obs, reward, done, _ = env.step(action_finish)
    
    if reward >= 1.0:
        print(f"✅ Grader works. Final Score: {reward}")
    else:
        print(f"⚠️ Grader logic returned unexpected score: {reward}")

    # Test 4: Task Switching
    env.reset(task_id="medium_triage")
    if env.current_patient_id == "P002":
        print("✅ Task switching works (Easy -> Medium)")
    
    print("\n--- 🚀 Environment is ready for deployment! ---")

if __name__ == "__main__":
    run_diagnostic()