from EHR_ENV.server.app import EHR_Environment
from schema import Action

def test_flow():
    env = EHR_Environment()
    
    # 1. Test Reset
    obs = env.reset(task_id="easy_coding")
    print(f"Initial Obs: {obs.record_data.patient_id} - {obs.record_data.raw_notes[:30]}...")

    # 2. Test Step (Adding a Diagnosis)
    action = Action(command="ADD_DIAGNOSIS", payload="I10")
    obs, reward, done, info = env.step(action)
    print(f"Step Reward (Partial): {reward}")

    # 3. Test Finish (Grader)
    action_finish = Action(command="FINISH", payload="")
    obs, reward, done, info = env.step(action_finish)
    print(f"Final Reward: {reward}")
    print(f"Is Done: {done}")

if __name__ == "__main__":
    test_flow()