import sys
from EHR_ENV.server.app import EHR_Environment
from schema import Action, Observation

def validate():
    print("🔍 Starting OpenEnv Manual Validation...")
    env = EHR_Environment()

    # 1. Test Reset
    try:
        obs = env.reset(task_id="easy_coding")
        if not isinstance(obs, Observation):
            print("❌ FAIL: reset() must return an Observation object.")
            return
        print("✅ reset() returned a valid Observation.")
    except Exception as e:
        print(f"❌ FAIL: reset() crashed: {e}")
        return

    # 2. Test Step
    try:
        action = Action(command="ADD_DIAGNOSIS", payload="I10")
        obs, reward, done, info = env.step(action)
        
        if not isinstance(reward, (int, float)):
            print("❌ FAIL: reward must be a number.")
            return
        if not isinstance(done, bool):
            print("❌ FAIL: done must be a boolean.")
            return
        print("✅ step() returned valid (Obs, Reward, Done, Info) types.")
    except Exception as e:
        print(f"❌ FAIL: step() crashed: {e}")
        return

    # 3. Test State
    try:
        s = env.state()
        if not isinstance(s, Observation):
            print("❌ FAIL: state() must return an Observation.")
            return
        print("✅ state() is functional.")
    except Exception as e:
        print(f"❌ FAIL: state() crashed: {e}")
        return

    print("\n🚀 All OpenEnv interface checks passed!")

if __name__ == "__main__":
    validate()