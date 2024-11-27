from stable_baselines3 import DQN
from environment import Touhou14Env
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path", "-f", type=str, required=True, help="Path to model save dir"
)
args = parser.parse_args()

try:
    env = Touhou14Env()

    model = DQN("MultiInputPolicy", env, buffer_size=1)
    model.load(args.save_path)

    total_reward = 0
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"\033[96mTotal reward: {total_reward}\033[0m")
            break
finally:
    if "env" in locals():
        print("\033[96mQuitting...\033[0m")
        env.close()
