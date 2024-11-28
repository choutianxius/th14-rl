from stable_baselines3 import DQN
from environment import Touhou14Env
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path", "-f", type=str, required=True, help="Path to model save dir"
)
parser.add_argument(
    "--episodes", "-n", type=int, default=1, help="Number of episodes to evaluate"
)
args = parser.parse_args()

try:
    env = Touhou14Env()

    model = DQN.load(args.save_path)

    total_reward = 0
    obs, info = env.reset()
    for _ in range(args.episodes):
        while True:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"\033[96mTotal reward: {total_reward}\033[0m")
                obs, info = env.reset()
                total_reward = 0
                break
finally:
    if "env" in locals():
        print("\033[96mQuitting...\033[0m")
        env.close()
