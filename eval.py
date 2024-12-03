from stable_baselines3 import DQN
from ddpg import DDPG
from environment import Touhou14Env
from ddpg_action_wrapper import DiscretizeActionWrapper
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path", "-f", type=str, required=True, help="Path to model save zip file"
)
parser.add_argument(
    "--episodes", "-n", type=int, default=1, help="Number of episodes to evaluate"
)
parser.add_argument(
    "--algorithm", "-a", type=str, default="dqn", help="Algorithm, dqn or ddpg"
)
args = parser.parse_args()

try:
    env = Touhou14Env()

    if args.algorithm == "dqn":
        model = DQN.load(args.save_path)
    elif args.algorithm == "ddpg":
        env = DiscretizeActionWrapper(env)
        model = DDPG.load(args.save_path, env=env)
    else:
        raise ValueError("Invalid algorithm, should be dqn or ddpg")

    total_rewards = []
    total_damages = []
    clear_times = 0
    for _ in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        while True:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        final_boss_hp = info["boss_hp"]
        total_damage = 1500 if final_boss_hp == 9999 else 1500 - final_boss_hp
        print(
            f"\033[96mTotal reward: {total_reward}, total damage: {total_damage}\033[0m"
        )
        total_rewards.append(total_reward)
        total_damages.append(total_damage)
        if terminated:
            clear_times += 1
    with open(args.save_path.split(".")[0] + "-eval_results.json", "w") as f:
        json.dump(
            {
                "total_rewards": total_rewards,
                "total_damages": total_damages,
                "clear_times": clear_times,
            },
            f,
        )
finally:
    if "env" in locals():
        print("\033[96mQuitting...\033[0m")
        env.close()
