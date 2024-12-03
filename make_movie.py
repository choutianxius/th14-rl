from environment import Touhou14Env
import argparse
import os
from moviepy import ImageSequenceClip
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent",
    type=str,
    required=True,
    help="Agent class to use for recording, should be dqn|duel|ddpg",
)
parser.add_argument(
    "--model_path", type=str, default="", help="Path to the model save file"
)
parser.add_argument(
    "--save_dir", type=str, required=True, help="Folder to put the video in"
)
parser.add_argument(
    "--save_name",
    type=str,
    default=datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S") + ".mp4",
    help="Video filename. It should include the extension of the desired video format, e.g., .mp4",
)
parser.add_argument(
    "--episodes", type=int, default=5, help="Number of episodes to record"
)
args = parser.parse_args()

try:
    if (args.agent != "random") and (not os.path.exists(args.model_path)):
        raise ValueError("Save path doesn't exist")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.agent == "dqn" or args.agent == "duel":
        from stable_baselines3 import DQN

        model = DQN.load(args.model_path)
    elif args.agent == "ddpg":
        from ddpg import DDPG

        model = DDPG.load(args.model_path)
    elif args.agent == "random":
        from random_walk import RandomWalk

        model = RandomWalk()
    else:
        raise ValueError("Invalid agent type, should be dqn or ddpg or random")

    env = Touhou14Env()
    frames = []

    for _ in range(args.episodes):
        obs, info = env.reset()
        frames.extend(env.frame_buffer)
        while True:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.extend(env.frame_buffer)
            if terminated or truncated:
                break

    clip = ImageSequenceClip(frames, fps=60)

    filename = os.path.join(
        args.save_dir,
        args.save_name,
    )
    clip.write_videofile(filename=filename)
    print(f"\033[92mSaved video to {filename}\033[0m")
finally:
    if "env" in locals():
        print("\033[96mQuitting...\033[0m")
        env.close()
