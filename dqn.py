from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import DQN
from environment import Touhou14Env
from datetime import datetime
import os
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument(
    "--memory",
    "-m",
    type=int,
    default=10000,
    help="Replay memory size. This should be limited to your available RAM",
)
parser.add_argument(
    "--steps", "-n", type=int, default=100000, help="Number of training steps"
)
parser.add_argument(
    "--target_update_interval",
    "-t",
    type=int,
    default=5000,
    help="Number of steps before synchronizing target and online networks",
)
parser.add_argument(
    "--n_save_chkpts", "-N", type=int, default=10, help="Number of checkpoints to save"
)
args = parser.parse_args()


try:
    env = Touhou14Env()

    # save dir
    save_dir = f"./save/dqn_{datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # record training config
    metadata = {
        "memory": args.memory,
        "steps": args.steps,
        "target_update_interval": args.target_update_interval,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # setup model
    logger = configure(save_dir, ["csv", "stdout"])
    chkpt_callback = CheckpointCallback(
        save_freq=args.steps // args.n_save_chkpts,
        save_path=save_dir,
        name_prefix="model",
        verbose=2,
    )
    model = DQN(
        "MultiInputPolicy",
        env,
        buffer_size=args.memory,
        target_update_interval=args.target_update_interval,
        device="cuda",
    )
    model.set_logger(logger)

    # learn
    model.learn(total_timesteps=args.steps, log_interval=1, callback=chkpt_callback)

    # final save
    model.save(os.path.join(save_dir, "model_final"))
finally:
    if "env" in locals():
        print("\033[96mQuitting...\033[0m")
        env.close()
