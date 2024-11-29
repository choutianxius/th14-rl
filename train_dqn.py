from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3 import DQN
from dueling_dqn import DuelingDQNPolicy
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
    default=25000,
    help="Replay memory size. This should be limited to your available RAM",
)
parser.add_argument(
    "--steps", "-n", type=int, default=100000, help="Number of training steps"
)
parser.add_argument(
    "--target_update_interval",
    "-t",
    type=int,
    default=2500,
    help="Number of steps before synchronizing target and online networks",
)
parser.add_argument(
    "--n_save_chkpts", "-N", type=int, default=10, help="Number of checkpoints to save"
)
parser.add_argument(
    "--dueling", action="store_true", help="Use dueling architecture or not"
)
args = parser.parse_args()


try:
    env = Touhou14Env()

    # save dir
    save_dir = f"./save/dqn_{datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # record training config
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(vars(args), f)

    # setup model
    logger = configure(save_dir, ["csv", "stdout"])
    chkpt_callback = CheckpointCallback(
        save_freq=args.steps // args.n_save_chkpts,
        save_path=save_dir,
        name_prefix="model",
        verbose=2,
    )
    model = DQN(
        DuelingDQNPolicy if args.dueling else "MultiInputPolicy",
        env,
        buffer_size=args.memory,
        target_update_interval=args.target_update_interval,
        device="cuda",
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        policy_kwargs=dict(
            net_arch=(256, 256), features_extractor_class=CombinedExtractor
        ),
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
