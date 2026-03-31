from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from environment import Touhou14Env
from ddpg import DDPG
from ddpg_action_wrapper import DiscretizeActionWrapper
from datetime import datetime
import os
import numpy as np

# Set up hyperparameters similar to DQN
buffer_size = 10000  # Replay memory size similar to DQN
batch_size = 64  # Mini-batch size
learning_rate = 0.005  # Learning rate similar to DQN
train_freq = (10, "step")  # Training frequency
total_timesteps = 50000  # Number of training steps
exploration_noise = 0.1  # Action noise to promote exploration

# Set up save directory
save_dir = f"./save/ddpg_{datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set up environment and wrapper
env = Touhou14Env()
wrapped_env = DiscretizeActionWrapper(env)

# Configure logger and checkpoint callback
logger = configure(save_dir, ["csv", "stdout"])
chkpt_callback = CheckpointCallback(
    save_freq=total_timesteps // 10, save_path=save_dir, name_prefix="model", verbose=2
)

# Set up action noise for exploration
action_dim = wrapped_env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(action_dim),
    sigma=exploration_noise * np.ones(action_dim) * wrapped_env.action_space.high[0],
)

# Initialize the DDPG model with hyperparameters similar to the DQN
model = DDPG(
    "MultiInputPolicy",
    wrapped_env,
    buffer_size=buffer_size,
    batch_size=batch_size,
    train_freq=train_freq,
    learning_rate=learning_rate,
    action_noise=action_noise,
    verbose=1,
    device="cuda",
    stats_window_size=5,
)

# Set logger
model.set_logger(logger)

# Train the model
try:
    model.learn(
        total_timesteps=total_timesteps, log_interval=1, callback=chkpt_callback
    )

    # Save the final trained model
    model.save(os.path.join(save_dir, "model_final"))
    print("Model training completed and saved successfully!")

except Exception as e:
    print(f"An error occurred during training: {e}")

finally:
    # Ensure the environment is properly closed
    if "wrapped_env" in locals() and wrapped_env is not None:
        wrapped_env.close()
    print("Environment closed.")
