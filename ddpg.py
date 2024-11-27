from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch
from environment import Touhou14Env

# Set up the environment
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(
        f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected'}"
    )

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Initialize the environment
    env = Touhou14Env()

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch version: {torch.__version__}")

    # Define action noise for exploration
    action_dim = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim), sigma=0.3 * np.ones(action_dim)
    )  # Increase noise

    # Initialize the DDPG model
    model = DDPG(
        "CnnPolicy",
        env,
        action_noise=action_noise,
        buffer_size=50000,
        batch_size=64,
        train_freq=(10, "step"),
        learning_rate=0.0005,
        device="cuda",
        verbose=1,
    )

    print(f"Stable-Baselines3 is using device: {model.device}")

    # Train the model
    model.learn(total_timesteps=1000, log_interval=4)

    # Save the model
    model.save("./save/ddpg")

    print("model saved")


except Exception as e:
    print(f"An error occurred: {e}")

finally:
    env.close()
    print("Environment closed.")
