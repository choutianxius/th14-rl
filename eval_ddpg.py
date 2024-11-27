from stable_baselines3 import DDPG
from environment import Touhou14Env
import numpy as np

# Number of episodes to simulate
episodes = 2

# Initialize the environment
env = Touhou14Env()

# Load the trained model
try:
    model = DDPG.load("./save/ddpg", env=env)  # Ensure the environment is passed
    print("Model loaded successfully!")

    # Adding exploration noise during inference
    noise_sigma = 0.1  # Exploration noise level for random walk

    # Start simulation for the specified number of episodes
    for episode in range(episodes):
        obs, info = env.reset()
        step = 1
        total_reward = 0

        while True:
            # 10% chance of taking a random action for exploration
            if np.random.rand() < 0.1:
                action = env.action_space.sample()
            else:
                # Predict the next action (using non-deterministic behavior)
                action, _ = model.predict(obs, deterministic=False)

                # Add uniform exploration noise to encourage movement during random walk
                action += np.random.uniform(-0.2, 0.2, size=action.shape)

            # Clip action to remain within valid action bounds
            action = np.clip(action, -1, 1)

            print(f"Episode {episode + 1}, Step {step}, Action: {action}")  # Debugging action

            # Take the action in the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Check if the episode is done
            if done or truncated:
                print(f"Episode {episode + 1} finished. Total Reward: {total_reward}")
                break

            step += 1

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure the environment is properly closed
    if 'env' in locals() and env is not None:
        env.close()
    print("Environment closed.")
