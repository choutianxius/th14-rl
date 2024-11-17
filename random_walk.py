from environment import Touhou14Env
from random import choice
import numpy as np

episodes = 2

env = Touhou14Env()
try:
    for episode in range(episodes):
        state, info = env.reset()
        step = 1
        while True:
            action = np.array([choice(range(5)), choice(range(2))])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Episode {episode + 1}, step {step}, reward = {reward}")
            step += 1
            if done:
                break
except Exception as e:
    print(e)
finally:
    env.close()
