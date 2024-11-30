import gymnasium as gym
import numpy as np


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super(DiscretizeActionWrapper, self).__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "The wrapped environment must have a Discrete action space"
            )
        self.n = env.action_space.n
        self.action_space = gym.spaces.Box(
            low=0.0, high=float(self.n), dtype=np.float32
        )

    def action(self, action: np.ndarray) -> int:
        if action.item() == self.n:
            return self.n - 1
        else:
            return int(action.item())
