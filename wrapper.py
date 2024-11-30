import gymnasium as gym
import numpy as np

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscretizeActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.discrete_actions = env.action_space.n
        self.step_count = 0

    def action(self, action: np.ndarray) -> int:
        # Debugging: Print the action details
        # print(f"Received action: {action}, Shape: {action.shape}")

        horizontal_value, vertical_value = action

        # Discretize the horizontal value
        if horizontal_value < -0.5:
            move_horizontal = 0  # Move left
        elif horizontal_value > 0.5:
            move_horizontal = 1  # Move right
        else:
            move_horizontal = 2  # Stay still horizontally

        # Discretize the vertical value
        if vertical_value < -0.5:
            move_vertical = 3  # Move down
        elif vertical_value > 0.5:
            move_vertical = 4  # Move up
        else:
            move_vertical = 5  # Stay still vertically

        # Alternate between horizontal and vertical to avoid bias
        if self.step_count % 2 == 0:
            # Prioritize vertical actions every other step
            if move_vertical in [3, 4]:
                action_mapped = move_vertical
            else:
                action_mapped = move_horizontal
        else:
            # Prioritize horizontal actions
            if move_horizontal in [0, 1]:
                action_mapped = move_horizontal
            else:
                action_mapped = move_vertical

        # Debugging: Print which action is taken
        print(f"Mapped action: {action_mapped}")

        self.step_count += 1
        return action_mapped
