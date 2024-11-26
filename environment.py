import gymnasium as gym
import numpy as np
import interface as I
from collections import deque
from typing import Any

class Touhou14Env(gym.Env):
    """
    Gymnasium environment for Touhou 14 (stage 1, normal difficulty, reimu B).

    Notice that frame stack is already included in this env, so don't wrap it
    with FrameStack again.
    """

    def __init__(self):
        self.n_frame_stack = 2
        self.frame_buffer = deque(maxlen=self.n_frame_stack)
        
        # Observation space remains unchanged
        self.observation_space = gym.spaces.Box(
            0, 255, (I.FRAME_HEIGHT, I.FRAME_WIDTH, self.n_frame_stack), dtype=np.uint8
        )
        
        # Define a continuous action space for DDPG
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Two continuous actions: [-1, 1] for movement, and [-1, 1] for slow mode intensity
        
        self.max_lost_lives = 2

        # Initialize the game interface
        I.init()
        I.suspend_game_process()
        self.info = self._get_info()
        # Used to truncate episode when losing too many lives
        self.initial_lives = self.info["lives"]

    def step(self, action: np.ndarray):
        """
        Step through the environment using continuous actions.
        """
        move = int((action[0] + 1) / 2 * 4)  # Map [-1, 1] to [0, 4]
        slow = 1 if action[1] > 0 else 0

        for _ in range(self.n_frame_stack):
            I.resume_game_process()
            I.act(move, slow)
            I.suspend_game_process()

            if I.read_game_status_int("game_state") != 2:  # End of run
                break

            if I.read_game_status_int("in_dialog") == -1:  # In dialog
                I.resume_game_process()
                I.skip_dialog()
                I.suspend_game_process()

            frame = I.capture_frame()
            self.frame_buffer.append(np.array(frame))

        next_state = self._get_stacked_frames()
        curr_info = self._get_info()

        # Calculate reward
        reward = self._calc_return(**curr_info) - self._calc_return(**self.info)

        # Penalize inactivity based on observation changes, action repetition, or position
        if self._is_inactive(action) or self._is_inactive_position(curr_info):
            reward -= 5  # Apply inactivity penalty

        terminated = curr_info["game_state"] != 2
        truncated = curr_info["lives"] < self.initial_lives - self.max_lost_lives
        self.info = curr_info
        return next_state, reward, terminated, truncated, curr_info

    def _is_inactive(self, action: np.ndarray):
        """
        Detect inactivity based on repeated actions or unchanged observations.
        """
        # Action repetition tracking
        if not hasattr(self, "last_action"):
            self.last_action = action
            self.repeated_action_count = 0

        # Check for repeated actions
        if np.allclose(action, self.last_action, atol=0.1):  # Allow small action variation
            self.repeated_action_count += 1
        else:
            self.repeated_action_count = 0  # Reset repetition counter if actions vary

        self.last_action = action

        # Threshold for inactivity due to repeated actions
        if self.repeated_action_count > 10:  # Change threshold as needed
            return True

        # Optional: Add observation-based inactivity (as above)
        return self._is_inactive_observation()
    
    def _is_inactive_observation(self):
        """
        Detect inactivity based on lack of changes in recent observations.
        """
        if len(self.frame_buffer) < self.n_frame_stack:
            return False  # Not enough frames to compare

        # Compare the most recent frames in the buffer
        recent_frames = list(self.frame_buffer)
        frame_differences = [
            np.sum(np.abs(recent_frames[i] - recent_frames[i + 1])) for i in range(len(recent_frames) - 1)
        ]

        # If differences are below a small threshold for all comparisons, consider inactive
        inactivity_threshold = 1e-3  # Adjust as needed
        return all(diff < inactivity_threshold for diff in frame_differences)


    def reset(self, seed: int | None = None):
        super().reset(seed=seed)

        # Reset the game state
        I.resume_game_process()
        I.release_all_keys()
        if I.read_game_status_int("game_state") == 1:  # End of run
            I.reset_from_end_of_run()
        else:
            I.force_reset()
        I.suspend_game_process()

        # Initialize the frame buffer
        frame = I.capture_frame()
        self.frame_buffer.clear()
        for _ in range(self.n_frame_stack):
            self.frame_buffer.append(np.array(frame))
        state = self._get_stacked_frames()
        info = self._get_info()
        self.info = info
        self.initial_lives = info["lives"]
        return state, info

    def close(self):
        I.clean_up()

    def _get_stacked_frames(self) -> np.ndarray:
        """
        Grayscale the frames and stack them on the last axis.
        """
        return np.clip(
            np.stack(
                np.dot(np.stack(self.frame_buffer, axis=0), [0.2989, 0.5870, 0.1140]),
                axis=-1,
            ),
            0,
            255,
        ).astype(np.uint8)

    def _get_info(self) -> dict[str, int]:
        info = {}
        for k in (
            "score",
            "lives",
            "life_fragments",
            "bombs",
            "bomb_fragments",
            "power",
            "game_state",
            "in_dialog",
        ):
            info[k] = I.read_game_status_int(k)
        return info

    def _calc_return(
        self,
        score: int,
        lives: int,
        life_fragments: int,
        bombs: int,
        bomb_fragments: int,
        power: int,
        **kwargs: dict[str, int],
    ):
        """
        Calculate the return from game states.

        The formula is designed to encourage grabbing life fragments, bomb
        fragments and power items, while penalizing life losses.

        1 life = 3 life fragments
        1 bomb = 8 bomb fragments
        """
        return (
            score
            + (lives + life_fragments / 3) * 200000
            + (bombs + bomb_fragments / 8) * 100000
            + power * 1000
        )