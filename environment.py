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
        self.n_frame_stack = 4
        self.frame_buffer = deque(maxlen=self.n_frame_stack)
        self.observation_space = gym.spaces.Box(
            0, 255, (I.FRAME_HEIGHT, I.FRAME_WIDTH, self.n_frame_stack), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(10)
        self.max_lost_lives = 2

        I.init()
        I.suspend_game_process()
        self.info = self._get_info()
        # used to truncate episode when losing too many lives
        self.initial_lives = self.info["lives"]

    def step(self, action: int | np.integer[Any]):
        move, slow = int(action % 5), int(action // 5)

        for _ in range(self.n_frame_stack):
            I.resume_game_process()
            I.act(move, slow)
            I.suspend_game_process()

            if I.read_game_int("game_state") != 2:  # end of run
                break

            if I.read_game_int("in_dialog") == -1:  # in dialog
                I.resume_game_process()
                I.skip_dialog()
                I.suspend_game_process()

            frame = I.capture_frame()
            self.frame_buffer.append(np.array(frame))

        next_state = self._get_stacked_frames()
        curr_info = self._get_info()
        reward = self._calc_return(**curr_info) - self._calc_return(**self.info)
        terminated = curr_info["game_state"] != 2
        truncated = curr_info["lives"] < self.initial_lives - self.max_lost_lives
        self.info = curr_info
        return next_state, reward, terminated, truncated, curr_info

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)

        I.resume_game_process()
        I.release_all_keys()
        if I.read_game_int("game_state") == 1:  # end of run
            I.reset_from_end_of_run()
        else:
            I.force_reset()
        I.suspend_game_process()

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
            info[k] = I.read_game_int(k)
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

        For reference, in-game clear-stage score bonus is 3 million.

        The current setting is quite arbitrary, and we might experiment with
        multiple settings to find one with high learning speed
        """
        return (
            score
            + (lives + life_fragments / 3) * 200000
            + (bombs + bomb_fragments / 8) * 100000
            + power * 1000
        )
