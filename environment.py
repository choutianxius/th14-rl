import gymnasium as gym
import numpy as np
import interface as I
from collections import deque
from typing import Any
import cv2
import logging
import sys


class Touhou14Env(gym.Env):
    """
    Gymnasium environment for Touhou 14 (stage 1, normal difficulty, reimu B).

    Notice that frame stack is already included in this env, so don't wrap it
    with FrameStack again.
    """

    def __init__(
        self,
        n_frame_stack: int = 4,
        frame_downsize_ratio: float = 1.0,
        max_lost_lives: int = 0,
        debug: bool = False,
    ):
        if n_frame_stack < 1:
            raise ValueError("Number of stacked frames should be positive")
        if frame_downsize_ratio <= 0.0 or frame_downsize_ratio > 1.0:
            raise ValueError("Invalid frame downsize ratio, should be 0-1")
        if max_lost_lives < 0:
            raise ValueError(
                "Maximum number of lost lives allowed should be non-negative"
            )
        if debug:
            self.logger = logging.getLogger("Touhou14Env")
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        else:
            self.logger = None
        self.n_frame_stack = n_frame_stack
        self.frame_downsize_ratio = frame_downsize_ratio
        self.frame_buffer = deque(maxlen=self.n_frame_stack)
        self.observation_space = gym.spaces.Dict(
            {
                "frames": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        int(I.FRAME_HEIGHT * self.frame_downsize_ratio),
                        int(I.FRAME_WIDTH * self.frame_downsize_ratio),
                        self.n_frame_stack,
                    ),
                    dtype=np.uint8,
                ),
                "player_position": gym.spaces.Box(
                    low=np.array((-184.0, 32.0), dtype=np.float32),
                    high=np.array((184.0, 432.0), dtype=np.float32),
                    shape=(2,),
                ),
                "boss_position": gym.spaces.Box(
                    low=np.array((-184.0, 32.0), dtype=np.float32),
                    high=np.array((184.0, 432.0), dtype=np.float32),
                    shape=(2,),
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(10)
        self.max_lost_lives = max_lost_lives

        I.init()
        I.suspend_game_process()
        self.info = self._get_game_info()
        # used to truncate episode when losing too many lives
        self.initial_lives = self.info["lives"]
        self.episode_time = 0
        self.finish_bonus_given = False
        self.prev_pos = None
        self.prev_boss_pos = []

    def step(self, action: int | np.integer[Any]):
        self.episode_time += 1
        move, slow = int(action % 5), int(action // 5)

        for _ in range(self.n_frame_stack):
            I.resume_game_process()
            I.act(move, slow)
            I.suspend_game_process()

            if I.read_game_val("game_state") != 2:  # end of run
                break

            if I.read_game_val("in_dialog") == -1:  # in dialog
                I.resume_game_process()
                I.skip_dialog()
                I.suspend_game_process()

            frame = I.capture_frame()
            self.frame_buffer.append(np.array(frame))

        next_state = self._get_state()
        curr_info = self._get_game_info()

        terminated = curr_info["game_state"] != 2
        truncated = curr_info["lives"] < self.initial_lives - self.max_lost_lives
        prev_info = self.info
        self.info = curr_info

        diff_life = (curr_info["lives"] - prev_info["lives"]) * 3 + (
            curr_info["life_fragments"] - prev_info["life_fragments"]
        )
        diff_boss_hp = min(0, curr_info["boss_hp"] - prev_info["boss_hp"])
        if diff_boss_hp < -100:
            diff_boss_hp = 0
        reward = diff_life * 500 - diff_boss_hp + 1
        # clear bonus
        if curr_info["boss_hp"] == 9999 and prev_info["boss_hp"] == 0:
            reward += 1500 * max(0, 500 - self.episode_time) / 500

        # penalize useless movement
        if np.all(next_state["player_position"] == self.prev_pos) and move != 0:
            reward -= 10
        self.prev_pos = next_state["player_position"]
        self.prev_boss_pos = next_state["boss_position"]

        if self.logger:
            self.logger.debug({"action": action.tolist(), "reward": reward})

        return next_state, reward, terminated, truncated, curr_info

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)

        I.resume_game_process()
        I.release_all_keys()
        if I.read_game_val("game_state") == 1:  # end of run
            I.reset_from_end_of_run()
        else:
            I.force_reset()
        I.suspend_game_process()

        frame = I.capture_frame()
        self.frame_buffer.clear()
        for _ in range(self.n_frame_stack):
            self.frame_buffer.append(np.array(frame))
        state = self._get_state()
        info = self._get_game_info()
        self.info = info
        self.initial_lives = info["lives"]
        self.episode_time = 0
        self.finish_bonus_given = False
        self.prev_pos = None
        self.prev_boss_pos = None
        return state, info

    def close(self):
        I.clean_up()

    def _get_state(self) -> dict:
        frames_gray_stacked = np.clip(
            np.stack(
                np.dot(np.stack(self.frame_buffer, axis=0), [0.2989, 0.5870, 0.1140]),
                axis=-1,
            ),
            0,
            255,
        ).astype(np.uint8)
        if self.frame_downsize_ratio == 1.0:
            resized_frames = frames_gray_stacked
        else:
            # note the new size param passed to cv2 is (width, hight)
            resized_frames = cv2.resize(
                frames_gray_stacked,
                (
                    int(I.FRAME_WIDTH * self.frame_downsize_ratio),
                    int(I.FRAME_HEIGHT * self.frame_downsize_ratio),
                ),
                interpolation=cv2.INTER_AREA,
            )

        pos_x = I.read_game_val("f_player_pos_x")
        pos_y = I.read_game_val("f_player_pos_y")
        boss_pos_x = I.read_game_val("f_boss_pos_x")
        boss_pos_y = I.read_game_val("f_boss_pos_y")
        if boss_pos_x is not None and boss_pos_x is not None:
            boss_position = np.array((boss_pos_x, boss_pos_y), dtype=np.float32)
        else:
            boss_position = self.prev_boss_pos

        return {
            "frames": resized_frames,
            "player_position": np.array((pos_x, pos_y), dtype=np.float32),
            "boss_position": boss_position,
        }

    def _get_game_info(self) -> dict[str, int]:
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
            "boss_hp",
        ):
            info[k] = I.read_game_val(k) or 0
        return info
