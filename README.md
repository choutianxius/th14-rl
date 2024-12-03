# Reinforcement Learning Agent for Touhou 14

![Formatting Check](https://github.com/choutianxius/th14-rl/actions/workflows/check-formatting.yaml/badge.svg)

<video width="384" height="448" src="https://github.com/user-attachments/assets/1567d045-3942-45bf-986f-c1e9bc32bfc0" type="video/mp4"></video>

Provides a reinforcement learning environment and agent for [Touhou 14](https://en.touhouwiki.net/wiki/Double_Dealing_Character).

## Quick Start

### Platform

The Touhou 14 game is only available on Windows. Windows 11 is tested.

### Game

You must own a copy of the Touhou 14 game (ver 1.00b) to work with this project.

### Setup

This project depends on Python 3.12. First, create a virtual environment via Powershell:

```shell
py -3.12 -m venv .venv
```

To activate the virtual environment, run:

```shell
.venv\Scripts\Activate.ps1
```

Install the dependencies by:

```shell
pip install -r requirements.txt
```

Notice that we are using PyTorch v2.5.1 which depends on [CUDA v12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive). You should install the exact CUDA version.

### Training

We use the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library for implementations of our RL agents (DQN, dueling DQN and DDPG). Please check the [`train_dqn.py`](train_dqn.py) and [`train_ddpg.py`](train_ddpg.py) scripts for details, which should be self-explanatory.

### Evaluation

Please check the [`eval.py`](eval.py) script.

### Misc

We also provide a script for recording videos for the environment, which is based on [`moviepy`](https://zulko.github.io/moviepy/). Check [`make_movie.py`](make_movie.py).

## Interface

The [game interface](./interface.py) wraps around the game binary and is responsible for:

- [x] Reading in-game status for reward calculation such as current score, remaining lives, etc.;
- [x] Sending agent actions to the game;
- [x] Capturing game frames for state calculation;
- [x] Handling setting up and maintaining the game, e.g., entering the selected game stages, resetting to stage beginnings, skipping dialogs, etc.

The interface uses the `pywin32` and `pygetwindow` libraries as well as the built-in `ctypes` library for handling the game process. The `pyscreeze` library is used for capturing screenshots for the game scene. You should run the game binary (`th14.exe`) in windowed mode with 640x480 resolution before using any utility from the interface.

The RL state is represented as the game scenes. Rewards are calculated from reading variables in the games memory, for example, current score, remaining lives, power level, etc. Memory offsets for most of the variables are taken from Guy-L's work ([Acknowledgement](#acknowledgement)), while some of them are found using CE.

Since we have no direct control over the game engine, the actions are applied by maintaining keyboard status using the `keyboard` library. The Touhou 14 game's clock is frame dependent, and we wait until at least the next frame before issuing another keyboard status change so that the actions can be effectively received by the game engine.

Utilities to suspend and resume the game process are also provided for the convenience of making the RL environment. For example, we want to suspend the game process when training the agent networks, which may take a lot of time compared with 1 frame in the game.

## Gymnasium Environment

### Problem Setting

For simplicity, the RL problem is currently restricted to:

- Character: Reimu A
- Using the "Spell Practice" mode
- Stage 1, spell card 2, No. 4
- Difficulty: Normal

Note that the interface can only load the desired game level if your save file has certain progress (namely, you should have finished this spell card but not the less challenging ones). We will fix this constraint in the future.

We expect the game process is active and you are on the title screen when initializing the environment by calling `Touhou14Env()`. The environment will be initialized to the start of the game run with the settings above. Also, the environment will keep the character always shooting, since there is no obvious advantage not doing so.

### Observation Space

We use a composite observation space containing both the stacked game frames (with sidebar areas cropped out) and extra in-game information (the positions of the character and the boss).

### Action Space

Please check the [Touhou wiki](https://en.touhouwiki.net/wiki/Double_Dealing_Character/Gameplay) if you are not familiar with the game play.

We use a discrete action space (`Discrete(10)`), which encodes two discrete dimensions:

- The first dimension represents the movement of the character;
- The second dimension represents whether the character is in "slow" mode.

For RL algorithms that expect continuous action spaces, we also provide an action wrapper class for our environment (see [`ddpg_action_wrapper.py`](ddpg_action_wrapper.py))

Values of the two dimensions are calculated by:

```math
\text{dim1} = \text{action} \% 5
```

```math
\text{dim1} = \left\lfloor \text{action} / 5 \right\rfloor
```

| Value of the First Dimension | Meaning     |
| ---------------------------- | ----------- |
| 0                            | No movement |
| 1                            | Move left   |
| 2                            | Move right  |
| 3                            | Move up     |
| 4                            | Move down   |

| Value of the Second Dimension | Meaning        |
| ----------------------------- | -------------- |
| 0                             | Normal         |
| 1                             | Slow (focused) |

The actions are mapped to keyboard press/release status by the interface.

### Reward

The reward is calculated using the following formula from in-game variables:

```math
\text{criterion} = \text{total lives} \times 1500 - \text{boss HP}
```

```math
\text{reward} = \Delta{\text{criterion}} + 1 + \text{clear bonus} - \text{risky y-pos penalty} - \text{useless action penalty}
```

The reward formula is designed so that the agent can focus on the goal of clearing the boss without dying. To encourage "cleaner" play styles, the clear bonus will be decreased based on the length to clear the boss. The two penalty items have been added to explicitly aid the agents to avoid suboptimal greedy policies. Finally, the positive 1 in the formula encourages the agent to stay alive longer if not completing the boss. 

The weights in the formula are to some extent arbitrary, and it's possible to experiment with various settings to find optimal ones for training agents.

Initially, we used the in-game score as the major component for calculating the criterion, and it turned out that the score is overly affected by collecting items, and it also has too significant absolute values, which hampers the formation of effective policies. This is a valuable lesson about the importance of a well-designed reward function when making RL agents to solve real problems. 

## Possible Enhancements

- [ ] Improve the game interface as a class.
- [ ] Dynamically get the "inner" game screenshot, instead of using fixed window offsets.
- [ ] Dynamically go to the desired game level by checking the game progress.
- [ ] Support multiple modes and levels.

## Acknowledgement

Much of the game interface is adapted from Guy-L's great work (see the [thgym](https://github.com/Guy-L/thgym/tree/master) repository and the [parakit](https://github.com/Guy-L/parakit) repository).

