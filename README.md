# Reinforcement Learning Agent for Touhou 14

![Formatting Check](https://github.com/choutianxius/th14-rl/actions/workflows/check-formatting.yaml/badge.svg)

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

- Character: Reimu B
- Stage: 1
- Difficulty: Normal
- Using the "practice start" mode

We expect the game process is active and you are on the title screen when initializing the environment by calling `Touhou14Env()`. The environment will be initialized to the start of the game run with the settings above. Also, the environment will keep the character always shooting, since there is no obvious advantage not doing so.

### Observation Space

We use a composite observation space containing both the stacked game frames (with sidebar areas cropped out) and extra in-game information (currently, the position of the character).

### Action Space

Please check the [Touhou wiki](https://en.touhouwiki.net/wiki/Double_Dealing_Character/Gameplay) if you are not familiar with the game play.

We use a discrete action space (`Discrete(10)`), which encodes two discrete dimensions:

- The first dimension represents the movement of the character;
- The second dimension represents whether the character is in "slow" mode.

Values of the two dimensions are calculated by:

$$ dim1 = action \ \% \ 5$$

$$ dim2 = action \ // \ 5$$

| Value of the First Dimension | Meaning     |
| ---------------------------- | ----------- |
| 0                            | No movement |
| 1                            | Move left   |
| 2                            | Move right  |
| 3                            | Move up     |
| 4                            | Move down   |

| Value of the Second Dimension | Meaning |
| ----------------------------- | ------- |
| 0                             | Normal  |
| 1                             | Slow    |

The actions are mapped to keyboard press/release status by the interface.

### Reward

The reward is calculated using a combination of in-game score and some custom logic. It's calculated using the following formula from in-game variables:

$$ \text{criterion} = \text{score} + \text{total lives} \times 20000 + \text{total bombs} \times 10000 + \text{power} \times 1000 - \text{time in auto-collect zone} \times 10000 $$

$$ \text{reward} = \Delta{\text{criterion}} $$

The in-game score is increased by hitting enemies, collecting items, moving close to enemy bullets (so called "graze"s) and clearing stages. Beyond that, we want to explicitly encourage the agent to collect items and penalize life losses, so the extra items are added in the formula.

We also penalize for staying in the auto-item-collect zone for too long. This term is introduced to prevent the agent from "lazily" learning a policy to always stay in that zone.

The weights in the formula are to some extent arbitrary, and it's possible to experiment with various settings to find optimal ones for training agents.

## Acknowledgement

Much of the game interface is adapted from Guy-L's great work (see the [thgym](https://github.com/Guy-L/thgym/tree/master) repository).
