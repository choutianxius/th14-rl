# Reinforcement Learning Agent for Touhou 14

## Interface

The game interface wraps around the game binary and is responsible for:

- [x] Reading in-game status for reward calculation such as current score, remaining lives, etc.;
- [ ] Sending agent actions to the game;
- [ ] Capturing game frames for state calculation.

## Acknowledgement

The game interface implementation is adapted from Guy-L's great work (see the [thgym](https://github.com/Guy-L/thgym/tree/master) repository).
