# `mia_rl`

## Package organization

- `mia_rl/core/`
  - generic abstractions such as `Environment`, `Policy`, `Agent`, `Episode`, `Transition`
- `mia_rl/envs/`
  - interactive environments for model-free methods
- `mia_rl/mdps/`
  - known-model MDP abstractions for dynamic programming classes
- `mia_rl/policies/`
  - reusable policies
- `mia_rl/agents/`
  - learning algorithms grouped by task
- `mia_rl/plots/`
  - reusable plotting functions
- `mia_rl/experiments/`
  - rollout, training, evaluation, and experiment helpers
- `mia_rl/scripts/`
  - runnable experiment scripts
- `mia_rl/outputs/`
  - generated plots and experiment results

## New files for this class

- `mia_rl/envs/tictactoe.py` — `TicTacToeEnv` (portfolio exercise, see below)
- `mia_rl/features/tictactoe.py` — `encode_state` (27-dim one-hot, perspective-relative), `random_action`
- `mia_rl/experiments/tictactoe.py` — `play_game` (evaluates two policies against each other)
- `mia_rl/notebooks/TicTacToe_Demo.ipynb` — demo notebook: watch two random agents play

## Current practical TODOs

1. **`mia_rl/features/tictactoe.py`** — implement `encode_state`:
   - Create a zero array of shape `(27,)` with dtype `float32`
   - Loop over each cell; set slot `i*3+0` (my piece), `i*3+1` (opponent), or `i*3+2` (empty)
   - Return the feature vector

2. **`mia_rl/experiments/tictactoe.py`** — inside `play_game`, select the correct policy for the current player and call it to get an action (2 lines, hint in the TODO comment)

After implementing the TODOs, open `mia_rl/notebooks/TicTacToe_Demo.ipynb` and run it to watch a game.

## Portfolio exercise

Implement the Tic-Tac-Toe environment in `mia_rl/envs/tictactoe.py`:
- `reset()` — reset the board and set X as the first player
- `available_actions(state)` — return indices of all empty cells
- `is_terminal(state)` — return True if the game is won or drawn
- `step(action)` — place the current player's mark, compute reward, switch turns
- `render(state)` — print a human-readable board (empty cells show their index 0–8)

