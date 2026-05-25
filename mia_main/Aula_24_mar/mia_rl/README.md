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

- `mia_rl/features/windy_gridworld.py` — tile-coding feature functions (`tile_features`, `state_action_features`)
- `mia_rl/agents/prediction/linear_td.py` — `LinearTD0` (semi-gradient TD(0))
- `mia_rl/agents/control/linear_sarsa.py` — `LinearSarsaControl` (semi-gradient SARSA, NumPy)
- `mia_rl/agents/control/torch_sarsa.py` — `TorchSarsaControl` (semi-gradient SARSA, PyTorch)
- `mia_rl/experiments/fa_training.py` — training helpers for FA agents
- `mia_rl/scripts/run_windy_gridworld_linear_td.py`
- `mia_rl/scripts/run_windy_gridworld_linear_sarsa.py`
- `mia_rl/scripts/run_windy_gridworld_torch_sarsa.py`

## Updated files for this class

- `mia_rl/plots/windy_gridworld.py` — added `plot_td_errors`, `plot_value_heatmap`, `plot_episode_length_comparison`

## Current practical TODOs

For the function-approximation practical, complete:
- `mia_rl/agents/control/linear_sarsa.py` — `update_transition`
- `mia_rl/agents/control/torch_sarsa.py` — `update_transition` (`use_optimizer=True` branch)


After implementing the TODOs, run:

- `python -m mia_rl.scripts.run_windy_gridworld_linear_td`
- `python -m mia_rl.scripts.run_windy_gridworld_linear_sarsa`
- `python -m mia_rl.scripts.run_windy_gridworld_torch_sarsa`

By default, generated figures are saved under `mia_rl/outputs/`.

## Portfolio exercise

Implement the Tic-Tac-Toe environment in `mia_rl/envs/tictactoe.py`:
- `reset()` — reset the board and set X as the first player
- `available_actions(state)` — return indices of all empty cells
- `is_terminal(state)` — return True if the game is won or drawn
- `step(action)` — place the current player's mark, compute reward, switch turns
- `render(state)` — print a human-readable board to stdout

