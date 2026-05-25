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

## New files for previous class (Practical 7 — TicTacToe environment)

- `mia_rl/envs/tictactoe.py` — `TicTacToeEnv` 
- `mia_rl/features/tictactoe.py` — `encode_state` (27-dim one-hot, perspective-relative), `random_action`
- `mia_rl/experiments/tictactoe.py` — `play_game` (evaluates two policies against each other)
- `mia_rl/notebooks/TicTacToe_Demo.ipynb` — demo notebook: watch two random agents play

## New files for this class (Practical 8 — Policy Gradient: REINFORCE)

- `mia_rl/agents/control/reinforce.py` — `ReinforceAgent`: linear softmax policy, MC returns, entropy regularisation
- `mia_rl/experiments/reinforce_tictactoe.py` — self-play training loop, evaluation vs random, opponent mixing
- `mia_rl/notebooks/TicTacToe_PolicyGradient.ipynb` — practical notebook

## Current practical TODOs (Practical 8)

1. **`mia_rl/agents/control/reinforce.py`** — `_probs()` — implement the masked softmax:
   - Compute logits `h(s, a) = θ[a] · φ(s)` for each available action
   - Subtract the max logit for numerical stability before exponentiating
   - Normalise so probabilities sum to 1

2. **`mia_rl/agents/control/reinforce.py`** — `update_episode()` — fill in the score function update inside the provided loop: `+φ·(1−π(a_t))` for the chosen action, `−φ·π(a')` for all others

3. **`mia_rl/experiments/reinforce_tictactoe.py`** — `run_reinforce_episode()` — inject `r = −1` into the losing player's last trajectory step after a win is detected

After implementing the TODOs, open `mia_rl/notebooks/TicTacToe_PolicyGradient.ipynb` and run it.



