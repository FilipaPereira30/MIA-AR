"""
run_explorer.py
---------------
Treina e compara três agentes no ambiente Explorer 5x5:
  - SARSA (on-policy TD control)
  - Q-Learning (off-policy TD control)
  - Monte Carlo Control (first-visit)

Gera três ficheiros em outputs/explorer/:
  - learning_curves.png
  - steps_per_episode.png
  - policy_qlearning.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.explorer import ExplorerEnv, ACTIONS
from agents.control.sarsa import SarsaAgent
from agents.control.q_learning import QLearningAgent
from agents.control.monte_carlo import MonteCarloControlAgent
from experiments.explorer import run_sarsa, run_qlearning, run_monte_carlo
from plots.explorer import plot_learning_curves, plot_steps_per_episode, plot_policy_grid

# =========================================================
# CONFIGURAÇÃO
# =========================================================

EPISODES  = 3000
MAX_STEPS = 200
SEED      = 42

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "explorer")

# =========================================================
# AMBIENTE
# =========================================================

env = ExplorerEnv()

# =========================================================
# AGENTES
# =========================================================

sarsa_agent = SarsaAgent(
    actions=ACTIONS,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.998,
    epsilon_min=0.05,
    seed=SEED,
)

qlearning_agent = QLearningAgent(
    actions=ACTIONS,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.998,
    epsilon_min=0.05,
    seed=SEED,
)

mc_agent = MonteCarloControlAgent(
    actions=ACTIONS,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.998,
    epsilon_min=0.05,
    seed=SEED,
)

# =========================================================
# TREINO
# =========================================================

print("A treinar SARSA...")
sarsa_rewards, sarsa_steps = run_sarsa(env, sarsa_agent, episodes=EPISODES, max_steps=MAX_STEPS)

print("A treinar Q-Learning...")
qlearning_rewards, qlearning_steps = run_qlearning(env, qlearning_agent, episodes=EPISODES, max_steps=MAX_STEPS)

print("A treinar Monte Carlo Control...")
mc_rewards, mc_steps = run_monte_carlo(env, mc_agent, episodes=EPISODES, max_steps=MAX_STEPS)

print("\nTreino concluído.")
print(f"  SARSA        — recompensa final (média últimos 100): {sum(sarsa_rewards[-100:]) / 100:.2f}")
print(f"  Q-Learning   — recompensa final (média últimos 100): {sum(qlearning_rewards[-100:]) / 100:.2f}")
print(f"  Monte Carlo  — recompensa final (média últimos 100): {sum(mc_rewards[-100:]) / 100:.2f}")

# =========================================================
# PLOTS
# =========================================================

plot_learning_curves(
    results={
        "SARSA":        sarsa_rewards,
        "Q-Learning":   qlearning_rewards,
        "Monte Carlo":  mc_rewards,
    },
    output_path=os.path.join(OUTPUT_DIR, "learning_curves.png"),
)

plot_steps_per_episode(
    results={
        "SARSA":        sarsa_steps,
        "Q-Learning":   qlearning_steps,
        "Monte Carlo":  mc_steps,
    },
    output_path=os.path.join(OUTPUT_DIR, "steps_per_episode.png"),
)

plot_policy_grid(
    env=env,
    agent=qlearning_agent,
    title="Política Greedy — Q-Learning",
    output_path=os.path.join(OUTPUT_DIR, "policy_qlearning.png"),
)

plot_policy_grid(
    env=env,
    agent=sarsa_agent,
    title="Política Greedy — SARSA",
    output_path=os.path.join(OUTPUT_DIR, "policy_sarsa.png"),
)
