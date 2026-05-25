from __future__ import annotations

from typing import List, Tuple

# =========================================================
# FUNÇÕES DE TREINO
# =========================================================


def run_qlearning(env, agent, episodes: int = 2000, max_steps: int = 200):
    """
    Treina o agente Q-Learning no ambiente Explorer.
    Devolve (rewards_per_episode, steps_per_episode).
    """
    rewards_history = []
    steps_history   = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action     = agent.select_action(state)
            next_state, reward, done = env.step(state, action)
            agent.update(state, action, reward, next_state, done)

            state         = next_state
            total_reward += reward
            steps        += 1

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        steps_history.append(steps)

    return rewards_history, steps_history


def run_sarsa(env, agent, episodes: int = 2000, max_steps: int = 200):
    """
    Treina o agente SARSA no ambiente Explorer.
    Devolve (rewards_per_episode, steps_per_episode).
    """
    rewards_history = []
    steps_history   = []

    for _ in range(episodes):
        state  = env.reset()
        action = agent.select_action(state)

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            next_state, reward, done = env.step(state, action)

            if done:
                next_action = action  # não importa, bootstrap=0
            else:
                next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            state  = next_state
            action = next_action

            total_reward += reward
            steps        += 1

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        steps_history.append(steps)

    return rewards_history, steps_history


def run_monte_carlo(env, agent, episodes: int = 2000, max_steps: int = 200):
    """
    Treina o agente Monte Carlo Control no ambiente Explorer.
    Devolve (rewards_per_episode, steps_per_episode).
    """
    rewards_history = []
    steps_history   = []

    for _ in range(episodes):
        state      = env.reset()
        trajectory = []
        steps      = 0

        for _ in range(max_steps):
            action              = agent.select_action(state)
            next_state, reward, done = env.step(state, action)
            trajectory.append((state, action, reward))

            state  = next_state
            steps += 1

            if done:
                break

        agent.update_episode(trajectory)
        agent.decay_epsilon()

        total_reward = sum(r for _, _, r in trajectory)
        rewards_history.append(total_reward)
        steps_history.append(steps)

    return rewards_history, steps_history


def smooth(values: List[float], window: int = 50) -> List[float]:
    """Média móvel para suavizar curvas de aprendizagem."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / (i - start + 1))
    return result
