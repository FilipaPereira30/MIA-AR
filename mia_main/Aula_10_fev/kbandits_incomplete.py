#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Environment: multi-armed bandit
# ============================================================

class KArmedBandit:
    def __init__(self, k=10, stationary=True, walk_std=0.01):
        self.k = k
        self.stationary = stationary
        self.walk_std = walk_std
        self.reset()

    def reset(self):
        self.q_true = np.random.randn(self.k)  # true action values
        self.optimal_action = np.argmax(self.q_true)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]

        # non-stationary random walk
        if not self.stationary:
            self.q_true += np.random.normal(0, self.walk_std, self.k)
            self.optimal_action = np.argmax(self.q_true)

        return reward


# ============================================================
# Base agent
# ============================================================

class BanditAgent:
    def __init__(self, k=10):
        self.k = k
        self.reset()

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        raise NotImplementedError


# ============================================================
# ε-greedy agent
# ============================================================

class EpsilonGreedy(BanditAgent):
    def __init__(self, k=10, epsilon=0.1, alpha=None, optimistic=0.0):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.optimistic = optimistic
        self.reset()

    def reset(self):
        super().reset()
        self.Q[:] = self.optimistic

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        # Escolhe a melhor ação estimada (exploração/greedy) [cite: 45, 53]
        else:
            return np.argmax(self.Q)

        
    def update(self, action, reward):
        self.t += 1
        self.N[action] += 1

        # Se alpha não for definido, usa a média aritmética simples [cite: 61]
        if self.alpha is None:
            step_size = 1 / self.N[action]
        # Se alpha for definido, usa um passo constante (bom para não-estacionários) [cite: 130]
        else:
            step_size = self.alpha

        # Fórmula: NovoEstimado = VelhoEstimado + Passo * [Alvo - VelhoEstimado] [cite: 62]
        self.Q[action] += step_size * (reward - self.Q[action])

# ============================================================
# UCB agent
# ============================================================

class UCB(BanditAgent):
    def __init__(self, k=10, c=2.0):
        super().__init__(k)
        self.c = c

    def select_action(self):
        self.t += 1

        # Garante que cada ação é tentada pelo menos uma vez
        for a in range(self.k):
            if self.N[a] == 0:
                return a

        # Aplica a fórmula do UCB [cite: 170]
        # At = argmax [ Qt(a) + c * sqrt(ln(t) / Nt(a)) ]
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# ============================================================
# Gradient bandit agent
# ============================================================

class GradientBandit(BanditAgent):
    def __init__(self, k=10, alpha=0.1, baseline=True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.reset()

    def reset(self):
        super().reset()
        self.H = np.zeros(self.k)
        self.avg_reward = 0.0

    def _policy(self):
        exp = np.exp(self.H - np.max(self.H))
        return exp / np.sum(exp)

    def select_action(self):
        probs = self._policy()
        probs = self._policy()
        # Seleciona a ação com base na distribuição de probabilidade calculada [cite: 194]
        return np.random.choice(self.k, p=probs)

    def update(self, action, reward):
        self.t += 1
        probs = self._policy()

        # Atualiza a média de recompensas (baseline) se estiver ativo [cite: 202, 204]
        if self.baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward
        else:
            baseline = 0

        # Atualiza as preferências H para todas as ações [cite: 199]
        for a in range(self.k):
            if a == action:
                # Para a ação tomada: H + alpha * (R - baseline) * (1 - pi)
                self.H[a] += self.alpha * (reward - baseline) * (1 - probs[a])
            else:
                # Para as outras ações: H - alpha * (R - baseline) * pi
                self.H[a] -= self.alpha * (reward - baseline) * probs[a]


# ============================================================
# Experiment runner
# ============================================================

def run_experiment(agent, env, steps=1000, runs=2000):
    rewards = np.zeros((runs, steps))
    optimal = np.zeros((runs, steps))

    for r in range(runs):
        env.reset()
        agent.reset()

        for t in range(steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update(action, reward)

            rewards[r, t] = reward
            optimal[r, t] = (action == env.optimal_action)

    return rewards.mean(axis=0), optimal.mean(axis=0)


# ============================================================
# Reproduce main Sutton & Barto plots
# ============================================================

def plot_epsilon_greedy():
    steps, runs = 1000, 2000
    env = KArmedBandit()

    epsilons = [0, 0.01, 0.1]

    plt.figure()
    for eps in epsilons:
        agent = EpsilonGreedy(epsilon=eps)
        rewards, _ = run_experiment(agent, env, steps, runs)
        plt.plot(rewards, label=f"ε={eps}")

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("ε-greedy comparison")
    plt.show()


def plot_optimistic_vs_ucb():
    steps, runs = 1000, 2000
    env = KArmedBandit()

    agents = {
        "Optimistic greedy": EpsilonGreedy(epsilon=0, optimistic=5),
        "UCB c=2": UCB(c=2),
    }

    plt.figure()
    for name, agent in agents.items():
        rewards, _ = run_experiment(agent, env, steps, runs)
        plt.plot(rewards, label=name)

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Optimistic vs UCB")
    plt.show()


def plot_gradient_bandit():
    steps, runs = 1000, 2000
    env = KArmedBandit()

    agents = {
        "α=0.1 baseline": GradientBandit(alpha=0.1, baseline=True),
        "α=0.4 baseline": GradientBandit(alpha=0.4, baseline=True),
        "α=0.1 no baseline": GradientBandit(alpha=0.1, baseline=False),
    }

    plt.figure()
    for name, agent in agents.items():
        rewards, _ = run_experiment(agent, env, steps, runs)
        plt.plot(rewards, label=name)

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Gradient bandit methods")
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    plot_epsilon_greedy()
    #plot_optimistic_vs_ucb()
    #plot_gradient_bandit()
