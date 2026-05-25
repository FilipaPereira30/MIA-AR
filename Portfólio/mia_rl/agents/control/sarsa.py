from __future__ import annotations

from collections import defaultdict
import random

# =========================================================
# SARSA AGENT
#
# On-policy TD control.
# Atualização: Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]
# onde a' é a ação realmente selecionada no próximo estado.
# =========================================================

class SarsaAgent:

    def __init__(
        self,
        actions,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int | None = None,
    ):
        self.actions       = actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.rng           = random.Random(seed)

        self.Q: dict = defaultdict(lambda: defaultdict(float))

    def reset(self):
        self.Q       = defaultdict(lambda: defaultdict(float))
        self.epsilon = 1.0

    def select_action(self, state) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.greedy_action(state)

    def greedy_action(self, state) -> str:
        return max(self.actions, key=lambda a: self.Q[state][a])

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        """
        bootstrap = 0.0 if done else self.Q[next_state][next_action]
        current = self.Q[state][action]
        self.Q[state][action] = current + self.alpha * (
            reward + self.gamma * bootstrap - current
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
