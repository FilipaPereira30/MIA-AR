from __future__ import annotations

from collections import defaultdict
import random

# =========================================================
# Q-LEARNING AGENT
#
# Off-policy TD control.
# Atualização: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
# =========================================================

class QLearningAgent:

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

        # Q[state][action]
        self.Q: dict = defaultdict(lambda: defaultdict(float))

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):
        """Reinicia a tabela Q e o epsilon."""
        self.Q       = defaultdict(lambda: defaultdict(float))
        self.epsilon = 1.0

    # --------------------------------------------------
    # SELEÇÃO DE AÇÃO (epsilon-greedy)
    # --------------------------------------------------

    def select_action(self, state) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.greedy_action(state)

    def greedy_action(self, state) -> str:
        return max(self.actions, key=lambda a: self.Q[state][a])

    # --------------------------------------------------
    # ATUALIZAÇÃO (Q-Learning off-policy)
    # --------------------------------------------------

    def update(self, state, action, reward, next_state, done):
        """
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        if done:
            best_next = 0.0
        else:
            best_next = max(self.Q[next_state][a] for a in self.actions)

        current = self.Q[state][action]
        self.Q[state][action] = current + self.alpha * (
            reward + self.gamma * best_next - current
        )

    # --------------------------------------------------
    # EPSILON DECAY
    # --------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
