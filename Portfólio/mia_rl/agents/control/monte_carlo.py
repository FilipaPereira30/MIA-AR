from __future__ import annotations

from collections import defaultdict
import random

# =========================================================
# MONTE CARLO CONTROL AGENT
#
# On-policy first-visit MC control com epsilon-greedy.
# Aprende da trajetória completa de cada episódio.
# Atualização: Q(s,a) <- média das recompensas acumuladas
# observadas após visitar (s,a).
# =========================================================

class MonteCarloControlAgent:

    def __init__(
        self,
        actions,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int | None = None,
    ):
        self.actions       = actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.rng           = random.Random(seed)

        self.Q: dict      = defaultdict(lambda: defaultdict(float))
        self._returns: dict = defaultdict(list)

    def reset(self):
        self.Q        = defaultdict(lambda: defaultdict(float))
        self._returns = defaultdict(list)
        self.epsilon  = 1.0

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
    # ATUALIZAÇÃO (first-visit MC)
    # --------------------------------------------------

    def update_episode(self, trajectory):
        """
        Recebe a trajetória completa do episódio:
            [(state, action, reward), ...]
        e atualiza Q com first-visit MC returns.
        """
        G = 0.0
        visited = set()

        for state, action, reward in reversed(trajectory):
            G = reward + self.gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                self._returns[(state, action)].append(G)
                self.Q[state][action] = sum(self._returns[(state, action)]) / len(
                    self._returns[(state, action)]
                )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
