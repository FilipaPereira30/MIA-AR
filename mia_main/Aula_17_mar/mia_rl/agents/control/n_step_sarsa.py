from __future__ import annotations

import random
from collections import defaultdict, deque

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.n_steps = n_steps
        self.alpha   = alpha
        self.epsilon = epsilon
        self.rng     = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        # Buffer circular com as últimas n transições ainda não processadas
        self._buffer: deque[Transition[StateT, ActionT]] = deque()

    # ------------------------------------------------------------------
    # Política ε-greedy (igual ao SARSA base)
    # ------------------------------------------------------------------

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            action = max(self.actions, key=lambda a: self.action_value_of(state, a))
        self._selected_actions[state] = action
        return action

    # ------------------------------------------------------------------
    # Update: acumula no buffer e atualiza quando tem n passos
    # ------------------------------------------------------------------

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        self._buffer.append(transition)

        # Só atualiza quando o buffer tem n transições OU o episódio terminou
        if len(self._buffer) < self.n_steps and not transition.done:
            return

        self._flush_one()

    def _flush_one(self) -> None:
        """Aplica a atualização n-step para a transição mais antiga do buffer."""
        if not self._buffer:
            return

        # Calcular o retorno G a partir das n transições no buffer
        G = 0.0
        for i, t in enumerate(self._buffer):
            G += (self.gamma ** i) * t.reward

        # Bootstrap com Q(S_{t+n}, A_{t+n}) se o último passo não for terminal
        last = self._buffer[-1]
        if not last.done:
            a_ = self._selected_actions.get(last.next_state)
            if a_ is not None:
                G += (self.gamma ** len(self._buffer)) * self.action_value_of(last.next_state, a_)

        # Atualizar Q para o estado/ação mais antigos do buffer
        oldest = self._buffer.popleft()
        s, a = oldest.state, oldest.action
        self.Q[(s, a)] += self.alpha * (G - self.Q[(s, a)])

    # ------------------------------------------------------------------
    # No fim do episódio, esvaziar o buffer com as transições restantes
    # ------------------------------------------------------------------

    def end_episode(self) -> None:
        while self._buffer:
            self._flush_one()

    # ------------------------------------------------------------------

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda a: self.action_value_of(state, a))