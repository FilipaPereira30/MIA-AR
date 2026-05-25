from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class SarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}

    def select_action(self, state: StateT) -> ActionT:
        #Choose an epsilon-greedy action and cache it for the SARSA bootstrap.

        #TODO:
        #1. With probability `self.epsilon`, choose a random action from `self.actions`.
        #2. Otherwise choose an action with the highest current action-value.
        #3. Store the chosen action in `self._selected_actions[state]` and return it.
               
     #raise NotImplementedError("TODO: implement epsilon-greedy action selection for Sarsa.")
        
        # 1. Exploração: escolhe ação aleatória com probabilidade epsilon
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        # 2. Exploração: escolhe a ação com maior valor Q atual
        else:
            action = max(self.actions, key=lambda a: self.action_value_of(state, a))

        # 3. Guarda a ação escolhida em cache e devolve
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        #Apply the SARSA update using the cached next action for the next state.

        #TODO:
        #1. Use a bootstrap value of `0.0` on terminal transitions.
        #2. Otherwise read the cached next action from `self._selected_actions[transition.next_state]`.
        #3. Compute the SARSA target `reward + gamma * Q(next_state, next_action)`.
        #4. Apply the incremental update with `self.alpha`.

        #raise NotImplementedError("TODO: implement the SARSA update.")
        
        
        s  = transition.state
        a  = transition.action
        r  = transition.reward
        s_ = transition.next_state

        # 1. Se terminal, bootstrap = 0; caso contrário usa Q(S', A')
        if transition.done:
            bootstrap = 0.0
        else:
            # 2. Lê a ação seguinte já escolhida pela política (on-policy!)
            a_ = self._selected_actions[s_]
            bootstrap = self.action_value_of(s_, a_)

        # 3. Target SARSA: R + γ * Q(S', A')
        target = r + self.gamma * bootstrap

        # 4. Atualização incremental
        self.Q[(s, a)] += self.alpha * (target - self.Q[(s, a)])
        
    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
