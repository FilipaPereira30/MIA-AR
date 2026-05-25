from __future__ import annotations
from collections import defaultdict
from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState

class NStepTDPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, n: int, alpha: float = 0.05, gamma: float = 1.0):
        self.n = n  # O valor de n define o horizonte de predição [cite: 24]
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update the value table using TD(0)."""
        
        for transition in episode.transitions:
            state = transition.state
            reward = transition.reward
            next_state = transition.next_state
            
            # ALTERA AQUI: de transition.terminal para transition.done
            done = transition.done 

            if done:
                v_next = 0.0
            else:
                v_next = self.V[next_state]

            target = reward + self.gamma * v_next
            self.V[state] += self.alpha * (target - self.V[state])
            
    def value_of(self, state: BlackjackState) -> float:
            """Devolve a estimativa atual do valor para um estado."""
            return float(self.V[state])