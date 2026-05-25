from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class TD0Prediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0):
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float) # Initialize the value table as a defaultdict of floats, which defaults to 0.0 for unseen states.

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
