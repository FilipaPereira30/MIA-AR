from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class FirstVisitMonteCarloPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def reset(self) -> None:
        self.V = defaultdict(float) #estimated value of the state
        self.N = defaultdict(int) #how many times have I already seen this state as a first visit?

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        #Update the value table using first-visit Monte Carlo prediction.

        #TODO:
        #1. Traverse the episode backwards to compute the return G for each time step.
        #2. Identify the first visit of each state within the episode.
        #3. Update the sample-average estimate using self.N[state].
            # Incremental mean update:
            # V_new = V_old + (G - V_old) / N
        
        returns = [0.0] * len(episode.transitions)
        G = 0.0
        visited_states = set()
        
        for i in range(len(episode.transitions) - 1, -1, -1):
            transition = episode.transitions[i]
            state = transition.state
            reward = transition.reward
            
            G = reward + G
            
            first_visit_idx = next(j for j, t in enumerate(episode.transitions) if t.state == state)
            
            if i == first_visit_idx:
                # 3. Update the sample-average estimate using self.N[state].
                self.N[state] += 1
                
                # Incremental mean update: V_new = V_old + (G - V_old) / N
                self.V[state] += (G - self.V[state]) / self.N[state]
                
        
    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])