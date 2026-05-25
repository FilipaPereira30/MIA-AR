from __future__ import annotations

import numpy as np

from mia_rl.features.tictactoe import STATE_FEATURE_DIM

# Number of board cells (= number of possible actions in TicTacToe)

N_ACTIONS: int = 9


class ReinforceAgent:
    #REINFORCE (Monte Carlo policy gradient) for TicTacToe.

    #Policy parameterization — softmax over available actions:

        #h(s, a)  =  θ[a] · φ(s)          logit for action a
        #π(a | s) =  softmax_available(h)  probability over legal moves only

    #φ(s) is the 27-dim perspective-relative feature vector from
    #mia_rl.features.tictactoe.encode_state``:
        #- 3 dims per cell × 9 cells = 27 dims
        #- [1,0,0] my piece  |  [0,1,0] opponent  |  [0,0,1] empty
    #Using a perspective-relative encoding allows the same θ to play
    #both X and O during self-play training.

    #θ ∈ R^{n_actions × n_features} — one weight vector per board cell.

    #REINFORCE update (applied at the end of each episode):

        #G_t = Σ_{k≥t} γ^{k−t} r_k          (discounted return from step t)

        #θ[a_t]  +=  α · γ^t · G_t · ∇_{θ[a_t]} log π(a_t | s_t)
        #θ[a']   +=  α · γ^t · G_t · ∇_{θ[a']}  log π(a_t | s_t)  ∀ a' ≠ a_t

    #Gradient of log π w.r.t. θ[a] for a linear softmax (score function):
        #a  =  a_t  →  +φ(s_t) · (1 − π(a_t | s_t))
        #a  ≠  a_t  →  −φ(s_t) ·  π(a   | s_t)

    #Optional entropy regularisation (entropy_beta > 0):
        #θ[a]  +=  α · β · ∂H/∂θ[a]
        #∂H/∂θ[a] = −π(a|s) · (log π(a|s) + H(π)) · φ(s)


    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        n_features: int = STATE_FEATURE_DIM,
        alpha: float = 0.01,
        gamma: float = 1.0,
        entropy_beta: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self._rng = np.random.default_rng(seed)
        self.reset()

    # ── Internals ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        #Initialise / reinitialise policy weights to zero.
        self.theta = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        self._episode: list[tuple[np.ndarray, int, list[int], float]] = []

    def _probs(self, phi: np.ndarray, available: list[int]) -> np.ndarray:
        #Softmax probabilities over *available* actions only.

        #Args:
            #phi: 27-dim feature vector for the current board/player.
            #available: list of legal action indices (0-based).

        #Returns:
            #1-D array of length ``len(available)`` that sums to 1.

        # TODO (1/3): Implement the masked softmax over available actions.
        #
        # Steps:
        #   1. Compute the logits: for each available action a, the logit is
        #      h(s, a) = θ[a] · φ(s).  You can do this in one matrix-vector
        #      product using self.theta[available] @ phi.
        #   2. Subtract the maximum logit before exponentiating for numerical
        #      stability (this doesn't change the output, just avoids overflow).
        #   3. Exponentiate and normalise so the probabilities sum to 1.
        #
        # Hint: the softmax policy from the slides is
        #   π(a|s) = exp(h(s,a)) / Σ_{b ∈ A(s)} exp(h(s,b))
        # Note that only *available* actions appear in the denominator.
        
        #raise NotImplementedError
    
        # 1. Logits para as ações disponíveis: h(s,a) = θ[a] · φ(s)
        logits = self.theta[available] @ phi

        # 2. Subtrai o máximo para estabilidade numérica
        logits -= np.max(logits)

        # 3. Softmax
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    # ── Action selection ───────────────────────────────────────────────────

    def select_action(self, phi: np.ndarray, available: list[int]) -> int:
        #Sample an action proportionally to the current stochastic policy.
        probs = self._probs(phi, available)
        idx = self._rng.choice(len(available), p=probs)
        return available[int(idx)]

    def greedy_action(self, phi: np.ndarray, available: list[int]) -> int:
        #Return the most probable available action (for evaluation / play).
        probs = self._probs(phi, available)
        return available[int(np.argmax(probs))]

    # ── Learning ───────────────────────────────────────────────────────────

    def store_step(
        self,
        phi: np.ndarray,
        action: int,
        available: list[int],
        reward: float,
    ) -> None:
        #Append one environment step to the episode buffer.
        self._episode.append((phi, action, available, reward))

    def update_episode(
        self,
        trajectory: list[tuple[np.ndarray, int, list[int], float]] | None = None,
    ) -> float:
        #Compute Monte Carlo returns and apply the REINFORCE gradient update.

        #Args:
            #trajectory: if provided, update from this trajectory instead of
                        #the internal buffer (useful for self-play).

        #Returns:
            #Mean policy-gradient loss over the episode (for monitoring).


        episode = trajectory if trajectory is not None else self._episode
        if not episode:
            return 0.0

        T = len(episode)

        # Backward pass — discounted returns
        returns = np.empty(T)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = episode[t][3] + self.gamma * G
            returns[t] = G

        total_loss = 0.0
        for t, (phi, action, available, _) in enumerate(episode):
            probs = self._probs(phi, available)
            action_idx = available.index(action)
            total_loss -= returns[t] * np.log(probs[action_idx] + 1e-8)

            # TODO (2/3): Apply the policy gradient update for every available action.
            #   scale = α · γ^t · G_t
            #   chosen action a_t  →  θ[a_t] += scale · φ · (1 − π(a_t))
            #   other action  a'   →  θ[a' ] -= scale · φ ·  π(a')
            
            scale = self.alpha * (self.gamma ** t) * returns[t]
            for i, a in enumerate(available):
                if a == action:
                    self.theta[a] += scale * phi * (1.0 - probs[i])
                else:
                    self.theta[a] -= scale * phi * probs[i]
            
                #raise NotImplementedError

            if self.entropy_beta > 0.0:
                H = -float(np.sum(probs * np.log(probs + 1e-8)))
                for i, a in enumerate(available):
                    log_p = np.log(probs[i] + 1e-8)
                    self.theta[a] += self.alpha * self.entropy_beta * (-probs[i] * (log_p + H)) * phi

        if trajectory is None:
            self._episode.clear()
        return total_loss / T
