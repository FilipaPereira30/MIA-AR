from __future__ import annotations

from mia_rl.core.base import Environment

WindyGridworldState = tuple[int, int]
WindyGridworldAction = str

ACTIONS: tuple[WindyGridworldAction, ...] = ("up", "down", "left", "right")
ACTION_TO_DELTA: dict[WindyGridworldAction, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class WindyGridworldEnv(Environment[WindyGridworldState, WindyGridworldAction]):
    def __init__(
        self,
        rows: int = 7,
        cols: int = 10,
        start: WindyGridworldState = (3, 0),
        goal: WindyGridworldState = (3, 7),
        wind: tuple[int, ...] = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0),
        reward_per_step: float = -1.0,
    ):
        if len(wind) != cols:
            raise ValueError("The wind specification must contain one strength per column.")

        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.wind = wind
        self.reward_per_step = reward_per_step
        self.current_state = start

    def states(self) -> list[WindyGridworldState]:
        return [(row, col) for row in range(self.rows) for col in range(self.cols)]

    def reset(self) -> WindyGridworldState:
        self.current_state = self.start
        return self.current_state

    def available_actions(self, state: WindyGridworldState) -> list[WindyGridworldAction]:
        return list(ACTIONS)

    def step_from_state(self, state: WindyGridworldState, action: WindyGridworldAction) -> tuple[WindyGridworldState, float, bool]:
        row, col = state

        # 1. Ler o delta da ação
        d_row, d_col = ACTION_TO_DELTA[action]

        # 2. Aplicar ação + vento da coluna atual (vento empurra para cima = diminui linha)
        new_row = row + d_row - self.wind[col]
        new_col = col + d_col

        # 3. Clampar dentro dos limites da grelha
        new_row = max(0, min(new_row, self.rows - 1))
        new_col = max(0, min(new_col, self.cols - 1))

        next_state = (new_row, new_col)

        # 4. Devolver (next_state, reward, done)
        return next_state, self.reward_per_step, next_state == self.goal
 
    # ------------------------------------------------------------------
    # step (atualiza o estado interno)
    # ------------------------------------------------------------------

        #Apply the chosen action and the column wind, keeping the agent inside the grid.

        #TODO:
        #1. Read `(delta_row, delta_col)` from `ACTION_TO_DELTA[action]`.
        #2. Use `self.wind[col]` to push the agent upward.
        #3. Clamp the resulting row and column to the grid bounds.
        #4. Return `(next_state, self.reward_per_step, next_state == self.goal)`.
        #raise NotImplementedError("TODO: implement Windy Gridworld dynamics.")

    def step(self, action: WindyGridworldAction) -> tuple[WindyGridworldState, float, bool]:
        next_state, reward, done = self.step_from_state(self.current_state, action)
        self.current_state = next_state
        return next_state, reward, done
