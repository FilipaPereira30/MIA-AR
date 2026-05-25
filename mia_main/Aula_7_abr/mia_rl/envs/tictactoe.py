from __future__ import annotations

from mia_rl.core.base import Environment

TicTacToeState  = tuple[int, ...]
TicTacToeAction = int

_WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def _winner(board: TicTacToeState) -> int:
    for i, j, k in _WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


class TicTacToeEnv(Environment[TicTacToeState, TicTacToeAction]):

    def __init__(self) -> None:
        self.board: TicTacToeState = (0,) * 9
        self.current_player: int = 1

    def reset(self) -> TicTacToeState:
        self.board = (0,) * 9
        self.current_player = 1
        return self.board

    def available_actions(self, state: TicTacToeState) -> list[TicTacToeAction]:
        return [i for i, cell in enumerate(state) if cell == 0]

    def is_terminal(self, state: TicTacToeState) -> bool:
        return _winner(state) != 0 or all(cell != 0 for cell in state)

    def step(self, action: TicTacToeAction) -> tuple[TicTacToeState, float, bool]:
        if self.board[action] != 0:
            raise ValueError(f"Cell {action} is already occupied.")

        new_board = list(self.board)
        new_board[action] = self.current_player
        new_board = tuple(new_board)

        winner = _winner(new_board)
        done = winner != 0 or all(cell != 0 for cell in new_board)
        reward = 1.0 if winner == self.current_player else 0.0

        self.current_player = -self.current_player
        self.board = new_board

        return new_board, reward, done

    def render(self, state: TicTacToeState | None = None) -> None:
        if state is None:
            state = self.board
        symbols = {1: "X", -1: "O", 0: None}
        rows = []
        for row in range(3):
            cells = []
            for col in range(3):
                i = row * 3 + col
                v = state[i]
                cells.append(symbols[v] if v != 0 else str(i))
            rows.append(" {} | {} | {}".format(*cells))
        divider = "---+---+---"
        print(f"\n{divider}\n".join(rows))