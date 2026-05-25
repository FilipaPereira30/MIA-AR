from __future__ import annotations

from mia_rl.core.base import Environment

# ── Type aliases ────────────────────────────────────────────────────────────
# The board is a 9-tuple of ints (one per cell, row-major):
#   0 = empty, 1 = player X, -1 = player O
# Actions are integers 0-8 identifying the cell to mark.
TicTacToeState  = tuple[int, ...]   # length-9
TicTacToeAction = int               # 0 … 8

# Indices of every winning line (rows, columns, diagonals)
_WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)


def _winner(board: TicTacToeState) -> int:
    """Return 1 if X wins, -1 if O wins, 0 otherwise."""
    for i, j, k in _WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


class TicTacToeEnv(Environment[TicTacToeState, TicTacToeAction]):
    """Two-player Tic-Tac-Toe environment.

    Conventions:
    - Player X always goes first (represented as +1 in the board).
    - Player O is represented as -1.
    - `current_player` alternates between 1 (X) and -1 (O) each step.
    - The state is a length-9 tuple representing all 9 cells row-major:
        indices  0 1 2
                 3 4 5
                 6 7 8
    - `step()` applies the current player's move, then switches turns.
    - Episode ends when a player wins or the board is full (draw).
    - Rewards from the perspective of the player who just moved:
        +1  for winning
        -1  for losing (opponent wins — not possible in one step, included for completeness)
         0  otherwise (ongoing or draw)

    For self-play, call `reset()` at the start of each game and alternate
    calling `step()` for player X and player O.
    """

    def __init__(self) -> None:
        self.board: TicTacToeState = (0,) * 9
        self.current_player: int = 1  # X starts

    def reset(self) -> TicTacToeState:
        """Reset the board to an empty state and set X as the first player.

        TODO:
        1. Set `self.board` to a tuple of nine zeros.
        2. Set `self.current_player` to 1 (player X).
        3. Return the initial board state.
        """
        raise NotImplementedError("TODO: implement reset.")

    def available_actions(self, state: TicTacToeState) -> list[TicTacToeAction]:
        """Return the indices of all empty cells in `state`.

        TODO:
        1. Return a list of all cell indices i where state[i] == 0.
        """
        raise NotImplementedError("TODO: implement available_actions.")

    def is_terminal(self, state: TicTacToeState) -> bool:
        """Return True if the game is over (win or draw).

        TODO:
        1. Use `_winner(state)` to check if any player has won.
        2. Also return True if there are no empty cells left (draw).
        """
        raise NotImplementedError("TODO: implement is_terminal.")

    def step(self, action: TicTacToeAction) -> tuple[TicTacToeState, float, bool]:
        """Place the current player's mark on cell `action` and advance the game.

        TODO:
        1. Validate that `action` is a legal move (cell must be empty).
           Raise `ValueError` if not.
        2. Build the new board by placing `self.current_player` at `action`.
           Hint: boards are tuples — use tuple slicing or `list` conversion.
        3. Check for a winner using `_winner`.
        4. Determine whether the episode is done:
           - done = True if there is a winner OR no empty cells remain.
        5. Compute the reward for the player who just moved:
           - +1 if that player won, 0 otherwise.
        6. Switch `self.current_player` to the other player (-1 ↔ +1).
        7. Update `self.board` to the new board.
        8. Return `(new_board, reward, done)`.
        """
        raise NotImplementedError("TODO: implement step.")

    def render(self, state: TicTacToeState | None = None) -> None:
        """Print a human-readable board to stdout.

        TODO:
        1. If `state` is None, use `self.board`.
        2. Map cell values: 1 → 'X', -1 → 'O', 0 → '.'.
        3. Print three rows of three symbols separated by spaces,
           with a blank line between rows for readability.
           Example output for an empty board:
               . . .
               . . .
               . . .
        """
        raise NotImplementedError("TODO: implement render.")
