from __future__ import annotations

import random

import numpy as np

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState

STATE_FEATURE_DIM: int = 27


def encode_state(board: TicTacToeState, current_player: int) -> np.ndarray:
    features = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    for i, cell in enumerate(board):
        if cell == current_player:
            features[i * 3 + 0] = 1.0
        elif cell == -current_player:
            features[i * 3 + 1] = 1.0
        else:
            features[i * 3 + 2] = 1.0
    return features


def random_action(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
    return random.choice(env.available_actions(state))