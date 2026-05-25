from __future__ import annotations

from typing import Callable

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState, _winner

Policy = Callable[[TicTacToeEnv, TicTacToeState], TicTacToeAction]


def play_game(
    env: TicTacToeEnv,
    policy_x: Policy,
    policy_o: Policy,
    render: bool = True,
) -> int:
    state = env.reset()
    if render:
        print("Initial board:")
        env.render(state)
        print()

    while not env.is_terminal(state):
        player_label = "X" if env.current_player == 1 else "O"

        # Selecionar a política correta e obter a ação
        policy = policy_x if env.current_player == 1 else policy_o
        action = policy(env, state)

        state, reward, done = env.step(action)

        if render:
            print(f"Player {player_label} plays cell {action}:")
            env.render(state)
            print()

    result = _winner(state)
    if render:
        if result == 1:
            print("X wins!")
        elif result == -1:
            print("O wins!")
        else:
            print("Draw!")
    return result