from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, FrozenSet

import numpy as np

# =========================================================
# ACTIONS
# =========================================================

ACTIONS = ("U", "D", "L", "R")

ACTION_TO_DELTA = {
    "U": (-1,  0),
    "D": ( 1,  0),
    "L": ( 0, -1),
    "R": ( 0,  1),
}

# =========================================================
# EXPLORER ENVIRONMENT
# 
# Grelha 5x5 onde o explorador parte de (0,0) e recolhe
# tesouros evitando armadilhas.
#
# Recompensas:
#   +10  ao recolher um tesouro
#   -5   ao cair numa armadilha (termina episódio)
#   -0.1 por cada passo (incentiva eficiência)
#
# Estado: (posição, tesouros_recolhidos)
#   posição          -> (linha, coluna)
#   tesouros_recolhidos -> frozenset de posições já recolhidas
# =========================================================

@dataclass
class ExplorerEnv:

    n_rows: int = 5
    n_cols: int = 5

    start: Tuple[int, int] = (0, 0)

    treasures: Tuple[Tuple[int, int], ...] = (
        (0, 4),
        (2, 2),
        (4, 0),
    )

    traps: Tuple[Tuple[int, int], ...] = (
        (1, 1),
        (3, 3),
        (0, 3),
    )

    reward_treasure: float = 10.0
    reward_trap: float    = -5.0
    reward_step: float    = -0.1

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):
        """Devolve o estado inicial: posição de partida, sem tesouros recolhidos."""
        return (self.start, frozenset())

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, state, action):
        """
        Aplica a ação ao estado atual.

        Devolve (next_state, reward, done).
        """
        position, collected = state

        dr, dc = ACTION_TO_DELTA[action]
        nr = np.clip(position[0] + dr, 0, self.n_rows - 1)
        nc = np.clip(position[1] + dc, 0, self.n_cols - 1)
        next_pos = (int(nr), int(nc))

        reward = self.reward_step
        done   = False

        # Armadilha
        if next_pos in self.traps:
            reward += self.reward_trap
            done    = True
            return (next_pos, collected), reward, done

        # Tesouro (só recolhe se ainda não tiver sido recolhido)
        new_collected = collected
        if next_pos in self.treasures and next_pos not in collected:
            reward       += self.reward_treasure
            new_collected = collected | {next_pos}

        next_state = (next_pos, new_collected)

        # Termina quando todos os tesouros forem recolhidos
        if len(new_collected) == len(self.treasures):
            done = True

        return next_state, reward, done

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def possible_actions(self, state) -> List[str]:
        """Todas as ações são sempre válidas (paredes bloqueiam via clip)."""
        return list(ACTIONS)

    def is_terminal(self, state) -> bool:
        position, collected = state
        all_treasures = len(collected) == len(self.treasures)
        in_trap = position in self.traps
        return all_treasures or in_trap

    # --------------------------------------------------
    # RENDER
    # --------------------------------------------------

    def render(self, state):
        """Imprime a grelha no terminal."""
        position, collected = state

        symbols = {
            "empty":     ".",
            "agent":     "A",
            "treasure":  "T",
            "collected": "✓",
            "trap":      "X",
            "agent_trap":"@",
        }

        print(f"\n  Tesouros recolhidos: {len(collected)}/{len(self.treasures)}")
        print("  " + " ".join(str(c) for c in range(self.n_cols)))
        print("  " + "-" * (self.n_cols * 2 - 1))

        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                cell = (r, c)
                if cell == position and cell in self.traps:
                    row.append(symbols["agent_trap"])
                elif cell == position:
                    row.append(symbols["agent"])
                elif cell in self.traps:
                    row.append(symbols["trap"])
                elif cell in collected:
                    row.append(symbols["collected"])
                elif cell in self.treasures:
                    row.append(symbols["treasure"])
                else:
                    row.append(symbols["empty"])
            print(f"{r} | " + " ".join(row))
        print()
