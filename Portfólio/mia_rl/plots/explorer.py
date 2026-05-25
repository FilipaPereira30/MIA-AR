from __future__ import annotations

from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# PLOTS DO EXPLORADOR
# =========================================================


def plot_learning_curves(
    results: Dict[str, List[float]],
    window: int = 50,
    title: str = "Curvas de Aprendizagem — Explorer",
    output_path: str | None = None,
):
    """
    Compara as recompensas médias por episódio para cada agente.
    results = {"SARSA": [...], "Q-Learning": [...], "Monte Carlo": [...]}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"SARSA": "#2196F3", "Q-Learning": "#F44336", "Monte Carlo": "#4CAF50"}

    for name, rewards in results.items():
        # Suavização com média móvel
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(rewards[start : i + 1]))

        ax.plot(smoothed, label=name, color=colors.get(name), linewidth=2)

    ax.set_xlabel("Episódio")
    ax.set_ylabel(f"Recompensa média (janela={window})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"  Guardado: {output_path}")

    plt.show()
    plt.close()


def plot_steps_per_episode(
    results: Dict[str, List[int]],
    window: int = 50,
    title: str = "Passos por Episódio — Explorer",
    output_path: str | None = None,
):
    """
    Mostra a evolução do número de passos por episódio (eficiência).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"SARSA": "#2196F3", "Q-Learning": "#F44336", "Monte Carlo": "#4CAF50"}

    for name, steps in results.items():
        smoothed = []
        for i in range(len(steps)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(steps[start : i + 1]))

        ax.plot(smoothed, label=name, color=colors.get(name), linewidth=2)

    ax.set_xlabel("Episódio")
    ax.set_ylabel(f"Passos médios (janela={window})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"  Guardado: {output_path}")

    plt.show()
    plt.close()


def plot_policy_grid(
    env,
    agent,
    title: str = "Política Greedy",
    output_path: str | None = None,
):
    """
    Visualiza a política greedy aprendida na grelha,
    com o estado sem tesouros recolhidos.
    """
    ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}

    collected = frozenset()  # estado base: sem tesouros

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(range(env.n_cols + 1))
    ax.set_yticks(range(env.n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            pos  = (r, c)
            cell_x = c + 0.5
            cell_y = r + 0.5

            if pos in env.traps:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#FFCDD2"))
                ax.text(cell_x, cell_y, "✕", ha="center", va="center",
                        fontsize=16, color="#C62828")
            elif pos in env.treasures:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#FFF9C4"))
                ax.text(cell_x, cell_y, "★", ha="center", va="center",
                        fontsize=16, color="#F9A825")
            elif pos == env.start:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#E3F2FD"))
                state  = (pos, collected)
                action = agent.greedy_action(state)
                ax.text(cell_x, cell_y, ARROW[action], ha="center", va="center",
                        fontsize=18)
            else:
                state  = (pos, collected)
                action = agent.greedy_action(state)
                ax.text(cell_x, cell_y, ARROW[action], ha="center", va="center",
                        fontsize=18)

    # Legenda manual
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#FFCDD2", label="Armadilha (-5)"),
        Patch(color="#FFF9C4", label="Tesouro (+10)"),
        Patch(color="#E3F2FD", label="Início"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"  Guardado: {output_path}")

    plt.show()
    plt.close()
