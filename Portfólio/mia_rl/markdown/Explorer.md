# Explorer — Novo Problema

## Descrição

O **Explorer** é um ambiente de grelha 5×5 onde um agente aprende a recolher tesouros evitando armadilhas, com o mínimo de passos possível.

```
  0 1 2 3 4
  ---------
0 | A . . X T    A = Agente (início)
1 | . X . . .    T = Tesouro (+10)
2 | . . T . .    X = Armadilha (-5, termina episódio)
3 | . . . X .    . = Célula vazia (-0.1 por passo)
4 | T . . . .
```

## Formulação MDP

| Elemento | Descrição |
|---|---|
| **Estado** | `(posição, tesouros_recolhidos)` — posição (linha, coluna) + frozenset das posições já recolhidas |
| **Ações** | U, D, L, R (cima, baixo, esquerda, direita) |
| **Recompensas** | +10 tesouro, −5 armadilha, −0.1 por passo |
| **Terminal** | Todos os tesouros recolhidos, ou agente numa armadilha |

## Ficheiros criados

```
envs/explorer.py                   ← Ambiente ExplorerEnv
agents/control/sarsa.py            ← SarsaAgent
agents/control/q_learning.py       ← QLearningAgent
agents/control/monte_carlo.py      ← MonteCarloControlAgent
experiments/explorer.py            ← run_sarsa, run_qlearning, run_monte_carlo
plots/explorer.py                  ← plot_learning_curves, plot_steps_per_episode, plot_policy_grid
scripts/run_explorer.py            ← Script principal
outputs/explorer/                  ← Resultados gerados
```

## Como executar

```bash
cd Portfolio/mia_rl
python -m scripts.run_explorer
```

## Algoritmos comparados

### SARSA (on-policy TD)
Atualiza Q com a ação *realmente tomada* no próximo estado:
```
Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]
```
Mais conservador — evita armadilhas porque aprende com a sua própria política exploratória.

### Q-Learning (off-policy TD)
Atualiza Q com o *máximo* do próximo estado:
```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
```
Mais agressivo — converge mais depressa mas pode sobrestimar regiões perigosas durante a exploração.

### Monte Carlo Control (first-visit)
Aprende da trajetória *completa* do episódio, calculando o retorno acumulado G:
```
G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
Q(s,a) ← média de todos os G observados após (s,a)
```
Não tem bias de bootstrap, mas tem maior variância e demora mais a aprender.

## Observações

- O espaço de estados é `5×5 × 2^3 = 200` estados distintos (posição × combinação de tesouros recolhidos).
- SARSA tende a ser mais cauteloso perto das armadilhas do que Q-Learning.
- Monte Carlo precisa de mais episódios para convergir porque só aprende no fim de cada episódio.
