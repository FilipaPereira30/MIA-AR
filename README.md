# MIA-AR
# Portfólio Individual - Aprendizagem por Reforço
**Mestrado em Inteligência Artificial 2025/26**

---

## Estrutura do repositório

```
Portefólio individual/
├── Aula_10_fev/        ← P1: K-Armed Bandits (script standalone)
├── Aula_24_fev/        ← P2: MDP Gridworld (script standalone)
├── Aula_3_mar/         ← P3: Dynamic Programming (scripts standalone)
├── Aula_10_mar/        ← P4: Blackjack — Predição (MC, TD, n-step TD)
├── Aula_17_mar/        ← P5: Windy Gridworld — SARSA & n-step SARSA
├── Aula_24_mar/        ← P6: Function Approximation — Linear SARSA & Torch SARSA
├── Aula_7_abr/         ← P7: TicTacToe Environment
├── Aula_14_abr/        ← P8: Policy Gradient — REINFORCE
├── Aula_21_abr/        ← P9: Planning — MCTS
└── Portfolio/          ← Projeto consolidado + Explorer (problema novo)
```

Cada pasta `Aula_XX/` é um snapshot do estado do projeto no final dessa aula. As pastas de P1 a P3 contêm scripts standalone (formato Colab), enquanto P4 a P9 seguem a estrutura modular do pacote `mia_rl`.

---

## Organização do package `mia_rl`

O package de código providenciado inclui os seguintes módulos:

* **core**
  Contém as abstrações base do projeto, incluindo ambientes, políticas, agentes, episódios e transições.

* **envs**
  Contém os ambientes de interação (Blackjack, Windy Gridworld, TicTacToe, Explorer).

* **mdps**
  Implementações de problemas baseados em Processos de Decisão de Markov (MDPs) com modelo conhecido, como os estudados em programação dinâmica.

* **policies**
  Implementações de políticas de seleção de ações (e.g. política de Blackjack, política de TicTacToe).

* **agents**
  Algoritmos de aprendizagem utilizados no projeto, organizados em três subpastas:
  * `prediction/` - algoritmos de predição da função de valor (MC, TD, n-step TD, Linear TD)
  * `control/` - algoritmos de controlo (SARSA, n-step SARSA, MC Control, Linear SARSA, Torch SARSA, REINFORCE, Q-Learning)
  * `planning/` - algoritmos de planeamento (MCTS)

* **features**
  Funções de codificação de estado (tile coding para Windy Gridworld, one-hot para TicTacToe).

* **experiments**
  Funções auxiliares para gerar episódios, executar treino e recolher resultados.

* **plots**
  Funções para visualização dos resultados.

* **scripts**
  Scripts executáveis para correr experiências.

* **notebooks**
  Notebooks Jupyter com demonstrações interativas (TicTacToe Demo, Policy Gradient, MCTS).

* **outputs**
  Pasta destinada ao armazenamento dos resultados e gráficos gerados.

---

## Práticos realizados

### Prático 1 — K-Armed Bandits
`Aula_10_fev/kbandits_incomplete.py`

Implementação de três estratégias de exploração para o problema do bandido multi-braço com k=10 armas:

- **ε-greedy** - com média aritmética simples (alpha=None) e passo constante (alpha fixo para ambientes não estacionários); testado com ε ∈ {0, 0.01, 0.1}
- **UCB** (Upper Confidence Bound) - seleção de ação com bónus de exploração `c·√(ln t / N(a))`
- **Gradient Bandit** - atualização de preferências H com e sem baseline de recompensa média

---

### Prático 2 — MDP e Gridworld
`Aula_24_fev/mdp_gridworld_incomplete.py`

Implementação do framework MDP num Gridworld 4×4 determinístico com estados terminais em (0,0) e (3,3):

- **Policy Evaluation** - Bellman expectation backup iterativo para política uniforme aleatória
- **Value Iteration** - Bellman optimality backup até convergência
- **Q-function** - avaliação iterativa de Q^π para verificação V^π(s) = Σ_a π(a|s)·Q^π(s,a)
- Exercícios: efeito de γ ∈ {0.5, 0.99}, ambiente com trap cell (−10), ambiente estocástico (0.8/0.1/0.1)

---

### Prático 3 — Programação Dinâmica
`Aula_3_mar/practical3_gridworld_incomplete.py` e `practical3_carrental_incomplete.py`

Aplicação de Policy Iteration a dois problemas:

- **Gridworld** — Policy Improvement (greedy w.r.t. V) + Policy Iteration completa; visualização da evolução da política em cada iteração exterior
- **Jack's Car Rental** — MDP estocástico com pedidos e devoluções Poisson; implementação de `q_from_v`, Policy Evaluation, Policy Improvement e Value Iteration; comparação PI vs VI

---

### Prático 4 — Blackjack: Predição
`Aula_10_mar/mia_rl/`

Predição da função de valor V^π para a política fixa de "hit até 20" no Blackjack:

- **Monte Carlo** (every-visit) - `agents/prediction/monte_carlo.py`
- **TD(0)** - `agents/prediction/td.py`
- **n-step TD** - `agents/prediction/nstep_td.py` *(extensão de portfólio)*

Outputs gerados: `blackjack_mc.png`, `blackjack_td0.png`, `blackjack_nstep.png`, `blackjack_nstep_minus_mc.png`

---

### Prático 5 — Windy Gridworld: Controlo
`Aula_17_mar/mia_rl/`

Controlo model-free num ambiente Windy Gridworld com colunas de vento variável:

- **SARSA** - on-policy TD control; `agents/control/sarsa.py`
- **n-step SARSA** - `agents/control/n_step_sarsa.py`
- **Monte Carlo Control** - `agents/control/monte_carlo.py`

Outputs gerados: `windy_gridworld_sarsa/`, `windy_gridworld_n_step_sarsa/`

---

### Prático 6 — Aproximação de Funções
`Aula_24_mar/mia_rl/`

Substituição de tabelas Q por representações paramétricas no Windy Gridworld:

- **Linear TD(0)** com tile coding - `agents/prediction/linear_td.py`
- **Linear SARSA** (semi-gradient) - `agents/control/linear_sarsa.py`
- **SARSA com PyTorch** - `agents/control/torch_sarsa.py`

Outputs gerados: `windy_gridworld_linear_td/`, `windy_gridworld_linear_sarsa/`, `windy_gridworld_torch_sarsa/`, `blackjack_control/`

---

### Prático 7 — TicTacToe Environment
`Aula_7_abr/mia_rl/`

Implementação do ambiente TicTacToe como classe `Environment` (exercício de portfólio):

- `reset()` - reinicia o tabuleiro e define X como primeiro jogador
- `available_actions(state)` - devolve índices das células vazias
- `is_terminal(state)` - deteta vitória ou empate
- `step(action)` - coloca a peça, calcula recompensa, troca o jogador
- `render(state)` - imprime o tabuleiro com índices nas células vazias

Codificação de estado: vetor one-hot de 27 dimensões (perspetiva relativa ao jogador atual) em `features/tictactoe.py`

---

### Prático 8 — Policy Gradient: REINFORCE
`Aula_14_abr/mia_rl/`

Algoritmo REINFORCE com política softmax linear aplicado ao TicTacToe:

- `_probs()` - softmax mascarado sobre ações disponíveis
- `update_episode()` - atualização pelo score function gradient
- Treino por self-play com mistura de adversário aleatório
- Avaliação contra agente aleatório

Notebook: `TicTacToe_PolicyGradient.ipynb`

---

### Prático 9 — Planning: MCTS
`Aula_21_abr/mia_rl/`

Monte Carlo Tree Search aplicado ao TicTacToe:

- **Seleção** - UCB1: `argmax [ Q(s,a) + c·√(ln N(s) / N(s,a)) ]`
- **Expansão** - adiciona nó filho para ação não visitada
- **Rollout** - política aleatória até estado terminal com `_apply`, `_available`, `_is_terminal`
- **Backup** - `backpropagate(value)`: incrementa `visit_count`, acumula `value_sum`, recursão ao pai com `-value`

Notebook: `TicTacToe_MCTS.ipynb`

---

### Problema novo — Explorer (Grelha 5×5)
`Portfolio/mia_rl/`

Ambiente criado de raiz para explorar e comparar algoritmos de controlo model-free. Um agente parte do canto (0,0) e tem de recolher 3 tesouros numa grelha 5×5 evitando 3 armadilhas, no menor número de passos possível.

**Formulação MDP:**
- **Estado** - `(posição, tesouros_recolhidos)`: posição (linha, coluna) + conjunto de tesouros já recolhidos
- **Ações** - U, D, L, R
- **Recompensas** - +10 por tesouro, −5 por armadilha (termina episódio), −0.1 por passo
- **Terminal** - todos os tesouros recolhidos, ou agente numa armadilha
- **Espaço de estados** - 5×5 × 2³ = 200 estados distintos

**Algoritmos comparados:**
- **SARSA** - on-policy TD; mais cauteloso perto das armadilhas porque aprende com a sua própria política exploratória
- **Q-Learning** - off-policy TD; converge mais depressa mas pode sobrestimar regiões perigosas durante a exploração: `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`
- **Monte Carlo Control** - first-visit; sem bias de bootstrap mas maior variância, aprende apenas no fim de cada episódio

Ficheiros criados: `envs/explorer.py`, `agents/control/q_learning.py`, `experiments/explorer.py`, `plots/explorer.py`, `scripts/run_explorer.py`

Outputs gerados: `outputs/explorer/learning_curves.png`, `steps_per_episode.png`, `policy_sarsa.png`, `policy_qlearning.png`

---

## Como executar

```bash
# Criar o ambiente conda (usar o environment.yml da aula mais recente)
conda env create -f Aula_21_abr/mia_rl/environment.yml
conda activate rl

# P4 — Blackjack prediction
cd Aula_10_mar
python -m mia_rl.scripts.run_blackjack_prediction

# P5 — Windy Gridworld
cd Aula_17_mar
python -m mia_rl.scripts.run_windy_gridworld_sarsa
python -m mia_rl.scripts.run_windy_gridworld_n_step_sarsa

# P6 — Function Approximation
cd Aula_24_mar
python -m mia_rl.scripts.run_windy_gridworld_linear_sarsa
python -m mia_rl.scripts.run_windy_gridworld_torch_sarsa
```

# Explorer (problema novo)
cd Portfolio
python -m mia_rl.scripts.run_explorer

Os gráficos gerados são guardados em `outputs/` dentro de cada pasta de aula.
