from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Blackjack model-free prediction experiments.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes for each algorithm.")
    parser.add_argument("--td-alpha", type=float, default=0.05, help="Step-size for TD(0).")
    parser.add_argument("--n-step", type=int, default=4, help="N value for n-step TD.")
    parser.add_argument("--threshold", type=int, default=20, help="Policy threshold: hit below this sum.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/blackjack_prediction", help="Directory where plots will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mia_rl.agents.prediction import FirstVisitMonteCarloPrediction, TD0Prediction, NStepTDPrediction
    from mia_rl.envs.blackjack import BlackjackEnv
    from mia_rl.experiments.training import train_prediction_agent
    from mia_rl.plots.blackjack import plot_value_difference, plot_value_function
    from mia_rl.policies.blackjack import ThresholdPolicy

    policy = ThresholdPolicy(threshold=args.threshold)
    checkpoints = sorted({cp for cp in (1000, 5000, args.episodes) if cp <= args.episodes})

    try:
        # Configurar ambientes e agentes
        mc_env, td_env, nstep_env = BlackjackEnv(seed=args.seed), BlackjackEnv(seed=args.seed), BlackjackEnv(seed=args.seed)
        
        mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
        td_agent = TD0Prediction(alpha=args.td_alpha, gamma=1.0)
        nstep_agent = NStepTDPrediction(n=args.n_step, alpha=args.td_alpha, gamma=1.0)

        # 1. Treinar Monte Carlo
        print(f"Training First-Visit Monte Carlo...")
        mc_history = train_prediction_agent(mc_env, policy, mc_agent, args.episodes, checkpoints=checkpoints)

        # 2. Treinar TD(0)
        print(f"Training TD(0)...")
        td_history = train_prediction_agent(td_env, policy, td_agent, args.episodes, checkpoints=checkpoints)

        # 3. Treinar n-step TD (ESTA LINHA FALTAVA)
        print(f"Training {args.n_step}-step TD...")
        nstep_history = train_prediction_agent(nstep_env, policy, nstep_agent, args.episodes, checkpoints=checkpoints)

        # Extrair resultados finais
        final_mc = mc_history[args.episodes]
        final_td = td_history[args.episodes]
        final_nstep = nstep_history[args.episodes]

        # Gerar os gráficos (Faltava criar as variáveis fig_mc, fig_td, fig_nstep)
        print("Generating plots...")
        fig_mc, _ = plot_value_function(final_mc, title="Monte Carlo", vmin=-1.0, vmax=1.0)
        fig_td, _ = plot_value_function(final_td, title="TD(0)", vmin=-1.0, vmax=1.0)
        fig_nstep, _ = plot_value_function(final_nstep, title=f"{args.n_step}-step TD", vmin=-1.0, vmax=1.0)
        fig_diff, _ = plot_value_difference(final_nstep, final_mc, title="n-step TD - MC", vmin=-1.0, vmax=1.0)

        # Guardar resultados
        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_mc.savefig(output_dir / "blackjack_mc.png")
        fig_td.savefig(output_dir / "blackjack_td0.png")
        fig_nstep.savefig(output_dir / "blackjack_nstep.png")
        fig_diff.savefig(output_dir / "blackjack_nstep_minus_mc.png")
        
        print(f"Success! Saved plots to {output_dir}")

        if not args.no_show:
            plt.show()

    except Exception as exc:
        print(f"\nAn error occurred: {exc}")
        return

if __name__ == "__main__":
    main()