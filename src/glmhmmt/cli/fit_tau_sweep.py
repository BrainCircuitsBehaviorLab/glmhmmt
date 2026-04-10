"""Sweep tau (action-trace half-life) from 1 to 100 and save a summary table.

For each tau the model is re-fit from scratch; results are stored under
    RESULTS/fits/tau_sweep/<model>_K<K>/tau_<tau>/

After the full sweep a consolidated metrics parquet is written to
    RESULTS/fits/tau_sweep/<model>_K<K>/tau_sweep_summary.parquet

with columns: subject, K, model_kind, tau, ll_per_trial, bic, acc

Usage
-----
# Full sweep glmhmm K=2, subjects A83 A84, taus 1-100
uv run glmhmmt-fit-tau-sweep --model glmhmm --K 2 --subjects A83 A84

# Full sweep glmhmmt K=2, all subjects, taus 10-60 step 5
uv run glmhmmt-fit-tau-sweep --model glmhmmt --K 2 --tau_min 10 --tau_max 60 --tau_step 5

# Both models
uv run glmhmmt-fit-tau-sweep --model glmhmm glmhmmt --K 2 3
"""
import argparse
import numpy as np
import polars as pl

from glmhmmt.runtime import add_runtime_path_args, configure_paths_from_args, get_data_dir, get_results_dir

def _sweep(
    model: str,
    subjects: list[str] | None,
    K_list: list[int],
    taus: list[int],
    num_iters: int,
    n_restarts: int,
    base_seed: int,
    emission_cols: list[str] | None,
    transition_cols: list[str] | None,
) -> None:
    if model == "glmhmm":
        from glmhmmt.cli.fit_glmhmm import main as fit_main
    elif model == "glmhmmt":
        from glmhmmt.cli.fit_glmhmmt import main as fit_main
    else:
        raise ValueError(f"Unknown model: {model!r}. Choose 'glmhmm' or 'glmhmmt'.")

    if subjects is None:
        df = pl.read_parquet(get_data_dir() / "df_filtered.parquet")
        subjects = df["subject"].unique().sort().to_list()

    for K in K_list:
        sweep_root = get_results_dir() / "fits" / "tau_sweep" / f"{model}_K{K}"
        sweep_root.mkdir(parents=True, exist_ok=True)

        for tau in taus:
            out_dir = sweep_root / f"tau_{tau}"
            print(f"[tau={tau:>3d}] {model} K={K} — subjects: {subjects}")

            kw = dict(
                subjects=subjects,
                K_list=[K],
                num_iters=num_iters,
                n_restarts=n_restarts,
                base_seed=base_seed,
                out_dir=out_dir,
                tau=float(tau),
            )
            if model == "glmhmmt":
                kw["emission_cols"] = emission_cols
                kw["transition_cols"] = transition_cols

            fit_main(**kw)

        # ── collate all per-tau metrics parquets into one summary ─────────────
        records = []
        suffix = f"_{model}_metrics.parquet"
        for tau in taus:
            tau_dir = sweep_root / f"tau_{tau}"
            for p in tau_dir.glob(f"*{suffix}"):
                df_m = pl.read_parquet(p)
                df_m = df_m.with_columns(pl.lit(tau).alias("tau").cast(pl.Int32))
                records.append(df_m)

        if records:
            summary = pl.concat(records).sort(["subject", "K", "tau"])
            out_path = sweep_root / "tau_sweep_summary.parquet"
            summary.write_parquet(out_path)
            print(f"\n✅ Summary written to {out_path}")
            _print_best(summary)


def _print_best(df: pl.DataFrame) -> None:
    """Print the best tau per subject/K according to BIC (lower is better)."""
    print("\nBest τ per subject × K (min BIC):")
    best = (
        df.sort("bic")
        .group_by(["subject", "K"])
        .first()
        .select(["subject", "K", "tau", "bic", "ll_per_trial", "acc"])
        .sort(["subject", "K"])
    )
    print(best)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep tau for GLM-HMM and/or GLM-HMMt models."
    )
    add_runtime_path_args(parser)
    parser.add_argument(
        "--model", nargs="+", choices=["glmhmm", "glmhmmt"], default=["glmhmm"],
        help="Model(s) to sweep.",
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--K", nargs="+", type=int, default=[2])
    parser.add_argument("--tau_min", type=int, default=1)
    parser.add_argument("--tau_max", type=int, default=100)
    parser.add_argument("--tau_step", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--emission_cols", nargs="+", default=None,
        help="glmhmmt only: emission regressors.",
    )
    parser.add_argument(
        "--transition_cols", nargs="+", default=None,
        help="glmhmmt only: transition regressors.",
    )
    args = parser.parse_args()
    configure_paths_from_args(args)

    taus = list(range(args.tau_min, args.tau_max + 1, args.tau_step))
    print(f"Sweeping τ ∈ {taus[:3]}…{taus[-3:]} ({len(taus)} values)")

    for model in args.model:
        _sweep(
            model=model,
            subjects=args.subjects,
            K_list=args.K,
            taus=taus,
            num_iters=args.num_iters,
            n_restarts=args.n_restarts,
            base_seed=args.seed,
            emission_cols=args.emission_cols,
            transition_cols=args.transition_cols,
        )


if __name__ == "__main__":
    main()
