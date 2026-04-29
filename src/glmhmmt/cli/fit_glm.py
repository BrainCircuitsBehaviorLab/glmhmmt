import numpy as np
import polars as pl
import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Callable

from glmhmmt.glm import fit_glm
from glmhmmt.runtime import (
    add_runtime_path_args,
    configure_paths_from_args,
    get_results_dir,
)
from glmhmmt.tasks import get_adapter

ProgressCallback = Callable[[dict[str, Any]], None]


def _normalize_condition_filter(task: str, condition_filter: str | None) -> str:
    options = get_adapter(task).condition_filter_options()
    if not options:
        return "all"
    value = str(condition_filter or "all").strip().lower()
    if value in options:
        return value
    if value == "saline" and "rest" in options:
        return "rest"
    return "all"


def _filter_condition_df(df: pl.DataFrame, task: str, condition_filter: str | None) -> pl.DataFrame:
    if df.is_empty():
        return df
    adapter = get_adapter(task)
    selected = _normalize_condition_filter(task, condition_filter)
    return adapter.filter_condition_df(df, selected)


def fit_subject(
    subject: str,
    emission_cols: list[str] | None = None,
    tau: float = 5.0,
    num_classes: int = 3,
    task: str = "MCDR",
    lapse_mode: str = "none",
    lapse_max: float = 1.0,
    n_restarts: int = 5,
    restart_noise_scale: float = 0.05,
    seed: int = 0,
    progress_callback: ProgressCallback | None = None,
    baseline_class_idx: int = 0,
    condition_filter: str = "all",
) -> dict:
    """Fit a GLM (K=1) to a single subject."""

    # Force binary for 2AFC
    adapter = get_adapter(task)
    num_classes = adapter.num_classes

    # 1. Load Data
    df = adapter.read_dataset()
    df = adapter.subject_filter(df)
    df = _filter_condition_df(df, task, condition_filter)
    df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)
    if len(df_sub) == 0:
        return None
    y, X, _, names = adapter.load_subject(df_sub, tau=tau, emission_cols=emission_cols)
    fit = fit_glm(
        X,
        y,
        num_classes=num_classes,
        baseline_class_idx=baseline_class_idx,
        lapse_mode=lapse_mode,
        lapse_max=lapse_max,
        n_restarts=n_restarts,
        restart_noise_scale=restart_noise_scale,
        seed=seed,
        progress_callback=progress_callback,
    )

    return {
        "subject": subject,
        "W": fit.weights,              # (C, M)
        "p_pred": fit.predictive_probs,         # (T, C)
        "lapse_rates": fit.lapse_rates,
        "lapse_mode": fit.lapse_mode,
        "lapse_labels": fit.lapse_labels,
        "nll": fit.negative_log_likelihood,
        "success": fit.success,
        "y": fit.y,
        "X": fit.X,
        "names": names,
        "T": fit.num_trials,
        "baseline_class_idx": int(baseline_class_idx),
    }

def save_results(result: dict, out_dir: Path, tau: float):
    if result is None: return
    
    subj = result["subject"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as {subj}_glm_arrays.npz
    prefix = out_dir / f"{subj}_glm"
    
    W_full = result["W"]
    C, M = W_full.shape
    baseline_class_idx = int(result.get("baseline_class_idx", 0))
    if not 0 <= baseline_class_idx < C:
        raise ValueError(f"baseline_class_idx={baseline_class_idx} is invalid for num_classes={C}.")
    W_save = np.delete(W_full, baseline_class_idx, axis=0)
    W_save = W_save[None, ...] # (1, C-1, M)

    np.savez(
        str(prefix) + "_arrays.npz",
        emission_weights=W_save,
        p_pred=result["p_pred"],
        y=result["y"],
        X=result["X"],
        smoothed_probs=np.ones((result["T"], 1)),  # K=1, prob=1 everywhere
        predictive_state_probs=np.ones((result["T"], 1)),  # K=1 predictive state mass is also 1 everywhere
        initial_probs=np.ones(1),
        transition_matrix=np.ones((1, 1)),
        X_cols=result["names"]["X_cols"] if "X_cols" in result["names"] else [],
        lapse_rates=result.get("lapse_rates", np.zeros(C)),
        lapse_mode=result.get("lapse_mode", "none"),
        lapse_labels=np.asarray(result.get("lapse_labels", []), dtype=str),
        baseline_class_idx=np.array(int(baseline_class_idx)),
        success=result["success"],
    )

    _lapse = np.asarray(result.get("lapse_rates", np.zeros(C)), dtype=float)
    _n_lapse_params = int(_lapse.size)
    acc = float(np.mean(np.argmax(result["p_pred"], axis=1) == result["y"])) if result["T"] > 0 else 0.0
    raw_ll = -float(result["nll"]) if result["T"] > 0 else np.nan
    ll_per_trial = raw_ll / result["T"] if result["T"] > 0 else np.nan
    k = (result["W"].shape[0] - 1) * result["W"].shape[1] + _n_lapse_params
    bic = k * np.log(result["T"]) + 2 * result["nll"] if result["T"] > 0 else np.nan
    
    pl.DataFrame({
        "subject": [subj],
        "model_kind": ["glm"],
        "tau": [tau],
        "nll": [result["nll"]],
        "raw_ll": [raw_ll],
        "ll_per_trial": [ll_per_trial],
        "bic": [bic],
        "acc": [acc],
        "k": [k],
        "n_trials": [result["T"]],
        "baseline_class_idx": [int(baseline_class_idx)],
    }).write_parquet(str(prefix) + "_metrics.parquet")


def generate_model_id(
    task,
    tau,
    emission_cols,
    lapse_mode: str = "none",
    lapse_max: float = 0.2,
    n_restarts: int = 5,
    restart_noise_scale: float = 0.05,
    seed: int | None = 0,
    baseline_class_idx: int = 0,
    condition_filter: str = "all",
):
    cols = sorted(emission_cols) if emission_cols else []
    config = {
        "task": task,
        "tau": float(tau),
        "emission_cols": cols,
        "lapse_mode": str(lapse_mode),
        "baseline_class_idx": int(baseline_class_idx),
    }
    if lapse_mode != "none":
        config["lapse_max"] = float(lapse_max)
        config["n_restarts"] = int(n_restarts)
        config["restart_noise_scale"] = float(restart_noise_scale)
        config["seed"] = None if seed is None else int(seed)
    if get_adapter(task).condition_filter_options():
        config["condition_filter"] = _normalize_condition_filter(task, condition_filter)
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def main(
    subjects: list[str] | None = None,
    out_dir: Path | None = None,
    tau: float = 5.0,
    emission_cols: list[str] | None = None,
    num_classes: int = 3,
    task: str = "MCDR",
    model_alias: str | None = None,
    lapse_mode: str = "none",
    lapse_max: float = 0.2,
    n_restarts: int = 5,
    restart_noise_scale: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    baseline_class_idx: int = 0,
    condition_filter: str = "all",
):
    # Compute base output directory
    base_out_dir = get_results_dir() / "fits" / task / "glm"
    requested_emission_cols = list(emission_cols) if emission_cols is not None else None

    # Generate Hash
    model_hash = generate_model_id(
        task,
        tau,
        emission_cols,
        lapse_mode=lapse_mode,
        lapse_max=lapse_max,
        n_restarts=n_restarts,
        restart_noise_scale=restart_noise_scale,
        seed=seed,
        baseline_class_idx=baseline_class_idx,
        condition_filter=condition_filter,
    )
    out_dirs = [base_out_dir / model_hash]

    if model_alias:
        out_dirs.append(base_out_dir / model_alias)
        # If out_dir provided as argument, it overrides only if model_alias is not set?
        # But wait, out_dir was previously just `paths.RESULTS / "fits" / task / "glm_baseline"`.
        # So I will overwrite out_dir based on logic now.

    if out_dir is not None:
         # If user explicitly passed out_dir, use it instead (legacy support or rigorous override)
         out_dirs = [out_dir]
         if model_alias:
             # If alias is also provided, perhaps save to both?
             out_dirs.append(base_out_dir / model_alias)
    
    # Ensure directories exist
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    if verbose:
        print(
            f"Fitting GLM | Task={task} Tau={tau} Hash={model_hash} "
            f"Alias={model_alias} N={len(subjects) if subjects else 'All'}"
        )

    adapter = get_adapter(task)
    num_classes = adapter.num_classes
    condition_filter = _normalize_condition_filter(task, condition_filter)

    df = adapter.read_dataset()
    df = adapter.subject_filter(df)
    df = _filter_condition_df(df, task, condition_filter)

    if subjects is None:
        subjects = df["subject"].unique().sort().to_list()

    df_for_names = df.filter(pl.col("subject").is_in(subjects)) if subjects else df.head(0)
    try:
        resolved_design_names = adapter.resolve_design_names(
            emission_cols=requested_emission_cols,
            df=df_for_names,
        )
        resolved_emission_cols = list(resolved_design_names.get("X_cols", []))
    except Exception:
        resolved_emission_cols = (
            list(requested_emission_cols)
            if requested_emission_cols is not None
            else None
        )

    for d in out_dirs:
        with open(d / "config.json", "w") as f:
            json.dump(
                {
                    "task": task,
                    "tau": tau,
                    "emission_cols": resolved_emission_cols,
                    "requested_emission_cols": requested_emission_cols,
                    "resolved_emission_cols": resolved_emission_cols,
                    "num_classes": num_classes,
                    "lapse_mode": lapse_mode,
                    "lapse_max": lapse_max,
                    "n_restarts": n_restarts,
                    "restart_noise_scale": restart_noise_scale,
                    "seed": seed,
                    "baseline_class_idx": int(baseline_class_idx),
                    "model_id": d.name,
                    **(
                        {"condition_filter": condition_filter}
                        if adapter.condition_filter_options()
                        else {}
                    ),
                },
                f,
                indent=4,
            )

    if verbose:
        print(f"Fitting GLM | Task={task} Tau={tau} N={len(subjects)}")
    
    for subj_idx, subj in enumerate(subjects, start=1):
        if verbose:
            print(f"  Fitting {subj}...")

        def _progress(info: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    **info,
                    "subject": subj,
                    "subject_index": subj_idx,
                    "subject_total": len(subjects),
                }
            )

        try:
            res = fit_subject(
                subj,
                tau=tau,
                emission_cols=requested_emission_cols,
                num_classes=num_classes,
                task=task,
                lapse_mode=lapse_mode,
                lapse_max=lapse_max,
                n_restarts=n_restarts,
                restart_noise_scale=restart_noise_scale,
                seed=seed,
                progress_callback=_progress if progress_callback is not None else None,
                baseline_class_idx=baseline_class_idx,
                condition_filter=condition_filter,
            )
            for d in out_dirs:
                save_results(res, d, tau)
        except Exception as e:
            if verbose:
                print(f"  Failed {subj}: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
    
    if verbose:
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_runtime_path_args(parser)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--tau", type=float, default=50.0) 
    parser.add_argument("--task", type=str, default="MCDR", choices=["MCDR", "2AFC", "2AFC_DRUG", "2AFC_delay", "nuo_auditory"])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--model_alias", type=str, default=None)
    parser.add_argument(
        "--lapse_mode",
        type=str,
        default="none",
        choices=["none", "class", "history", "history_conditioned"],
        help="Lapse model to fit: none, class, history, or history_conditioned.",
    )
    parser.add_argument("--lapse_max", type=float, default=0.2,
                        help="Upper bound for each lapse rate (default 0.20)")
    parser.add_argument("--n_restarts", type=int, default=5,
                        help="Number of noisy restarts for lapse fits (default 5)")
    parser.add_argument("--restart_noise_scale", type=float, default=0.05,
                        help="Stddev of Gaussian noise added to each parameter at each lapse-fit restart")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for lapse-fit restart initialization noise")
    parser.add_argument(
        "--baseline_class_idx",
        type=int,
        default=0,
        help="Choice class used as the implicit softmax reference.",
    )
    parser.add_argument(
        "--condition_filter",
        type=str,
        default="all",
        help="Task-specific condition subset, for example rest or drug for 2AFC_DRUG.",
    )

    args = parser.parse_args()
    configure_paths_from_args(args)

    main(
        subjects=args.subjects,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        tau=args.tau,
        task=args.task,
        num_classes=args.num_classes,
        model_alias=args.model_alias,
        lapse_mode=args.lapse_mode,
        lapse_max=args.lapse_max,
        n_restarts=args.n_restarts,
        restart_noise_scale=args.restart_noise_scale,
        seed=args.seed,
        baseline_class_idx=args.baseline_class_idx,
        condition_filter=args.condition_filter,
    )
