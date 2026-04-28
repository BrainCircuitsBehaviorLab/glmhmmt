import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import polars as pl

from glmhmmt.cli.fit_common import (
    apply_valid_trial_mask,
    build_session_kfold_split,
    fit_best_restart,
    normalize_cv_mode,
    score_split,
    stable_model_id,
)
from glmhmmt.model import SoftmaxGLMHMM, normalize_frozen_emissions, serialize_frozen_emissions
from glmhmmt.runtime import (
    add_runtime_path_args,
    configure_paths_from_args,
    get_results_dir,
)
from glmhmmt.tasks import get_adapter

ProgressCallback = Callable[[dict[str, Any]], None]


def generate_model_id(
    task: str,
    K: int,
    tau: float,
    emission_cols: list | None = None,
    frozen_emissions: dict | None = None,
    baseline_class_idx: int = 0,
    cv_mode: str = "none",
    cv_repeats: int = 0,
    condition_filter: str = "all",
) -> str:
    """Stable 8-char hash over the GLMHMM-defining model configuration."""
    base_config = {
        "task": task,
        "K": int(K),
        "tau": float(tau),
        "emission_cols": sorted(emission_cols) if emission_cols else [],
        "frozen_emissions": serialize_frozen_emissions(frozen_emissions),
        "baseline_class_idx": int(baseline_class_idx),
        "cv_mode": normalize_cv_mode(cv_mode),
        "cv_repeats": int(cv_repeats) if normalize_cv_mode(cv_mode) != "none" else 0,
    }
    if get_adapter(task).condition_filter_options():
        base_config["condition_filter"] = _normalize_condition_filter(task, condition_filter)
    return hashlib.md5(json.dumps(base_config, sort_keys=True).encode()).hexdigest()[:8]


def _normalize_condition_filter(task: str, condition_filter: str | None) -> str:
    options = get_adapter(task).condition_filter_options()
    if not options:
        return "all"
    value = str(condition_filter or "all").strip().lower()
    return value if value in options else "all"


def _filter_condition_df(df: pl.DataFrame, task: str, condition_filter: str | None) -> pl.DataFrame:
    if df.is_empty():
        return df
    adapter = get_adapter(task)
    selected = _normalize_condition_filter(task, condition_filter)
    return adapter.filter_condition_df(df, selected)


def _load_subject_feature_df(subject: str, task: str, tau: float, condition_filter: str = "all") -> tuple[Any, pl.DataFrame]:
    adapter = get_adapter(task)
    df = adapter.read_dataset()
    df = adapter.subject_filter(df)
    df = _filter_condition_df(df, task, condition_filter)
    df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)
    feature_df = adapter.build_feature_df(df_sub, tau=tau)
    return adapter, feature_df


def _build_model(
    K: int,
    num_classes: int,
    input_dim: int,
    m_step_num_iters: int,
    stickiness: float,
    frozen: dict | None,
    names: dict[str, Any],
    baseline_class_idx: int = 0,
) -> SoftmaxGLMHMM:
    return SoftmaxGLMHMM(
        num_states=K,
        num_classes=num_classes,
        emission_input_dim=input_dim,
        transition_input_dim=0,
        m_step_num_iters=m_step_num_iters,
        transition_matrix_stickiness=stickiness,
        frozen_emissions=frozen or None,
        emission_feature_names=names.get("X_cols", []),
        baseline_class_idx=baseline_class_idx,
    )


def _prepare_arrays(
    adapter,
    feature_df: pl.DataFrame,
    emission_cols,
    session_col: str,
    *,
    min_session_length: int = 2,
):
    y, X, _, names = adapter.build_design_matrices(feature_df, emission_cols=emission_cols)
    session_ids = feature_df[session_col].to_numpy()
    y, X, session_ids = apply_valid_trial_mask(session_ids, y, X, min_length=min_session_length)
    return jnp.asarray(y), jnp.asarray(X), np.asarray(session_ids), names


def fit_subject(
    subject: str,
    K: int,
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    m_step_num_iters: int = 100,
    emission_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    stickiness: float = 10.0,
    tau: float = 50.0,
    task: str = "MCDR",
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    condition_filter: str = "all",
    baseline_class_idx: int = 0,
) -> dict:
    """Fit a GLMHMM to a single subject's data, returning the best-fitting params and other info."""
    adapter, feature_df = _load_subject_feature_df(subject, task, tau, condition_filter=condition_filter)
    y, X, session_ids, names = _prepare_arrays(adapter, feature_df, emission_cols, adapter.session_col)
    num_classes = adapter.num_classes
    frozen = normalize_frozen_emissions(frozen_emissions)
    model = _build_model(
        K,
        num_classes,
        X.shape[1],
        m_step_num_iters,
        stickiness,
        frozen,
        names,
        baseline_class_idx=baseline_class_idx,
    )
    best_params, best_lps, best_restart = fit_best_restart(
        model,
        n_restarts=n_restarts,
        base_seed=base_seed,
        fit_once=lambda params, props: model.fit_em_multisession(
            params=params,
            props=props,
            emissions=y,
            inputs=X,
            session_ids=session_ids,
            num_iters=num_iters,
            verbose=verbose,
        ),
        failure_message="GLMHMM fitting did not produce valid parameters.",
        progress_callback=progress_callback,
        progress_payload={"subject": subject, "K": K},
    )

    smoothed_probs = model.smoother_multisession(params=best_params, emissions=y, inputs=X, session_ids=session_ids)
    p_pred = model.predict_choice_probs_multisession(best_params, y, X, session_ids=session_ids)
    predictive_state_probs = model.predict_state_probs_multisession(best_params, y, X, session_ids=session_ids)
    split_metrics = score_split(model, best_params, y, X, session_ids)
    T = int(y.shape[0])

    return {
        "subject": subject,
        "K": K,
        "num_classes": num_classes,
        "model": model,
        "fitted_params": best_params,
        "lps": best_lps,
        "best_restart": int(best_restart),
        "smoothed_probs": smoothed_probs,
        "p_pred": p_pred,
        "predictive_state_probs": predictive_state_probs,
        "raw_ll": float(split_metrics["raw_ll"]),
        "objective_lp": float(best_lps[-1]),
        "T": T,
        "names": names,
        "y": np.asarray(y),
        "X": np.asarray(X),
        "frozen_emissions": serialize_frozen_emissions(frozen),
        "baseline_class_idx": int(baseline_class_idx),
        "cv_mode": "none",
        "cv_repeats": 0,
    }


def fit_subject_cv(
    subject: str,
    K: int,
    num_iters: int = 50,
    n_restarts: int = 1,
    cv_repeats: int = 5,
    base_seed: int = 0,
    m_step_num_iters: int = 100,
    emission_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    stickiness: float = 10.0,
    tau: float = 50.0,
    task: str = "2AFC",
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    condition_filter: str = "all",
    baseline_class_idx: int = 0,
) -> dict:
    adapter, feature_df = _load_subject_feature_df(subject, task, tau, condition_filter=condition_filter)
    frozen = normalize_frozen_emissions(frozen_emissions)
    repeats: list[dict[str, Any]] = []
    best_repeat_idx = -1
    best_test_ll = -np.inf
    best_repeat_params = None
    best_repeat_lps = None
    best_repeat_model = None
    best_repeat_names = None
    n_folds = 5
    for repeat_idx in range(n_folds):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "cv_repeat_start",
                    "subject": subject,
                    "K": K,
                    "cv_repeat_index": repeat_idx + 1,
                    "cv_repeat_total": n_folds,
                }
            )
        train_df, test_df, split_meta = build_session_kfold_split(
            feature_df,
            adapter.session_col,
            fold_index=repeat_idx,
            n_splits=n_folds,
            seed=base_seed,
        )
        if verbose:
            print(
                f"[CV fold {repeat_idx + 1}/{n_folds}] "
                f"sessions train/test={split_meta['train_session_count']}/{split_meta['test_session_count']}"
            )

        y_train, X_train, session_train, names = _prepare_arrays(
            adapter,
            train_df,
            emission_cols,
            adapter.session_col,
        )
        y_test, X_test, session_test, _ = _prepare_arrays(
            adapter,
            test_df,
            emission_cols,
            adapter.session_col,
        )

        model = _build_model(
            K,
            adapter.num_classes,
            X_train.shape[1],
            m_step_num_iters,
            stickiness,
            frozen,
            names,
            baseline_class_idx=baseline_class_idx,
        )
        best_params, best_lps, best_restart = fit_best_restart(
            model,
            n_restarts=n_restarts,
            base_seed=base_seed + 1000 * (repeat_idx + 1),
            fit_once=lambda params, props: model.fit_em_multisession(
                params=params,
                props=props,
                emissions=y_train,
                inputs=X_train,
                session_ids=session_train,
                num_iters=num_iters,
                verbose=verbose,
            ),
            failure_message="GLMHMM fitting did not produce valid parameters.",
            progress_callback=progress_callback,
            progress_payload={
                "subject": subject,
                "K": K,
                "cv_repeat_index": repeat_idx + 1,
                "cv_repeat_total": n_folds,
            },
        )
        train_metrics = score_split(model, best_params, y_train, X_train, session_train)
        test_metrics = score_split(model, best_params, y_test, X_test, session_test)
        if float(test_metrics["ll_per_trial"]) > best_test_ll:
            best_test_ll = float(test_metrics["ll_per_trial"])
            best_repeat_idx = repeat_idx + 1
            best_repeat_params = best_params
            best_repeat_lps = np.asarray(best_lps)
            best_repeat_model = model
            best_repeat_names = names
        repeats.append(
            {
                "subject": subject,
                "repeat_index": repeat_idx + 1,
                "best_restart": int(best_restart),
                "train_T": int(train_metrics["T"]),
                "test_T": int(test_metrics["T"]),
                "train_raw_ll": float(train_metrics["raw_ll"]),
                "train_ll_per_trial": float(train_metrics["ll_per_trial"]),
                "train_acc": float(train_metrics["acc"]),
                "test_raw_ll": float(test_metrics["raw_ll"]),
                "test_ll_per_trial": float(test_metrics["ll_per_trial"]),
                "test_acc": float(test_metrics["acc"]),
                "objective_lp": float(best_lps[-1]),
                "train_session_count": int(split_meta["train_session_count"]),
                "test_session_count": int(split_meta["test_session_count"]),
                "train_label_counts_json": json.dumps(split_meta["train_label_counts"], sort_keys=True),
                "test_label_counts_json": json.dumps(split_meta["test_label_counts"], sort_keys=True),
                "balance_score": None,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "cv_repeat_complete",
                    "subject": subject,
                    "K": K,
                    "cv_repeat_index": repeat_idx + 1,
                    "cv_repeat_total": n_folds,
                    "test_ll_per_trial": float(test_metrics["ll_per_trial"]),
                    "test_acc": float(test_metrics["acc"]),
                }
            )

    if best_repeat_model is None or best_repeat_params is None or best_repeat_lps is None or best_repeat_names is None:
        raise ValueError("CV fitting did not produce a valid best repeat.")

    y_full, X_full, session_full, full_names = _prepare_arrays(
        adapter,
        feature_df,
        emission_cols,
        adapter.session_col,
    )
    smoothed_probs = best_repeat_model.smoother_multisession(
        params=best_repeat_params,
        emissions=y_full,
        inputs=X_full,
        session_ids=session_full,
    )
    p_pred = best_repeat_model.predict_choice_probs_multisession(
        best_repeat_params,
        y_full,
        X_full,
        session_ids=session_full,
    )
    predictive_state_probs = best_repeat_model.predict_state_probs_multisession(
        best_repeat_params,
        y_full,
        X_full,
        session_ids=session_full,
    )
    full_metrics = score_split(best_repeat_model, best_repeat_params, y_full, X_full, session_full)

    return {
        "subject": subject,
        "K": K,
        "num_classes": adapter.num_classes,
        "repeats": repeats,
        "best_repeat_index": int(best_repeat_idx),
        "best_repeat_test_ll_per_trial": float(best_test_ll),
        "model": best_repeat_model,
        "fitted_params": best_repeat_params,
        "lps": best_repeat_lps,
        "smoothed_probs": smoothed_probs,
        "p_pred": p_pred,
        "predictive_state_probs": predictive_state_probs,
        "raw_ll": float(full_metrics["raw_ll"]),
        "T": int(y_full.shape[0]),
        "names": full_names,
        "y": np.asarray(y_full),
        "X": np.asarray(X_full),
        "frozen_emissions": serialize_frozen_emissions(frozen),
        "baseline_class_idx": int(baseline_class_idx),
        "cv_mode": "balanced_session_holdout",
        "cv_repeats": n_folds,
    }


def save_results(result: dict, out_dir: Path) -> None:
    subj = result["subject"]
    K = result["K"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{subj}_K{K}_glmhmm"

    T = result["T"]
    p_pred = result["p_pred"]
    acc = float(np.mean(np.argmax(p_pred, axis=1) == result["y"]))
    raw_ll = float(result["raw_ll"])
    ll_per_trial = raw_ll / T
    num_classes = result["num_classes"]
    n_params = result["K"] * (result["K"] - 1) + result["K"] * (num_classes - 1) * result["X"].shape[1]
    bic = -2 * raw_ll + n_params * np.log(T)

    pl.DataFrame(
        {
            "subject": [subj],
            "K": [K],
            "model_kind": ["glmhmm"],
            "ll_per_trial": [ll_per_trial],
            "raw_ll": [raw_ll],
            "objective_lp": [float(result["objective_lp"])],
            "bic": [bic],
            "acc": [acc],
            "best_restart": [int(result.get("best_restart", 1))],
            "cv_mode": [result.get("cv_mode", "none")],
            "cv_repeats": [int(result.get("cv_repeats", 0))],
        }
    ).write_parquet(str(prefix) + "_metrics.parquet")

    _tp = result["fitted_params"].transitions
    if hasattr(_tp, "transition_matrix"):
        _A = np.asarray(_tp.transition_matrix)
    else:
        _A = np.asarray(jnn.softmax(_tp.bias, axis=-1))

    np.savez(
        str(prefix) + "_arrays.npz",
        lps=result["lps"],
        p_pred=p_pred,
        smoothed_probs=result["smoothed_probs"],
        predictive_state_probs=result["predictive_state_probs"],
        initial_probs=np.asarray(result["fitted_params"].initial.probs),
        emission_weights=np.asarray(result["fitted_params"].emissions.weights),
        transition_matrix=_A,
        y=result["y"],
        X=result["X"],
        X_cols=np.array(result["names"].get("X_cols", []), dtype=object),
        frozen_emissions_json=np.array(json.dumps(result["frozen_emissions"], sort_keys=True)),
        baseline_class_idx=np.array(int(result.get("baseline_class_idx", 0))),
    )


def save_cv_results(result: dict, out_dir: Path) -> None:
    subj = result["subject"]
    K = result["K"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{subj}_K{K}_glmhmm"
    repeats_df = pl.DataFrame(result["repeats"])
    repeats_df.write_parquet(str(prefix) + "_cv_repeats.parquet")

    test_ll = repeats_df["test_ll_per_trial"].to_numpy()
    test_acc = repeats_df["test_acc"].to_numpy()
    train_ll = repeats_df["train_ll_per_trial"].to_numpy()
    train_acc = repeats_df["train_acc"].to_numpy()
    T = int(result["T"])
    p_pred = result["p_pred"]
    raw_ll = float(result["raw_ll"])
    ll_per_trial = raw_ll / T
    acc = float(np.mean(np.argmax(p_pred, axis=1) == result["y"]))
    num_classes = result["num_classes"]
    n_params = result["K"] * (result["K"] - 1) + result["K"] * (num_classes - 1) * result["X"].shape[1]
    bic = -2 * raw_ll + n_params * np.log(T)
    pl.DataFrame(
        {
            "subject": [subj],
            "K": [K],
            "model_kind": ["glmhmm"],
            "ll_per_trial": [ll_per_trial],
            "raw_ll": [raw_ll],
            "objective_lp": [float(np.mean(repeats_df["objective_lp"].to_numpy()))],
            "bic": [bic],
            "acc": [acc],
            "best_repeat_index": [int(result["best_repeat_index"])],
            "best_repeat_test_ll_per_trial": [float(result["best_repeat_test_ll_per_trial"])],
            "cv_mode": [result["cv_mode"]],
            "cv_repeats": [int(result["cv_repeats"])],
            "test_ll_per_trial_mean": [float(np.mean(test_ll))],
            "test_ll_per_trial_std": [float(np.std(test_ll, ddof=0))],
            "test_acc_mean": [float(np.mean(test_acc))],
            "test_acc_std": [float(np.std(test_acc, ddof=0))],
            "train_ll_per_trial_mean": [float(np.mean(train_ll))],
            "train_ll_per_trial_std": [float(np.std(train_ll, ddof=0))],
            "train_acc_mean": [float(np.mean(train_acc))],
            "train_acc_std": [float(np.std(train_acc, ddof=0))],
        }
    ).write_parquet(str(prefix) + "_metrics.parquet")

    _tp = result["fitted_params"].transitions
    if hasattr(_tp, "transition_matrix"):
        _A = np.asarray(_tp.transition_matrix)
    else:
        _A = np.asarray(jnn.softmax(_tp.bias, axis=-1))

    np.savez(
        str(prefix) + "_arrays.npz",
        lps=result["lps"],
        p_pred=result["p_pred"],
        smoothed_probs=result["smoothed_probs"],
        predictive_state_probs=result["predictive_state_probs"],
        initial_probs=np.asarray(result["fitted_params"].initial.probs),
        emission_weights=np.asarray(result["fitted_params"].emissions.weights),
        transition_matrix=_A,
        y=result["y"],
        X=result["X"],
        X_cols=np.array(result["names"].get("X_cols", []), dtype=object),
        frozen_emissions_json=np.array(json.dumps(result["frozen_emissions"], sort_keys=True)),
        baseline_class_idx=np.array(int(result.get("baseline_class_idx", 0))),
        cv_selected_repeat=np.array(int(result["best_repeat_index"])),
        cv_selected_metric=np.array(float(result["best_repeat_test_ll_per_trial"])),
    )


def main(
    subjects: list[str] | None = None,
    K_list: list[int] = [2, 3],
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    out_dir: Path | None = None,
    emission_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    tau: float = 50.0,
    task: str = "MCDR",
    cv_mode: str = "none",
    cv_repeats: int = 0,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    condition_filter: str = "all",
    baseline_class_idx: int = 0,
):
    adapter = get_adapter(task)
    cv_mode = normalize_cv_mode(cv_mode)
    cv_repeats = int(cv_repeats) if cv_mode != "none" else 0
    frozen_spec = serialize_frozen_emissions(frozen_emissions)
    condition_filter = _normalize_condition_filter(task, condition_filter)
    df = None
    if emission_cols is None or subjects is None:
        df = adapter.read_dataset()
        df = adapter.subject_filter(df)
        df = _filter_condition_df(df, task, condition_filter)
    resolved_emission_cols = emission_cols or adapter.default_emission_cols(df)
    if out_dir is None:
        model_id = generate_model_id(
            task=task,
            K=K_list[0],
            tau=tau,
            emission_cols=resolved_emission_cols,
            frozen_emissions=frozen_spec,
            baseline_class_idx=baseline_class_idx,
            cv_mode=cv_mode,
            cv_repeats=cv_repeats,
            condition_filter=condition_filter,
        )
        out_dir = get_results_dir() / "fits" / task / "glmhmm" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as _f:
        json.dump(
            {
                "task": task,
                "tau": tau,
                "subjects": subjects,
                "emission_cols": resolved_emission_cols,
                "frozen_emissions": frozen_spec,
                "baseline_class_idx": int(baseline_class_idx),
                "K_list": K_list,
                "model_id": out_dir.name,
                "cv_mode": cv_mode,
                "cv_repeats": int(cv_repeats),
                "condition_filter": condition_filter,
            },
            _f,
            indent=4,
        )
    if subjects is None:
        if df is None:
            df = adapter.read_dataset()
            df = adapter.subject_filter(df)
            df = _filter_condition_df(df, task, condition_filter)
        subjects = df["subject"].unique().sort().to_list()

    for subj_idx, subj in enumerate(subjects, start=1):
        for k_idx, K in enumerate(K_list, start=1):
            if verbose:
                print(f"Fitting glmhmm | subject={subj} K={K} tau={tau} task={task} cv_mode={cv_mode} ...")

            def _progress(info: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    {
                        **info,
                        "subject_index": subj_idx,
                        "subject_total": len(subjects),
                        "k_index": k_idx,
                        "k_total": len(K_list),
                    }
                )

            if cv_mode == "balanced_session_holdout":
                result = fit_subject_cv(
                    subj,
                    K,
                    num_iters=num_iters,
                    n_restarts=n_restarts,
                    cv_repeats=cv_repeats,
                    base_seed=base_seed,
                    emission_cols=emission_cols,
                    frozen_emissions=frozen_spec,
                    baseline_class_idx=baseline_class_idx,
                    tau=tau,
                    task=task,
                    verbose=verbose,
                    progress_callback=_progress if progress_callback is not None else None,
                    condition_filter=condition_filter,
                )
                save_cv_results(result, out_dir)
            else:
                result = fit_subject(
                    subj,
                    K,
                    num_iters=num_iters,
                    n_restarts=n_restarts,
                    base_seed=base_seed,
                    tau=tau,
                    emission_cols=emission_cols,
                    frozen_emissions=frozen_spec,
                    baseline_class_idx=baseline_class_idx,
                    task=task,
                    verbose=verbose,
                    progress_callback=_progress if progress_callback is not None else None,
                    condition_filter=condition_filter,
                )
                save_results(result, out_dir)
            if verbose:
                print(f"  ✓ saved to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_runtime_path_args(parser)
    parser.add_argument("--subjects", nargs="+", default=None, help="List of subject IDs. If None, fits all subjects.")
    parser.add_argument("--K", nargs="+", type=int, default=[2, 3], help="List of number of states to fit.")
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--tau", type=float, default=50.0, help="Half-life for exponential action traces.")
    parser.add_argument("--task", type=str, default="MCDR", help="Task to fit: 'MCDR', '2AFC', or 'nuo_auditory'.")
    parser.add_argument(
        "--emission_cols",
        nargs="+",
        default=None,
        help="Emission regressors to use. If omitted, uses the task defaults.",
    )
    parser.add_argument(
        "--cv_mode",
        type=str,
        default="none",
        choices=["none", "balanced_session_holdout", "balanced_holdout"],
    )
    parser.add_argument("--cv_repeats", type=int, default=0)
    parser.add_argument(
        "--frozen_emissions",
        type=str,
        default=None,
        help='JSON object mapping state indices to {feature: fixed_value}, e.g. \'{"0":{"SL":0.0}}\'.',
    )
    parser.add_argument(
        "--condition_filter",
        type=str,
        default="all",
        choices=["all", "saline", "drug"],
        help="Condition subset for 2AFC_DRUG. Ignored for other tasks.",
    )
    parser.add_argument(
        "--baseline_class_idx",
        type=int,
        default=0,
        help="Choice class used as the implicit softmax reference.",
    )
    args = parser.parse_args()
    configure_paths_from_args(args)
    frozen_emissions = json.loads(args.frozen_emissions) if args.frozen_emissions else None
    main(
        subjects=args.subjects,
        K_list=args.K,
        num_iters=args.num_iters,
        n_restarts=args.n_restarts,
        base_seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        tau=args.tau,
        emission_cols=args.emission_cols,
        task=args.task,
        cv_mode=args.cv_mode,
        cv_repeats=args.cv_repeats,
        frozen_emissions=frozen_emissions,
        condition_filter=args.condition_filter,
        baseline_class_idx=args.baseline_class_idx,
    )
