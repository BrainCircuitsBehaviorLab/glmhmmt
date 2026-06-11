import numpy as np
import polars as pl
import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Callable

from glmhmmt.cli.fit_common import (
    align_design_matrix_to_columns,
    build_session_kfold_split,
    normalize_cv_mode,
)
from glmhmmt.glm import (
    fit_glm,
    fit_private_alternative_glm,
    predict_glm_probs,
    private_alternative_probs,
)
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


def _load_subject_feature_df(
    subject: str,
    task: str,
    tau: float,
    condition_filter: str = "all",
) -> tuple[Any, pl.DataFrame]:
    adapter = get_adapter(task)
    df = adapter.read_dataset()
    df = adapter.subject_filter(df)
    df = _filter_condition_df(df, task, condition_filter)
    df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)
    return adapter, adapter.build_feature_df(df_sub, tau=tau)


def _prepare_standard_arrays(
    adapter,
    feature_df: pl.DataFrame,
    emission_cols: list[str] | None,
    *,
    expected_x_cols: list[str] | None = None,
):
    y, X, _, names = adapter.build_design_matrices(feature_df, emission_cols=emission_cols)
    if expected_x_cols is not None:
        X, x_cols = align_design_matrix_to_columns(X, list(names.get("X_cols", [])), expected_x_cols)
        names = {**names, "X_cols": x_cols}
    return np.asarray(y), np.asarray(X), names


def _align_private_design_tensor_to_columns(
    values,
    value_cols: list[str],
    reference_cols: list[str],
):
    value_cols = [str(col) for col in value_cols]
    reference_cols = [str(col) for col in reference_cols]
    if value_cols == reference_cols:
        return values, reference_cols
    arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"Private design tensor must be 3D; got shape {arr.shape}.")
    dtype = arr.dtype if arr.size else np.float32
    aligned = np.zeros((arr.shape[0], arr.shape[1], len(reference_cols)), dtype=dtype)
    col_to_idx = {col: idx for idx, col in enumerate(value_cols)}
    for dst_idx, col in enumerate(reference_cols):
        src_idx = col_to_idx.get(col)
        if src_idx is not None:
            aligned[:, :, dst_idx] = arr[:, :, src_idx]
    return aligned, reference_cols


def _prepare_private_arrays(
    adapter,
    feature_df: pl.DataFrame,
    *,
    expected_x_cols: list[str] | None = None,
):
    y, X_private, _, names = adapter.build_private_design_matrices(feature_df)
    if expected_x_cols is not None:
        X_private, x_cols = _align_private_design_tensor_to_columns(
            X_private,
            list(names.get("X_private_cols", [])),
            expected_x_cols,
        )
        names = {**names, "X_private_cols": x_cols}
    return np.asarray(y), np.asarray(X_private), names


def _score_probabilities(y, p_pred) -> dict[str, Any]:
    y_np = np.asarray(y, dtype=int)
    probs = np.asarray(p_pred, dtype=float)
    T = int(y_np.shape[0])
    if T == 0:
        return {
            "raw_ll": 0.0,
            "ll_per_trial": np.nan,
            "acc": np.nan,
            "T": 0,
            "p_pred": probs,
        }
    clipped = np.clip(probs, 1e-10, 1.0)
    raw_ll = float(np.sum(np.log(clipped[np.arange(T), y_np])))
    acc = float(np.mean(np.argmax(probs, axis=1) == y_np))
    return {
        "raw_ll": raw_ll,
        "ll_per_trial": raw_ll / T,
        "acc": acc,
        "T": T,
        "p_pred": probs,
    }


def _score_standard_split(fit, y, X, baseline_class_idx: int) -> dict[str, Any]:
    p_pred = predict_glm_probs(
        X,
        fit.weights,
        y=y,
        baseline_class_idx=baseline_class_idx,
        num_classes=fit.num_classes,
        lapse_mode=fit.lapse_mode,
        lapse_rates=fit.lapse_rates,
    )
    return _score_probabilities(y, p_pred)


def _score_private_split(fit, y, X_private) -> dict[str, Any]:
    p_pred = private_alternative_probs(X_private, fit.private_weights, fit.private_bias)
    return _score_probabilities(y, p_pred)


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
    emission_model: str = "standard",
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
    emission_model = str(emission_model or "standard").strip().lower()
    if emission_model == "private_alternative":
        if not hasattr(adapter, "build_private_design_matrices"):
            raise ValueError(f"Task {task!r} does not support emission_model='private_alternative'.")
        feature_df = adapter.build_feature_df(df_sub, tau=tau)
        y, X_private, _, names = adapter.build_private_design_matrices(feature_df)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_start",
                    "restart_index": 1,
                    "restart_total": 1,
                }
            )
        fit = fit_private_alternative_glm(
            X_private,
            y,
            num_classes=num_classes,
            include_bias=True,
            bias_reference_class_idx=getattr(adapter, "baseline_class_idx", baseline_class_idx),
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_complete",
                    "restart_index": 1,
                    "restart_total": 1,
                    "negative_log_likelihood": fit.negative_log_likelihood,
                    "success": fit.success,
                }
            )
        return {
            "subject": subject,
            "emission_model": "private_alternative",
            "private_weights": fit.private_weights,
            "private_bias": fit.private_bias,
            "p_pred": fit.predictive_probs,
            "nll": fit.negative_log_likelihood,
            "success": fit.success,
            "y": fit.y,
            "X_private": fit.X_private,
            "names": names,
            "T": fit.num_trials,
            "baseline_class_idx": int(getattr(adapter, "baseline_class_idx", baseline_class_idx)),
            "n_iter": fit.n_iter,
            "nll_history": np.asarray(fit.nll_history, dtype=float),
            "cv_mode": "none",
            "cv_repeats": 0,
        }

    if emission_model != "standard":
        raise ValueError("emission_model must be one of: standard, private_alternative.")

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
        "emission_model": "standard",
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
        "cv_mode": "none",
        "cv_repeats": 0,
    }


def fit_subject_cv(
    subject: str,
    emission_cols: list[str] | None = None,
    tau: float = 5.0,
    num_classes: int = 3,
    task: str = "MCDR",
    lapse_mode: str = "none",
    lapse_max: float = 1.0,
    n_restarts: int = 5,
    restart_noise_scale: float = 0.05,
    cv_repeats: int = 5,
    seed: int = 0,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    baseline_class_idx: int = 0,
    condition_filter: str = "all",
    emission_model: str = "standard",
) -> dict:
    """Fit GLM session-fold CV for a single subject."""
    del cv_repeats
    adapter, feature_df = _load_subject_feature_df(subject, task, tau, condition_filter=condition_filter)
    if feature_df.height == 0:
        return None

    num_classes = adapter.num_classes
    emission_model = str(emission_model or "standard").strip().lower()
    if emission_model == "private_alternative":
        if not hasattr(adapter, "build_private_design_matrices"):
            raise ValueError(f"Task {task!r} does not support emission_model='private_alternative'.")
        baseline_class_idx = int(getattr(adapter, "baseline_class_idx", baseline_class_idx))
    elif emission_model != "standard":
        raise ValueError("emission_model must be one of: standard, private_alternative.")

    repeats: list[dict[str, Any]] = []
    best_repeat_idx = -1
    best_test_ll = -np.inf
    best_fit = None
    best_names = None
    n_folds = 5

    for repeat_idx in range(n_folds):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "cv_repeat_start",
                    "subject": subject,
                    "K": 1,
                    "cv_repeat_index": repeat_idx + 1,
                    "cv_repeat_total": n_folds,
                }
            )

        train_df, test_df, split_meta = build_session_kfold_split(
            feature_df,
            adapter.session_col,
            fold_index=repeat_idx,
            n_splits=n_folds,
            seed=seed,
        )
        if verbose:
            print(
                f"[CV fold {repeat_idx + 1}/{n_folds}] "
                f"sessions train/test={split_meta['train_session_count']}/{split_meta['test_session_count']}"
            )

        if emission_model == "private_alternative":
            y_train, X_train, names = _prepare_private_arrays(adapter, train_df)
            y_test, X_test, _ = _prepare_private_arrays(
                adapter,
                test_df,
                expected_x_cols=list(names.get("X_private_cols", [])),
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "restart_start",
                        "restart_index": 1,
                        "restart_total": 1,
                        "subject": subject,
                        "K": 1,
                        "cv_repeat_index": repeat_idx + 1,
                        "cv_repeat_total": n_folds,
                    }
                )
            fit = fit_private_alternative_glm(
                X_train,
                y_train,
                num_classes=num_classes,
                include_bias=True,
                bias_reference_class_idx=baseline_class_idx,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "restart_complete",
                        "restart_index": 1,
                        "restart_total": 1,
                        "negative_log_likelihood": fit.negative_log_likelihood,
                        "success": fit.success,
                        "subject": subject,
                        "K": 1,
                        "cv_repeat_index": repeat_idx + 1,
                        "cv_repeat_total": n_folds,
                    }
                )
            train_metrics = _score_private_split(fit, y_train, X_train)
            test_metrics = _score_private_split(fit, y_test, X_test)
            objective_lp = -float(fit.negative_log_likelihood)
        else:
            y_train, X_train, names = _prepare_standard_arrays(adapter, train_df, emission_cols)
            y_test, X_test, _ = _prepare_standard_arrays(
                adapter,
                test_df,
                emission_cols,
                expected_x_cols=list(names.get("X_cols", [])),
            )

            def _fold_progress(info: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    {
                        **info,
                        "subject": subject,
                        "K": 1,
                        "cv_repeat_index": repeat_idx + 1,
                        "cv_repeat_total": n_folds,
                    }
                )

            fit = fit_glm(
                X_train,
                y_train,
                num_classes=num_classes,
                baseline_class_idx=baseline_class_idx,
                lapse_mode=lapse_mode,
                lapse_max=lapse_max,
                n_restarts=n_restarts,
                restart_noise_scale=restart_noise_scale,
                seed=seed + 1000 * (repeat_idx + 1),
                progress_callback=_fold_progress if progress_callback is not None else None,
            )
            train_metrics = _score_standard_split(fit, y_train, X_train, baseline_class_idx)
            test_metrics = _score_standard_split(fit, y_test, X_test, baseline_class_idx)
            objective_lp = -float(fit.negative_log_likelihood)

        if float(test_metrics["ll_per_trial"]) > best_test_ll:
            best_test_ll = float(test_metrics["ll_per_trial"])
            best_repeat_idx = repeat_idx + 1
            best_fit = fit
            best_names = names

        repeats.append(
            {
                "subject": subject,
                "repeat_index": repeat_idx + 1,
                "best_restart": 1,
                "train_T": int(train_metrics["T"]),
                "test_T": int(test_metrics["T"]),
                "train_raw_ll": float(train_metrics["raw_ll"]),
                "train_ll_per_trial": float(train_metrics["ll_per_trial"]),
                "train_acc": float(train_metrics["acc"]),
                "test_raw_ll": float(test_metrics["raw_ll"]),
                "test_ll_per_trial": float(test_metrics["ll_per_trial"]),
                "test_acc": float(test_metrics["acc"]),
                "objective_lp": objective_lp,
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
                    "K": 1,
                    "cv_repeat_index": repeat_idx + 1,
                    "cv_repeat_total": n_folds,
                    "test_ll_per_trial": float(test_metrics["ll_per_trial"]),
                    "test_acc": float(test_metrics["acc"]),
                }
            )

    if best_fit is None or best_names is None:
        raise ValueError("CV fitting did not produce a valid best repeat.")

    if emission_model == "private_alternative":
        y_full, X_full, full_names = _prepare_private_arrays(
            adapter,
            feature_df,
            expected_x_cols=list(best_names.get("X_private_cols", [])),
        )
        full_metrics = _score_private_split(best_fit, y_full, X_full)
        return {
            "subject": subject,
            "K": 1,
            "num_classes": num_classes,
            "emission_model": "private_alternative",
            "private_weights": best_fit.private_weights,
            "private_bias": best_fit.private_bias,
            "p_pred": full_metrics["p_pred"],
            "nll": -float(full_metrics["raw_ll"]),
            "success": best_fit.success,
            "y": y_full,
            "X_private": X_full,
            "names": full_names,
            "T": int(full_metrics["T"]),
            "baseline_class_idx": int(baseline_class_idx),
            "n_iter": best_fit.n_iter,
            "nll_history": np.asarray(best_fit.nll_history, dtype=float),
            "raw_ll": float(full_metrics["raw_ll"]),
            "repeats": repeats,
            "best_repeat_index": int(best_repeat_idx),
            "best_repeat_test_ll_per_trial": float(best_test_ll),
            "cv_mode": "balanced_session_holdout",
            "cv_repeats": n_folds,
        }

    y_full, X_full, full_names = _prepare_standard_arrays(
        adapter,
        feature_df,
        emission_cols,
        expected_x_cols=list(best_names.get("X_cols", [])),
    )
    full_metrics = _score_standard_split(best_fit, y_full, X_full, baseline_class_idx)
    return {
        "subject": subject,
        "K": 1,
        "num_classes": num_classes,
        "emission_model": "standard",
        "W": best_fit.weights,
        "p_pred": full_metrics["p_pred"],
        "lapse_rates": best_fit.lapse_rates,
        "lapse_mode": best_fit.lapse_mode,
        "lapse_labels": best_fit.lapse_labels,
        "nll": -float(full_metrics["raw_ll"]),
        "success": best_fit.success,
        "y": y_full,
        "X": X_full,
        "names": full_names,
        "T": int(full_metrics["T"]),
        "baseline_class_idx": int(baseline_class_idx),
        "raw_ll": float(full_metrics["raw_ll"]),
        "repeats": repeats,
        "best_repeat_index": int(best_repeat_idx),
        "best_repeat_test_ll_per_trial": float(best_test_ll),
        "cv_mode": "balanced_session_holdout",
        "cv_repeats": n_folds,
    }


def save_results(result: dict, out_dir: Path, tau: float):
    if result is None: return
    
    subj = result["subject"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as {subj}_glm_arrays.npz
    prefix = out_dir / f"{subj}_glm"
    emission_model = str(result.get("emission_model", "standard"))

    if emission_model == "private_alternative":
        p_pred = np.asarray(result["p_pred"], dtype=float)
        y = np.asarray(result["y"], dtype=int)
        T = int(result["T"])
        C = int(p_pred.shape[1])
        private_weights = np.asarray(result["private_weights"], dtype=float)
        private_bias = np.asarray(result["private_bias"], dtype=float)
        baseline_class_idx = int(result.get("baseline_class_idx", 1))
        dummy_emission_weights = np.zeros((1, max(C - 1, 1), private_weights.shape[0]), dtype=np.float32)

        np.savez(
            str(prefix) + "_arrays.npz",
            emission_model=np.array("private_alternative"),
            private_weights=private_weights,
            private_bias=private_bias,
            X_private=np.asarray(result["X_private"], dtype=np.float32),
            X=np.asarray(result["X_private"], dtype=np.float32).mean(axis=1),
            X_private_cols=np.array(result["names"].get("X_private_cols", []), dtype=object),
            X_cols=np.array(result["names"].get("X_private_cols", []), dtype=object),
            alternative_order=np.array(result["names"].get("alternative_order", ["L", "C", "R"]), dtype=object),
            emission_weights=dummy_emission_weights,
            p_pred=p_pred,
            y=y,
            smoothed_probs=np.ones((T, 1)),
            predictive_state_probs=np.ones((T, 1)),
            initial_probs=np.ones(1),
            transition_matrix=np.ones((1, 1)),
            baseline_class_idx=np.array(baseline_class_idx),
            success=result["success"],
            nll_history=np.asarray(result.get("nll_history", []), dtype=float),
            n_iter=np.array(int(result.get("n_iter", 0))),
            cv_selected_repeat=np.array(int(result.get("best_repeat_index", 0))),
            cv_selected_metric=np.array(float(result.get("best_repeat_test_ll_per_trial", np.nan))),
        )

        raw_ll = -float(result["nll"]) if T > 0 else np.nan
        ll_per_trial = raw_ll / T if T > 0 else np.nan
        acc = float(np.mean(np.argmax(p_pred, axis=1) == y)) if T > 0 else 0.0
        k = int(private_weights.shape[0] + 2)
        bic = k * np.log(T) + 2 * result["nll"] if T > 0 else np.nan
        pl.DataFrame({
            "subject": [subj],
            "model_kind": ["glm"],
            "emission_model": ["private_alternative"],
            "tau": [tau],
            "nll": [result["nll"]],
            "raw_ll": [raw_ll],
            "ll_per_trial": [ll_per_trial],
            "bic": [bic],
            "acc": [acc],
            "k": [k],
            "n_trials": [T],
            "baseline_class_idx": [baseline_class_idx],
            "success": [bool(result["success"])],
            "n_iter": [int(result.get("n_iter", 0))],
            "cv_mode": [result.get("cv_mode", "none")],
            "cv_repeats": [int(result.get("cv_repeats", 0))],
        }).write_parquet(str(prefix) + "_metrics.parquet")
        return
    
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
        emission_model=np.array("standard"),
        baseline_class_idx=np.array(int(baseline_class_idx)),
        success=result["success"],
        cv_selected_repeat=np.array(int(result.get("best_repeat_index", 0))),
        cv_selected_metric=np.array(float(result.get("best_repeat_test_ll_per_trial", np.nan))),
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
        "emission_model": ["standard"],
        "tau": [tau],
        "nll": [result["nll"]],
        "raw_ll": [raw_ll],
        "ll_per_trial": [ll_per_trial],
        "bic": [bic],
        "acc": [acc],
        "k": [k],
        "n_trials": [result["T"]],
        "baseline_class_idx": [int(baseline_class_idx)],
        "cv_mode": [result.get("cv_mode", "none")],
        "cv_repeats": [int(result.get("cv_repeats", 0))],
    }).write_parquet(str(prefix) + "_metrics.parquet")


def save_cv_results(result: dict, out_dir: Path) -> None:
    if result is None:
        return

    subj = result["subject"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{subj}_glm"
    repeats_df = pl.DataFrame(result["repeats"])
    repeats_df.write_parquet(str(prefix) + "_cv_repeats.parquet")

    test_ll = repeats_df["test_ll_per_trial"].to_numpy()
    test_acc = repeats_df["test_acc"].to_numpy()
    train_ll = repeats_df["train_ll_per_trial"].to_numpy()
    train_acc = repeats_df["train_acc"].to_numpy()
    T = int(result["T"])
    p_pred = np.asarray(result["p_pred"], dtype=float)
    y = np.asarray(result["y"], dtype=int)
    raw_ll = float(result.get("raw_ll", -float(result["nll"])))
    ll_per_trial = raw_ll / T if T > 0 else np.nan
    acc = float(np.mean(np.argmax(p_pred, axis=1) == y)) if T > 0 else np.nan
    emission_model = str(result.get("emission_model", "standard"))
    if emission_model == "private_alternative":
        private_weights = np.asarray(result["private_weights"], dtype=float)
        n_params = int(private_weights.shape[0] + 2)
    else:
        lapse = np.asarray(result.get("lapse_rates", []), dtype=float)
        n_params = int((result["W"].shape[0] - 1) * result["W"].shape[1] + lapse.size)
    bic = -2 * raw_ll + n_params * np.log(T) if T > 0 else np.nan
    summary = {
        "subject": [subj],
        "K": [1],
        "model_kind": ["glm"],
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

    save_results(result, out_dir, tau=np.nan)
    pl.DataFrame(summary).write_parquet(str(prefix) + "_metrics.parquet")


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
    emission_model: str = "standard",
    cv_mode: str = "none",
    cv_repeats: int = 0,
):
    cols = sorted(emission_cols) if emission_cols else []
    cv_mode = normalize_cv_mode(cv_mode)
    config = {
        "task": task,
        "tau": float(tau),
        "emission_cols": cols,
        "lapse_mode": str(lapse_mode),
        "baseline_class_idx": int(baseline_class_idx),
        "emission_model": str(emission_model),
        "cv_mode": cv_mode,
        "cv_repeats": int(cv_repeats) if cv_mode != "none" else 0,
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
    emission_model: str = "standard",
    cv_mode: str = "none",
    cv_repeats: int = 0,
):
    # Compute base output directory
    base_out_dir = get_results_dir() / "fits" / task / "glm"
    requested_emission_cols = list(emission_cols) if emission_cols is not None else None
    emission_model = str(emission_model or "standard").strip().lower()
    cv_mode = normalize_cv_mode(cv_mode)
    cv_repeats = int(cv_repeats) if cv_mode != "none" else 0
    if emission_model == "private_alternative":
        baseline_class_idx = int(getattr(get_adapter(task), "baseline_class_idx", baseline_class_idx))

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
        emission_model=emission_model,
        cv_mode=cv_mode,
        cv_repeats=cv_repeats,
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
                    "emission_model": emission_model,
                    "model_id": d.name,
                    "cv_mode": cv_mode,
                    "cv_repeats": int(cv_repeats),
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
            if cv_mode == "balanced_session_holdout":
                res = fit_subject_cv(
                    subj,
                    tau=tau,
                    emission_cols=requested_emission_cols,
                    num_classes=num_classes,
                    task=task,
                    lapse_mode=lapse_mode,
                    lapse_max=lapse_max,
                    n_restarts=n_restarts,
                    restart_noise_scale=restart_noise_scale,
                    cv_repeats=cv_repeats,
                    seed=seed,
                    verbose=verbose,
                    progress_callback=_progress if progress_callback is not None else None,
                    baseline_class_idx=baseline_class_idx,
                    condition_filter=condition_filter,
                    emission_model=emission_model,
                )
            else:
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
                    emission_model=emission_model,
                )
            for d in out_dirs:
                if cv_mode == "balanced_session_holdout":
                    save_cv_results(res, d)
                else:
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
    parser.add_argument(
        "--emission_model",
        type=str,
        default="standard",
        choices=["standard", "private_alternative"],
        help="Emission model to fit. private_alternative is currently supported for MCDR GLMs.",
    )
    parser.add_argument(
        "--cv_mode",
        type=str,
        default="none",
        choices=["none", "balanced_session_holdout", "balanced_holdout"],
    )
    parser.add_argument("--cv_repeats", type=int, default=0)

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
        emission_model=args.emission_model,
        cv_mode=args.cv_mode,
        cv_repeats=args.cv_repeats,
    )
