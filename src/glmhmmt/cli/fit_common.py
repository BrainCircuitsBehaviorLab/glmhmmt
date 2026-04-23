import hashlib
import json
from typing import Any, Callable

import jax.random as jr
import jax.numpy as jnp
import numpy as np
import polars as pl

from glmhmmt.model import serialize_frozen_emissions

FIT_ROW_ID_COL = "_fit_row_id"
CV_SESSION_ID_COL = "_cv_session_id"


def normalize_cv_mode(cv_mode: str) -> str:
    """Map legacy CV mode names to the current canonical values."""
    cv_mode = str(cv_mode)
    if cv_mode == "balanced_holdout":
        return "balanced_session_holdout"
    return cv_mode


def valid_trial_mask(session_ids: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Return a boolean mask keeping only trials from sessions with >= min_length trials."""
    ids, counts = np.unique(session_ids, return_counts=True)
    keep = set(ids[counts >= min_length])
    return np.array([session_id in keep for session_id in session_ids])


def apply_valid_trial_mask(
    session_ids: np.ndarray,
    *arrays: np.ndarray,
    min_length: int = 2,
) -> tuple[np.ndarray, ...]:
    """Mask all arrays using the requested minimum-session-length rule."""
    mask = valid_trial_mask(session_ids, min_length=min_length)
    masked = [np.asarray(arr)[mask] for arr in arrays]
    return (*masked, np.asarray(session_ids)[mask])


def stable_model_id(
    task: str,
    K: int,
    tau: float,
    emission_cols: list | None = None,
    transition_cols: list | None = None,
    frozen_emissions: dict | None = None,
    baseline_class_idx: int = 0,
    cv_mode: str = "none",
    cv_repeats: int = 0,
) -> str:
    """Stable 8-char MD5 hash over the fit-defining model configuration."""
    cv_mode = normalize_cv_mode(cv_mode)
    cv_repeats = int(cv_repeats) if cv_mode != "none" else 0
    config = {
        "task": task,
        "K": int(K),
        "tau": float(tau),
        "emission_cols": sorted(emission_cols) if emission_cols else [],
        "transition_cols": sorted(transition_cols) if transition_cols else [],
        "frozen_emissions": serialize_frozen_emissions(frozen_emissions),
        "baseline_class_idx": int(baseline_class_idx),
        "cv_mode": cv_mode,
        "cv_repeats": cv_repeats,
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def fit_best_restart(
    model,
    *,
    n_restarts: int,
    base_seed: int,
    fit_once: Callable[[Any, Any], tuple[Any, Any]],
    failure_message: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_payload: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, int]:
    """Run repeated seeded fits and keep the restart with the best final log-probability."""
    best_lp = -np.inf
    best_params = None
    best_lps = None
    best_restart = -1
    payload = progress_payload or {}

    for restart_idx in range(int(n_restarts)):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_start",
                    "restart_index": restart_idx + 1,
                    "restart_total": int(n_restarts),
                    **payload,
                }
            )

        key = jr.PRNGKey(int(base_seed) + restart_idx)
        params, props = model.initialize(key=key)
        fitted_params, lps = fit_once(params, props)
        final_lp = float(np.asarray(lps)[-1])

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_complete",
                    "restart_index": restart_idx + 1,
                    "restart_total": int(n_restarts),
                    "log_prob": final_lp,
                    **payload,
                }
            )

        if final_lp > best_lp:
            best_lp = final_lp
            best_params = fitted_params
            best_lps = np.asarray(lps)
            best_restart = restart_idx + 1

    if best_params is None or best_lps is None:
        raise ValueError(failure_message)
    return best_params, best_lps, int(best_restart)


def attach_feature_row_ids(feature_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a stable row-id column exists for later split bookkeeping."""
    if FIT_ROW_ID_COL in feature_df.columns:
        return feature_df
    return feature_df.with_row_index(FIT_ROW_ID_COL)


def _sort_columns(sort_col: str | list[str]) -> list[str]:
    return list(sort_col) if isinstance(sort_col, list) else [sort_col]


def _trial_order_column(sort_col: str | list[str]) -> str:
    cols = _sort_columns(sort_col)
    return cols[-1]


def _label_sort_key(label: Any) -> tuple[int, Any]:
    if isinstance(label, (int, float, np.integer, np.floating)):
        return (0, float(label))
    return (1, str(label))


def build_balanced_session_holdout(
    feature_df: pl.DataFrame,
    labels: pl.Series | np.ndarray | list[Any],
    session_col: str,
    seed: int,
    test_fraction: float = 0.5,
    n_candidates: int = 512,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Split full sessions into train/test while balancing aggregate label counts."""
    df = feature_df
    label_arr = labels.to_numpy() if isinstance(labels, pl.Series) else np.asarray(labels)
    if label_arr.shape[0] != df.height:
        raise ValueError(
            f"CV labels length {label_arr.shape[0]} does not match feature_df height {df.height}."
        )

    session_arr = np.asarray(df[session_col].to_numpy())
    by_session_label: dict[Any, dict[Any, int]] = {}
    by_label: dict[Any, int] = {}
    for session_id, label in zip(session_arr, label_arr, strict=True):
        if label is None:
            continue
        if isinstance(label, (float, np.floating)) and not np.isfinite(float(label)):
            continue
        by_session_label.setdefault(session_id, {})
        by_session_label[session_id][label] = by_session_label[session_id].get(label, 0) + 1
        by_label[label] = by_label.get(label, 0) + 1

    if len(by_label) < 2:
        raise ValueError("Balanced CV requires at least two non-empty condition bins.")
    sessions = list(by_session_label)
    n_sessions = len(sessions)
    if n_sessions < 2:
        raise ValueError(
            "Balanced session CV requires at least two sessions with valid labeled trials."
        )

    label_keys = sorted(by_label, key=_label_sort_key)
    label_index = {label: idx for idx, label in enumerate(label_keys)}
    counts = np.zeros((n_sessions, len(label_keys)), dtype=np.int32)
    for sess_idx, session_id in enumerate(sessions):
        for label, count in by_session_label[session_id].items():
            counts[sess_idx, label_index[label]] = int(count)

    total_counts = counts.sum(axis=0).astype(np.float64)
    target_test = total_counts * float(test_fraction)
    test_n_sessions = int(round(n_sessions * float(test_fraction)))
    test_n_sessions = max(1, min(n_sessions - 1, test_n_sessions))

    rng = np.random.default_rng(seed)
    best_test_idx: np.ndarray | None = None
    best_score = np.inf

    for _ in range(max(1, int(n_candidates))):
        test_idx = np.sort(rng.choice(n_sessions, size=test_n_sessions, replace=False))
        train_mask = np.ones(n_sessions, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.flatnonzero(train_mask)
        test_counts = counts[test_idx].sum(axis=0).astype(np.float64)
        train_counts = counts[train_idx].sum(axis=0).astype(np.float64)
        if np.any(test_counts <= 0) or np.any(train_counts <= 0):
            zero_penalty = 1e9 + float(np.sum(test_counts <= 0) + np.sum(train_counts <= 0))
            score = zero_penalty
        else:
            scale = np.maximum(total_counts, 1.0)
            balance_error = np.sum(np.abs(test_counts - target_test) / scale)
            ratio_error = abs(len(test_idx) - (n_sessions / 2.0)) / max(n_sessions, 1)
            score = float(balance_error + ratio_error)
        if score < best_score:
            best_score = score
            best_test_idx = test_idx

    if best_test_idx is None:
        raise ValueError("Balanced session CV could not construct a valid session split.")

    if best_score >= 1e9:
        raise ValueError(
            "Balanced session CV could not find a train/test split that preserves "
            "all balance labels in both partitions. Reduce the number of labels or "
            "use subjects with more sessions."
        )

    test_sessions = [sessions[idx] for idx in best_test_idx.tolist()]
    test_session_set = set(test_sessions)
    train_sessions = [session for session in sessions if session not in test_session_set]
    train_session_set = set(train_sessions)
    train_df = df.filter(pl.col(session_col).is_in(train_sessions))
    test_df = df.filter(pl.col(session_col).is_in(test_sessions))

    train_label_counts: dict[str, int] = {}
    test_label_counts: dict[str, int] = {}
    for label in label_keys:
        idx = label_index[label]
        train_label_counts[str(label)] = int(
            counts[[i for i, session in enumerate(sessions) if session in train_session_set], idx].sum()
        )
        test_label_counts[str(label)] = int(
            counts[[i for i, session in enumerate(sessions) if session in test_session_set], idx].sum()
        )

    return train_df, test_df, {
        "labels": [
            float(lbl) if isinstance(lbl, (int, float, np.integer, np.floating)) else str(lbl)
            for lbl in label_keys
        ],
        "train_session_count": int(len(train_sessions)),
        "test_session_count": int(len(test_sessions)),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "balance_score": float(best_score),
    }


def build_session_kfold_split(
    feature_df: pl.DataFrame,
    session_col: str,
    *,
    fold_index: int,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Split sessions into k folds, training on k-1 folds and testing on one."""
    df = feature_df
    session_series = df.get_column(session_col).unique(maintain_order=True)
    sessions = session_series.to_list()
    n_sessions = len(sessions)
    if n_sessions < int(n_splits):
        raise ValueError(
            f"{n_splits}-fold CV requires at least {n_splits} sessions with valid trials; "
            f"found {n_sessions}."
        )

    rng = np.random.default_rng(seed)
    shuffled_sessions = list(np.asarray(sessions, dtype=object)[rng.permutation(n_sessions)])
    folds = [list(fold) for fold in np.array_split(np.asarray(shuffled_sessions, dtype=object), int(n_splits))]
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index must be in [0, {len(folds) - 1}], got {fold_index}.")

    test_sessions = [session.item() if isinstance(session, np.generic) else session for session in folds[fold_index]]
    train_sessions = [
        session.item() if isinstance(session, np.generic) else session
        for idx, fold in enumerate(folds)
        if idx != fold_index
        for session in fold
    ]
    train_df = df.filter(pl.col(session_col).is_in(train_sessions))
    test_df = df.filter(pl.col(session_col).is_in(test_sessions))

    return train_df, test_df, {
        "fold_index": int(fold_index + 1),
        "n_splits": int(n_splits),
        "train_session_count": int(len(train_sessions)),
        "test_session_count": int(len(test_sessions)),
        "train_label_counts": {},
        "test_label_counts": {},
        "balance_score": float("nan"),
    }


def format_balance_label_stats(
    train_label_counts: dict[str, int],
    test_label_counts: dict[str, int],
) -> str:
    """Return a compact human-readable summary of per-label train/test counts."""
    def _sort_key(label: str) -> tuple[int, Any]:
        try:
            return (0, float(label))
        except Exception:
            return (1, label)

    parts: list[str] = []
    for label in sorted(set(train_label_counts) | set(test_label_counts), key=_sort_key):
        train_count = int(train_label_counts.get(label, 0))
        test_count = int(test_label_counts.get(label, 0))
        total = train_count + test_count
        test_frac = (100.0 * test_count / total) if total > 0 else np.nan
        parts.append(f"{label}: {train_count}|{test_count} ({test_frac:.1f}% test)")
    return ", ".join(parts)


def resegment_subsampled_trials(
    feature_df: pl.DataFrame,
    session_col: str,
    sort_col: str | list[str],
    output_session_col: str = CV_SESSION_ID_COL,
) -> pl.DataFrame:
    """Create synthetic session ids so removed trials do not imply fake adjacency."""
    sort_cols = _sort_columns(sort_col)
    trial_col = _trial_order_column(sort_col)
    df_sorted = feature_df.sort(sort_cols)
    if df_sorted.height == 0:
        return df_sorted.with_columns(pl.Series(output_session_col, [], dtype=pl.Int32))

    sess = np.asarray(df_sorted[session_col].to_numpy())
    trial = np.asarray(df_sorted[trial_col].to_numpy())
    seg_ids = np.zeros(df_sorted.height, dtype=np.int32)
    current_seg = 0
    for idx in range(1, df_sorted.height):
        prev_trial = float(trial[idx - 1])
        curr_trial = float(trial[idx])
        is_gap = (
            sess[idx] != sess[idx - 1]
            or not np.isfinite(prev_trial)
            or not np.isfinite(curr_trial)
            or curr_trial != prev_trial + 1
        )
        if is_gap:
            current_seg += 1
        seg_ids[idx] = current_seg
    return df_sorted.with_columns(pl.Series(output_session_col, seg_ids))


def raw_loglik_multisession(model, params, emissions, inputs, session_ids) -> float:
    """Return the summed data log-likelihood without any parameter prior."""
    if int(np.asarray(emissions).shape[0]) == 0:
        return 0.0
    sessions = model._split_by_session(emissions, inputs, session_ids)
    if not sessions:
        return 0.0
    e_pad, i_pad, _ = model._pad_sessions(sessions)
    _batch_stats, ll_batch = model._batched_e_step_jit(params, e_pad, i_pad)
    return float(jnp.sum(ll_batch))


def score_split(model, params, emissions, inputs, session_ids) -> dict[str, Any]:
    """Evaluate one fitted model on a train or test split."""
    T = int(np.asarray(emissions).shape[0])
    if T == 0:
        return {
            "raw_ll": 0.0,
            "ll_per_trial": np.nan,
            "acc": np.nan,
            "T": 0,
            "p_pred": np.empty((0, getattr(model, "num_classes", 0))),
        }
    raw_ll = raw_loglik_multisession(model, params, emissions, inputs, session_ids)
    p_pred = np.asarray(
        model.predict_choice_probs_multisession(params, emissions, inputs, session_ids=session_ids)
    )
    acc = float(np.mean(np.argmax(p_pred, axis=1) == np.asarray(emissions))) if T else np.nan
    return {
        "raw_ll": raw_ll,
        "ll_per_trial": raw_ll / T if T else np.nan,
        "acc": acc,
        "T": T,
        "p_pred": p_pred,
    }
