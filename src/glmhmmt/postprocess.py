"""postprocess.py — Standard trial-level DataFrame builders.

All DataFrames use **Polars**.  These are the canonical intermediate
representations consumed by all plotting functions.

Key invariant
-------------
``p_state_k`` columns in :func:`build_trial_df` output come **directly** from
``view.smoothed_probs[:, k]`` (the HMM forward-backward posterior), **never**
from a re-application of the emission model.  This guarantees that posterior
plots show the correct, temporally-smooth HMM quantities.

``p_model_correct`` (MAP-state emission) and ``p_model_correct_marginal``
(marginal across states) are computed from the emission weights and are used
only for accuracy analyses — not for posterior visualisation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from glmhmmt.tasks import TaskAdapter

from glmhmmt.views import SubjectFitView, _LABEL_RANK


# ── helpers ────────────────────────────────────────────────────────────────────


def _emission_probs(
    W: np.ndarray,  # (K, C-1, F)
    X: np.ndarray,  # (T, F)
    map_k: np.ndarray,  # (T,) int — MAP state per trial
    C: int,
) -> np.ndarray:
    """Return MAP-state emission probabilities: softmax(W[map_k[t]] @ X[t]).

    Returns
    -------
    p_map : (T, C)
    """
    W_map = W[map_k]  # (T, C-1, F)
    logits_ce = np.einsum("tcf,tf->tc", W_map, X)  # (T, C-1)
    logits_map = _insert_reference(logits_ce, C)  # (T, C)
    return _stable_softmax(logits_map)  # (T, C)


def _insert_reference(logits_ce: np.ndarray, C: int) -> np.ndarray:
    """Insert the reference class (logit = 0) into a (*, C-1) array → (*, C).

    Convention
    ----------
    The model always appends the reference class last
    (SoftmaxGLMHMMEmissions: ``logits = [eta..., 0]``).

    * C == 2 (binary): logits_ce = [logit_L]           →  [logit_L, 0]
    * C == 3 (3-AFC):  logits_ce = [logit_L, logit_C]  →  [logit_L, logit_C, 0_R]
    """
    shape_prefix = logits_ce.shape[:-1]
    return np.concatenate([logits_ce, np.zeros((*shape_prefix, 1))], axis=-1)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax, input shape ``(N, C)``."""
    lse = logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits - lse)
    return exp / exp.sum(axis=-1, keepdims=True)


# ── public builders ────────────────────────────────────────────────────────────


def build_trial_df(
    view: SubjectFitView,
    adapter: TaskAdapter,
    df_behavioral: pl.DataFrame,
    behavioral_cols: dict[str, str],
) -> pl.DataFrame:
    """Build a trial-level Polars DataFrame for one subject.

    The caller must ensure that *df_behavioral* has been:

    * filtered to this subject only,
    * sorted by trial index (``sort_col``),
    * filtered to sessions with ≥ 2 trials (same mask as used during fitting).

    This guarantees ``df_behavioral.height == view.T`` and row ``i`` of the
    behavioral DataFrame aligns exactly with row ``i`` of ``view.smoothed_probs``.

    Parameters
    ----------
    view :
        :class:`~glmhmmt.views.SubjectFitView` for this subject.
    df_behavioral :
        Pre-filtered behavioral DataFrame (see constraints above).
    behavioral_cols :
        Mapping from canonical column name → actual column name in
        *df_behavioral*.  Required keys: ``"trial_idx"``, ``"session"``,
        ``"stimulus"``, ``"response"``, ``"performance"``.  All other columns
        in *df_behavioral* are preserved as-is (e.g. ``stimd_n``, ``ttype_n``).

    Returns
    -------
    pl.DataFrame with columns:

    * All columns from *df_behavioral* (standard ones renamed to canonical
      names).
    * ``subject``
    * ``p_state_0`` … ``p_state_{K-1}``  — HMM posterior (direct copy)
    * ``p_state_pred_0`` … ``p_state_pred_{K-1}``  — one-step predictive
      state probabilities, when available
    * ``state_idx``, ``state_rank``, ``state_label``  — MAP assignment
    * ``pL`` [, ``pC``], ``pR``  — marginal class probabilities from p_pred
    * ``pL_state_0`` …, ``pR_state_0`` …  — per-trial state-conditional
      class probabilities ``P(y_t = c | z_t = k, x_t)``
    * ``p_model_correct``  — MAP-state emission P(correct class)
    * ``p_model_correct_marginal``  — marginal P(correct class)
    * ``correct_bool``
    """
    T = view.T
    if df_behavioral.height != T:
        raise ValueError(
            f"Subject {view.subject!r}: df_behavioral has {df_behavioral.height} "
            f"rows but view.smoothed_probs has T={T}. "
            "Ensure the same session-length filter was applied to both."
        )

    # ── rename standard behavioral columns to canonical names ─────────────────
    rename_map: dict[str, str] = {}
    primary_name_by_actual: dict[str, str] = {}
    alias_pairs: list[tuple[str, str]] = []
    for canonical, actual in behavioral_cols.items():
        if actual not in df_behavioral.columns:
            continue
        if actual not in primary_name_by_actual:
            primary = canonical if actual != canonical else actual
            primary_name_by_actual[actual] = primary
            if actual != canonical:
                rename_map[actual] = canonical
        else:
            alias_pairs.append((canonical, primary_name_by_actual[actual]))
    if rename_map:
        df_out = df_behavioral.rename(rename_map)
    else:
        df_out = df_behavioral.clone()
    if alias_pairs:
        df_out = df_out.with_columns(
            [
                pl.col(src).alias(alias)
                for alias, src in alias_pairs
                if src in df_out.columns and alias not in df_out.columns and alias != src
            ]
        )

    # ── add emission-feature columns from the fitted design matrix ───────────
    # This keeps trial_df aligned with the exact regressors used to build X,
    # so downstream plotting can access columns such as `at_choice`.
    X_arr = np.asarray(view.X, dtype=np.float64)
    feature_cols = []
    for fi, fname in enumerate(list(view.feat_names or [])):
        if fi >= X_arr.shape[1] or fname in df_out.columns:
            continue
        feature_cols.append(pl.Series(fname, X_arr[:, fi]))
    if feature_cols:
        df_out = df_out.with_columns(feature_cols)

    # ── p_state_k  — HMM posterior, direct copy (NEVER recomputed) ───────────
    posterior_series = [pl.Series(f"p_state_{k}", view.smoothed_probs[:, k].astype(np.float64)) for k in range(view.K)]
    predictive_state_series = [
        pl.Series(f"p_state_pred_{k}", np.asarray(view.predictive_state_probs[:, k], dtype=np.float64))
        for k in range(view.K)
    ]
    # ── MAP state assignment ───────────────────────────────────────────────────
    map_k = view.map_states()  # (T,) int
    state_rank_arr = np.array([view.state_rank_by_idx.get(int(ki), ki) for ki in map_k], dtype=np.int32)
    state_label_arr = np.array([view.state_name_by_idx.get(int(ki), f"State {ki}") for ki in map_k])

    # ── state-conditional emission probabilities on the observed trials ──────
    C = view.num_classes
    p_state_conditional = view.state_conditional_probs()  # (T, K, C)

    # ── MAP-state emission probabilities ──────────────────────────────────────
    p_map = _emission_probs(view.emission_weights, view.X, map_k, C)  # (T, C)

    correct_class = adapter.get_correct_class(df_out)
    if np.any((correct_class < 0) | (correct_class >= C)):
        bad = np.unique(correct_class[(correct_class < 0) | (correct_class >= C)])
        raise ValueError(
            f"Subject {view.subject!r}: adapter returned invalid correct_class indices {bad.tolist()} for C={C}."
        )

    p_model_correct_map = p_map[np.arange(T), correct_class]

    # ── marginal class probabilities (from view.p_pred if available) ──────────
    if view.p_pred is None:
        raise ValueError(
            f"Subject {view.subject!r}: missing p_pred. "
            "Predictions must be precomputed upstream and stored in the fit arrays."
        )

    p_marginal = np.asarray(view.p_pred, dtype=np.float64)
    C = p_marginal.shape[1]

    p_marginal_correct = p_marginal[np.arange(T), correct_class]

    # ── assemble all new columns ───────────────────────────────────────────────
    new_cols = [
        pl.Series("subject", [view.subject] * T),
        *posterior_series,
        *predictive_state_series,
        pl.Series("state_idx", map_k.astype(np.int32)),
        pl.Series("state_rank", state_rank_arr),
        pl.Series("state_label", state_label_arr),
        pl.Series("p_model_correct", p_model_correct_map.astype(np.float64)),
        pl.Series("p_model_correct_marginal", p_marginal_correct.astype(np.float64)),
    ]

    for k in range(view.K):
        new_cols.append(pl.Series(f"p_state_model_0_{k}", p_state_conditional[:, k, 0].astype(np.float64)))
        if C >= 2:
            new_cols.append(pl.Series(f"p_state_model_1_{k}", p_state_conditional[:, k, 1].astype(np.float64)))
        if C >= 3:
            new_cols.append(pl.Series(f"p_state_model_2_{k}", p_state_conditional[:, k, 2].astype(np.float64)))

    prob_cols = list(adapter.probability_columns)
    if len(prob_cols) != C:
        prob_cols = [f"p_{idx}" for idx in range(C)]

    if C == 2:
        new_cols += [
            pl.Series(prob_cols[0], p_marginal[:, 0].astype(np.float64)),
            pl.Series(prob_cols[1], p_marginal[:, 1].astype(np.float64)),
        ]
        for k in range(view.K):
            new_cols += [
                pl.Series(f"{prob_cols[0]}_state_{k}", p_state_conditional[:, k, 0].astype(np.float64)),
                pl.Series(f"{prob_cols[1]}_state_{k}", p_state_conditional[:, k, 1].astype(np.float64)),
            ]
    else:
        new_cols += [
            pl.Series(prob_cols[0], p_marginal[:, 0].astype(np.float64)),
            pl.Series(prob_cols[1], p_marginal[:, 1].astype(np.float64)),
            pl.Series(prob_cols[2], p_marginal[:, 2].astype(np.float64)),
        ]
        for k in range(view.K):
            new_cols += [
                pl.Series(f"{prob_cols[0]}_state_{k}", p_state_conditional[:, k, 0].astype(np.float64)),
                pl.Series(f"{prob_cols[1]}_state_{k}", p_state_conditional[:, k, 1].astype(np.float64)),
                pl.Series(f"{prob_cols[2]}_state_{k}", p_state_conditional[:, k, 2].astype(np.float64)),
            ]

    # overwrite/add; drop pre-existing computed cols (pL/pC/pR, subject)
    _computed_names = {s.name for s in new_cols}
    df_out = df_out.select([c for c in df_out.columns if c not in _computed_names])
    df_out = df_out.with_columns(new_cols)

    # ── correct_bool from performance ─────────────────────────────────────────
    if "performance" in df_out.columns and "correct_bool" not in df_out.columns:
        df_out = df_out.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))

    return df_out


def build_emission_weights_df(views: dict[str, SubjectFitView]) -> pl.DataFrame:
    """Long-format Polars DataFrame of all subjects' emission weights.

    Columns
    -------
    subject, state_idx, state_label, state_rank, class_idx, feature, weight
    """
    records: list[dict] = []
    for subj, view in views.items():
        W = view.emission_weights  # (K, C-1, F)
        for k in range(view.K):
            lbl = view.state_name_by_idx.get(k, f"State {k}")
            rank = view.state_rank_by_idx.get(k, k)
            for c in range(W.shape[1]):
                for fi, fname in enumerate(view.feat_names):
                    records.append(
                        {
                            "subject": subj,
                            "state_idx": k,
                            "state_label": lbl,
                            "state_rank": rank,
                            "class_idx": c,
                            "feature": fname,
                            "weight": float(W[k, c, fi]),
                        }
                    )
    if not records:
        return pl.DataFrame()
    return pl.DataFrame(records)


def build_posterior_df(views: dict[str, SubjectFitView]) -> pl.DataFrame:
    """Long-format Polars DataFrame of HMM posterior probabilities.

    The ``probability`` column contains the raw HMM posterior
    γ(z_t = k | y_{1:T}) — the **same** values as ``smoothed_probs``.

    Columns
    -------
    subject, trial_idx, state_idx, state_label, state_rank, probability
    """
    frames: list[pl.DataFrame] = []
    for subj, view in views.items():
        T = view.T
        for k in range(view.K):
            lbl = view.state_name_by_idx.get(k, f"State {k}")
            rank = view.state_rank_by_idx.get(k, k)
            frames.append(
                pl.DataFrame(
                    {
                        "subject": [subj] * T,
                        "trial_idx": list(range(T)),
                        "state_idx": [k] * T,
                        "state_label": [lbl] * T,
                        "state_rank": [rank] * T,
                        "probability": view.smoothed_probs[:, k].tolist(),
                    }
                )
            )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)
