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

import re

import numpy as np
import polars as pl
from glmhmmt.tasks import TaskAdapter

from glmhmmt.views import SubjectFitView, _LABEL_RANK, get_state_color

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


_BIAS_FAMILY_PATTERN = re.compile(r"^bias_\d+$")
_BIAS_FAMILY_LABEL = "|bias|"


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


def _default_state_labels(K: int, C: int = 2) -> list[str]:
    if K == 1:
        return ["State 0"]
    if K == 2:
        return ["Disengaged", "Engaged"]
    if K == 3:
        return ["Engaged", "Biased L", "Biased R"]
    return [f"State {k}" for k in range(K)]


def _resolve_state_colors(
    state_labels: list[str],
    state_colors,
) -> list[str]:
    if state_colors is not None:
        if isinstance(state_colors, str) or np.isscalar(state_colors):
            return [str(state_colors)] * len(state_labels)
        colors = [str(color) for color in state_colors]
        if len(colors) == 1 and len(state_labels) > 1:
            colors *= len(state_labels)
        if len(colors) != len(state_labels):
            raise ValueError(
                f"state_colors must have length {len(state_labels)}, got {len(colors)}."
            )
        return colors
    return [
        get_state_color(label, idx, K=len(state_labels) or None)
        for idx, label in enumerate(state_labels)
    ]


def _weights_boxplot_payload_from_array(
    weights,
    feature_names=None,
    state_labels=None,
    state_colors=None,
) -> dict:
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[None, :, None, :]
    elif W.ndim == 3:
        W = W[:, :, None, :]
    if W.ndim != 4:
        raise ValueError(
            "weights must have shape (N, K, C-1, M), (N, K, M), or (K, M)."
        )

    _, K, C_m1, M = W.shape
    labels = (
        [str(label) for label in state_labels]
        if state_labels is not None
        else _default_state_labels(K, C_m1 + 1)
    )
    if len(labels) != K:
        raise ValueError(f"state_labels must have length {K}, got {len(labels)}.")

    features = (
        [str(name) for name in feature_names]
        if feature_names is not None
        else [f"Feature {idx}" for idx in range(M)]
    )
    if len(features) != M:
        raise ValueError(f"feature_names must have length {M}, got {len(features)}.")

    colors = _resolve_state_colors(labels, state_colors)
    W_avg = np.nanmean(W, axis=2)
    values_by_state_feature = [
        [
            np.asarray(W_avg[:, state_idx, feat_idx], dtype=float)
            for feat_idx in range(M)
        ]
        for state_idx in range(K)
    ]
    subject_lines = [
        np.asarray(W_avg[:, :, feat_idx], dtype=float)
        for feat_idx in range(M)
    ]
    return {
        "values_by_state_feature": values_by_state_feature,
        "feature_names": features,
        "state_labels": labels,
        "state_colors": colors,
        "subject_lines": subject_lines,
    }


def _weights_boxplot_payload_from_frame(
    weights_df,
    feature_names=None,
    state_labels=None,
    state_colors=None,
) -> dict:
    if isinstance(weights_df, pl.DataFrame):
        df = weights_df.clone()
    elif pd is not None and isinstance(weights_df, pd.DataFrame):
        df = pl.from_pandas(weights_df)
    else:
        raise TypeError("weights_df must be a Polars or pandas DataFrame.")

    if df.is_empty():
        return {
            "values_by_state_feature": [],
            "feature_names": [],
            "state_labels": [],
            "state_colors": [],
            "subject_lines": [],
        }

    if "class_idx" in df.columns:
        df = df.filter(pl.col("class_idx") == 0)
    if df.is_empty():
        return {
            "values_by_state_feature": [],
            "feature_names": [],
            "state_labels": [],
            "state_colors": [],
            "subject_lines": [],
        }

    required = {"feature", "weight"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "weights dataframe must contain at least 'feature' and 'weight'. "
            f"Missing: {missing}."
        )

    df = df.with_columns(
        pl.col("feature").cast(pl.Utf8).alias("feature"),
        pl.col("weight").cast(pl.Float64, strict=False).alias("weight"),
    ).drop_nulls("weight")
    if df.is_empty():
        return {
            "values_by_state_feature": [],
            "feature_names": [],
            "state_labels": [],
            "state_colors": [],
            "subject_lines": [],
        }

    feature_name_order = (
        [str(name) for name in feature_names]
        if feature_names is not None
        else df["feature"].unique(maintain_order=True).to_list()
    )
    normalized_feature_names: list[str] = []
    for feature_name in feature_name_order:
        normalized_name = (
            _BIAS_FAMILY_LABEL
            if _BIAS_FAMILY_PATTERN.match(feature_name)
            else feature_name
        )
        if normalized_name not in normalized_feature_names:
            normalized_feature_names.append(normalized_name)

    if any(_BIAS_FAMILY_PATTERN.match(name) for name in df["feature"].unique().to_list()):
        group_cols = [col for col in df.columns if col not in {"feature", "weight"}]
        df = (
            df.with_columns(
                pl.when(pl.col("feature").str.contains(r"^bias_\d+$"))
                .then(pl.lit(_BIAS_FAMILY_LABEL))
                .otherwise(pl.col("feature"))
                .alias("feature"),
                pl.when(pl.col("feature").str.contains(r"^bias_\d+$"))
                .then(pl.col("weight").abs())
                .otherwise(pl.col("weight"))
                .alias("weight"),
            )
            .group_by([*group_cols, "feature"], maintain_order=True)
            .agg(pl.col("weight").mean().alias("weight"))
        )

    if "subject" not in df.columns:
        df = df.with_columns(pl.lit("subject-0").alias("subject"))
    else:
        df = df.with_columns(pl.col("subject").cast(pl.Utf8).alias("subject"))

    if "state_label" not in df.columns:
        df = df.with_columns(pl.lit("State 0").alias("state_label"))
    else:
        df = df.with_columns(pl.col("state_label").cast(pl.Utf8).alias("state_label"))

    if "state_rank" not in df.columns:
        if "state_idx" in df.columns:
            fallback = (
                df.select(pl.col("state_idx").cast(pl.Int64, strict=False))
                .to_series()
                .fill_null(0)
                .to_list()
            )
        else:
            fallback = [0] * df.height
        labels = df["state_label"].to_list()
        ranks = [
            int(_LABEL_RANK.get(label, fallback_idx))
            for label, fallback_idx in zip(labels, fallback, strict=False)
        ]
        df = df.with_columns(pl.Series("state_rank", ranks, dtype=pl.Int64))

    if feature_names is None:
        features = df["feature"].unique(maintain_order=True).to_list()
    else:
        requested = normalized_feature_names
        present = set(df["feature"].unique().to_list())
        features = [name for name in requested if name in present]
        features.extend(
            name
            for name in df["feature"].unique(maintain_order=True).to_list()
            if name not in features
        )

    state_meta = (
        df.select(["state_rank", "state_label"])
        .unique(maintain_order=True)
        .sort(["state_rank", "state_label"])
    )
    available_labels = state_meta["state_label"].to_list()
    if state_labels is None:
        labels = available_labels
    else:
        requested_labels = [str(label) for label in state_labels]
        labels = [label for label in requested_labels if label in set(available_labels)]
        labels.extend(label for label in available_labels if label not in labels)

    colors = _resolve_state_colors(labels, state_colors)

    values_by_state_feature: list[list[np.ndarray]] = []
    for state_name in labels:
        state_df = df.filter(pl.col("state_label") == state_name)
        feature_values = []
        for feature_name in features:
            vals = (
                state_df.filter(pl.col("feature") == feature_name)["weight"]
                .to_numpy()
                .astype(float, copy=False)
            )
            feature_values.append(vals)
        values_by_state_feature.append(feature_values)

    feature_frames = [df.filter(pl.col("feature") == feature_name) for feature_name in features]
    subject_lines: list[np.ndarray] = []
    for feat_df in feature_frames:
        if feat_df.is_empty():
            subject_lines.append(np.empty((0, len(labels)), dtype=float))
            continue
        subjects = feat_df["subject"].unique(maintain_order=True).to_list()
        per_subject = np.full((len(subjects), len(labels)), np.nan, dtype=float)
        for subj_idx, subject in enumerate(subjects):
            subj_df = feat_df.filter(pl.col("subject") == subject)
            for state_idx, label in enumerate(labels):
                state_df = subj_df.filter(pl.col("state_label") == label)
                if state_df.is_empty():
                    continue
                per_subject[subj_idx, state_idx] = float(np.mean(state_df["weight"].to_numpy()))
        subject_lines.append(per_subject)

    return {
        "values_by_state_feature": values_by_state_feature,
        "feature_names": features,
        "state_labels": labels,
        "state_colors": colors,
        "subject_lines": subject_lines,
    }


def build_weights_boxplot_payload(
    weights,
    feature_names=None,
    state_labels=None,
    state_colors=None,
) -> dict:
    if isinstance(weights, pl.DataFrame) or (pd is not None and isinstance(weights, pd.DataFrame)):
        return _weights_boxplot_payload_from_frame(
            weights,
            feature_names=feature_names,
            state_labels=state_labels,
            state_colors=state_colors,
        )
    return _weights_boxplot_payload_from_array(
        weights,
        feature_names=feature_names,
        state_labels=state_labels,
        state_colors=state_colors,
    )


def _resolve_transition_matrix_from_arrays(
    arrays_store: dict,
    subject: str,
) -> np.ndarray | None:
    arr = arrays_store.get(subject, {})
    if "transition_matrix" in arr:
        return np.asarray(arr["transition_matrix"], dtype=float)
    if "transition_bias" in arr:
        bias = np.asarray(arr["transition_bias"], dtype=float)
        exp_bias = np.exp(bias - bias.max(axis=-1, keepdims=True))
        return exp_bias / exp_bias.sum(axis=-1, keepdims=True)
    return None


def build_transition_matrix_payload(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
) -> dict:
    selected = [
        subject
        for subject in subjects
        if _resolve_transition_matrix_from_arrays(arrays_store, subject) is not None
    ]
    if not selected:
        return {
            "matrix": np.asarray([], dtype=float),
            "tick_labels": [],
            "title": "No transition matrices found.",
        }

    mean_matrix = np.mean(
        [
            _resolve_transition_matrix_from_arrays(arrays_store, subject)
            for subject in selected
        ],
        axis=0,
    )
    label_map = state_labels.get(selected[0], {k: f"S{k}" for k in range(K)})
    tick_labels = [str(label_map.get(k, f"S{k}")) for k in range(K)]
    return {
        "matrix": mean_matrix,
        "tick_labels": tick_labels,
        "title": f"Mean transition matrix  (n={len(selected)} subjects)",
    }


def build_transition_matrix_by_subject_payload(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
) -> dict:
    selected = [
        subject
        for subject in subjects
        if _resolve_transition_matrix_from_arrays(arrays_store, subject) is not None
    ]
    if not selected:
        return {
            "matrices": [],
            "subject_ids": [],
            "tick_labels_by_subject": [],
        }

    matrices = [
        _resolve_transition_matrix_from_arrays(arrays_store, subject)
        for subject in selected
    ]
    tick_labels_by_subject = [
        [str(state_labels.get(subject, {}).get(k, f"S{k}")) for k in range(K)]
        for subject in selected
    ]
    return {
        "matrices": matrices,
        "subject_ids": selected,
        "tick_labels_by_subject": tick_labels_by_subject,
    }


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


def _as_pandas_df(df_like, *, name: str):
    if pd is not None and isinstance(df_like, pd.DataFrame):
        return df_like.copy()
    if isinstance(df_like, pl.DataFrame):
        return df_like.to_pandas()
    raise TypeError(f"{name} must be a Polars or pandas DataFrame.")


def _posterior_cols(df) -> list[str]:
    cols = [
        col for col in df.columns
        if isinstance(col, str) and col.startswith("p_state_") and col.removeprefix("p_state_").isdigit()
    ]
    return sorted(cols, key=lambda col: int(col.rsplit("_", 1)[1]))


def _require_columns(df, required: list[str], *, name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")


def _state_meta_from_trial_df(df) -> tuple[list[str], dict[int, str], dict[int, int]]:
    _require_columns(df, ["state_idx", "state_label"], name="trial_df")
    meta = (
        df[["state_idx", "state_label", *([ "state_rank" ] if "state_rank" in df.columns else [])]]
        .dropna(subset=["state_idx", "state_label"])
        .drop_duplicates()
    )
    label_by_idx = {
        int(row.state_idx): str(row.state_label)
        for row in meta.itertuples(index=False)
    }
    if "state_rank" in meta.columns:
        rank_by_idx = {
            int(row.state_idx): int(row.state_rank)
            for row in meta.itertuples(index=False)
        }
    else:
        rank_by_idx = {idx: idx for idx in label_by_idx}
    state_order = [
        label_by_idx[idx]
        for idx in sorted(label_by_idx, key=lambda state_idx: (rank_by_idx.get(state_idx, state_idx), state_idx))
    ]
    return state_order, label_by_idx, rank_by_idx


def _transition_matrix_from_view(view) -> np.ndarray | None:
    transition_matrix = getattr(view, "transition_matrix", None)
    if transition_matrix is not None:
        matrix = np.asarray(transition_matrix, dtype=float)
        return matrix if matrix.ndim == 2 else None

    transition_bias = getattr(view, "transition_bias", None)
    transition_weights = getattr(view, "transition_weights", None)
    if transition_bias is None or transition_weights is not None:
        return None
    bias = np.asarray(transition_bias, dtype=float)
    if bias.ndim != 2:
        return None
    shifted = bias - bias.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def _geometric_dwell_pmf(self_prob: float, max_dwell: int) -> np.ndarray:
    p = float(np.clip(self_prob, 0.0, np.nextafter(1.0, 0.0)))
    dwell = np.arange(1, max_dwell + 1, dtype=float)
    return (1.0 - p) * np.power(p, dwell - 1.0)


def _normal_z_for_ci(ci_level: float) -> float:
    ci_level = float(ci_level)
    if ci_level <= 0.70:
        return 1.0
    if ci_level <= 0.90:
        return 1.645
    if ci_level <= 0.95:
        return 1.96
    return 2.576


def _binomial_interval(counts: np.ndarray, total: int, ci_level: float) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    if total <= 0:
        zeros = np.zeros_like(counts, dtype=float)
        return zeros, zeros
    phat = counts / float(total)
    z = _normal_z_for_ci(ci_level)
    half_width = z * np.sqrt(np.clip(phat * (1.0 - phat) / float(total), 0.0, None))
    return np.clip(phat - half_width, 0.0, 1.0), np.clip(phat + half_width, 0.0, 1.0)


def _binned_dwell_summary(
    dwell_lengths: np.ndarray,
    *,
    max_dwell: int,
    bin_width: int,
    ci_level: float,
    self_prob: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    n_bins = int(np.ceil(max_dwell / float(bin_width)))
    centers = bin_width * np.arange(n_bins, dtype=float) + (bin_width / 2.0)
    predicted = None
    if self_prob is not None:
        predicted_full = _geometric_dwell_pmf(float(self_prob), max_dwell)
        predicted = np.zeros(n_bins, dtype=float)
        for bin_idx in range(n_bins):
            lo = bin_idx * bin_width
            hi = min((bin_idx + 1) * bin_width, max_dwell)
            predicted[bin_idx] = float(np.sum(predicted_full[lo:hi]))

    dwell_lengths = np.asarray(dwell_lengths, dtype=int)
    total_runs = int(dwell_lengths.size)
    if total_runs == 0:
        zeros = np.zeros(n_bins, dtype=float)
        return centers, predicted, zeros, np.vstack([zeros, zeros])

    in_range = dwell_lengths[(dwell_lengths >= 1) & (dwell_lengths <= max_dwell)]
    if in_range.size == 0:
        zeros = np.zeros(n_bins, dtype=float)
        return centers, predicted, zeros, np.vstack([zeros, zeros])

    bin_idx = (in_range - 1) // int(bin_width)
    counts = np.bincount(bin_idx, minlength=n_bins)[:n_bins]
    empirical = counts / float(total_runs)
    ci_low, ci_high = _binomial_interval(counts, total_runs, ci_level)
    yerr = np.vstack(
        [
            np.clip(empirical - ci_low, 0.0, None),
            np.clip(ci_high - empirical, 0.0, None),
        ]
    )
    return centers, predicted, empirical, yerr


def _infer_engaged_state_idx(label_by_idx: dict[int, str], rank_by_idx: dict[int, int]) -> int:
    for idx, label in label_by_idx.items():
        normalized = label.lower()
        if "engaged" in normalized and "dis" not in normalized:
            return int(idx)
    return int(sorted(label_by_idx, key=lambda state_idx: (rank_by_idx.get(state_idx, state_idx), state_idx))[0])


def _confident_change_events_by_direction(
    probs: np.ndarray,
    *,
    engaged_idx: int,
    posterior_threshold: float | None,
) -> dict[str, np.ndarray]:
    empty = {
        "into_engaged": np.array([], dtype=int),
        "out_of_engaged": np.array([], dtype=int),
    }
    if len(probs) <= 1:
        return empty

    if posterior_threshold is None:
        keep_idx = np.arange(len(probs), dtype=int)
        confident_states = np.argmax(probs, axis=1)
    else:
        keep_idx = np.flatnonzero(np.max(probs, axis=1) >= float(posterior_threshold))
        if keep_idx.size <= 1:
            return empty
        confident_states = np.argmax(probs[keep_idx], axis=1)

    change_mask = np.diff(confident_states) != 0
    if not np.any(change_mask):
        return empty

    prev_states = confident_states[:-1][change_mask]
    next_states = confident_states[1:][change_mask]
    change_idx = keep_idx[1:][change_mask]
    return {
        "into_engaged": change_idx[(prev_states != engaged_idx) & (next_states == engaged_idx)],
        "out_of_engaged": change_idx[(prev_states == engaged_idx) & (next_states != engaged_idx)],
    }


def build_state_accuracy_payload(
    trial_df,
    *,
    performance_col: str = "correct_bool",
    chance_level: float | None = None,
    title: str = "Accuracy by state",
) -> dict:
    """Prepare the payload consumed by ``plot_state_accuracy``."""
    df = _as_pandas_df(trial_df, name="trial_df")
    _require_columns(df, ["subject", "state_label", performance_col], name="trial_df")
    state_order, _, _ = _state_meta_from_trial_df(df)
    work = df.dropna(subset=["subject", "state_label", performance_col]).copy()
    work[performance_col] = work[performance_col].astype(float)
    accuracy_df = (
        work.groupby(["subject", "state_label"], observed=True)
        .agg(accuracy=(performance_col, "mean"), n_trials=(performance_col, "size"))
        .reset_index()
    )
    all_df = (
        work.groupby("subject", observed=True)
        .agg(accuracy=(performance_col, "mean"), n_trials=(performance_col, "size"))
        .reset_index()
    )
    all_df["state_label"] = "All"
    accuracy_df = pd.concat([all_df[["subject", "state_label", "accuracy", "n_trials"]], accuracy_df], ignore_index=True)
    return {
        "accuracy_df": accuracy_df,
        "state_order": state_order,
        "chance_level": chance_level,
        "title": title,
    }


def build_session_trajectories_payload(
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    title: str = "Session trajectories",
) -> dict:
    """Prepare long posterior trajectories averaged over sessions."""
    df = _as_pandas_df(trial_df, name="trial_df")
    _require_columns(df, ["subject", session_col, sort_col], name="trial_df")
    pcols = _posterior_cols(df)
    if not pcols:
        raise ValueError("trial_df must contain posterior columns named p_state_0, p_state_1, ...")
    state_order, label_by_idx, rank_by_idx = _state_meta_from_trial_df(df)
    work = df.sort_values(["subject", session_col, sort_col]).copy()
    work["trial_in_session"] = work.groupby(["subject", session_col], observed=True).cumcount()
    long = work.melt(
        id_vars=["subject", session_col, "trial_in_session"],
        value_vars=pcols,
        var_name="state_col",
        value_name="probability",
    )
    long["state_idx"] = long["state_col"].str.rsplit("_", n=1).str[-1].astype(int)
    long["state_label"] = long["state_idx"].map(label_by_idx).fillna(long["state_col"])
    long["state_rank"] = long["state_idx"].map(rank_by_idx).fillna(long["state_idx"]).astype(int)
    long = long.rename(columns={session_col: "session"})
    trajectory_df = (
        long.groupby(["subject", "session", "trial_in_session", "state_idx", "state_label", "state_rank"], observed=True)
        .agg(probability=("probability", "mean"))
        .reset_index()
    )
    return {
        "trajectory_df": trajectory_df,
        "state_order": state_order,
        "title": title,
    }


def build_change_triggered_posteriors_payload(
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    switch_posterior_threshold: float | None = None,
    window: int = 15,
    title: str = "Change-triggered posteriors",
) -> dict:
    """Prepare posterior traces around confident MAP-state changes."""
    df = _as_pandas_df(trial_df, name="trial_df")
    _require_columns(df, ["subject", session_col, sort_col, "state_idx"], name="trial_df")
    pcols = _posterior_cols(df)
    if not pcols:
        raise ValueError("trial_df must contain posterior columns named p_state_0, p_state_1, ...")
    state_order, label_by_idx, rank_by_idx = _state_meta_from_trial_df(df)
    engaged_idx = _infer_engaged_state_idx(label_by_idx, rank_by_idx)
    non_engaged_candidates = [
        idx
        for idx in sorted(label_by_idx, key=lambda state_idx: (rank_by_idx.get(state_idx, state_idx), state_idx))
        if idx != engaged_idx
    ]
    if not non_engaged_candidates:
        raise ValueError("Change-triggered posteriors require at least two states.")
    non_engaged_idx = int(non_engaged_candidates[0])
    engaged_label = label_by_idx[engaged_idx]
    non_engaged_label = label_by_idx[non_engaged_idx] if len(non_engaged_candidates) == 1 else "Non-engaged"

    rows = []
    work = df.sort_values(["subject", session_col, sort_col]).copy()
    for (subject, session), group in work.groupby(["subject", session_col], observed=True):
        probs = group[pcols].to_numpy(dtype=float)
        change_idx_by_direction = _confident_change_events_by_direction(
            probs,
            engaged_idx=engaged_idx,
            posterior_threshold=switch_posterior_threshold,
        )
        for direction, change_indices in change_idx_by_direction.items():
            for event_idx, change_trial in enumerate(change_indices):
                change_id = f"{subject}:{session}:{direction}:{event_idx}"
                lo = max(0, int(change_trial) - int(window))
                hi = min(len(group), int(change_trial) + int(window) + 1)
                for row_idx in range(lo, hi):
                    rel = int(row_idx - int(change_trial))
                    p_engaged = float(probs[row_idx, engaged_idx])
                    for state_idx, state_label, probability in (
                        (engaged_idx, engaged_label, p_engaged),
                        (non_engaged_idx, non_engaged_label, 1.0 - p_engaged),
                    ):
                        rows.append(
                            {
                                "subject": subject,
                                "session": session,
                                "change_id": change_id,
                                "direction": direction,
                                "relative_trial": rel,
                                "state_idx": state_idx,
                                "state_label": state_label,
                                "probability": probability,
                            }
                        )
    return {
        "change_df": pd.DataFrame(
            rows,
            columns=[
                "subject",
                "session",
                "change_id",
                "direction",
                "relative_trial",
                "state_idx",
                "state_label",
                "probability",
            ],
        ),
        "state_order": [engaged_label, non_engaged_label],
        "directions": ["into_engaged", "out_of_engaged"],
        "window": int(window),
        "title": title,
    }


def build_state_occupancy_payload(
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    title: str = "State occupancy",
) -> dict:
    """Prepare occupancy and MAP-switch summaries."""
    df = _as_pandas_df(trial_df, name="trial_df")
    _require_columns(df, ["subject", session_col, sort_col, "state_label"], name="trial_df")
    state_order, _, _ = _state_meta_from_trial_df(df)
    work = df.sort_values(["subject", session_col, sort_col]).copy()

    subj_totals = work.groupby("subject", observed=True).size().rename("n_total")
    occupancy_df = (
        work.groupby(["subject", "state_label"], observed=True)
        .size()
        .rename("n_trials")
        .reset_index()
        .merge(subj_totals.reset_index(), on="subject", how="left")
    )
    occupancy_df["occupancy"] = occupancy_df["n_trials"] / occupancy_df["n_total"]

    session_totals = work.groupby(["subject", session_col], observed=True).size().rename("n_total")
    session_occupancy_df = (
        work.groupby(["subject", session_col, "state_label"], observed=True)
        .size()
        .rename("n_trials")
        .reset_index()
        .merge(session_totals.reset_index(), on=["subject", session_col], how="left")
    )
    session_occupancy_df["occupancy"] = session_occupancy_df["n_trials"] / session_occupancy_df["n_total"]
    session_occupancy_df = session_occupancy_df.rename(columns={session_col: "session"})

    state_source = "state_idx" if "state_idx" in work.columns else "state_label"
    switches = []
    for (subject, session), group in work.groupby(["subject", session_col], observed=True):
        values = group[state_source].to_numpy()
        n_switches = int((values[1:] != values[:-1]).sum()) if len(values) > 1 else 0
        switches.append({"subject": subject, "session": session, "n_switches": n_switches})
    switches_df = pd.DataFrame(switches)
    return {
        "occupancy_df": occupancy_df,
        "session_occupancy_df": session_occupancy_df,
        "switches_df": switches_df,
        "state_order": state_order,
        "title": title,
    }


def build_state_dwell_times_payload(
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    transition_matrices: dict | None = None,
    views: dict | None = None,
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    bin_width: int = 10,
    title: str = "Dwell times",
) -> dict:
    """Prepare empirical MAP-state dwell lengths split at session boundaries."""
    df = _as_pandas_df(trial_df, name="trial_df")
    _require_columns(df, ["subject", session_col, sort_col, "state_label"], name="trial_df")
    state_order, label_by_idx, rank_by_idx = _state_meta_from_trial_df(df)
    work = df.sort_values(["subject", session_col, sort_col]).copy()
    rows = []
    state_source = "state_idx" if "state_idx" in work.columns else "state_label"
    for (subject, session), group in work.groupby(["subject", session_col], observed=True):
        values = group[state_source].to_list()
        labels = group["state_label"].astype(str).to_list()
        ranks = group["state_idx"].map(rank_by_idx).fillna(group.get("state_rank", 0)).to_list() if "state_idx" in group.columns else [None] * len(group)
        if not values:
            continue
        start = 0
        for idx in range(1, len(values) + 1):
            if idx == len(values) or values[idx] != values[start]:
                state_idx_value = int(values[start]) if state_source == "state_idx" else None
                rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "state_idx": state_idx_value,
                        "state_rank": rank_by_idx.get(state_idx_value, ranks[start]) if state_idx_value is not None else ranks[start],
                        "state_label": labels[start],
                        "dwell": idx - start,
                    }
                )
                start = idx
    dwell_df = pd.DataFrame(rows)
    if max_dwell is None:
        if dwell_df.empty:
            max_dwell = 1
        else:
            session_lengths = work.groupby(["subject", session_col], observed=True).size()
            max_dwell = int(max(1, session_lengths.max()))

    matrix_by_subject = dict(transition_matrices or {})
    if views:
        for subject, view in views.items():
            matrix = _transition_matrix_from_view(view)
            if matrix is not None:
                matrix_by_subject[subject] = matrix

    summary_rows = []
    y_max = 0.0
    for state_label in state_order:
        state_rows = dwell_df[dwell_df["state_label"].astype(str) == state_label]
        if state_rows.empty:
            continue
        state_idx_values = state_rows["state_idx"].dropna().astype(int).unique().tolist() if "state_idx" in state_rows.columns else []
        state_idx = state_idx_values[0] if state_idx_values else None
        dwell_lengths = state_rows["dwell"].to_numpy(dtype=int)
        self_probs = []
        if state_idx is not None:
            for subject in pd.unique(state_rows["subject"]):
                matrix = matrix_by_subject.get(subject)
                if matrix is not None and state_idx < matrix.shape[0]:
                    n_runs = int((state_rows["subject"] == subject).sum())
                    self_probs.extend([float(matrix[state_idx, state_idx])] * max(1, n_runs))
        self_prob = float(np.mean(self_probs)) if self_probs else None
        centers, predicted, empirical, yerr = _binned_dwell_summary(
            dwell_lengths,
            max_dwell=int(max_dwell),
            bin_width=int(bin_width),
            ci_level=float(ci_level),
            self_prob=self_prob,
        )
        for idx, center in enumerate(centers):
            pred_value = float(predicted[idx]) if predicted is not None else np.nan
            y_max = max(y_max, pred_value if np.isfinite(pred_value) else 0.0, float(empirical[idx] + yerr[1, idx]))
            summary_rows.append(
                {
                    "state_label": state_label,
                    "state_idx": state_idx,
                    "bin_center": float(center),
                    "predicted": pred_value,
                    "empirical": float(empirical[idx]),
                    "yerr_low": float(yerr[0, idx]),
                    "yerr_high": float(yerr[1, idx]),
                    "n_runs": int(dwell_lengths.size),
                }
            )
    return {
        "dwell_df": dwell_df,
        "dwell_summary_df": pd.DataFrame(summary_rows),
        "state_order": state_order,
        "max_dwell": int(max_dwell),
        "y_max": float(y_max if y_max > 0 else 1.0),
        "title": title,
    }


def build_state_posterior_count_payload(
    trial_df,
    *,
    title: str = "Posterior count distribution",
) -> dict:
    df = _as_pandas_df(trial_df, name="trial_df")
    _require_columns(df, ["subject", "state_label"], name="trial_df")
    pcols = _posterior_cols(df)
    if not pcols:
        raise ValueError("trial_df must contain posterior columns named p_state_0, p_state_1, ...")
    state_order, label_by_idx, rank_by_idx = _state_meta_from_trial_df(df)
    count_df = (
        df.groupby(["subject", "state_label"], observed=True)
        .size()
        .rename("n_trials")
        .reset_index()
    )
    posterior_df = df.melt(
        id_vars=["subject"],
        value_vars=pcols,
        var_name="state_col",
        value_name="probability",
    )
    posterior_df["state_idx"] = posterior_df["state_col"].str.rsplit("_", n=1).str[-1].astype(int)
    posterior_df["state_label"] = posterior_df["state_idx"].map(label_by_idx).fillna(posterior_df["state_col"])
    posterior_df["state_rank"] = posterior_df["state_idx"].map(rank_by_idx).fillna(posterior_df["state_idx"]).astype(int)
    return {
        "count_df": count_df,
        "posterior_df": posterior_df,
        "state_order": state_order,
        "bins": 20,
        "title": title,
    }


def build_session_deepdive_payload(
    trial_df,
    *,
    subject,
    session,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    trace_cols: list[str] | None = None,
    engaged_window: int = 20,
    engaged_trace_mode: str = "rolling",
    rolling_accuracy_window: int = 20,
    chance_level: float | None = None,
    num_classes: int | None = None,
    title: str | None = None,
) -> dict:
    """Prepare one session for ``plot_session_deepdive``."""
    df = _as_pandas_df(trial_df, name="trial_df")
    _require_columns(df, ["subject", session_col, sort_col], name="trial_df")
    pcols = _posterior_cols(df)
    if not pcols:
        raise ValueError("trial_df must contain posterior columns named p_state_0, p_state_1, ...")
    state_order, label_by_idx, rank_by_idx = _state_meta_from_trial_df(df)
    engaged_idx = _infer_engaged_state_idx(label_by_idx, rank_by_idx)
    work = df[(df["subject"].astype(str) == str(subject)) & (df[session_col] == session)].copy()
    if work.empty:
        raise ValueError(f"No rows found for subject={subject!r}, session={session!r}.")
    work = work.sort_values(sort_col).copy()
    work["trial_in_session"] = range(len(work))
    posterior_df = work.melt(
        id_vars=["trial_in_session"],
        value_vars=pcols,
        var_name="state_col",
        value_name="probability",
    )
    posterior_df["state_idx"] = posterior_df["state_col"].str.rsplit("_", n=1).str[-1].astype(int)
    posterior_df["state_label"] = posterior_df["state_idx"].map(label_by_idx).fillna(posterior_df["state_col"])
    posterior_df["state_rank"] = posterior_df["state_idx"].map(rank_by_idx).fillna(posterior_df["state_idx"]).astype(int)

    change_trials = []
    if "state_idx" in work.columns:
        values = work["state_idx"].to_numpy()
        change_trials = (np.flatnonzero(values[1:] != values[:-1]) + 1).astype(int).tolist()
    if trace_cols is None:
        default_trace_cols = ["A_plus", "A_minus", "A_L", "A_C", "A_R"]
        trace_cols = [column for column in default_trace_cols if column in work.columns]
    if num_classes is None and "response" in work.columns:
        response_vals = pd.to_numeric(work["response"], errors="coerce").dropna()
        if not response_vals.empty:
            num_classes = int(response_vals.max()) + 1
    return {
        "trial_df": work,
        "posterior_df": posterior_df,
        "state_order": state_order,
        "change_trials": change_trials,
        "trace_cols": list(trace_cols or []),
        "engaged_state_label": label_by_idx.get(engaged_idx),
        "engaged_window": int(engaged_window),
        "engaged_trace_mode": engaged_trace_mode,
        "rolling_accuracy_window": int(rolling_accuracy_window),
        "chance_level": chance_level,
        "num_classes": num_classes,
        "title": title or f"Subject {subject} - session {session}  ({len(work)} trials, {len(change_trials)} state changes)",
    }
