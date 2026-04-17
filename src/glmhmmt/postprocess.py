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
