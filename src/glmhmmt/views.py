"""views.py — SubjectFitView dataclass and pipeline-wide constants.

This module is the **single source of truth** for:

* :class:`SubjectFitView` — a task-agnostic, per-subject representation of a
  fitted GLM-HMM model.
* :data:`_LABEL_RANK` — canonical mapping from semantic state label to integer
  rank (Engaged = 0, Disengaged variants = 1, 2, …).
* :data:`_STATE_HEX` — default rank-indexed hex palette loaded from
  ``config.toml``.
* :func:`get_state_palette` — choose a K-specific palette from
  ``config.toml`` when one is defined, else fall back to :data:`_STATE_HEX`.
* :func:`build_views` — factory that combines ``arrays_store`` with a
  :class:`~tasks.TaskAdapter` to produce a ``{subject: SubjectFitView}`` dict.

Design contract
---------------
State labelling is **not** performed here.  The labelling strategy (e.g.
engagement scoring) lives on the task adapter and is applied when building a
view via :func:`build_views`, so this module stays task-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from glmhmmt.runtime import load_app_config

# ── palette & rank constants ──────────────────────────────────────────────────

_cfg = load_app_config()

_STATE_HEX: list[str] = _cfg.get("palettes", {}).get(
    "states_hex",
    [
        "#66A61E",
        "#7f7f7f",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#E6AB02",
    ],
)

_STATE_HEX_BY_K: dict[int, list[str]] = {
    2: list(_cfg.get("palettes", {}).get("states_hex_k2", [])),
    3: list(_cfg.get("palettes", {}).get("states_hex_k3", [])),
}

_LABEL_RANK: dict[str, int] = {
    "Engaged": 0,
    "Disengaged": 1,
    "Biased L": 1,
    "Biased R": 2,
    "Disengaged L": 1,
    "Disengaged R": 2,
    "Disengaged C": 3,
    **{f"Disengaged {i}": i for i in range(1, 10)},
}


def get_state_palette(K: Optional[int] = None) -> list[str]:
    """Return the configured state palette for ``K`` or the default palette."""
    if K is not None:
        palette = _STATE_HEX_BY_K.get(int(K), [])
        if len(palette) >= int(K):
            return list(palette)
    return list(_STATE_HEX)


def get_state_rank(label: str, fallback_idx: int = 0) -> int:
    """Return the canonical rank for a state label."""
    return _LABEL_RANK.get(label, int(fallback_idx))


def get_state_color(
    label: str,
    fallback_idx: int = 0,
    *,
    K: Optional[int] = None,
    palette: Optional[list[str]] = None,
) -> str:
    """Return the configured colour for a state label."""
    resolved_palette = list(palette) if palette is not None else get_state_palette(K)
    rank = get_state_rank(label, fallback_idx=fallback_idx)
    return resolved_palette[rank % len(resolved_palette)]


def build_state_palette(
    state_labels_per_subj: Mapping[object, Mapping[int, str]],
    K: Optional[int] = None,
) -> tuple[dict[str, str], list[str]]:
    """Return a rank-ordered ``(palette_dict, hue_order)`` pair."""
    seen: dict[str, int] = {}
    for labels_by_state in state_labels_per_subj.values():
        for state_idx, label in labels_by_state.items():
            if label not in seen:
                seen[label] = get_state_rank(label, fallback_idx=int(state_idx))

    ordered = sorted(seen, key=lambda label: seen[label])
    resolved_palette = get_state_palette(K if K is not None else len(ordered) or None)
    palette = {
        label: get_state_color(label, seen[label], palette=resolved_palette)
        for label in ordered
    }
    return palette, ordered


# ── dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class SubjectFitView:
    """Task-agnostic view of a fitted GLM-HMM model for one subject.

    All arrays are plain NumPy (never JAX) so they can be pickled, serialised,
    and used in plotting without any JAX dependency.

    Parameters
    ----------
    subject :
        Subject identifier string.
    K :
        Number of hidden states.
    smoothed_probs :
        HMM posterior  γ(z_t = k | y_{1:T}),  shape ``(T, K)``.
        **This is the canonical quantity for posterior plots.**  It is saved
        directly from ``model.smoother_multisession`` and loaded verbatim —
        never recomputed.
    emission_weights :
        Softmax regression weight tensor,  shape ``(K, C-1, F)``.
    X :
        Emission design matrix,  shape ``(T, F)``.
    y :
        Integer choices,  shape ``(T,)``.
    feat_names :
        Emission feature names, length ``F``.
    state_name_by_idx :
        Mapping ``{state_int_k: label_str}`` — e.g.
        ``{0: "Engaged", 1: "Disengaged"}``.
    state_idx_order :
        State indices sorted by engagement rank (Engaged first).
    state_rank_by_idx :
        Mapping ``{state_int_k: rank_int}`` where rank 0 = Engaged.

    Optional
    --------
    p_pred :
        Marginal model predictions shape ``(T, C)``.  Pre-computed during
        fitting; if absent, :func:`~glmhmmt.postprocess.build_trial_df` will
        recompute it.
    predictive_state_probs :
        One-step-ahead predictive state probabilities shape ``(T, K)``,
        i.e. ``p(z_t = k | y_{1:t-1}, x_{1:t})``. Unlike ``smoothed_probs``,
        these do not use the current choice ``y_t`` and are therefore the
        right quantity for empirical state-conditioned choice summaries.
    initial_probs :
        Initial state distribution shape ``(K,)``.
    transition_matrix :
        Homogeneous transition matrix shape ``(K, K)`` when available.
    transition_bias :
        Input-driven transition bias shape ``(K, K)`` — GLM-HMM-t only.
    transition_weights :
        Input-driven transition weight tensor, shape ``(K, K, D)`` —
        GLM-HMM-t only.
    U :
        Transition design matrix, shape ``(T, D)`` — GLM-HMM-t only.
    U_cols :
        Transition feature names, length ``D``.
    """

    subject: str
    K: int
    smoothed_probs: np.ndarray  # (T, K)
    emission_weights: np.ndarray  # (K, C-1, F)
    X: np.ndarray  # (T, F)
    y: np.ndarray  # (T,)
    feat_names: list[str]
    state_name_by_idx: dict[int, str]
    state_idx_order: list[int]
    state_rank_by_idx: dict[int, int]
    predictive_state_probs: np.ndarray  # (T, K)
    lapse_rates: Optional[np.ndarray] = None
    lapse_mode: str = "none"
    lapse_labels: tuple[str, ...] = field(default_factory=tuple)
    # optional
    p_pred: Optional[np.ndarray] = None  # (T, C)
    initial_probs: Optional[np.ndarray] = None  # (K,)
    transition_matrix: Optional[np.ndarray] = None  # (K, K) or (T-1, K, K)
    transition_bias: Optional[np.ndarray] = None  # (K, K)
    transition_weights: Optional[np.ndarray] = None  # (K, K, D)
    U: Optional[np.ndarray] = None  # (T, D)
    U_cols: list[str] = field(default_factory=list)

    # ── derived helpers ───────────────────────────────────────────────────────

    @property
    def T(self) -> int:
        """Number of trials."""
        return int(self.smoothed_probs.shape[0])

    @property
    def num_classes(self) -> int:
        """Number of choice classes (C)."""
        return self.emission_weights.shape[1] + 1  # C-1 rows + reference class

    def map_states(self) -> np.ndarray:
        """Return MAP state assignment ``(T,)`` = argmax(smoothed_probs, axis=1)."""
        return np.argmax(self.smoothed_probs, axis=1).astype(int)

    def engaged_k(self) -> int:
        """Index of the Engaged state (rank 0)."""
        for k, rank in self.state_rank_by_idx.items():
            if rank == 0:
                return k
        return self.state_idx_order[0]

    def state_conditional_probs(self) -> np.ndarray:
        """Return per-trial emission class probabilities for every state.

        Shape is ``(T, K, C)`` where entry ``[t, k, c]`` equals
        ``P(y_t = c | z_t = k, x_t)`` evaluated on the observed design row
        ``X[t]``. This is the trial-matched model quantity needed when
        comparing posterior-weighted empirical summaries against model
        predictions within a latent state.
        """
        logits_ce = np.einsum("kcf,tf->tkc", self.emission_weights, self.X)  # (T, K, C-1)
        zeros = np.zeros((self.T, self.K, 1), dtype=logits_ce.dtype)
        logits = np.concatenate([logits_ce, zeros], axis=-1)  # (T, K, C)
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


# ── factory ───────────────────────────────────────────────────────────────────


def build_views(
    arrays_store: dict,
    adapter,
    K: int,
    subjects: list[str],
) -> dict[str, SubjectFitView]:
    """Build a ``{subject: SubjectFitView}`` dict from *arrays_store*.

    Parameters
    ----------
    arrays_store :
        Dict ``{subj: npz-dict}`` as loaded by the notebook I/O cell.
        Each entry must contain at least ``smoothed_probs``, ``emission_weights``,
        ``X``, ``y``, and ``X_cols``.
    adapter :
        Any :class:`~tasks.TaskAdapter` instance.  Used to apply the
        task-specific labelling strategy (``adapter.label_states``).
    K :
        Number of states.
    subjects :
        Subject IDs to include (those absent from *arrays_store* are silently
        skipped).

    Returns
    -------
    dict[str, SubjectFitView]
    """
    _selected = [s for s in subjects if s in arrays_store]
    if not _selected:
        return {}

    # State labelling is task-specific and performed here, outside the dataclass.
    # names dict is populated from the first available subject's X_cols.
    _names: dict = {}
    for s in _selected:
        _cols = arrays_store[s].get("X_cols")
        if _cols is not None:
            _names = {"X_cols": list(_cols)}
            break

    state_labels, state_order = adapter.label_states(arrays_store, _names, K, _selected)

    views: dict[str, SubjectFitView] = {}
    for subj in _selected:
        d = arrays_store[subj]
        feat_names = list(d.get("X_cols", []))
        _W = np.asarray(d["emission_weights"])
        feat_names = feat_names[: _W.shape[2]]
        slbls = state_labels.get(subj, {k: f"State {k}" for k in range(K)})
        sorder = state_order.get(subj, list(range(K)))

        # rank map: ordered position in sorder → LABEL_RANK (semantic)
        srank: dict[int, int] = {}
        for k in range(K):
            lbl = slbls.get(k, "")
            srank[k] = _LABEL_RANK.get(lbl, sorder.index(k) if k in sorder else k)

        if "predictive_state_probs" not in d:
            raise KeyError(
                f"Subject {subj!r}: missing 'predictive_state_probs' in fitted arrays. "
                "Re-run the fit/export pipeline with the updated code."
            )
        predictive_state_probs = np.asarray(d["predictive_state_probs"])
        if predictive_state_probs.shape != np.asarray(d["smoothed_probs"]).shape:
            raise ValueError(
                f"Subject {subj!r}: predictive_state_probs has shape {predictive_state_probs.shape}, "
                f"expected {np.asarray(d['smoothed_probs']).shape}."
            )
        transition_weights = np.asarray(d["transition_weights"]) if "transition_weights" in d else None
        u_cols = list(d.get("U_cols", []))
        if transition_weights is not None:
            u_cols = u_cols[: transition_weights.shape[2]]

        views[subj] = SubjectFitView(
            subject=subj,
            K=K,
            smoothed_probs=np.asarray(d["smoothed_probs"]),
            emission_weights=_W,
            X=np.asarray(d["X"]),
            y=np.asarray(d["y"]),
            feat_names=feat_names,
            state_name_by_idx={int(k): v for k, v in slbls.items()},
            state_idx_order=[int(k) for k in sorder],
            state_rank_by_idx={int(k): v for k, v in srank.items()},
            lapse_rates=np.asarray(d["lapse_rates"]) if "lapse_rates" in d else None,
            lapse_mode=str(d.get("lapse_mode", "none")),
            lapse_labels=tuple(np.asarray(d.get("lapse_labels", []), dtype=str).tolist()),
            p_pred=np.asarray(d["p_pred"]) if "p_pred" in d else None,
            predictive_state_probs=predictive_state_probs,
            initial_probs=np.asarray(d["initial_probs"]) if "initial_probs" in d else None,
            transition_matrix=np.asarray(d["transition_matrix"]) if "transition_matrix" in d else None,
            transition_bias=np.asarray(d["transition_bias"]) if "transition_bias" in d else None,
            transition_weights=transition_weights,
            U=np.asarray(d["U"]) if "U" in d else None,
            U_cols=u_cols,
        )

    return views
