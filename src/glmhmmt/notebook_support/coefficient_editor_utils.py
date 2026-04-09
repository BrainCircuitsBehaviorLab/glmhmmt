from __future__ import annotations

from copy import copy

import numpy as np
import polars as pl


def build_editor_payload(
    stored_weights: np.ndarray,
    *,
    choice_labels: list[str],
    stored_class_indices: list[int],
    reference_class_idx: int,
    display_reference_class_idx: int | None = None,
) -> dict:
    weights = np.asarray(stored_weights, dtype=float)
    num_classes = len(choice_labels)
    if len(stored_class_indices) != weights.shape[0]:
        raise ValueError("stored_class_indices must match the number of weight rows.")

    if display_reference_class_idx is None:
        display_reference_class_idx = reference_class_idx

    display_class_indices = [
        idx for idx in range(num_classes) if idx != display_reference_class_idx
    ]
    if len(display_class_indices) != weights.shape[0]:
        raise ValueError("Display and stored parametrizations must have the same number of explicit classes.")

    row_by_stored_class = {
        class_idx: weights[row_idx]
        for row_idx, class_idx in enumerate(stored_class_indices)
    }
    display_weights = []
    for class_idx in display_class_indices:
        if class_idx == reference_class_idx:
            row = np.zeros(weights.shape[1], dtype=float)
        else:
            row = row_by_stored_class[class_idx].copy()

        if display_reference_class_idx != reference_class_idx:
            if display_reference_class_idx in row_by_stored_class:
                row = row - row_by_stored_class[display_reference_class_idx]
            else:
                raise ValueError("Display reference must be either the stored reference or one stored explicit class.")
        display_weights.append(row)

    display_weights = np.asarray(display_weights, dtype=float)
    channel_labels = [choice_labels[idx] for idx in display_class_indices]

    if num_classes == 2:
        reference_label = choice_labels[display_reference_class_idx]
        return {
            "weights": display_weights,
            "channel_labels": channel_labels,
            "subtitle": f"{reference_label} is implied by the binary logit.",
            "explicit_class_indices": display_class_indices,
            "reference_class_idx": display_reference_class_idx,
            "stored_class_indices": list(stored_class_indices),
            "stored_reference_class_idx": reference_class_idx,
        }

    reference_label = choice_labels[display_reference_class_idx]

    return {
        "weights": display_weights,
        "channel_labels": channel_labels,
        "subtitle": f"Displayed with {reference_label} implicit.",
        "explicit_class_indices": display_class_indices,
        "reference_class_idx": display_reference_class_idx,
        "stored_class_indices": list(stored_class_indices),
        "stored_reference_class_idx": reference_class_idx,
    }


def stored_weights_from_editor_weights(
    editor_weights: np.ndarray,
    *,
    explicit_class_indices: list[int],
    reference_class_idx: int,
    stored_class_indices: list[int],
    stored_reference_class_idx: int,
) -> np.ndarray:
    """Map editor/display weights back to the stored explicit-class parametrization."""
    weights = np.asarray(editor_weights, dtype=float)
    if len(explicit_class_indices) != weights.shape[0]:
        raise ValueError("explicit_class_indices must match the number of editor weight rows.")
    if len(stored_class_indices) != weights.shape[0]:
        raise ValueError("stored_class_indices must match the number of stored weight rows.")

    explicit_class_indices = [int(idx) for idx in explicit_class_indices]
    stored_class_indices = [int(idx) for idx in stored_class_indices]
    reference_class_idx = int(reference_class_idx)
    stored_reference_class_idx = int(stored_reference_class_idx)

    if (
        explicit_class_indices == stored_class_indices
        and reference_class_idx == stored_reference_class_idx
    ):
        return weights.copy()

    row_by_display_class = {
        class_idx: weights[row_idx].copy()
        for row_idx, class_idx in enumerate(explicit_class_indices)
    }
    zeros = np.zeros(weights.shape[1], dtype=float)
    abs_by_class = {stored_reference_class_idx: zeros}

    if reference_class_idx == stored_reference_class_idx:
        for class_idx in stored_class_indices:
            if class_idx not in row_by_display_class:
                raise ValueError(f"Missing editor row for class {class_idx}.")
            abs_by_class[class_idx] = row_by_display_class[class_idx]
    else:
        if stored_reference_class_idx not in row_by_display_class:
            raise ValueError("Stored reference class must be explicit in the editor parametrization.")
        ref_shift = -row_by_display_class[stored_reference_class_idx]
        abs_by_class[reference_class_idx] = ref_shift
        for class_idx, row in row_by_display_class.items():
            if class_idx == stored_reference_class_idx:
                continue
            abs_by_class[class_idx] = row + ref_shift

    missing = [class_idx for class_idx in stored_class_indices if class_idx not in abs_by_class]
    if missing:
        raise ValueError(f"Could not reconstruct stored weights for classes {missing}.")

    return np.asarray([abs_by_class[class_idx] for class_idx in stored_class_indices], dtype=float)


def emission_probs_from_editor_weights(
    editor_weights: np.ndarray,
    X: np.ndarray,
    *,
    num_classes: int,
    explicit_class_indices: list[int],
    reference_class_idx: int,
) -> np.ndarray:
    weights = np.asarray(editor_weights, dtype=float)
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    if len(explicit_class_indices) != weights.shape[0]:
        raise ValueError("explicit_class_indices must match the number of weight rows.")

    logits = np.zeros((T, num_classes), dtype=float)
    for row_idx, class_idx in enumerate(explicit_class_indices):
        logits[:, class_idx] = X @ weights[row_idx]
    logits[:, reference_class_idx] = 0.0

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_state_tweak_to_view(
    view,
    *,
    state_idx: int,
    edited_weights: np.ndarray,
    explicit_class_indices: list[int],
    reference_class_idx: int,
    stored_class_indices: list[int],
    stored_reference_class_idx: int,
):
    """Return a shallow-copied view with one state's stored emission weights replaced."""
    tweaked_view = copy(view)
    emission_weights = np.asarray(view.emission_weights, dtype=float).copy()
    stored_weights = stored_weights_from_editor_weights(
        edited_weights,
        explicit_class_indices=list(explicit_class_indices),
        reference_class_idx=int(reference_class_idx),
        stored_class_indices=list(stored_class_indices),
        stored_reference_class_idx=int(stored_reference_class_idx),
    )
    state_idx = int(state_idx)
    if emission_weights[state_idx].shape != stored_weights.shape:
        raise ValueError("Edited weights do not match the stored emission weight shape.")
    emission_weights[state_idx] = stored_weights
    tweaked_view.emission_weights = emission_weights
    return tweaked_view


def apply_state_tweak_to_trial_df(
    trial_df_sub: pl.DataFrame,
    *,
    adapter,
    view,
    state_idx: int,
    edited_weights: np.ndarray,
    original_weights: np.ndarray,
    explicit_class_indices: list[int],
    reference_class_idx: int,
) -> pl.DataFrame:
    """Update a subject trial_df after editing one state's emission weights."""
    correct_class = np.asarray(adapter.get_correct_class(trial_df_sub), dtype=int)
    num_classes = int(view.num_classes)
    trial_idx = np.arange(view.T)
    valid_correct = (correct_class >= 0) & (correct_class < num_classes)

    probs_new = emission_probs_from_editor_weights(
        edited_weights,
        view.X,
        num_classes=num_classes,
        explicit_class_indices=explicit_class_indices,
        reference_class_idx=reference_class_idx,
    )
    probs_old = emission_probs_from_editor_weights(
        original_weights,
        view.X,
        num_classes=num_classes,
        explicit_class_indices=explicit_class_indices,
        reference_class_idx=reference_class_idx,
    )

    p_correct_new = np.zeros(view.T, dtype=float)
    p_correct_old = np.zeros(view.T, dtype=float)
    p_correct_new[valid_correct] = probs_new[trial_idx[valid_correct], correct_class[valid_correct]]
    p_correct_old[valid_correct] = probs_old[trial_idx[valid_correct], correct_class[valid_correct]]

    map_mask = trial_df_sub["state_idx"].to_numpy().astype(int) == int(state_idx)
    state_prob_col = f"p_state_{int(state_idx)}"
    if state_prob_col in trial_df_sub.columns:
        state_prob = trial_df_sub[state_prob_col].to_numpy().astype(float)
    else:
        state_prob = map_mask.astype(float)

    p_model_correct = trial_df_sub["p_model_correct"].to_numpy().astype(float).copy()
    p_model_correct[map_mask] = p_correct_new[map_mask]

    p_model_correct_marginal = (
        trial_df_sub["p_model_correct_marginal"].to_numpy().astype(float)
        + state_prob * (p_correct_new - p_correct_old)
    )

    updated_cols = [
        pl.Series("p_model_correct", np.clip(p_model_correct, 0.0, 1.0)),
        pl.Series(
            "p_model_correct_marginal",
            np.clip(p_model_correct_marginal, 0.0, 1.0),
        ),
    ]

    class_cols = list(adapter.probability_columns)
    for class_idx, class_col in enumerate(class_cols):
        current = trial_df_sub[class_col].to_numpy().astype(float)
        updated = current + state_prob * (probs_new[:, class_idx] - probs_old[:, class_idx])
        updated_cols.append(pl.Series(class_col, np.clip(updated, 0.0, 1.0)))

    return trial_df_sub.with_columns(updated_cols)
