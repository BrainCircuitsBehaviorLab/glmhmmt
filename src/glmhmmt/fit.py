"""Array-first, I/O-free fitting for GLM-HMM and GLM-HMM-T models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from glmhmmt._fit_utils import fit_best_restart, restore_trial_order, score_split
from glmhmmt.model import SoftmaxGLMHMM, normalize_frozen_emissions

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class CVFoldResult:
    """Generalization metrics from one session-level cross-validation fold."""

    fold_index: int
    training_session_ids: np.ndarray
    test_session_ids: np.ndarray
    training_trial_count: int
    test_trial_count: int
    training_log_likelihood_per_trial: float
    training_accuracy: float
    test_log_likelihood_per_trial: float
    test_accuracy: float
    best_restart: int
    final_training_objective: float


@dataclass
class FitResult:
    """Fitted model, aligned trial-level outputs, and scientific fit metrics."""

    model: SoftmaxGLMHMM
    fitted_params: Any
    log_probability: np.ndarray
    best_restart: int

    y: np.ndarray
    X: np.ndarray
    U: np.ndarray
    session_ids: np.ndarray

    emission_names: list[str]
    transition_names: list[str]

    smoothed_state_probs: np.ndarray
    predictive_state_probs: np.ndarray
    choice_probs: np.ndarray

    raw_log_likelihood: float
    log_likelihood_per_trial: float
    accuracy: float
    objective_log_probability: float

    num_states: int
    num_classes: int
    baseline_class_idx: int
    frozen_emissions: dict[int, dict[str, float]]

    cv_metrics: list[CVFoldResult]


@dataclass
class _NumericalFit:
    model: SoftmaxGLMHMM
    fitted_params: Any
    log_probability: np.ndarray
    best_restart: int
    smoothed_state_probs: np.ndarray
    predictive_state_probs: np.ndarray
    choice_probs: np.ndarray
    raw_log_likelihood: float
    log_likelihood_per_trial: float
    accuracy: float


def _integer_parameter(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}.")
    return value


def _validate_feature_names(
    names: list[str] | None,
    width: int,
    *,
    name: str,
    default_prefix: str,
) -> list[str]:
    if names is None:
        return [f"{default_prefix}_{index}" for index in range(width)]
    if not isinstance(names, list):
        raise ValueError(f"{name} must be a list of feature names or None.")
    resolved = list(names)
    if len(resolved) != width:
        raise ValueError(
            f"{name} must contain exactly {width} names to match the matrix width; "
            f"got {len(resolved)}."
        )
    if not all(isinstance(feature_name, str) and feature_name for feature_name in resolved):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate feature names.")
    return resolved


def _ordered_unique_sessions(session_ids: np.ndarray) -> np.ndarray:
    try:
        _, first_indices = np.unique(session_ids, return_index=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "session_ids must contain mutually comparable scalar identifiers."
        ) from exc
    return session_ids[np.sort(first_indices)]


def _validate_inputs(
    y: np.ndarray,
    X: np.ndarray,
    U: np.ndarray | None,
    session_ids: np.ndarray,
    *,
    num_states: int,
    num_classes: int,
    emission_names: list[str] | None,
    transition_names: list[str] | None,
    baseline_class_idx: int,
    frozen_emissions: dict[int, dict[str, float]] | None,
    num_iters: int,
    m_step_num_iters: int,
    n_restarts: int,
    stickiness: float,
    seed: int,
    cv_folds: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    dict[int, dict[str, float]],
    np.ndarray,
]:
    num_states = _integer_parameter("num_states", num_states, minimum=1)
    num_classes = _integer_parameter("num_classes", num_classes, minimum=2)
    _integer_parameter("num_iters", num_iters, minimum=1)
    _integer_parameter("m_step_num_iters", m_step_num_iters, minimum=1)
    _integer_parameter("n_restarts", n_restarts, minimum=1)
    _integer_parameter("seed", seed, minimum=0)
    cv_folds = _integer_parameter("cv_folds", cv_folds, minimum=0)
    if cv_folds == 1:
        raise ValueError("cv_folds must be 0 (disabled) or at least 2; got 1.")
    if not np.isfinite(stickiness):
        raise ValueError(f"stickiness must be finite; got {stickiness!r}.")
    if isinstance(baseline_class_idx, bool) or not isinstance(baseline_class_idx, Integral):
        raise ValueError(
            f"baseline_class_idx must be an integer; got {baseline_class_idx!r}."
        )
    baseline_class_idx = int(baseline_class_idx)
    if not 0 <= baseline_class_idx < num_classes:
        raise ValueError(
            f"baseline_class_idx must be in [0, {num_classes - 1}]; "
            f"got {baseline_class_idx}."
        )

    y_array = np.asarray(y)
    X_array = np.asarray(X)
    session_array = np.asarray(session_ids)
    if y_array.ndim != 1:
        raise ValueError(
            f"y must be one-dimensional with shape (T,); got {y_array.shape}."
        )
    if X_array.ndim != 2:
        raise ValueError(
            f"X must be two-dimensional with shape (T, P); got {X_array.shape}."
        )
    if session_array.ndim != 1:
        raise ValueError(
            "session_ids must be one-dimensional with shape (T,); "
            f"got {session_array.shape}."
        )
    trial_count = y_array.shape[0]
    if trial_count == 0:
        raise ValueError("At least one trial is required for fitting.")

    if U is None:
        U_array = np.empty((trial_count, 0), dtype=X_array.dtype)
    else:
        U_array = np.asarray(U)
        if U_array.ndim != 2:
            raise ValueError(
                f"U must be two-dimensional with shape (T, Q); got {U_array.shape}."
            )

    first_dimensions = {
        "y": y_array.shape[0],
        "X": X_array.shape[0],
        "U": U_array.shape[0],
        "session_ids": session_array.shape[0],
    }
    if len(set(first_dimensions.values())) != 1:
        detail = ", ".join(f"{name}={size}" for name, size in first_dimensions.items())
        raise ValueError(
            f"Input arrays must have the same number of trials; got {detail}."
        )

    if not np.issubdtype(y_array.dtype, np.integer) or np.issubdtype(
        y_array.dtype, np.bool_
    ):
        raise ValueError("y must contain integer-encoded choices.")
    invalid_choices = (y_array < 0) | (y_array >= num_classes)
    if np.any(invalid_choices):
        invalid_values = np.unique(y_array[invalid_choices]).tolist()
        raise ValueError(
            f"y must contain class indices in [0, {num_classes - 1}]; "
            f"found {invalid_values}."
        )

    for matrix_name, matrix in (("X", X_array), ("U", U_array)):
        if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
            matrix.dtype, np.complexfloating
        ):
            raise ValueError(f"{matrix_name} must contain real numeric regressors.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{matrix_name} contains non-finite regressor values.")

    if session_array.dtype.kind in "fc" and not np.all(np.isfinite(session_array)):
        raise ValueError("session_ids contains non-finite identifiers.")
    if session_array.dtype.kind == "O":
        for session_id in session_array:
            if session_id is None or (
                isinstance(session_id, (float, np.floating))
                and not np.isfinite(session_id)
            ):
                raise ValueError("session_ids contains missing or non-finite identifiers.")

    ordered_sessions = _ordered_unique_sessions(session_array)
    session_counts = np.asarray(
        [np.count_nonzero(session_array == session_id) for session_id in ordered_sessions]
    )
    short_sessions = ordered_sessions[session_counts < 2]
    if short_sessions.size:
        raise ValueError(
            "Every session must contain at least 2 trials; sessions with fewer trials: "
            f"{short_sessions.tolist()}."
        )
    if cv_folds > ordered_sessions.shape[0]:
        raise ValueError(
            f"cv_folds={cv_folds} exceeds the number of available sessions "
            f"({ordered_sessions.shape[0]})."
        )

    resolved_emission_names = _validate_feature_names(
        emission_names,
        X_array.shape[1],
        name="emission_names",
        default_prefix="emission",
    )
    resolved_transition_names = _validate_feature_names(
        transition_names,
        U_array.shape[1],
        name="transition_names",
        default_prefix="transition",
    )
    if frozen_emissions is not None and not isinstance(frozen_emissions, dict):
        raise ValueError("frozen_emissions must be a dictionary or None.")
    frozen = normalize_frozen_emissions(frozen_emissions)
    for state_index, feature_map in frozen.items():
        if not 0 <= state_index < num_states:
            raise ValueError(
                f"Frozen emission state index {state_index} is outside "
                f"[0, {num_states - 1}]."
            )
        unknown_features = sorted(set(feature_map) - set(resolved_emission_names))
        if unknown_features:
            raise ValueError(
                "Frozen emission features must appear in emission_names; unknown features: "
                f"{unknown_features}."
            )

    return (
        y_array,
        X_array,
        U_array,
        session_array,
        resolved_emission_names,
        resolved_transition_names,
        frozen,
        ordered_sessions,
    )


def _fit_arrays(
    y: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    session_ids: np.ndarray,
    *,
    num_states: int,
    num_classes: int,
    emission_names: list[str],
    baseline_class_idx: int,
    frozen_emissions: dict[int, dict[str, float]],
    num_iters: int,
    m_step_num_iters: int,
    n_restarts: int,
    stickiness: float,
    seed: int,
    verbose: bool,
    progress_callback: ProgressCallback | None,
    progress_payload: dict[str, Any],
) -> _NumericalFit:
    model = SoftmaxGLMHMM(
        num_states=num_states,
        num_classes=num_classes,
        emission_input_dim=X.shape[1],
        transition_input_dim=U.shape[1],
        m_step_num_iters=m_step_num_iters,
        transition_matrix_stickiness=stickiness,
        frozen_emissions=frozen_emissions or None,
        emission_feature_names=emission_names,
        baseline_class_idx=baseline_class_idx,
    )
    y_jax = jnp.asarray(y)
    inputs = jnp.concatenate([jnp.asarray(X), jnp.asarray(U)], axis=1)
    fitted_params, log_probability, best_restart = fit_best_restart(
        model,
        n_restarts=n_restarts,
        base_seed=seed,
        fit_once=lambda params, properties: model.fit_em_multisession(
            params=params,
            props=properties,
            emissions=y_jax,
            inputs=inputs,
            session_ids=session_ids,
            num_iters=num_iters,
            verbose=verbose,
        ),
        failure_message=(
            "GLM-HMM fitting did not produce finite parameters in any restart. "
            "Check regressor scaling and fitting settings."
        ),
        progress_callback=progress_callback,
        progress_payload=progress_payload,
    )
    grouped_smoothed = model.smoother_multisession(
        fitted_params, y_jax, inputs, session_ids=session_ids
    )
    grouped_predictive = model.predict_state_probs_multisession(
        fitted_params, y_jax, inputs, session_ids=session_ids
    )
    split_metrics = score_split(model, fitted_params, y_jax, inputs, session_ids)
    return _NumericalFit(
        model=model,
        fitted_params=fitted_params,
        log_probability=np.asarray(log_probability),
        best_restart=best_restart,
        smoothed_state_probs=restore_trial_order(grouped_smoothed, session_ids),
        predictive_state_probs=restore_trial_order(grouped_predictive, session_ids),
        choice_probs=np.asarray(split_metrics["p_pred"]),
        raw_log_likelihood=float(split_metrics["raw_ll"]),
        log_likelihood_per_trial=float(split_metrics["ll_per_trial"]),
        accuracy=float(split_metrics["acc"]),
    )


def _session_mask(session_ids: np.ndarray, selected_sessions: np.ndarray) -> np.ndarray:
    mask = np.zeros(session_ids.shape[0], dtype=bool)
    for session_id in selected_sessions:
        mask |= session_ids == session_id
    return mask


def fit_glmhmm(
    y: np.ndarray,
    X: np.ndarray,
    session_ids: np.ndarray,
    *,
    U: np.ndarray | None = None,
    num_states: int,
    num_classes: int,
    emission_names: list[str] | None = None,
    transition_names: list[str] | None = None,
    baseline_class_idx: int = 0,
    frozen_emissions: dict[int, dict[str, float]] | None = None,
    num_iters: int = 50,
    m_step_num_iters: int = 100,
    n_restarts: int = 5,
    stickiness: float = 10.0,
    seed: int = 0,
    cv_folds: int = 0,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit a GLM-HMM or GLM-HMM-T from caller-prepared numerical arrays.

    Choices must already be integer encoded. ``X`` and optional ``U`` must
    already contain the desired emission and transition regressors. No data is
    loaded, transformed, filtered, or persisted by this function.
    """
    (
        y_array,
        X_array,
        U_array,
        session_array,
        resolved_emission_names,
        resolved_transition_names,
        frozen,
        ordered_sessions,
    ) = _validate_inputs(
        y,
        X,
        U,
        session_ids,
        num_states=num_states,
        num_classes=num_classes,
        emission_names=emission_names,
        transition_names=transition_names,
        baseline_class_idx=baseline_class_idx,
        frozen_emissions=frozen_emissions,
        num_iters=num_iters,
        m_step_num_iters=m_step_num_iters,
        n_restarts=n_restarts,
        stickiness=stickiness,
        seed=seed,
        cv_folds=cv_folds,
    )

    cv_metrics: list[CVFoldResult] = []
    if cv_folds >= 2:
        shuffled_sessions = np.random.default_rng(seed).permutation(ordered_sessions)
        session_folds = np.array_split(shuffled_sessions, cv_folds)
        for fold_index, test_sessions in enumerate(session_folds):
            test_mask = _session_mask(session_array, test_sessions)
            training_mask = ~test_mask
            training_sessions = _ordered_unique_sessions(session_array[training_mask])
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "cv_fold_start",
                        "fold_index": fold_index,
                        "fold_total": cv_folds,
                    }
                )
            fold_fit = _fit_arrays(
                y_array[training_mask],
                X_array[training_mask],
                U_array[training_mask],
                session_array[training_mask],
                num_states=num_states,
                num_classes=num_classes,
                emission_names=resolved_emission_names,
                baseline_class_idx=baseline_class_idx,
                frozen_emissions=frozen,
                num_iters=num_iters,
                m_step_num_iters=m_step_num_iters,
                n_restarts=n_restarts,
                stickiness=stickiness,
                seed=seed + (fold_index + 1) * 10_000,
                verbose=verbose,
                progress_callback=progress_callback,
                progress_payload={
                    "fit_phase": "cross_validation",
                    "fold_index": fold_index,
                },
            )
            test_inputs = jnp.concatenate(
                [jnp.asarray(X_array[test_mask]), jnp.asarray(U_array[test_mask])], axis=1
            )
            test_score = score_split(
                fold_fit.model,
                fold_fit.fitted_params,
                jnp.asarray(y_array[test_mask]),
                test_inputs,
                session_array[test_mask],
            )
            cv_metrics.append(
                CVFoldResult(
                    fold_index=fold_index,
                    training_session_ids=np.asarray(training_sessions).copy(),
                    test_session_ids=np.asarray(test_sessions).copy(),
                    training_trial_count=int(np.count_nonzero(training_mask)),
                    test_trial_count=int(test_score["T"]),
                    training_log_likelihood_per_trial=fold_fit.log_likelihood_per_trial,
                    training_accuracy=fold_fit.accuracy,
                    test_log_likelihood_per_trial=float(test_score["ll_per_trial"]),
                    test_accuracy=float(test_score["acc"]),
                    best_restart=fold_fit.best_restart,
                    final_training_objective=float(fold_fit.log_probability[-1]),
                )
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "cv_fold_complete",
                        "fold_index": fold_index,
                        "fold_total": cv_folds,
                    }
                )

    full_fit = _fit_arrays(
        y_array,
        X_array,
        U_array,
        session_array,
        num_states=num_states,
        num_classes=num_classes,
        emission_names=resolved_emission_names,
        baseline_class_idx=baseline_class_idx,
        frozen_emissions=frozen,
        num_iters=num_iters,
        m_step_num_iters=m_step_num_iters,
        n_restarts=n_restarts,
        stickiness=stickiness,
        seed=seed,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_payload={"fit_phase": "full_data"},
    )
    return FitResult(
        model=full_fit.model,
        fitted_params=full_fit.fitted_params,
        log_probability=full_fit.log_probability,
        best_restart=full_fit.best_restart,
        y=y_array,
        X=X_array,
        U=U_array,
        session_ids=session_array,
        emission_names=resolved_emission_names,
        transition_names=resolved_transition_names,
        smoothed_state_probs=full_fit.smoothed_state_probs,
        predictive_state_probs=full_fit.predictive_state_probs,
        choice_probs=full_fit.choice_probs,
        raw_log_likelihood=full_fit.raw_log_likelihood,
        log_likelihood_per_trial=full_fit.log_likelihood_per_trial,
        accuracy=full_fit.accuracy,
        objective_log_probability=float(full_fit.log_probability[-1]),
        num_states=int(num_states),
        num_classes=int(num_classes),
        baseline_class_idx=int(baseline_class_idx),
        frozen_emissions=frozen,
        cv_metrics=cv_metrics,
    )


# Concise public alias requested for callers that fit either model variant.
fit_hmm = fit_glmhmm


__all__ = ["CVFoldResult", "FitResult", "fit_glmhmm", "fit_hmm"]
