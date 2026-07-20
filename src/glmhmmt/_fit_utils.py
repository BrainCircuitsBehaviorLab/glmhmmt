"""Small numerical helpers shared by public and legacy fitting interfaces."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np


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
    """Run repeated seeded fits and retain the best final objective."""
    best_log_probability = -np.inf
    best_params = None
    best_log_probabilities = None
    best_restart = -1
    payload = progress_payload or {}

    for restart_index in range(int(n_restarts)):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_start",
                    "restart_index": restart_index + 1,
                    "restart_total": int(n_restarts),
                    **payload,
                }
            )

        params, properties = model.initialize(
            key=jr.PRNGKey(int(base_seed) + restart_index)
        )
        fitted_params, log_probabilities = fit_once(params, properties)
        log_probabilities = np.asarray(log_probabilities)
        final_log_probability = float(log_probabilities[-1])

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_complete",
                    "restart_index": restart_index + 1,
                    "restart_total": int(n_restarts),
                    "log_prob": final_log_probability,
                    **payload,
                }
            )

        if (
            np.isfinite(final_log_probability)
            and final_log_probability > best_log_probability
        ):
            best_log_probability = final_log_probability
            best_params = fitted_params
            best_log_probabilities = log_probabilities
            best_restart = restart_index + 1

    if best_params is None or best_log_probabilities is None:
        raise ValueError(failure_message)
    return best_params, best_log_probabilities, int(best_restart)


def raw_loglik_multisession(model, params, emissions, inputs, session_ids) -> float:
    """Return the summed data log likelihood without a parameter prior."""
    if int(np.asarray(emissions).shape[0]) == 0:
        return 0.0
    sessions = model._split_by_session(emissions, inputs, session_ids)
    if not sessions:
        return 0.0
    padded_emissions, padded_inputs, _ = model._pad_sessions(sessions)
    _batch_stats, log_likelihoods = model._batched_e_step_jit(
        params, padded_emissions, padded_inputs
    )
    return float(jnp.sum(log_likelihoods))


def _session_grouped_trial_indices(session_ids: np.ndarray) -> np.ndarray:
    """Return trial indices in the ordering used by model multisession methods."""
    session_ids = np.asarray(session_ids)
    _, first_indices = np.unique(session_ids, return_index=True)
    ordered_sessions = session_ids[np.sort(first_indices)]
    return np.concatenate(
        [np.flatnonzero(session_ids == session_id) for session_id in ordered_sessions]
    )


def restore_trial_order(values, session_ids: np.ndarray) -> np.ndarray:
    """Undo the session-grouped ordering produced by multisession predictions."""
    values = np.asarray(values)
    grouped_indices = _session_grouped_trial_indices(session_ids)
    restored = np.empty_like(values)
    restored[grouped_indices] = values
    return restored


def score_split(model, params, emissions, inputs, session_ids) -> dict[str, Any]:
    """Evaluate one fitted model on a train or test split."""
    trial_count = int(np.asarray(emissions).shape[0])
    if trial_count == 0:
        return {
            "raw_ll": 0.0,
            "ll_per_trial": np.nan,
            "acc": np.nan,
            "T": 0,
            "p_pred": np.empty((0, getattr(model, "num_classes", 0))),
        }

    raw_log_likelihood = raw_loglik_multisession(
        model, params, emissions, inputs, session_ids
    )
    grouped_choice_probs = model.predict_choice_probs_multisession(
        params, emissions, inputs, session_ids=session_ids
    )
    choice_probs = restore_trial_order(grouped_choice_probs, session_ids)
    accuracy = float(
        np.mean(np.argmax(choice_probs, axis=1) == np.asarray(emissions))
    )
    return {
        "raw_ll": raw_log_likelihood,
        "ll_per_trial": raw_log_likelihood / trial_count,
        "acc": accuracy,
        "T": trial_count,
        "p_pred": choice_probs,
    }
