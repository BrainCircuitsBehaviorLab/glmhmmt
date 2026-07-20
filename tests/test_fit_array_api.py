from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from glmhmmt import FitResult, fit_glmhmm, fit_hmm
from glmhmmt._fit_utils import fit_best_restart, restore_trial_order
from glmhmmt.model import SoftmaxGLMHMM


def _synthetic_arrays():
    session_ids = np.repeat(np.arange(4), 4)
    stimulus = np.asarray(
        [-1.2, -0.4, 0.3, 1.1, -0.9, -0.2, 0.5, 1.3] * 2,
        dtype=np.float32,
    )
    X = np.column_stack([np.ones(stimulus.size), stimulus]).astype(np.float32)
    U = np.linspace(-1.0, 1.0, stimulus.size, dtype=np.float32)[:, None]
    y = (stimulus > 0).astype(np.int32)
    return y, X, U, session_ids


def _fit_kwargs():
    return {
        "num_states": 2,
        "num_classes": 2,
        "emission_names": ["bias", "stimulus"],
        "num_iters": 1,
        "m_step_num_iters": 1,
        "n_restarts": 1,
        "seed": 7,
        "verbose": False,
    }


@pytest.fixture(scope="module")
def standard_results():
    y, X, _U, session_ids = _synthetic_arrays()
    first = fit_glmhmm(y, X, session_ids, U=None, **_fit_kwargs())
    second = fit_hmm(y, X, session_ids, U=None, **_fit_kwargs())

    model = SoftmaxGLMHMM(
        num_states=2,
        num_classes=2,
        emission_input_dim=X.shape[1],
        transition_input_dim=0,
        m_step_num_iters=1,
        transition_matrix_stickiness=10.0,
        emission_feature_names=["bias", "stimulus"],
        baseline_class_idx=0,
    )
    direct_params, direct_log_probability, direct_restart = fit_best_restart(
        model,
        n_restarts=1,
        base_seed=7,
        fit_once=lambda params, properties: model.fit_em_multisession(
            params,
            properties,
            jnp.asarray(y),
            jnp.asarray(X),
            session_ids,
            num_iters=1,
            verbose=False,
        ),
        failure_message="direct fit failed",
    )
    return first, second, direct_params, direct_log_probability, direct_restart


@pytest.fixture(scope="module")
def transition_result():
    y, X, U, session_ids = _synthetic_arrays()
    return fit_glmhmm(
        y,
        X,
        session_ids,
        U=U,
        transition_names=["reward_history"],
        **_fit_kwargs(),
    )


@pytest.fixture(scope="module")
def cv_results():
    y, X, U, session_ids = _synthetic_arrays()
    events: list[dict] = []
    cross_validated = fit_glmhmm(
        y,
        X,
        session_ids,
        U=U,
        transition_names=["reward_history"],
        cv_folds=2,
        progress_callback=events.append,
        **_fit_kwargs(),
    )
    full_only = fit_glmhmm(
        y,
        X,
        session_ids,
        U=U,
        transition_names=["reward_history"],
        cv_folds=0,
        **_fit_kwargs(),
    )
    return cross_validated, full_only, events


def test_glmhmm_with_u_none_returns_standard_model(standard_results):
    result = standard_results[0]

    assert isinstance(result, FitResult)
    assert result.U.shape == (result.y.size, 0)
    assert result.model.transition_input_dim == 0
    assert result.cv_metrics == []
    assert result.smoothed_state_probs.shape == (result.y.size, 2)
    assert result.predictive_state_probs.shape == (result.y.size, 2)
    assert result.choice_probs.shape == (result.y.size, 2)


def test_glmhmmt_with_transition_regressors(transition_result):
    assert transition_result.U.shape[1] == 1
    assert transition_result.model.transition_input_dim == 1
    assert transition_result.transition_names == ["reward_history"]
    assert hasattr(transition_result.fitted_params.transitions, "weights")


def test_repeated_calls_with_same_seed_are_identical(standard_results):
    first, second = standard_results[:2]

    np.testing.assert_allclose(first.log_probability, second.log_probability)
    np.testing.assert_allclose(
        first.fitted_params.emissions.weights,
        second.fitted_params.emissions.weights,
    )
    np.testing.assert_allclose(first.choice_probs, second.choice_probs)
    assert first.best_restart == second.best_restart


def test_no_cv_returns_all_trials_and_aligned_names(standard_results):
    y, X, _U, session_ids = _synthetic_arrays()
    result = standard_results[0]

    np.testing.assert_array_equal(result.y, y)
    np.testing.assert_array_equal(result.X, X)
    np.testing.assert_array_equal(result.session_ids, session_ids)
    assert result.emission_names == ["bias", "stimulus"]
    assert result.transition_names == []
    assert result.cv_metrics == []


def test_session_kfold_has_no_leakage_and_returns_every_fold(cv_results):
    result = cv_results[0]
    expected_test_folds = np.array_split(
        np.random.default_rng(7).permutation(np.arange(4)), 2
    )

    assert [fold.fold_index for fold in result.cv_metrics] == [0, 1]
    assert len(result.cv_metrics) == 2
    all_test_sessions: list[int] = []
    for fold, expected_test_sessions in zip(
        result.cv_metrics, expected_test_folds, strict=True
    ):
        training_sessions = set(fold.training_session_ids.tolist())
        test_sessions = set(fold.test_session_ids.tolist())
        assert training_sessions.isdisjoint(test_sessions)
        np.testing.assert_array_equal(fold.test_session_ids, expected_test_sessions)
        assert fold.training_trial_count + fold.test_trial_count == result.y.size
        all_test_sessions.extend(fold.test_session_ids.tolist())
    assert sorted(all_test_sessions) == [0, 1, 2, 3]


def test_cross_validation_fits_a_separate_full_data_model(cv_results):
    cross_validated, full_only, events = cv_results

    np.testing.assert_array_equal(cross_validated.y, full_only.y)
    np.testing.assert_allclose(
        cross_validated.log_probability, full_only.log_probability
    )
    np.testing.assert_allclose(
        cross_validated.fitted_params.emissions.weights,
        full_only.fitted_params.emissions.weights,
    )
    restart_phases = [
        event.get("fit_phase")
        for event in events
        if event.get("event") == "restart_start"
    ]
    assert restart_phases == ["cross_validation", "cross_validation", "full_data"]


def test_frozen_emission_coefficients_resolve_by_feature_name():
    y, X, _U, session_ids = _synthetic_arrays()
    result = fit_glmhmm(
        y,
        X,
        session_ids,
        U=None,
        frozen_emissions={0: {"stimulus": 0.25}},
        **_fit_kwargs(),
    )

    np.testing.assert_allclose(
        np.asarray(result.fitted_params.emissions.weights)[0, :, 1],
        0.25,
    )
    assert result.frozen_emissions == {0: {"stimulus": 0.25}}


def test_no_cv_matches_previous_direct_numerical_fit(standard_results):
    result, _repeated, direct_params, direct_log_probability, direct_restart = standard_results

    np.testing.assert_allclose(result.log_probability, direct_log_probability)
    np.testing.assert_allclose(
        result.fitted_params.initial.probs, direct_params.initial.probs
    )
    np.testing.assert_allclose(
        result.fitted_params.transitions.transition_matrix,
        direct_params.transitions.transition_matrix,
    )
    np.testing.assert_allclose(
        result.fitted_params.emissions.weights, direct_params.emissions.weights
    )
    assert result.best_restart == direct_restart


def test_session_grouped_outputs_are_restored_to_original_trial_order():
    session_ids = np.asarray(["b", "a", "b", "a"])
    grouped_values = np.asarray([[10.0], [30.0], [20.0], [40.0]])

    restored = restore_trial_order(grouped_values, session_ids)

    np.testing.assert_array_equal(
        restored,
        np.asarray([[10.0], [20.0], [30.0], [40.0]]),
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"X": np.ones((15, 2))}, "same number of trials"),
        ({"X": np.ones(16)}, "X must be two-dimensional"),
        ({"U": np.ones(16)}, "U must be two-dimensional"),
        ({"y": np.asarray([0] * 15 + [2])}, "class indices"),
        ({"X": np.asarray([[np.nan, 0.0]] * 16)}, "non-finite"),
        ({"U": np.asarray([[np.inf]] * 16)}, "non-finite"),
        ({"emission_names": ["bias"]}, "exactly 2 names"),
        ({"transition_names": []}, "exactly 1 names"),
        ({"cv_folds": 1}, r"0 \(disabled\) or at least 2"),
        ({"cv_folds": 5}, "exceeds the number"),
        (
            {"session_ids": np.asarray([0] + [1] * 15)},
            "at least 2 trials",
        ),
    ],
)
def test_invalid_inputs_fail_before_fitting(updates, message):
    y, X, U, session_ids = _synthetic_arrays()
    arguments = {
        "y": y,
        "X": X,
        "U": U,
        "session_ids": session_ids,
        "num_states": 2,
        "num_classes": 2,
        "emission_names": ["bias", "stimulus"],
        "transition_names": ["reward_history"],
        "num_iters": 1,
        "m_step_num_iters": 1,
        "n_restarts": 1,
        "verbose": False,
    }
    arguments.update(updates)

    with pytest.raises(ValueError, match=message):
        fit_glmhmm(**arguments)
