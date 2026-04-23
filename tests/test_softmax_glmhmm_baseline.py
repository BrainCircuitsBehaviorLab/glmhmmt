from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from glmhmmt.model import SoftmaxGLMHMM, SoftmaxGLMHMMEmissions
from glmhmmt.postprocess import _emission_probs, build_emission_weights_df
from glmhmmt.views import SubjectFitView, build_views


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def test_default_baseline_is_first_class_for_binary_emissions():
    emissions = SoftmaxGLMHMMEmissions(
        num_states=1,
        num_classes=2,
        input_dim=1,
        emission_input_dim=1,
    )
    params, _ = emissions.initialize(
        jnp.array([0, 0], dtype=jnp.uint32),
        emission_weights=jnp.asarray([[[2.0]]], dtype=jnp.float32),
    )

    probs = np.asarray(
        emissions.distribution(params, 0, jnp.asarray([1.0], dtype=jnp.float32)).probs_parameter()
    )

    np.testing.assert_allclose(probs, _softmax(np.asarray([0.0, 2.0])), rtol=1e-6)
    assert probs[1] > probs[0]


def test_selected_baseline_inserts_zero_logit_at_that_class():
    emissions = SoftmaxGLMHMMEmissions(
        num_states=1,
        num_classes=3,
        input_dim=1,
        emission_input_dim=1,
        baseline_class_idx=1,
    )
    params, _ = emissions.initialize(
        jnp.array([0, 0], dtype=jnp.uint32),
        emission_weights=jnp.asarray([[[1.0], [3.0]]], dtype=jnp.float32),
    )

    probs = np.asarray(
        emissions.distribution(params, 0, jnp.asarray([1.0], dtype=jnp.float32)).probs_parameter()
    )

    np.testing.assert_allclose(probs, _softmax(np.asarray([1.0, 0.0, 3.0])), rtol=1e-6)


def test_model_rejects_invalid_baseline_class_index():
    with pytest.raises(ValueError, match="baseline_class_idx"):
        SoftmaxGLMHMM(
            num_states=1,
            num_classes=2,
            emission_input_dim=1,
            transition_input_dim=0,
            baseline_class_idx=2,
        )


def test_view_and_postprocess_probabilities_use_baseline_metadata():
    view = SubjectFitView(
        subject="s1",
        K=1,
        smoothed_probs=np.ones((1, 1)),
        emission_weights=np.asarray([[[1.0], [3.0]]], dtype=float),
        X=np.asarray([[1.0]], dtype=float),
        y=np.asarray([2], dtype=int),
        feat_names=["x"],
        state_name_by_idx={0: "State 0"},
        state_idx_order=[0],
        state_rank_by_idx={0: 0},
        predictive_state_probs=np.ones((1, 1)),
        p_pred=np.asarray([[0.0, 0.0, 1.0]], dtype=float),
        baseline_class_idx=1,
    )

    expected = _softmax(np.asarray([[1.0, 0.0, 3.0]]))

    np.testing.assert_allclose(view.state_conditional_probs()[0, 0], expected[0], rtol=1e-6)
    np.testing.assert_allclose(
        _emission_probs(view.emission_weights, view.X, np.asarray([0]), view.num_classes, baseline_class_idx=1),
        expected,
        rtol=1e-6,
    )


def test_build_emission_weights_df_reports_actual_class_indices():
    view = SubjectFitView(
        subject="s1",
        K=1,
        smoothed_probs=np.ones((1, 1)),
        emission_weights=np.asarray([[[2.0]]], dtype=float),
        X=np.asarray([[1.0]], dtype=float),
        y=np.asarray([1], dtype=int),
        feat_names=["x"],
        state_name_by_idx={0: "State 0"},
        state_idx_order=[0],
        state_rank_by_idx={0: 0},
        predictive_state_probs=np.ones((1, 1)),
        p_pred=np.asarray([[0.1, 0.9]], dtype=float),
        baseline_class_idx=0,
    )

    row = build_emission_weights_df({"s1": view}).row(0, named=True)

    assert row["class_idx"] == 1
    assert row["weight_row_idx"] == 0
    assert row["baseline_class_idx"] == 0


class _DummyAdapter:
    def label_states(self, arrays_store, names, K, subjects):
        return (
            {subject: {0: "State 0"} for subject in subjects},
            {subject: [0] for subject in subjects},
        )


def test_build_views_uses_saved_baseline_and_legacy_last_class_fallback():
    base_arrays = {
        "smoothed_probs": np.ones((1, 1)),
        "predictive_state_probs": np.ones((1, 1)),
        "emission_weights": np.asarray([[[2.0]]], dtype=float),
        "X": np.asarray([[1.0]], dtype=float),
        "y": np.asarray([1], dtype=int),
        "X_cols": np.asarray(["x"], dtype=object),
        "p_pred": np.asarray([[0.1, 0.9]], dtype=float),
    }

    new_view = build_views(
        {"new": {**base_arrays, "baseline_class_idx": np.asarray(0)}},
        _DummyAdapter(),
        K=1,
        subjects=["new"],
    )["new"]
    legacy_view = build_views(
        {"old": base_arrays},
        _DummyAdapter(),
        K=1,
        subjects=["old"],
    )["old"]

    assert new_view.baseline_class_idx == 0
    assert legacy_view.baseline_class_idx == 1
    assert new_view.state_conditional_probs()[0, 0, 1] > new_view.state_conditional_probs()[0, 0, 0]
    assert legacy_view.state_conditional_probs()[0, 0, 0] > legacy_view.state_conditional_probs()[0, 0, 1]


def test_build_views_recognizes_legacy_three_class_glm_center_reference():
    arrays = {
        "smoothed_probs": np.ones((1, 1)),
        "predictive_state_probs": np.ones((1, 1)),
        "emission_weights": np.asarray([[[1.0], [3.0]]], dtype=float),
        "X": np.asarray([[1.0]], dtype=float),
        "y": np.asarray([2], dtype=int),
        "X_cols": np.asarray(["x"], dtype=object),
        "p_pred": np.asarray([[0.1, 0.2, 0.7]], dtype=float),
        "lapse_mode": np.asarray("none"),
    }

    view = build_views({"glm": arrays}, _DummyAdapter(), K=1, subjects=["glm"])["glm"]

    assert view.baseline_class_idx == 1
    np.testing.assert_allclose(
        view.state_conditional_probs()[0, 0],
        _softmax(np.asarray([1.0, 0.0, 3.0])),
        rtol=1e-6,
    )
