from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from glmhmmt.model import InputDrivenSoftmaxTransitions, ParamsInputDrivenTransitions
from glmhmmt.plots.transitions import (
    transition_weights_by_subject,
    transition_weights_summary_boxplot,
)
from glmhmmt.postprocess import build_transition_weights_df
from glmhmmt.views import SubjectFitView


def _view(subject: str, offset: float = 0.0) -> SubjectFitView:
    transition_weights = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=float,
    )
    return SubjectFitView(
        subject=subject,
        K=2,
        smoothed_probs=np.ones((1, 2)) / 2,
        emission_weights=np.zeros((2, 1, 1), dtype=float),
        X=np.asarray([[1.0]], dtype=float),
        y=np.asarray([0], dtype=int),
        feat_names=["x"],
        state_name_by_idx={0: "Disengaged", 1: "Engaged"},
        state_idx_order=[0, 1],
        state_rank_by_idx={0: 1, 1: 0},
        predictive_state_probs=np.ones((1, 2)) / 2,
        transition_weights=transition_weights + offset,
        U_cols=["A_plus", "A_minus"],
    )


def test_build_transition_weights_df_matches_legacy_standardization():
    weights_df = build_transition_weights_df({"s1": _view("s1")})

    assert weights_df.columns == [
        "subject",
        "state_idx",
        "state_label",
        "state_rank",
        "feature",
        "feature_idx",
        "weight",
    ]

    rows = weights_df.sort(["state_idx", "feature_idx"]).to_dicts()
    assert [row["feature"] for row in rows] == ["A_plus", "A_minus", "A_plus", "A_minus"]
    assert np.allclose(
        [row["weight"] for row in rows],
        [1.0 / 3.0, 2.0 / 3.0, 7.0 / 3.0, 8.0 / 3.0],
    )


def test_build_transition_weights_df_expands_alternative_formulation_weights():
    view = _view("s1")
    view.transition_weights = np.asarray([[2.0, 4.0]], dtype=float)

    rows = build_transition_weights_df({"s1": view}).sort(["state_idx", "feature_idx"]).to_dicts()

    assert [row["feature"] for row in rows] == ["A_plus", "A_minus", "A_plus", "A_minus"]
    assert np.allclose([row["weight"] for row in rows], [1.0, 2.0, -1.0, -2.0])


def test_input_driven_transitions_use_destination_inputs_and_k_minus_one_weights():
    transitions = InputDrivenSoftmaxTransitions(
        num_states=2,
        emission_input_dim=1,
        transition_input_dim=1,
        stickiness=0.0,
    )
    params = ParamsInputDrivenTransitions(
        bias=jnp.zeros((2, 2), dtype=jnp.float32),
        weights=jnp.asarray([[2.0]], dtype=jnp.float32),
    )
    inputs = jnp.asarray(
        [
            [1.0, -3.0],
            [1.0, 0.5],
            [1.0, -1.0],
        ],
        dtype=jnp.float32,
    )

    matrices = np.asarray(transitions._compute_transition_matrices(params, inputs))
    _, suff_inputs = transitions.collect_suff_stats(
        params,
        SimpleNamespace(trans_probs=jnp.ones((2, 2, 2), dtype=jnp.float32)),
        inputs,
    )

    np.testing.assert_allclose(np.asarray(suff_inputs), np.asarray(inputs[1:]))
    np.testing.assert_allclose(
        matrices[:, 0, :],
        np.asarray(
            [
                [np.exp(1.0) / (np.exp(1.0) + 1.0), 1.0 / (np.exp(1.0) + 1.0)],
                [np.exp(-2.0) / (np.exp(-2.0) + 1.0), 1.0 / (np.exp(-2.0) + 1.0)],
            ]
        ),
        rtol=1e-6,
    )
    assert params.weights.shape == (1, 1)


def test_transition_weight_plots_use_emission_style_dataframe():
    weights_df = build_transition_weights_df(
        {"s1": _view("s1"), "s2": _view("s2", offset=1.0)}
    )

    fig_subjects, axes = transition_weights_by_subject(weights_df, K=2)
    ax_summary = transition_weights_summary_boxplot(weights_df, K=2, show_ttests=True)

    assert axes.shape == (1, 2)
    assert axes.ravel()[0].get_ylabel() == "Transition weight"
    assert ax_summary.get_ylabel() == "Transition weight toward Engaged"
    assert [tick.get_text() for tick in ax_summary.get_xticklabels()] == ["A_plus", "A_minus"]
    assert [text.get_text() for text in ax_summary.get_legend().get_texts()] == ["Engaged"]

    fig_subjects.clf()
    ax_summary.figure.clf()


def test_transition_weight_plots_forward_tick_rotation():
    weights_df = build_transition_weights_df(
        {"s1": _view("s1"), "s2": _view("s2", offset=1.0)}
    )

    ax_summary = transition_weights_summary_boxplot(weights_df, K=2, tick_rotation=0)

    assert ax_summary.get_xticklabels()[0].get_rotation() == 0
    assert ax_summary.get_xticklabels()[0].get_ha() == "center"

    ax_summary.figure.clf()
