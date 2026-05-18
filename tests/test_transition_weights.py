from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

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


def test_transition_weight_plots_use_emission_style_dataframe():
    weights_df = build_transition_weights_df(
        {"s1": _view("s1"), "s2": _view("s2", offset=1.0)}
    )

    fig_subjects, axes = transition_weights_by_subject(weights_df, K=2)
    ax_summary = transition_weights_summary_boxplot(weights_df, K=2, show_ttests=True)

    assert axes.shape == (1, 2)
    assert axes.ravel()[0].get_ylabel() == "Transition weight"
    assert ax_summary.get_ylabel() == "Transition weight"
    assert [tick.get_text() for tick in ax_summary.get_xticklabels()] == ["A_plus", "A_minus"]

    fig_subjects.clf()
    ax_summary.figure.clf()
