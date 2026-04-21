from __future__ import annotations

import pandas as pd

from glmhmmt.model_plots import plot_binary_emission_weights_summary_boxplot


def test_binary_emission_summary_boxplot_accepts_weights_df_and_feature_labels():
    weights_df = pd.DataFrame(
        {
            "subject": ["mouse-1", "mouse-1", "mouse-2", "mouse-2"] * 2,
            "state_label": ["Disengaged", "Engaged", "Disengaged", "Engaged"] * 2,
            "state_rank": [1, 0, 1, 0] * 2,
            "class_idx": [0] * 8,
            "feature": ["stim_vals"] * 4 + ["bias"] * 4,
            "weight": [-2.0, -3.0, -1.0, -4.0, -0.5, -0.25, -0.75, -0.1],
        }
    )

    fig = plot_binary_emission_weights_summary_boxplot(
        weights_df,
        K=2,
        weight_sign=-1.0,
        feature_order=("bias", "stim_vals"),
        abs_features=("bias",),
        feature_labeler=lambda name: {"bias": "Bias", "stim_vals": "Stimulus"}[name],
        state_label_order=("Disengaged",),
    )

    ax = fig.axes[0]
    xticks = [tick.get_text() for tick in ax.get_xticklabels()]
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()] if legend else []

    assert xticks == ["Bias", "Stimulus"]
    assert legend_labels == ["Disengaged", "Engaged"]
