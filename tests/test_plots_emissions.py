from __future__ import annotations

import pandas as pd
import matplotlib

matplotlib.use("Agg")

from glmhmmt.plots.emissions import (
    _fold_three_choice_raw_weights,
    emission_weights_summary_boxplot,
)


def _three_choice_weights_df() -> pd.DataFrame:
    records = []
    for subject, scale in (("mouse-1", 1.0), ("mouse-2", 1.5)):
        for state_label, state_rank, state_shift in (("Engaged", 0, 0.2), ("Biased R", 1, -0.1)):
            for class_idx, class_sign in ((0, 1.0), (2, -1.0)):
                for feature, base_weight in (("SL", 1.0), ("SR", 0.5), ("SC", -0.3)):
                    records.append(
                        {
                            "subject": subject,
                            "state_label": state_label,
                            "state_rank": state_rank,
                            "class_idx": class_idx,
                            "baseline_class_idx": 1,
                            "feature": feature,
                            "weight": scale * class_sign * base_weight + state_shift,
                        }
                    )
    return pd.DataFrame.from_records(records)


def test_emission_weights_summary_boxplot_auto_folds_three_choice_weights():
    ax = emission_weights_summary_boxplot(_three_choice_weights_df(), K=2)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["S"]
    assert ax.get_ylabel() == "Weight"


def test_three_choice_fold_keeps_raw_sign_aligned_weights():
    folded = _fold_three_choice_raw_weights(_three_choice_weights_df())

    assert folded is not None
    assert folded["feature"].unique().tolist() == ["S"]
    row = folded[
        (folded["subject"] == "mouse-1")
        & (folded["state_label"] == "Engaged")
    ].iloc[0]
    assert abs(row["weight"] - 0.25) < 1e-12


def test_emission_weights_summary_boxplot_class_idx_keeps_raw_weights():
    ax = emission_weights_summary_boxplot(_three_choice_weights_df(), K=2, class_idx=0)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["SL", "SR", "SC"]
    assert ax.get_ylabel() == "Weight"
