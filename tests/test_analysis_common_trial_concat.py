from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from glmhmmt.notebook_support.analysis_common import _align_trial_frames_for_concat


def test_align_trial_frames_pads_missing_feature_columns_with_zero() -> None:
    frames = [
        pl.DataFrame({"subject": ["a"], "trial": [1], "bias_0": [1.0]}),
        pl.DataFrame({"subject": ["b"], "trial": [1], "bias_1": [1.0]}),
    ]
    views = {
        "a": SimpleNamespace(feat_names=["bias_0"]),
        "b": SimpleNamespace(feat_names=["bias_1"]),
    }

    aligned = _align_trial_frames_for_concat(frames, views=views)
    out = pl.concat(aligned, how="vertical_relaxed")

    assert out["bias_0"].to_list() == [1.0, 0.0]
    assert out["bias_1"].to_list() == [0.0, 1.0]

