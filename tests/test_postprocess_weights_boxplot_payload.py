from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _WORKSPACE_ROOT / "glmhmmt" / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from glmhmmt.postprocess import build_weights_boxplot_payload


def test_build_weights_boxplot_payload_collapses_bias_family_to_abs_mean():
    weights_df = pl.DataFrame(
        {
            "subject": ["mouse-1", "mouse-1", "mouse-1", "mouse-1", "mouse-2", "mouse-2", "mouse-2", "mouse-2"],
            "state_label": ["Engaged", "Disengaged", "Engaged", "Disengaged", "Engaged", "Disengaged", "Engaged", "Disengaged"],
            "state_rank": [0, 1, 0, 1, 0, 1, 0, 1],
            "class_idx": [0] * 8,
            "feature": ["bias_0", "bias_0", "bias_1", "bias_1", "bias_0", "bias_0", "bias_1", "bias_1"],
            "weight": [-1.0, -3.0, 3.0, -6.0, -2.0, -5.0, -4.0, 8.0],
        }
    )

    payload = build_weights_boxplot_payload(weights_df)

    assert payload["feature_names"] == ["|bias|"]
    assert payload["state_labels"] == ["Engaged", "Disengaged"]
    np.testing.assert_allclose(
        payload["subject_lines"][0],
        np.asarray(
            [
                [2.0, 4.5],
                [3.0, 6.5],
            ]
        ),
    )
    np.testing.assert_allclose(
        payload["values_by_state_feature"][0][0],
        np.asarray([2.0, 3.0]),
    )
    np.testing.assert_allclose(
        payload["values_by_state_feature"][1][0],
        np.asarray([4.5, 6.5]),
    )
