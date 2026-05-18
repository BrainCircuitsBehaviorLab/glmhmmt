from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE_ROOT / "src"))

from glmhmmt.cli.fit_common import align_design_matrix_to_columns


def test_align_design_matrix_zero_fills_missing_one_hot_columns():
    values = np.asarray(
        [
            [1.0, 0.2],
            [1.0, 0.4],
        ],
        dtype=np.float32,
    )

    aligned, columns = align_design_matrix_to_columns(
        values,
        ["bias", "bias_1"],
        ["bias", "bias_0", "bias_1"],
    )

    assert columns == ["bias", "bias_0", "bias_1"]
    np.testing.assert_allclose(
        aligned,
        np.asarray(
            [
                [1.0, 0.0, 0.2],
                [1.0, 0.0, 0.4],
            ],
            dtype=np.float32,
        ),
    )


def test_align_design_matrix_ignores_columns_unseen_during_training():
    values = np.asarray([[1.0, 0.2, 1.0]], dtype=np.float32)

    aligned, columns = align_design_matrix_to_columns(
        values,
        ["bias", "stim", "bias_7"],
        ["bias", "stim"],
    )

    assert columns == ["bias", "stim"]
    np.testing.assert_allclose(aligned, np.asarray([[1.0, 0.2]], dtype=np.float32))
