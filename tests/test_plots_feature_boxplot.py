from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _WORKSPACE_ROOT / "glmhmmt" / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from glmhmmt.plots import plot_feature_boxplot


def test_plot_feature_boxplot_draws_subject_lines_without_scatter():
    fig = plot_feature_boxplot(
        [
            np.asarray([1.0, 2.0]),
            np.asarray([3.0, 4.0]),
        ],
        ["2", "4"],
        title="stim_hot",
        xlabel="Stimulus level",
        subject_lines=np.asarray(
            [
                [1.0, 3.0],
                [2.0, 4.0],
            ]
        ),
        subject_line_color="#123456",
    )

    ax = fig.axes[0]

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["2", "4"]
    assert len(ax.collections) == 0
    assert sum(mcolors.to_hex(line.get_color()) == "#123456" for line in ax.lines) == 2
