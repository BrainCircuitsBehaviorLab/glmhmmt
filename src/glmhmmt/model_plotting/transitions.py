from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt

from glmhmmt.plots import (
    plot_transition_matrix as _draw_transition_matrix,
    plot_transition_matrix_by_subject as _draw_transition_matrix_by_subject,
)


def plot_transition_matrix(
    matrix,
    tick_labels: Sequence[str],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "bone",
    ax: plt.Axes | None = None,
    cbar: bool = True,
    cbar_label: str = "probability",
) -> plt.Figure:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        raise ValueError("matrix is empty; build a transition payload before plotting.")
    return _draw_transition_matrix(
        matrix,
        tick_labels,
        title=title,
        figsize=figsize,
        cmap=cmap,
        ax=ax,
        cbar=cbar,
        cbar_label=cbar_label,
    )


def plot_transition_matrix_by_subject(
    matrices,
    subject_ids: Sequence[str],
    tick_labels_by_subject: Sequence[Sequence[str]] | dict[str, Sequence[str]],
    *,
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    cmap: str = "bone",
    cbar_label: str = "probability",
) -> plt.Figure:
    matrices = [np.asarray(matrix, dtype=float) for matrix in matrices]
    if not matrices:
        raise ValueError("matrices is empty; build a transition-by-subject payload before plotting.")
    return _draw_transition_matrix_by_subject(
        matrices,
        subject_ids,
        tick_labels_by_subject,
        n_cols=n_cols,
        figsize=figsize,
        cmap=cmap,
        cbar_label=cbar_label,
    )
