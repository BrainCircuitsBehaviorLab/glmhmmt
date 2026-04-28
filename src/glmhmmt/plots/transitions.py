from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
) -> plt.Axes:
    """Plot a transition matrix on a single axis.

    If ``ax`` is not provided, a new figure and axis are created internally. If
    ``ax`` is provided, the heatmap is drawn into the supplied axis.

    Parameters
    ----------
    matrix
        Square transition matrix to display.
    tick_labels
        Labels for rows and columns of the matrix.
    title
        Optional axis title.
    figsize
        Figure size used only when creating a new figure internally.
    cmap
        Colormap used for the heatmap.
    ax
        Optional matplotlib axis to draw into.
    cbar
        Whether to draw a colorbar.
    cbar_label
        Label for the colorbar.

    Returns
    -------
    ax
        Axis containing the plotted transition matrix.

    Raises
    ------
    ValueError
        If ``matrix`` is empty, not two-dimensional, not square, or if
        ``tick_labels`` has the wrong length.
    """
    matrix = np.asarray(matrix, dtype=float)
    labels = [str(label) for label in tick_labels]

    if matrix.size == 0:
        raise ValueError("matrix is empty; build a transition payload before plotting.")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got ndim={matrix.ndim}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"tick_labels must have length {matrix.shape[0]}, got {len(labels)}."
        )

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize or (4.4, 3.8))
    else:
        fig = ax.figure

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels,
        cbar=cbar,
        cbar_kws={"shrink": 0.8, "label": cbar_label} if cbar else None,
    )
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")

    if created_fig:
        fig.tight_layout()

    return ax


def plot_transition_matrix_by_subject(
    matrices,
    subject_ids: Sequence[str],
    tick_labels_by_subject: Sequence[Sequence[str]] | dict[str, Sequence[str]] | None,
    *,
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    cmap: str = "bone",
    cbar_label: str = "probability",
    axes: Sequence[plt.Axes] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot one transition matrix per subject.

    If ``axes`` is not provided, a figure and grid of subplots are created
    internally. If ``axes`` is provided, the subject matrices are drawn into the
    supplied axes.

    Parameters
    ----------
    matrices
        Sequence of transition matrices, one per subject.
    subject_ids
        Subject identifiers in the same order as ``matrices``.
    tick_labels_by_subject
        Tick labels for each subject matrix, either as a sequence aligned with
        ``subject_ids``, as a dictionary keyed by subject id, or ``None`` to
        use default labels.
    n_cols
        Number of subplot columns when creating a new figure internally.
    figsize
        Figure size used only when creating a new figure internally.
    cmap
        Colormap used for the heatmaps.
    cbar_label
        Label for the colorbar.
    axes
        Optional sequence of matplotlib axes to draw into.

    Returns
    -------
    fig
        Figure containing the subject panels.
    axes
        Array of axes arranged as ``(n_rows, n_cols)``.

    Raises
    ------
    ValueError
        If ``matrices`` is empty, if its length does not match
        ``subject_ids``, if the label specification is invalid, or if fewer
        axes are provided than required.
    """
    subject_ids = [str(subject_id) for subject_id in subject_ids]
    matrices = [np.asarray(matrix, dtype=float) for matrix in matrices]

    if not matrices:
        raise ValueError(
            "matrices is empty; build a transition-by-subject payload before plotting."
        )
    if len(matrices) != len(subject_ids):
        raise ValueError(
            f"matrices must have length {len(subject_ids)}, got {len(matrices)}."
        )

    if tick_labels_by_subject is None:
        labels_per_subject = [
            [f"S{k}" for k in range(matrix.shape[0])]
            for matrix in matrices
        ]
    elif isinstance(tick_labels_by_subject, dict):
        labels_per_subject = [
            [
                str(label)
                for label in tick_labels_by_subject.get(
                    subject_id,
                    [f"S{k}" for k in range(matrix.shape[0])],
                )
            ]
            for subject_id, matrix in zip(subject_ids, matrices, strict=False)
        ]
    else:
        if len(tick_labels_by_subject) != len(subject_ids):
            raise ValueError(
                "tick_labels_by_subject must provide one label sequence per subject."
            )
        labels_per_subject = [
            [str(label) for label in labels]
            for labels in tick_labels_by_subject
        ]

    n_cols = max(1, min(int(n_cols), len(matrices)))
    n_rows = int(math.ceil(len(matrices) / n_cols))
    n_panels = n_rows * n_cols

    created_fig = axes is None
    if created_fig:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize or (4.2 * n_cols, 3.4 * n_rows),
            squeeze=False,
        )
        axes = np.asarray(axes, dtype=object).ravel()
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) < n_panels:
            raise ValueError(f"Expected at least {n_panels} axes, got {len(axes)}.")
        fig = axes[0].figure

    for idx, (matrix, subject_id, labels) in enumerate(
        zip(matrices, subject_ids, labels_per_subject, strict=False)
    ):
        ax = axes[idx]
        plot_transition_matrix(
            matrix,
            labels,
            title=f"Subject {subject_id}",
            ax=ax,
            cmap=cmap,
            cbar=idx == 0,
            cbar_label=cbar_label,
        )

    for idx in range(len(matrices), n_panels):
        axes[idx].set_visible(False)

    if created_fig:
        fig.tight_layout()

    return fig, axes.reshape(n_rows, n_cols)