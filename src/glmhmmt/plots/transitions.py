from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_1samp

from glmhmmt.plots.emissions import (
    _prepare_weights_df,
    emission_weights_by_subject,
    emission_weights_summary_boxplot,
    emission_weights_summary_lineplot,
)
from glmhmmt.plots.common import (
    custom_boxplot,
    resolve_single_axis,
    significance_label,
    state_legend_handles,
)
from glmhmmt.views import get_state_color


def _add_binary_zero_ttest_labels(
    ax: plt.Axes,
    df,
    *,
    features: Sequence[str],
    reference_state: str,
) -> None:
    """Annotate binary centered transition weights with one-sample tests."""
    y_positions: list[float] = []
    for feat_idx, feature in enumerate(features):
        feature_mask = df["feature"].astype(str) == str(feature)
        feature_values = df.loc[feature_mask, "weight"].to_numpy(dtype=float)
        finite_feature_values = feature_values[np.isfinite(feature_values)]
        if finite_feature_values.size == 0:
            continue

        values = df.loc[
            feature_mask & (df["state_label"].astype(str) == reference_state),
            "weight",
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue

        y_range = float(finite_feature_values.max() - finite_feature_values.min())
        if not np.isfinite(y_range) or y_range <= 0:
            y_range = 1.0
        y = float(finite_feature_values.max()) + y_range * 0.12
        pvalue = float(ttest_1samp(values, popmean=0.0).pvalue)
        ax.text(
            feat_idx,
            y,
            significance_label(pvalue),
            ha="center",
            va="bottom",
            color="black",
        )
        y_positions.append(y)

    if y_positions:
        _, top = ax.get_ylim()
        ax.set_ylim(top=max(top, max(y_positions) * 1.08))


def _resolve_engaged_state_label(state_order: Sequence[str]) -> str | None:
    """Return the engaged state label, falling back to the first binary target."""
    labels = [str(label) for label in state_order]
    for label in labels:
        normalized = label.casefold()
        if "engaged" in normalized and "disengaged" not in normalized:
            return label
    return labels[0] if labels else None


def _binary_engaged_weights_df(
    weights_df,
    *,
    K: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
):
    """Return prepared binary weights filtered to the engaged target state."""
    df, features, display_features, state_order, _ = _prepare_weights_df(
        weights_df,
        K=K,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    engaged_state = (
        _resolve_engaged_state_label(state_order)
        if len(state_order) == 2
        else None
    )
    if engaged_state is None:
        return df, features, display_features, state_order, None
    plot_df = df[df["state_label"].astype(str) == engaged_state].copy()
    return plot_df, features, display_features, [engaged_state], engaged_state


def _binary_engaged_summary_boxplot(
    weights_df,
    *,
    K: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show_ttests: bool = False,
    tick_rotation: float = 35,
) -> tuple[plt.Axes, str | None]:
    """Draw the binary engaged transition contrast with the K=2 state color."""
    (
        df,
        features,
        display_features,
        state_order,
        engaged_state,
    ) = _binary_engaged_weights_df(
        weights_df,
        K=K,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    if engaged_state is None:
        return ax, None

    fig, ax, created_fig = resolve_single_axis(
        ax=ax,
        figsize=figsize or (max(6.0, 1.4 * max(1, len(display_features))), 4.2),
    )
    color = get_state_color(engaged_state, K=2)
    values = [
        df.loc[df["feature"] == feature, "weight"].to_numpy(dtype=float)
        for feature in features
    ]

    custom_boxplot(
        ax,
        values,
        positions=range(len(features)),
        widths=0.62,
        median_colors=[color] * len(features),
        box_facecolor="white",
        box_alpha=0.85,
        showfliers=False,
        showcaps=False,
    )
    if show_ttests:
        _add_binary_zero_ttest_labels(
            ax,
            df,
            features=features,
            reference_state=engaged_state,
        )

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(
        display_features,
        rotation=tick_rotation,
        ha="right" if tick_rotation else "center",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Transition weight")
    if title is not None:
        ax.set_title(title)
    ax.legend(
        state_legend_handles(state_order, [color]),
        state_order,
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    if created_fig:
        fig.tight_layout()
    return ax, engaged_state


def transition_weights_by_subject(
    weights_df,
    *,
    K: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    tick_rotation: float = 35,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot subject-wise transition weights with the emission-weight style."""
    fig, axes_grid = emission_weights_by_subject(
        weights_df,
        K=K,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
        axes=axes,
        figsize=figsize,
        title=title,
        tick_rotation=tick_rotation,
    )
    for ax in np.asarray(axes_grid, dtype=object).ravel():
        if ax.get_visible():
            ax.set_ylabel("Transition weight")
    return fig, axes_grid


def transition_weights_summary_lineplot(
    weights_df,
    *,
    K: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    tick_rotation: float = 35,
) -> plt.Axes:
    """Plot transition-weight means with the emission summary-line style."""
    (
        _plot_weights,
        _features,
        _display_features,
        _state_order,
        _engaged_state,
    ) = _binary_engaged_weights_df(
        weights_df,
        K=K,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    if _engaged_state is not None:
        fig, ax, created_fig = resolve_single_axis(
            ax=ax,
            figsize=figsize or (max(6.0, 1.6 * max(1, len(_display_features))), 4.2),
        )
        sns.lineplot(
            data=_plot_weights,
            x="Feature",
            y="weight",
            hue="State",
            hue_order=_state_order,
            palette={_engaged_state: get_state_color(_engaged_state, K=2)},
            marker="o",
            errorbar="se",
            err_kws={"edgecolor": "none", "linewidth": 0},
            ax=ax,
        )
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("")
        ax.set_ylabel("Transition weight")
        plt.setp(
            ax.get_xticklabels(),
            rotation=tick_rotation,
            ha="right" if tick_rotation else "center",
        )
        if title is not None:
            ax.set_title(title)
        ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        if created_fig:
            fig.tight_layout()
        return ax

    ax = emission_weights_summary_lineplot(
        weights_df,
        K=K,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
        ax=ax,
        figsize=figsize,
        title=title,
        tick_rotation=tick_rotation,
    )
    ax.set_ylabel("Transition weight")
    return ax


def transition_weights_summary_boxplot(
    weights_df,
    *,
    K: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    connect_subjects: bool = True,
    show_ttests: bool = False,
    subject_line_color: str = "#7A7A7A",
    subject_line_alpha: float = 0.15,
    subject_line_width: float = 1.0,
    tick_rotation: float = 35,
) -> plt.Axes:
    """Plot transition-weight distributions with the emission boxplot style.

    For two-state centered transition weights, only the engaged target-state
    contrast is plotted. ``show_ttests=True`` annotates a one-sample t-test
    against zero across subjects.
    """
    ax_binary, _engaged_state = _binary_engaged_summary_boxplot(
        weights_df,
        K=K,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
        ax=ax,
        figsize=figsize,
        title=title,
        show_ttests=show_ttests,
        tick_rotation=tick_rotation,
    )
    if _engaged_state is not None:
        return ax_binary

    ax = emission_weights_summary_boxplot(
        weights_df,
        K=K,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
        ax=ax,
        figsize=figsize,
        title=title,
        connect_subjects=connect_subjects,
        show_ttests=show_ttests and _engaged_state is None,
        subject_line_color=subject_line_color,
        subject_line_alpha=subject_line_alpha,
        subject_line_width=subject_line_width,
        tick_rotation=tick_rotation,
    )
    ax.set_ylabel("Transition weight")
    return ax


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
