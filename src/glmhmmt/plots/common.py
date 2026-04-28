from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel

from glmhmmt.views import get_state_color


def resolve_single_axis(
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes, bool]:
    """Resolve a single plotting axis.

    Returns the figure, axis, and a flag indicating whether the figure was
    created inside the function.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax, created_fig

def resolve_axes_grid(
    *,
    axes: Sequence[plt.Axes] | np.ndarray | None = None,
    n_panels: int,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] | None = None,
    squeeze: bool = False,
    sharex: bool = False,
    sharey: bool = False,
    gridspec_kw: dict | None = None,
) -> tuple[plt.Figure, np.ndarray, bool]:
    """Resolve a grid of axes for multi-panel plotting.

    Returns
    -------
    fig
        Owning figure.
    axes_array
        Axes reshaped as ``(nrows, ncols)``.
    created_fig
        True if the figure was created internally.
    """
    if n_panels > nrows * ncols:
        raise ValueError(
            f"Grid shape ({nrows}, {ncols}) has room for {nrows * ncols} panels, "
            f"but n_panels={n_panels}."
        )

    created_fig = axes is None

    if created_fig:
        fig, axes_obj = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=squeeze,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=gridspec_kw,
        )
        flat_axes = np.asarray(axes_obj, dtype=object).ravel()
    else:
        flat_axes = np.asarray(axes, dtype=object).ravel()
        if len(flat_axes) < n_panels:
            raise ValueError(f"Expected at least {n_panels} axes, got {len(flat_axes)}.")
        fig = flat_axes[0].figure

    axes_array = flat_axes[: nrows * ncols].reshape(nrows, ncols)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    return fig, axes_array, created_fig


def resolve_colors(
    colors: Sequence[str] | str | None,
    n_artists: int,
    *,
    name: str,
) -> list[str]:
    """Resolve a color specification to one color per artist."""
    if colors is None:
        raise ValueError(f"{name} cannot be None.")
    if isinstance(colors, str) or np.isscalar(colors):
        return [str(colors)] * n_artists

    resolved = list(colors)
    if len(resolved) == 1 and n_artists > 1:
        resolved *= n_artists
    if len(resolved) != n_artists:
        raise ValueError(f"{name} must have length {n_artists}, got {len(resolved)}.")
    return [str(color) for color in resolved]


def resolve_state_colors(
    state_labels: Sequence[str],
    state_colors: Sequence[str] | str | None = None,
) -> list[str]:
    """Resolve one color per state label."""
    labels = [str(label) for label in state_labels]
    if state_colors is not None:
        return resolve_colors(state_colors, len(labels), name="state_colors")
    return [
        get_state_color(label, idx, K=len(labels) or None)
        for idx, label in enumerate(labels)
    ]


def custom_boxplot(
    ax: plt.Axes,
    values,
    *,
    positions,
    widths,
    median_colors,
    box_facecolor: str = "white",
    box_edgecolor: str = "#666666",
    box_alpha: float = 0.85,
    box_linewidth: float = 1.1,
    whisker_color: str = "#666666",
    whisker_linewidth: float = 1.0,
    median_linewidth: float = 3.0,
    line_values: np.ndarray | None = None,
    line_color: str = "#B0B0B0",
    line_alpha: float = 0.15,
    line_linewidth: float = 1.25,
    line_zorder: float = 2.0,
    showfliers: bool = False,
    showcaps: bool = False,
    zorder: float = 0.0,
    **kwargs,
):
    """Draw boxplots with project defaults and optional subject-connecting lines."""
    positions = list(np.atleast_1d(np.asarray(positions, dtype=float)))
    resolved_median_colors = resolve_colors(
        median_colors,
        len(positions),
        name="median_colors",
    )

    if len(positions) == 1:
        grouped_values = [np.asarray(values, dtype=float)]
    else:
        grouped_values = [np.asarray(vals, dtype=float) for vals in values]
        if len(grouped_values) != len(positions):
            raise ValueError(
                f"values must have length {len(positions)}, got {len(grouped_values)}."
            )

    valid_triplets = [
        (vals[np.isfinite(vals)], pos, color)
        for vals, pos, color in zip(
            grouped_values,
            positions,
            resolved_median_colors,
            strict=False,
        )
        if np.isfinite(vals).any()
    ]

    if valid_triplets:
        valid_values, valid_positions, valid_median_colors = zip(
            *valid_triplets,
            strict=False,
        )
        box = ax.boxplot(
            list(valid_values),
            positions=list(valid_positions),
            widths=widths,
            patch_artist=True,
            showfliers=showfliers,
            showcaps=showcaps,
            zorder=zorder,
            **kwargs,
        )

        for patch in box["boxes"]:
            patch.set(
                facecolor=box_facecolor,
                edgecolor=box_edgecolor,
                alpha=box_alpha,
                linewidth=box_linewidth,
            )

        for elem in ("whiskers", "caps"):
            for artist in box[elem]:
                artist.set(color=whisker_color, linewidth=whisker_linewidth)

        for median, color in zip(box["medians"], valid_median_colors, strict=False):
            median.set(color=color, linewidth=median_linewidth)
    else:
        box = {"boxes": [], "whiskers": [], "caps": [], "medians": []}

    if line_values is not None:
        line_values = np.asarray(line_values, dtype=float)
        if line_values.ndim == 1:
            line_values = line_values[None, :]
        if line_values.shape[1] != len(positions):
            raise ValueError(
                "line_values must have one column per box position. "
                f"Expected {len(positions)}, got {line_values.shape[1]}."
            )

        line_positions = np.asarray(positions, dtype=float)
        for ys in line_values:
            valid_idx = np.flatnonzero(np.isfinite(ys))
            if valid_idx.size < 2:
                continue
            split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
            for segment in np.split(valid_idx, split_points):
                if segment.size < 2:
                    continue
                ax.plot(
                    line_positions[segment],
                    ys[segment],
                    color=line_color,
                    alpha=line_alpha,
                    linewidth=line_linewidth,
                    zorder=line_zorder,
                )

    return box


def significance_label(pvalue: float) -> str:
    """Map a p-value to a star significance label."""
    if not np.isfinite(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def add_pairwise_significance_brackets(
    ax: plt.Axes,
    *,
    grouped_values: list[list[np.ndarray]],
    subject_lines: list[np.ndarray] | None,
    n_features: int,
    n_states: int,
    hue_width: float,
) -> None:
    """Add paired t-test significance brackets across states within each feature."""
    if subject_lines is None or n_states < 2:
        return

    state_pairs = list(combinations(range(n_states), 2))
    for feat_idx, per_subject_values in enumerate(subject_lines):
        finite_groups = [
            values[np.isfinite(values)]
            for values in (grouped_values[state_idx][feat_idx] for state_idx in range(n_states))
            if np.isfinite(values).any()
        ]
        if not finite_groups:
            continue

        finite_values = np.concatenate(finite_groups)
        y_range = float(np.max(finite_values) - np.min(finite_values))
        if not np.isfinite(y_range) or y_range <= 0:
            y_range = 1.0
        y_base = float(np.max(finite_values))
        step = y_range * 0.08
        bracket_height = y_range * 0.02

        for pair_idx, (state_a, state_b) in enumerate(state_pairs):
            paired = per_subject_values[
                np.isfinite(per_subject_values[:, state_a])
                & np.isfinite(per_subject_values[:, state_b])
            ]
            if paired.shape[0] < 2:
                continue

            pvalue = float(ttest_rel(paired[:, state_a], paired[:, state_b]).pvalue)
            star = significance_label(pvalue)
            x1 = feat_idx + (state_a - (n_states - 1) / 2.0) * hue_width
            x2 = feat_idx + (state_b - (n_states - 1) / 2.0) * hue_width
            y = y_base + step * (pair_idx + 1)

            ax.plot(
                [x1, x1, x2, x2],
                [y, y + bracket_height, y + bracket_height, y],
                lw=1.0,
                c="black",
            )
            ax.text(
                (x1 + x2) / 2.0,
                y + bracket_height,
                star,
                ha="center",
                va="bottom",
                color="black",
            )


def state_legend_handles(
    state_labels: Sequence[str],
    colors: Sequence[str],
) -> list[Line2D]:
    """Build simple line handles for a state legend."""
    labels = [str(label) for label in state_labels]
    return [
        Line2D([0], [0], color=colors[idx], lw=2, label=label)
        for idx, label in enumerate(labels)
    ]