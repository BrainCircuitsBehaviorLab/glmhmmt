from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel

from glmhmmt.views import get_state_color


def _empty_plot(message: str = "No data") -> plt.Figure:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return fig


def _resolve_colors(colors, n_artists: int, name: str) -> list[str]:
    if isinstance(colors, str) or np.isscalar(colors):
        return [colors] * n_artists

    resolved = list(colors)
    if len(resolved) == 1 and n_artists > 1:
        resolved *= n_artists
    if len(resolved) != n_artists:
        raise ValueError(f"{name} must have length {n_artists}, got {len(resolved)}.")
    return resolved


def _resolve_state_colors(
    state_labels: Sequence[str],
    state_colors: Sequence[str] | str | None,
) -> list[str]:
    if state_colors is not None:
        return _resolve_colors(state_colors, len(state_labels), "state_colors")
    return [
        get_state_color(label, idx, K=len(state_labels) or None)
        for idx, label in enumerate(state_labels)
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
    zorder: float = 0,
    **kwargs,
):
    """Small wrapper around ``ax.boxplot`` with project defaults."""
    positions = list(np.atleast_1d(np.asarray(positions, dtype=float)))
    resolved_median_colors = _resolve_colors(
        median_colors,
        len(positions),
        "median_colors",
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


def _as_grouped_values(
    values_by_state_feature,
    n_states: int,
    n_features: int,
) -> list[list[np.ndarray]]:
    if len(values_by_state_feature) != n_states:
        raise ValueError(
            f"values_by_state_feature must have length {n_states}, got {len(values_by_state_feature)}."
        )

    grouped: list[list[np.ndarray]] = []
    for state_idx, per_feature in enumerate(values_by_state_feature):
        if len(per_feature) != n_features:
            raise ValueError(
                "Each state entry in values_by_state_feature must contain one "
                f"array per feature. State {state_idx} has {len(per_feature)}, expected {n_features}."
            )
        grouped.append([np.asarray(vals, dtype=float) for vals in per_feature])
    return grouped


def _as_subject_lines(
    subject_lines,
    n_features: int,
    n_states: int,
) -> list[np.ndarray] | None:
    if subject_lines is None:
        return None
    if len(subject_lines) != n_features:
        raise ValueError(
            f"subject_lines must have length {n_features}, got {len(subject_lines)}."
        )

    resolved: list[np.ndarray] = []
    for feature_idx, line_values in enumerate(subject_lines):
        lines = np.asarray(line_values, dtype=float)
        if lines.ndim == 1:
            lines = lines[None, :]
        if lines.size == 0:
            lines = np.empty((0, n_states), dtype=float)
        if lines.shape[1] != n_states:
            raise ValueError(
                "Each subject_lines entry must have one column per state. "
                f"Feature {feature_idx} has shape {lines.shape}, expected (*, {n_states})."
            )
        resolved.append(lines)
    return resolved


def _significance_label(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def plot_feature_boxplot(
    values_by_feature,
    feature_names: Sequence[str],
    *,
    subject_lines=None,
    figsize: tuple[float, float] | None = None,
    title: str = "Feature weights",
    xlabel: str = "",
    ylabel: str = "Weight",
    median_color: str = "#1f77b4",
    subject_line_color: str = "#7A7A7A",
    subject_line_alpha: float = 0.15,
    subject_line_width: float = 1.0,
    box_width: float = 0.72,
) -> plt.Figure:
    features = [str(name) for name in feature_names]
    if not features:
        return _empty_plot()

    grouped_values = [np.asarray(vals, dtype=float) for vals in values_by_feature]
    if len(grouped_values) != len(features):
        raise ValueError(
            "values_by_feature must have one entry per feature. "
            f"Expected {len(features)}, got {len(grouped_values)}."
        )

    resolved_subject_lines = None
    if subject_lines is not None:
        resolved_subject_lines = np.asarray(subject_lines, dtype=float)
        if resolved_subject_lines.ndim == 1:
            resolved_subject_lines = resolved_subject_lines[None, :]
        if resolved_subject_lines.shape[1] != len(features):
            raise ValueError(
                "subject_lines must have one column per feature. "
                f"Expected {len(features)}, got {resolved_subject_lines.shape[1]}."
            )

    positions = np.arange(len(features), dtype=float)
    fig, ax = plt.subplots(
        figsize=figsize or (max(5.0, 0.9 * len(features)), 4.0)
    )
    custom_boxplot(
        ax,
        grouped_values,
        positions=positions,
        widths=box_width,
        median_colors=median_color,
        line_values=resolved_subject_lines,
        line_color=subject_line_color,
        line_alpha=subject_line_alpha,
        line_linewidth=subject_line_width,
        showfliers=False,
        showcaps=False,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(features)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_weights_boxplot(
    values_by_state_feature,
    feature_names: Sequence[str],
    state_labels: Sequence[str],
    state_colors: Sequence[str] | str | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "GLM-HMM weights (across subjects)",
    ylabel: str = "Weight",
    connect_subjects: bool = True,
    subject_lines=None,
    show_ttests: bool = True,
    subject_line_color: str = "#7A7A7A",
    subject_line_alpha: float = 0.15,
    subject_line_width: float = 1.0,
) -> plt.Figure:
    features = [str(name) for name in feature_names]
    labels = [str(label) for label in state_labels]
    if not features or not labels:
        return _empty_plot()

    n_features = len(features)
    n_states = len(labels)
    colors = _resolve_state_colors(labels, state_colors)
    grouped_values = _as_grouped_values(values_by_state_feature, n_states, n_features)
    resolved_subject_lines = _as_subject_lines(subject_lines, n_features, n_states)

    fig, ax = plt.subplots(
        figsize=figsize or (max(6.0, 1.4 * max(1, n_features)), 4.2)
    )
    group_width = 0.8
    hue_width = group_width / max(1, n_states)

    for feat_idx in range(n_features):
        positions = [
            feat_idx + (state_idx - (n_states - 1) / 2.0) * hue_width
            for state_idx in range(n_states)
        ]
        values = [grouped_values[state_idx][feat_idx] for state_idx in range(n_states)]
        line_values = None
        if connect_subjects and resolved_subject_lines is not None:
            line_values = resolved_subject_lines[feat_idx]
        custom_boxplot(
            ax,
            values,
            positions=positions,
            widths=hue_width * 0.78,
            median_colors=colors,
            box_facecolor="white",
            box_alpha=0.85,
            line_values=line_values,
            line_color=subject_line_color,
            line_alpha=subject_line_alpha,
            line_linewidth=subject_line_width,
            showfliers=False,
            showcaps=False,
        )

    if show_ttests and resolved_subject_lines is not None and n_states >= 2:
        state_pairs = list(combinations(range(n_states), 2))
        for feat_idx, per_subject_values in enumerate(resolved_subject_lines):
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
                star = _significance_label(pvalue)
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

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(features, rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        [
            Line2D([0], [0], color=colors[idx], lw=2, label=label)
            for idx, label in enumerate(labels)
        ],
        labels,
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


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
    labels = [str(label) for label in tick_labels]
    if matrix.size == 0 or matrix.ndim != 2:
        return _empty_plot("No transition matrices found.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"tick_labels must have length {matrix.shape[0]}, got {len(labels)}."
        )

    if ax is None:
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
    ax.set_title(title or "Transition matrix")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    fig.tight_layout()
    return fig


def plot_transition_matrix_by_subject(
    matrices,
    subject_ids: Sequence[str],
    tick_labels_by_subject: Sequence[Sequence[str]] | dict[str, Sequence[str]] | None,
    *,
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    cmap: str = "bone",
    cbar_label: str = "probability",
) -> plt.Figure:
    subject_ids = [str(subject_id) for subject_id in subject_ids]
    matrices = [np.asarray(matrix, dtype=float) for matrix in matrices]
    if not matrices:
        return _empty_plot("No transition matrices found.")
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
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (4.2 * n_cols, 3.4 * n_rows),
        squeeze=False,
    )

    for idx, (matrix, subject_id, labels) in enumerate(
        zip(matrices, subject_ids, labels_per_subject, strict=False)
    ):
        ax = axes[idx // n_cols, idx % n_cols]
        plot_transition_matrix(
            matrix,
            labels,
            title=f"Subject {subject_id}",
            ax=ax,
            cmap=cmap,
            cbar=idx == 0,
            cbar_label=cbar_label,
        )

    for idx in range(len(matrices), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.tight_layout()
    return fig


__all__ = [
    "custom_boxplot",
    "plot_feature_boxplot",
    "plot_transition_matrix",
    "plot_transition_matrix_by_subject",
    "plot_weights_boxplot",
]
