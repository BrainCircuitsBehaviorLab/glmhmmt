from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel

from glmhmmt.plots.common import (
    add_pairwise_significance_brackets,
    custom_boxplot,
    resolve_single_axis,
    state_legend_handles,
)
from glmhmmt.plots.utils import require_columns, resolve_state_order, state_palette, to_pandas_df


def _infer_feature_side(feature: str) -> tuple[str, str | None]:
    """Infer a base feature family and optional L/C/R side from a feature name."""
    name = str(feature)
    if len(name) > 2 and name[-2] == "_" and name[-1] in {"L", "C", "R"}:
        return name[:-2], name[-1]
    if len(name) >= 2 and name[0] == "S" and name[1] in {"L", "C", "R"}:
        suffix = name[2:]
        if not suffix or suffix.startswith("x"):
            return f"S{suffix}", name[1]
    if len(name) >= 2 and name[-1] in {"L", "C", "R"}:
        return name[:-1], name[-1]
    return name, None


def _fold_three_choice_raw_weights(weights_df) -> pd.DataFrame | None:
    """Fold three-choice L/R logits into sign-aligned raw weights.

    The center class is assumed to be the implicit baseline. The first explicit
    logit row is treated as the left logit and the second as the right logit.
    For side-coded L/R feature families, coherent raw weights are kept as
    positive contributions and incoherent raw weights are sign-flipped before
    averaging. Non-side features are averaged across the two explicit logits.
    """
    df = to_pandas_df(weights_df, name="weights_df")
    required = ["subject", "state_label", "feature", "weight"]
    if any(column not in df.columns for column in required):
        return None

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["state_label"] = df["state_label"].astype(str)
    df["feature"] = df["feature"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"]).copy()
    if df.empty:
        return None

    if "baseline_class_idx" in df.columns:
        baseline_values = pd.to_numeric(df["baseline_class_idx"], errors="coerce").dropna().astype(int)
        if not baseline_values.empty and set(baseline_values.tolist()) != {1}:
            return None

    if "weight_row_idx" in df.columns:
        df["logit_idx"] = pd.to_numeric(df["weight_row_idx"], errors="coerce")
    elif "class_idx" in df.columns:
        class_values = sorted(
            pd.to_numeric(df["class_idx"], errors="coerce").dropna().astype(int).unique().tolist()
        )
        if len(class_values) != 2:
            return None
        class_to_logit = {class_idx: idx for idx, class_idx in enumerate(class_values)}
        df["logit_idx"] = pd.to_numeric(df["class_idx"], errors="coerce").map(class_to_logit)
    else:
        return None

    df = df.dropna(subset=["logit_idx"]).copy()
    if df.empty:
        return None
    df["logit_idx"] = df["logit_idx"].astype(int)
    if sorted(pd.unique(df["logit_idx"]).tolist()) != [0, 1]:
        return None

    if "state_rank" not in df.columns:
        df["state_rank"] = 0
    else:
        df["state_rank"] = pd.to_numeric(df["state_rank"], errors="coerce").fillna(0).astype(int)

    feature_order = list(pd.unique(df["feature"]).tolist())
    family_sides: dict[str, dict[str, str]] = {}
    for feature in feature_order:
        base, side = _infer_feature_side(feature)
        if side is not None:
            family_sides.setdefault(base, {})[side] = feature

    output_order: list[tuple[str, str]] = []
    seen_families: set[str] = set()
    for feature in feature_order:
        base, side = _infer_feature_side(feature)
        sides = family_sides.get(base, {})
        if side in {"L", "R"} and {"L", "R"} & set(sides):
            if base not in seen_families:
                output_order.append(("family", base))
                seen_families.add(base)
        elif side == "C" and {"L", "R"} & set(sides):
            continue
        else:
            output_order.append(("feature", feature))

    records: list[dict[str, object]] = []
    group_cols = ["subject", "state_label", "state_rank"]
    for (subject, state_label, state_rank), group in df.groupby(group_cols, sort=False):
        weight_by_feature_logit = {
            (str(row.feature), int(row.logit_idx)): float(row.weight)
            for row in (
                group.groupby(["feature", "logit_idx"], as_index=False, observed=False)["weight"]
                .mean()
                .itertuples(index=False)
            )
        }

        for kind, label in output_order:
            values: list[float] = []
            if kind == "family":
                sides = family_sides.get(label, {})
                left_feature = sides.get("L")
                right_feature = sides.get("R")
                if left_feature is not None:
                    left_logit = weight_by_feature_logit.get((left_feature, 0))
                    right_logit = weight_by_feature_logit.get((left_feature, 1))
                    if left_logit is not None:
                        values.append(left_logit)
                    if right_logit is not None:
                        values.append(-right_logit)
                if right_feature is not None:
                    left_logit = weight_by_feature_logit.get((right_feature, 0))
                    right_logit = weight_by_feature_logit.get((right_feature, 1))
                    if right_logit is not None:
                        values.append(right_logit)
                    if left_logit is not None:
                        values.append(-left_logit)
            else:
                values.extend(
                    value
                    for value in (
                        weight_by_feature_logit.get((label, 0)),
                        weight_by_feature_logit.get((label, 1)),
                    )
                    if value is not None
                )

            finite_values = [value for value in values if np.isfinite(value)]
            if finite_values:
                records.append(
                    {
                        "subject": subject,
                        "state_label": state_label,
                        "state_rank": int(state_rank),
                        "feature": label,
                        "weight": float(np.mean(finite_values)),
                    }
                )

    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _prepare_weights_df(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], dict[str, str]]:
    """Prepare emission weights data for plotting a selected class.

    Converts the input weights table to a pandas DataFrame, optionally filters
    one class, applies feature-wise absolute-value transforms, resolves plotting
    order for states and features, and constructs both raw and display-ready
    feature representations for downstream plotting.

    Parameters
    ----------
    weights_df
        Long-format table of emission weights. It must contain at least
        ``subject``, ``state_label``, ``feature``, and ``weight``. If
        ``class_idx`` is provided, it must also contain ``class_idx``.
    K
        Optional number of states to use when constructing the default state
        palette. If omitted, all states present in ``weights_df`` are used.
    class_idx
        Index of the class whose emission weights should be plotted. In
        multinomial models with a baseline class, this selects the explicit
        non-baseline class to visualize.
    state_label_order
        Optional preferred order for state labels. Labels not listed are
        appended afterwards in their resolved default order.
    feature_order
        Optional preferred order for features. Features not listed are appended
        afterwards in their observed order.
    abs_features
        Feature names whose weights should be plotted in absolute value.
    feature_labeler
        Optional callable used to map raw feature names to display labels.

    Returns
    -------
    df
        Prepared pandas DataFrame containing cleaned weights and ordered
        categorical columns for plotting.
    features
        Raw feature names in plotting order.
    display_features
        Display labels for the plotted features, in plotting order.
    states
        State labels in plotting order.
    palette
        Mapping from state label to color.
    """
    df = to_pandas_df(weights_df, name="weights_df")
    require_columns(df, ["subject", "state_label", "feature", "weight"], name="weights_df")

    if class_idx is not None:
        require_columns(df, ["class_idx"], name="weights_df")
        df = df[df["class_idx"].astype(int) == int(class_idx)].copy()

    if df.empty:
        raise ValueError("weights_df has no rows for the requested emission plot.")

    df["subject"] = df["subject"].astype(str)
    df["state_label"] = df["state_label"].astype(str)
    df["feature"] = df["feature"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"]).copy()

    if df.empty:
        raise ValueError("weights_df has no finite weights.")

    abs_feature_set = {str(feature) for feature in abs_features}
    if abs_feature_set:
        mask = df["feature"].isin(abs_feature_set)
        df.loc[mask, "weight"] = df.loc[mask, "weight"].abs()

    available_features = list(pd.unique(df["feature"]))
    requested_features = [
        str(feature)
        for feature in (feature_order or ())
        if str(feature) in available_features
    ]
    features = requested_features + [
        feature for feature in available_features if feature not in requested_features
    ]
    if not features:
        raise ValueError("weights_df does not contain plottable features.")

    base_state_order = resolve_state_order(df)
    if state_label_order is not None:
        requested_states = [str(label) for label in state_label_order]
        states = [label for label in requested_states if label in base_state_order]
        states.extend(label for label in base_state_order if label not in states)
    else:
        states = base_state_order

    display_features = [
        feature_labeler(feature) if feature_labeler else feature
        for feature in features
    ]
    label_map = dict(zip(features, display_features, strict=False))

    df = df[df["feature"].isin(features)].copy()
    df["Feature"] = pd.Categorical(
        df["feature"].map(label_map),
        categories=display_features,
        ordered=True,
    )
    df["State"] = pd.Categorical(
        df["state_label"],
        categories=states,
        ordered=True,
    )

    palette = state_palette(states[: int(K) if K else len(states)])
    fallback_palette = state_palette(states)
    palette.update({
        label: palette.get(label, fallback_palette.get(label, "#333333"))
        for label in states
    })

    return df, features, display_features, states, palette


def _emission_boxplot_payload(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    states: Sequence[str],
) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    """Build grouped values and subject-wise lines for emission boxplots.

    Converts the prepared long-format emission weights table into the grouped
    arrays expected by ``custom_boxplot``. For each feature, it also builds a
    subject-by-state matrix so values from the same subject can be connected
    across states.

    Parameters
    ----------
    df
        Prepared emission weights DataFrame.
    features
        Raw feature names in plotting order.
    states
        State labels in plotting order.

    Returns
    -------
    values_by_state_feature
        Nested list where each state contains one array of weights per feature.
    subject_lines
        One array per feature with shape ``(n_subjects, n_states)``, used to
        connect values from the same subject across states.
    """
    values_by_state_feature: list[list[np.ndarray]] = []
    subject_lines: list[np.ndarray] = []

    subjects = sorted(pd.unique(df["subject"]).tolist())

    for state in states:
        per_feature: list[np.ndarray] = []
        for feature in features:
            vals = df.loc[
                (df["state_label"] == state) & (df["feature"] == feature),
                "weight",
            ].to_numpy(dtype=float)
            per_feature.append(vals)
        values_by_state_feature.append(per_feature)

    for feature in features:
        pivot = (
            df.loc[df["feature"] == feature, ["subject", "state_label", "weight"]]
            .pivot(index="subject", columns="state_label", values="weight")
            .reindex(index=subjects, columns=list(states))
        )
        subject_lines.append(pivot.to_numpy(dtype=float))

    return values_by_state_feature, subject_lines


def emission_weights_by_subject(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot subject-wise emission weights for one selected class.

    Creates one bar plot per subject, with features on the x-axis and weights on
    the y-axis, colored by state. If ``axes`` is not provided, a figure and grid
    of subplots are created automatically. If ``axes`` is provided, the plots
    are drawn into the supplied axes.

    Parameters
    ----------
    weights_df
        Long-format table of emission weights.
    K
        Optional number of states to use when constructing the default state
        palette.
    class_idx
        Index of the class whose emission weights should be plotted. In
        multinomial models, this selects the explicit non-baseline class to
        visualize.
    state_label_order
        Optional preferred order for state labels.
    feature_order
        Optional preferred order for features.
    abs_features
        Feature names whose weights should be plotted in absolute value.
    feature_labeler
        Optional callable used to map raw feature names to display labels.
    axes
        Optional sequence of matplotlib axes to draw into. It must contain at
        least as many axes as required subject panels.
    figsize
        Figure size used only when creating a new figure internally.
    title
        Optional figure title.

    Returns
    -------
    fig
        Matplotlib figure containing the subject panels.
    axes
        Array of axes arranged as ``(n_rows, n_cols)``.

    Raises
    ------
    ValueError
        If fewer axes are provided than the number of required subject panels.
    """
    df, features, display_features, state_order, palette = _prepare_weights_df(
        weights_df,
        K=K,
        class_idx=class_idx,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )

    subjects = sorted(pd.unique(df["subject"]).tolist())
    n_cols = min(3, len(subjects))
    n_rows = int(math.ceil(len(subjects) / n_cols))
    n_panels = n_rows * n_cols

    default_figsize = (
        max(5.0, 0.7 * max(1, len(display_features))) * n_cols,
        3.5 * n_rows,
    )

    created_fig = axes is None
    if created_fig:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize or default_figsize,
            sharey=True,
            squeeze=False,
        )
        axes = np.asarray(axes, dtype=object).ravel()
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) < n_panels:
            raise ValueError(f"Expected at least {n_panels} axes, got {len(axes)}.")
        fig = axes[0].figure

    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        sns.barplot(
            data=df[df["subject"] == subject],
            x="Feature",
            y="weight",
            hue="State",
            hue_order=state_order,
            palette=palette,
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("")
        ax.set_ylabel("Weight")
        ax.set_title(f"Subject {subject}")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    for idx in range(len(subjects), n_panels):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles[: len(state_order)], labels[: len(state_order)], frameon=False)

    if title is not None:
        fig.suptitle(title)

    if created_fig:
        fig.tight_layout(rect=(0, 0, 1, 0.96) if title is not None else None)

    return fig, axes.reshape(n_rows, n_cols)


def emission_weights_summary_lineplot(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot a summary line plot of emission weights for one selected class.

    Displays mean weights by feature and state, with standard-error intervals
    across subjects. If ``ax`` is not provided, a new figure and axis are
    created automatically. If ``ax`` is provided, the plot is drawn into the
    supplied axis.

    Parameters
    ----------
    weights_df
        Long-format table of emission weights.
    K
        Optional number of states to use when constructing the default state
        palette.
    class_idx
        Index of the class whose emission weights should be plotted. In
        multinomial models, this selects the explicit non-baseline class to
        visualize.
    state_label_order
        Optional preferred order for state labels.
    feature_order
        Optional preferred order for features.
    abs_features
        Feature names whose weights should be plotted in absolute value.
    feature_labeler
        Optional callable used to map raw feature names to display labels.
    ax
        Optional matplotlib axis to draw into.
    figsize
        Figure size used only when creating a new figure internally.
    title
        Optional axis title.

    Returns
    -------
    ax
        Axis containing the plotted summary line plot.
    """
    df, _features, display_features, state_order, palette = _prepare_weights_df(
        weights_df,
        K=K,
        class_idx=class_idx,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )

    fig, ax, created_fig = resolve_single_axis(
        ax=ax,
        figsize=figsize or (max(6.0, 1.6 * max(1, len(display_features))), 4.2),
    )

    sns.lineplot(
        data=df,
        x="Feature",
        y="weight",
        hue="State",
        hue_order=state_order,
        palette=palette,
        marker="o",
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        ax=ax,
    )
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("Weight")
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    if created_fig:
        fig.tight_layout()

    return ax


def emission_weights_summary_boxplot(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
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
) -> plt.Axes:
    """Plot a summary boxplot of emission weights.

    Displays the distribution of weights by feature and state using custom
    boxplots, with optional subject-wise lines connecting values from the same
    subject across states within each feature. Optional paired significance
    brackets can also be added across states. For three-choice center-baseline
    tables, the two explicit left/right logits are automatically folded into
    sign-aligned raw weights when ``class_idx`` is not selected.

    Parameters
    ----------
    weights_df
        Long-format table of emission weights.
    K
        Optional number of states to use when constructing the default state
        palette.
    class_idx
        Index of the class whose emission weights should be plotted. In
        multinomial models, this selects the explicit non-baseline class to
        visualize.
    state_label_order
        Optional preferred order for state labels.
    feature_order
        Optional preferred order for features.
    abs_features
        Feature names whose weights should be plotted in absolute value.
    feature_labeler
        Optional callable used to map raw feature names to display labels.
    ax
        Optional matplotlib axis to draw into.
    figsize
        Figure size used only when creating a new figure internally.
    title
        Optional axis title.
    connect_subjects
        Whether to draw lines connecting values from the same subject across
        states within each feature.
    show_ttests
        Whether to add paired t-test significance brackets across states within
        each feature.
    subject_line_color
        Color used for subject-connecting lines.
    subject_line_alpha
        Alpha used for subject-connecting lines.
    subject_line_width
        Line width used for subject-connecting lines.

    Returns
    -------
    ax
        Axis containing the plotted summary boxplot.
    """
    plot_weights = weights_df
    folded = _fold_three_choice_raw_weights(weights_df) if class_idx is None else None
    if folded is not None:
        plot_weights = folded

    df, features, display_features, state_order, palette = _prepare_weights_df(
        plot_weights,
        K=K,
        class_idx=class_idx,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )

    fig, ax, created_fig = resolve_single_axis(
        ax=ax,
        figsize=figsize or (max(6.0, 1.4 * max(1, len(display_features))), 4.2),
    )

    grouped_values, subject_lines = _emission_boxplot_payload(
        df,
        features=features,
        states=state_order,
    )

    n_features = len(features)
    n_states = len(state_order)
    group_width = 0.8
    hue_width = group_width / max(1, n_states)
    colors = [palette[state] for state in state_order]

    for feat_idx in range(n_features):
        positions = [
            feat_idx + (state_idx - (n_states - 1) / 2.0) * hue_width
            for state_idx in range(n_states)
        ]
        values = [grouped_values[state_idx][feat_idx] for state_idx in range(n_states)]
        line_values = subject_lines[feat_idx] if connect_subjects else None

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

    if show_ttests and connect_subjects:
        add_pairwise_significance_brackets(
            ax,
            grouped_values=grouped_values,
            subject_lines=subject_lines,
            n_features=n_features,
            n_states=n_states,
            hue_width=hue_width,
        )

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(display_features, rotation=35, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Weight")
    if title is not None:
        ax.set_title(title)

    ax.legend(
        state_legend_handles(state_order, colors),
        state_order,
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    if created_fig:
        fig.tight_layout()

    return ax


def emission_regressor_lr_boxplot(
    lr_df,
    *,
    feature_order: Sequence[str] | None = None,
    feature_labeler: Callable[[str], str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    connect_subjects: bool = True,
    subject_line_color: str = "#7A7A7A",
    subject_line_alpha: float = 0.15,
    subject_line_width: float = 1.0,
    median_color: str | None = None,
) -> plt.Axes:
    """Plot subject-wise emission regressor LR chi2 values as a summary boxplot.

    Displays the per-subject likelihood-ratio chi2 statistic assigned to each
    regressor using one boxplot per feature. Optional subject-wise lines can be
    drawn across features so each subject remains traceable within the summary.

    Parameters
    ----------
    lr_df
        Long-format table containing at least ``subject``, ``feature``, and
        ``lr_chi2`` columns.
    feature_order
        Optional preferred order for features. Features not listed are
        appended afterwards in their observed order.
    feature_labeler
        Optional callable used to map raw feature names to display labels.
    ax
        Optional matplotlib axis to draw into.
    figsize
        Figure size used only when creating a new figure internally.
    title
        Optional axis title.
    connect_subjects
        Whether to draw lines connecting the same subject across features.
    subject_line_color
        Color used for subject-connecting lines.
    subject_line_alpha
        Alpha used for subject-connecting lines.
    subject_line_width
        Line width used for subject-connecting lines.
    median_color
        Optional median color for the boxplots. If omitted, the first project
        state color is used.

    Returns
    -------
    ax
        Axis containing the plotted LR chi2 summary boxplot.
    """
    df = to_pandas_df(lr_df, name="lr_df")
    require_columns(df, ["subject", "feature", "lr_chi2"], name="lr_df")

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["feature"] = df["feature"].astype(str)
    df["lr_chi2"] = pd.to_numeric(df["lr_chi2"], errors="coerce")
    df = df.dropna(subset=["lr_chi2"]).copy()

    if df.empty:
        raise ValueError("lr_df has no finite LR chi2 values.")

    available_features = list(pd.unique(df["feature"]))
    requested_features = [
        str(feature)
        for feature in (feature_order or ())
        if str(feature) in available_features
    ]
    features = requested_features + [
        feature for feature in available_features if feature not in requested_features
    ]
    if not features:
        raise ValueError("lr_df does not contain plottable features.")

    display_features = [
        feature_labeler(feature) if feature_labeler else feature
        for feature in features
    ]
    label_map = dict(zip(features, display_features, strict=False))

    df = df[df["feature"].isin(features)].copy()
    df["Feature"] = pd.Categorical(
        df["feature"].map(label_map),
        categories=display_features,
        ordered=True,
    )

    fig, ax, created_fig = resolve_single_axis(
        ax=ax,
        figsize=figsize or (max(6.0, 1.35 * max(1, len(display_features))), 4.2),
    )

    grouped_values = [
        df.loc[df["feature"] == feature, "lr_chi2"].to_numpy(dtype=float)
        for feature in features
    ]

    subject_lines = None
    if connect_subjects:
        subjects = sorted(pd.unique(df["subject"]).tolist())
        pivot = (
            df[["subject", "feature", "lr_chi2"]]
            .pivot(index="subject", columns="feature", values="lr_chi2")
            .reindex(index=subjects, columns=features)
        )
        subject_lines = pivot.to_numpy(dtype=float)

    resolved_median_color = median_color or state_palette(["State 1"])["State 1"]
    custom_boxplot(
        ax,
        grouped_values,
        positions=range(len(features)),
        widths=0.62,
        median_colors=[resolved_median_color] * len(features),
        box_facecolor="white",
        box_alpha=0.85,
        line_values=subject_lines,
        line_color=subject_line_color,
        line_alpha=subject_line_alpha,
        line_linewidth=subject_line_width,
        showfliers=False,
        showcaps=False,
    )

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(display_features, rotation=35, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("LR chi2")
    ax.set_ylim(bottom=0.0)
    if title is not None:
        ax.set_title(title)

    if created_fig:
        fig.tight_layout()

    return ax



def _default_lapse_choice_labels(num_classes: int) -> list[str]:
    if num_classes == 2:
        return ["Left", "Right"]
    if num_classes == 3:
        return ["Left", "Center", "Right"]
    return [f"Class {idx}" for idx in range(num_classes)]


def _format_lapse_axis_labels(
    lapse_mode: str,
    lapse_labels: Sequence[str],
    *,
    choice_labels: Sequence[str],
) -> list[str]:
    formatted: list[str] = []
    for idx, raw in enumerate(lapse_labels):
        raw = str(raw)
        if lapse_mode == "history":
            if raw == "repeat":
                formatted.append("Repeat")
                continue
            if raw == "alternate":
                formatted.append("Alternate")
                continue
        choice_idx = idx
        if "_prev_" in raw:
            try:
                choice_idx = int(str(raw).rsplit("_prev_", 1)[1])
            except ValueError:
                choice_idx = idx
        choice = choice_labels[choice_idx] if choice_idx < len(choice_labels) else f"Class {choice_idx}"
        if lapse_mode == "class":
            formatted.append(choice)
        elif lapse_mode == "history_conditioned" and raw.startswith("repeat_prev_"):
            formatted.append(f"Repeat after {choice}")
        elif lapse_mode == "history_conditioned" and raw.startswith("alternate_prev_"):
            formatted.append(f"Alternate after {choice}")
        else:
            formatted.append(raw)
    return formatted


def lapse_rates_boxplot(
    views: dict,
    K: int | None = None,
    ax : plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    choice_labels: Sequence[str] | None = None,
    collapse_history_choices: bool = False,
    annotate_significance: bool = True,
    title: str | None = None,
) -> plt.Figure:
    _ = K
    records: list[dict[str, object]] = []
    selected = [view for view in views.values() if view is not None]
    if not selected:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    num_classes = max(
        (
            len(np.asarray(getattr(view, "lapse_rates", []), dtype=float))
            for view in selected
            if getattr(view, "lapse_rates", None) is not None
        ),
        default=0,
    )
    resolved_choice_labels = list(choice_labels) if choice_labels is not None else _default_lapse_choice_labels(num_classes)

    for view in selected:
        lapse_rates = np.asarray(getattr(view, "lapse_rates", []), dtype=float)
        if lapse_rates.size == 0 or not np.any(np.isfinite(lapse_rates)):
            continue
        lapse_mode = str(getattr(view, "lapse_mode", "none"))
        if lapse_mode == "none" or not np.any(lapse_rates > 0):
            continue
        raw_labels = tuple(getattr(view, "lapse_labels", ())) or tuple(f"class_{idx}" for idx in range(lapse_rates.size))
        display_labels = _format_lapse_axis_labels(
            lapse_mode,
            raw_labels,
            choice_labels=resolved_choice_labels,
        )
        for label, rate in zip(display_labels, lapse_rates, strict=False):
            collapsed_label = label
            if collapse_history_choices and lapse_mode == "history_conditioned":
                if str(label).startswith("Repeat after "):
                    collapsed_label = "Repeat"
                elif str(label).startswith("Alternate after "):
                    collapsed_label = "Alternate"
            records.append(
                {
                    "Subject": getattr(view, "subject", ""),
                    "Lapse": label,
                    "CollapsedLapse": collapsed_label,
                    "Rate": float(rate),
                    "Mode": lapse_mode,
                }
            )

    if not records:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.text(0.5, 0.5, "No fitted lapses", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(records)
    label_col = "CollapsedLapse" if collapse_history_choices else "Lapse"
    order = list(dict.fromkeys(df[label_col].tolist()))
    if collapse_history_choices:
        df = df.groupby(["Subject", label_col], as_index=False)["Rate"].mean()


    fig, ax, created_fig = resolve_single_axis(ax=ax, figsize= figsize or (4.0, 4.0))

    grouped_rates = [
        df.loc[df[label_col] == label, "Rate"].dropna().to_numpy(dtype=float)
        for label in order
    ]
    subject_lines = (
        df.pivot_table(index="Subject", columns=label_col, values="Rate", aggfunc="first")
        .reindex(columns=order)
        .to_numpy(dtype=float)
    )
    custom_boxplot(
        ax,
        grouped_rates,
        positions=np.arange(len(order)),
        widths=0.58,
        median_colors=["#355C7D"] * len(order),
        box_facecolor="white",
        box_alpha=1.0,
        line_values=subject_lines,
        line_color="#355C7D",
        line_alpha=0.25,
        line_linewidth=1.1,
        showfliers=False,
        zorder=1,
    )
    if annotate_significance and len(order) >= 2:
        def _star(pval: float) -> str:
            if pval < 0.001:
                return "***"
            if pval < 0.01:
                return "**"
            if pval < 0.05:
                return "*"
            return "ns"

        paired = df.pivot_table(
            index="Subject", columns=label_col, values="Rate", aggfunc="first"
        ).reindex(columns=order)
        finite_groups = [vals[np.isfinite(vals)] for vals in grouped_rates if vals.size > 0]
        finite = np.concatenate(finite_groups) if finite_groups else np.array([], dtype=float)
        y_min = float(finite.min()) if finite.size else 0.0
        y_max = float(finite.max()) if finite.size else 1.0
        y_range = y_max - y_min
        if y_range <= 0:
            y_range = max(y_max, 1.0)
        y_offset_step = y_range * 0.08
        bar_height = y_range * 0.03
        current_y = y_max + y_offset_step
        positions = np.arange(len(order), dtype=float)

        for left_idx in range(len(order)):
            for right_idx in range(left_idx + 1, len(order)):
                pair = paired.iloc[:, [left_idx, right_idx]].dropna()
                if len(pair) < 2:
                    continue
                _, pval = ttest_rel(pair.iloc[:, 0], pair.iloc[:, 1])
                label = _star(float(pval))
                x1 = positions[left_idx]
                x2 = positions[right_idx]
                ax.plot(
                    [x1, x1, x2, x2],
                    [current_y, current_y + bar_height, current_y + bar_height, current_y],
                    lw=1.0,
                    c="k",
                )
                ax.text(
                    (x1 + x2) / 2,
                    current_y + bar_height,
                    label,
                    ha="center",
                    va="bottom",
                    color="k",
                )
                current_y += y_offset_step

    ax.set_xlabel("")
    ax.set_ylabel("Lapse rate")
    ax.set_ylim(bottom=0.0)
    top_ylim = ax.get_ylim()[1]
    if annotate_significance and len(order) >= 2:
        ax.set_ylim(0.0, max(top_ylim, current_y + y_offset_step * 0.5))
    ax.set_title(title or "Lapse rates")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=18)
    if created_fig:
        fig.tight_layout()
    return fig