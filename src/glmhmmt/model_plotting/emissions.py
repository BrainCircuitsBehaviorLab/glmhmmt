from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from glmhmmt.model_plotting.utils import require_columns, resolve_state_order, state_palette, to_pandas_df


def _prepare_weights_df(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
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
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce") * float(weight_sign)
    df = df.dropna(subset=["weight"]).copy()
    if df.empty:
        raise ValueError("weights_df has no finite weights.")

    abs_feature_set = {str(feature) for feature in abs_features}
    if abs_feature_set:
        mask = df["feature"].isin(abs_feature_set)
        df.loc[mask, "weight"] = df.loc[mask, "weight"].abs()

    available_features = list(pd.unique(df["feature"]))
    requested_features = [str(feature) for feature in (feature_order or ()) if str(feature) in available_features]
    features = requested_features + [feature for feature in available_features if feature not in requested_features]
    if not features:
        raise ValueError("weights_df does not contain plottable features.")

    base_state_order = resolve_state_order(df)
    if state_label_order is not None:
        requested_states = [str(label) for label in state_label_order]
        states = [label for label in requested_states if label in base_state_order]
        states.extend(label for label in base_state_order if label not in states)
    else:
        states = base_state_order

    labels = [feature_labeler(feature) if feature_labeler else feature for feature in features]
    label_map = dict(zip(features, labels, strict=False))
    df = df[df["feature"].isin(features)].copy()
    df["Feature"] = pd.Categorical(df["feature"].map(label_map), categories=labels, ordered=True)
    df["State"] = pd.Categorical(df["state_label"], categories=states, ordered=True)

    palette = state_palette(states[: int(K) if K else len(states)])
    palette.update({label: palette.get(label, state_palette(states).get(label, "#333333")) for label in states})
    return df, labels, states, palette


def plot_emission_weights_by_subject(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    df, display_features, state_order, palette = _prepare_weights_df(
        weights_df,
        K=K,
        class_idx=class_idx,
        weight_sign=weight_sign,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    subjects = sorted(pd.unique(df["subject"]).tolist())
    n_cols = min(3, len(subjects))
    n_rows = int(math.ceil(len(subjects) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5.0, 0.7 * max(1, len(display_features))) * n_cols, 3.5 * n_rows),
        sharey=True,
        squeeze=False,
    )

    for idx, subject in enumerate(subjects):
        ax = axes[idx // n_cols, idx % n_cols]
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

    for idx in range(len(subjects), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles[: len(state_order)], labels[: len(state_order)], frameon=False)
    fig.suptitle(title or "Emission weights by subject", y=1.01)
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_emission_weights_summary_lineplot(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    df, display_features, state_order, palette = _prepare_weights_df(
        weights_df,
        K=K,
        class_idx=class_idx,
        weight_sign=weight_sign,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * max(1, len(display_features))), 4.2))
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
    ax.set_title(title or "Emission weights summary")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_emission_weights_summary_boxplot(
    weights_df,
    *,
    K: int | None = None,
    class_idx: int | None = None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    df, display_features, state_order, palette = _prepare_weights_df(
        weights_df,
        K=K,
        class_idx=class_idx,
        weight_sign=weight_sign,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * max(1, len(display_features))), 4.2))
    sns.boxplot(
        data=df,
        x="Feature",
        y="weight",
        hue="State",
        hue_order=state_order,
        palette=palette,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="Feature",
        y="weight",
        hue="State",
        hue_order=state_order,
        palette=palette,
        dodge=True,
        alpha=0.35,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("Weight")
    ax.set_title(title or "Emission weights summary")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[: len(state_order)], labels[: len(state_order)], frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_emission_weights_summary(weights_df, **kwargs) -> plt.Figure:
    df = to_pandas_df(weights_df, name="weights_df")
    require_columns(df, ["subject"], name="weights_df")
    n_subjects = df["subject"].astype(str).nunique()
    if n_subjects > 1:
        return plot_emission_weights_summary_boxplot(weights_df, **kwargs)
    return plot_emission_weights_by_subject(weights_df, **kwargs)


def plot_emission_weights(weights_df, **kwargs) -> tuple[plt.Figure, plt.Figure]:
    return (
        plot_emission_weights_by_subject(weights_df, **kwargs),
        plot_emission_weights_summary(weights_df, **kwargs),
    )


def plot_binary_emission_weights_by_subject(weights_df, *, K: int | None = None, **kwargs) -> plt.Figure:
    return plot_emission_weights_by_subject(weights_df, K=K, class_idx=0, **kwargs)


def plot_binary_emission_weights_summary_lineplot(weights_df, *, K: int | None = None, **kwargs) -> plt.Figure:
    return plot_emission_weights_summary_lineplot(weights_df, K=K, class_idx=0, **kwargs)


def plot_binary_emission_weights_summary_boxplot(weights_df, *, K: int | None = None, **kwargs) -> plt.Figure:
    return plot_emission_weights_summary_boxplot(weights_df, K=K, class_idx=0, **kwargs)


def plot_binary_emission_weights_summary(weights_df, *, K: int | None = None, **kwargs) -> plt.Figure:
    return plot_emission_weights_summary(weights_df, K=K, class_idx=0, **kwargs)


def plot_binary_emission_weights(weights_df, *, K: int | None = None, **kwargs) -> tuple[plt.Figure, plt.Figure]:
    return (
        plot_binary_emission_weights_by_subject(weights_df, K=K, **kwargs),
        plot_binary_emission_weights_summary(weights_df, K=K, **kwargs),
    )
