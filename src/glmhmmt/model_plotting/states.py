from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from glmhmmt.model_plotting.sessions import plot_session_trajectories
from glmhmmt.model_plotting.utils import require_columns, resolve_state_order, state_palette, to_pandas_df


def plot_state_accuracy(payload: dict) -> tuple[plt.Figure, pd.DataFrame]:
    accuracy_df = to_pandas_df(payload.get("accuracy_df"), name="accuracy_df")
    require_columns(accuracy_df, ["subject", "state_label", "accuracy"], name="accuracy_df")
    if accuracy_df.empty:
        raise ValueError("accuracy_df is empty.")

    states = payload.get("state_order") or resolve_state_order(accuracy_df)
    palette = state_palette(states)
    fig, ax = plt.subplots(figsize=payload.get("figsize", (max(5, 1.2 * len(states)), 4)))
    sns.boxplot(
        data=accuracy_df,
        x="state_label",
        y="accuracy",
        order=states,
        palette=palette,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=accuracy_df,
        x="state_label",
        y="accuracy",
        order=states,
        palette=palette,
        alpha=0.55,
        ax=ax,
    )
    chance = payload.get("chance_level")
    if chance is not None:
        ax.axhline(float(chance), color="grey", ls="--", lw=1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_title(payload.get("title", "Accuracy by state"))
    sns.despine(fig=fig)
    fig.tight_layout()

    summary = (
        accuracy_df.groupby("state_label", observed=True)
        .agg(accuracy=("accuracy", "mean"), n_subjects=("subject", "nunique"))
        .reindex(states)
        .reset_index()
    )
    return fig, summary


def plot_state_occupancy_overall_boxplot(payload: dict) -> plt.Figure:
    occupancy_df = to_pandas_df(payload.get("occupancy_df"), name="occupancy_df")
    require_columns(occupancy_df, ["subject", "state_label", "occupancy"], name="occupancy_df")
    if occupancy_df.empty:
        raise ValueError("occupancy_df is empty.")
    states = payload.get("state_order") or resolve_state_order(occupancy_df)
    palette = state_palette(states)
    fig, ax = plt.subplots(figsize=payload.get("figsize", (max(5, 1.2 * len(states)), 4)))
    sns.boxplot(
        data=occupancy_df,
        x="state_label",
        y="occupancy",
        order=states,
        palette=palette,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=occupancy_df,
        x="state_label",
        y="occupancy",
        order=states,
        palette=palette,
        alpha=0.55,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of trials")
    ax.set_title(payload.get("title", "State occupancy"))
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_state_occupancy(payload: dict) -> plt.Figure:
    occupancy_df = to_pandas_df(payload.get("occupancy_df"), name="occupancy_df")
    session_df = to_pandas_df(payload.get("session_occupancy_df"), name="session_occupancy_df")
    require_columns(occupancy_df, ["subject", "state_label", "occupancy"], name="occupancy_df")
    require_columns(session_df, ["subject", "session", "state_label", "occupancy"], name="session_occupancy_df")
    if occupancy_df.empty or session_df.empty:
        raise ValueError("state occupancy payload is empty.")

    states = payload.get("state_order") or resolve_state_order(occupancy_df)
    palette = state_palette(states)
    fig, axes = plt.subplots(1, 3, figsize=payload.get("figsize", (14, 4)))

    sns.boxplot(data=occupancy_df, x="state_label", y="occupancy", order=states, palette=palette, showfliers=False, ax=axes[0])
    sns.stripplot(data=occupancy_df, x="state_label", y="occupancy", order=states, palette=palette, alpha=0.5, ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Fraction of trials")
    axes[0].set_title("Overall")

    sns.lineplot(data=session_df, x="session", y="occupancy", hue="state_label", hue_order=states, palette=palette, errorbar="se", ax=axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of trials")
    axes[1].set_title("By session")
    axes[1].legend(frameon=False)

    switches_df = payload.get("switches_df")
    if switches_df is not None:
        switches = to_pandas_df(switches_df, name="switches_df")
        require_columns(switches, ["subject", "session", "n_switches"], name="switches_df")
        sns.histplot(data=switches, x="n_switches", bins=15, color="0.25", ax=axes[2])
        axes[2].set_xlabel("MAP switches per session")
    else:
        axes[2].axis("off")
    axes[2].set_title("Switches")

    fig.suptitle(payload.get("title", "State occupancy"), y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_state_dwell_times_by_subject(payload: dict) -> plt.Figure:
    dwell_df = to_pandas_df(payload.get("dwell_df"), name="dwell_df")
    require_columns(dwell_df, ["subject", "state_label", "dwell"], name="dwell_df")
    if dwell_df.empty:
        raise ValueError("dwell_df is empty.")
    states = payload.get("state_order") or resolve_state_order(dwell_df)
    palette = state_palette(states)
    fig, ax = plt.subplots(figsize=payload.get("figsize", (9, 4)))
    sns.ecdfplot(data=dwell_df, x="dwell", hue="state_label", hue_order=states, palette=palette, complementary=True, ax=ax)
    ax.set_xlabel("Dwell length (trials)")
    ax.set_ylabel("Survival probability")
    ax.set_title(payload.get("title", "Dwell times by subject"))
    ax.legend(frameon=False)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_state_dwell_times_summary(payload: dict) -> plt.Figure:
    dwell_df = to_pandas_df(payload.get("dwell_df"), name="dwell_df")
    require_columns(dwell_df, ["state_label", "dwell"], name="dwell_df")
    if dwell_df.empty:
        raise ValueError("dwell_df is empty.")
    states = payload.get("state_order") or resolve_state_order(dwell_df)
    palette = state_palette(states)
    fig, ax = plt.subplots(figsize=payload.get("figsize", (max(5, 1.2 * len(states)), 4)))
    sns.boxplot(data=dwell_df, x="state_label", y="dwell", order=states, palette=palette, showfliers=False, ax=ax)
    sns.stripplot(data=dwell_df, x="state_label", y="dwell", order=states, palette=palette, alpha=0.35, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Dwell length (trials)")
    ax.set_title(payload.get("title", "Dwell-time summary"))
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_state_dwell_times(payload: dict) -> plt.Figure:
    return plot_state_dwell_times_by_subject(payload)


def plot_state_posterior_count_kde(payload: dict) -> plt.Figure:
    count_df = to_pandas_df(payload.get("count_df"), name="count_df")
    require_columns(count_df, ["subject", "state_label", "n_trials"], name="count_df")
    if count_df.empty:
        raise ValueError("count_df is empty.")
    states = payload.get("state_order") or resolve_state_order(count_df)
    palette = state_palette(states)
    fig, ax = plt.subplots(figsize=payload.get("figsize", (7, 4)))
    sns.kdeplot(data=count_df, x="n_trials", hue="state_label", hue_order=states, palette=palette, fill=False, ax=ax)
    ax.set_xlabel("Trials assigned to state")
    ax.set_title(payload.get("title", "Posterior count distribution"))
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_change_triggered_posteriors_summary(payload: dict) -> plt.Figure:
    return plot_session_trajectories(payload)


def plot_change_triggered_posteriors_by_subject(payload: dict) -> plt.Figure:
    return plot_session_trajectories(payload)
