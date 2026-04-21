from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glmhmmt.model_plotting.utils import require_columns, resolve_state_order, state_palette, to_pandas_df


def plot_posterior_probs(
    posterior_df,
    *,
    subject: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    df = to_pandas_df(posterior_df, name="posterior_df")
    require_columns(df, ["subject", "trial_idx", "state_label", "probability"], name="posterior_df")
    if subject is not None:
        df = df[df["subject"].astype(str) == str(subject)].copy()
    if df.empty:
        raise ValueError("posterior_df has no rows for the requested subject.")

    if subject is None:
        subject = str(df["subject"].iloc[0])
        df = df[df["subject"].astype(str) == subject].copy()

    states = resolve_state_order(df)
    palette = state_palette(states)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for state in states:
        sub = df[df["state_label"].astype(str) == state].sort_values("trial_idx")
        ax.plot(sub["trial_idx"], sub["probability"], lw=1.5, color=palette[state], label=state)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Posterior probability")
    ax.set_title(title or f"Posterior probabilities - {subject}")
    ax.legend(frameon=False, ncol=min(4, len(states)))
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_session_trajectories(payload: dict) -> plt.Figure:
    trajectory_df = to_pandas_df(payload.get("trajectory_df"), name="trajectory_df")
    require_columns(
        trajectory_df,
        ["subject", "session", "trial_in_session", "state_label", "probability"],
        name="trajectory_df",
    )
    if trajectory_df.empty:
        raise ValueError("trajectory_df is empty.")

    states = payload.get("state_order") or resolve_state_order(trajectory_df)
    palette = state_palette(states)
    fig, ax = plt.subplots(figsize=payload.get("figsize", (9, 4)))
    sns.lineplot(
        data=trajectory_df,
        x="trial_in_session",
        y="probability",
        hue="state_label",
        hue_order=states,
        palette=palette,
        errorbar="se",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial within session")
    ax.set_ylabel("Mean posterior probability")
    ax.set_title(payload.get("title", "Session trajectories"))
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_session_deepdive(payload: dict) -> plt.Figure:
    trial_df = to_pandas_df(payload.get("trial_df"), name="trial_df")
    posterior_df = to_pandas_df(payload.get("posterior_df"), name="posterior_df")
    require_columns(trial_df, ["trial_in_session"], name="trial_df")
    require_columns(posterior_df, ["trial_in_session", "state_label", "probability"], name="posterior_df")
    if trial_df.empty or posterior_df.empty:
        raise ValueError("session deepdive payload is empty.")

    states = payload.get("state_order") or resolve_state_order(posterior_df)
    palette = state_palette(states)
    fig, axes = plt.subplots(3, 1, figsize=payload.get("figsize", (11, 7)), sharex=True)

    ax = axes[0]
    for state in states:
        sub = posterior_df[posterior_df["state_label"].astype(str) == state].sort_values("trial_in_session")
        ax.plot(sub["trial_in_session"], sub["probability"], color=palette[state], lw=1.5, label=state)
    for change_trial in payload.get("change_trials", []):
        ax.axvline(change_trial, color="crimson", lw=0.8, alpha=0.45)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Posterior")
    ax.legend(frameon=False, ncol=min(4, len(states)))

    x = trial_df["trial_in_session"].to_numpy(dtype=float)
    if "correct_bool" in trial_df.columns:
        y = pd.to_numeric(trial_df["correct_bool"], errors="coerce").rolling(15, min_periods=1).mean()
        axes[1].plot(x, y, color="black", lw=1.3)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Rolling accuracy")
    else:
        axes[1].axis("off")

    trace_cols: Sequence[str] = payload.get("trace_cols", [])
    if trace_cols:
        for col in trace_cols:
            if col in trial_df.columns:
                axes[2].plot(x, pd.to_numeric(trial_df[col], errors="coerce"), lw=1.0, label=str(col))
        axes[2].legend(frameon=False, ncol=min(4, len(trace_cols)))
        axes[2].set_ylabel("Regressor")
    elif "state_rank" in trial_df.columns:
        axes[2].step(x, trial_df["state_rank"], where="mid", color="black", lw=1.0)
        axes[2].set_ylabel("MAP state rank")
    else:
        axes[2].axis("off")

    axes[2].set_xlabel("Trial within session")
    fig.suptitle(payload.get("title", "Session deepdive"), y=0.99)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig
