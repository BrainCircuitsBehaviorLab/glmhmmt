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
    x = trial_df["trial_in_session"].to_numpy(dtype=float)
    T = len(x)
    posterior_wide = (
        posterior_df.pivot_table(index="trial_in_session", columns="state_label", values="probability", aggfunc="mean")
        .reindex(index=x, columns=states)
        .fillna(0.0)
    )
    probs = posterior_wide.to_numpy(dtype=float)

    trace_cols: Sequence[str] = payload.get("trace_cols", [])
    n_rows = 2 if trace_cols else 1
    height_ratios = [2, 1.5] if trace_cols else [1]
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=payload.get("figsize", (14, 5 + 2.5 * (n_rows - 1))),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes)
    ax = axes[0]

    engaged_window = int(payload.get("engaged_window", 20))
    engaged_trace_mode = str(payload.get("engaged_trace_mode", "rolling")).strip().lower()
    engaged_label = payload.get("engaged_state_label")
    if engaged_label not in states:
        engaged_matches = [state for state in states if "engaged" in state.lower() and "dis" not in state.lower()]
        engaged_label = engaged_matches[0] if engaged_matches else states[0]
    engaged_idx = states.index(engaged_label)
    if engaged_trace_mode == "rolling":
        probs_display = pd.DataFrame(probs).rolling(engaged_window, min_periods=1).mean().to_numpy(dtype=float)
        engaged_trace = pd.Series(probs[:, engaged_idx]).rolling(engaged_window, min_periods=1).mean().to_numpy(dtype=float)
        engaged_trace_label = f"P({engaged_label}) rolling ({engaged_window} trials)"
    else:
        probs_display = probs
        engaged_trace = probs[:, engaged_idx]
        engaged_trace_label = f"P({engaged_label}) raw"

    bottom = np.zeros(T)
    for state_idx, state in enumerate(states):
        ax.fill_between(
            x,
            bottom,
            bottom + probs_display[:, state_idx],
            alpha=0.7,
            color=palette[state],
            label=state,
        )
        bottom += probs_display[:, state_idx]

    engaged_color = palette[engaged_label]
    ax.fill_between(x, 0.0, engaged_trace, color=engaged_color, alpha=0.18)
    ax.plot(x, engaged_trace, color=engaged_color, lw=2, label=engaged_trace_label)
    x = trial_df["trial_in_session"].to_numpy(dtype=float)
    change_trials = np.asarray(payload.get("change_trials", []), dtype=int)
    if change_trials.size:
        valid_change_trials = change_trials[(change_trials >= 0) & (change_trials < T)]
        ax.scatter(
            x[valid_change_trials],
            engaged_trace[valid_change_trials],
            s=34,
            color="crimson",
            linewidths=0,
            zorder=6,
            label="State change",
        )

    if "response" in trial_df.columns:
        response = pd.to_numeric(trial_df["response"], errors="coerce").to_numpy(dtype=float)
        choice_colors = payload.get("choice_colors")
        choice_labels = payload.get("choice_labels")
        if choice_colors is None or choice_labels is None:
            num_classes = int(payload.get("num_classes", max(2, int(np.nanmax(response)) + 1 if np.isfinite(response).any() else 2)))
            if num_classes == 2:
                choice_colors = {0: "royalblue", 1: "tomato"}
                choice_labels = {0: "L", 1: "R"}
            else:
                choice_colors = {0: "royalblue", 1: "gold", 2: "tomato"}
                choice_labels = {0: "L", 1: "C", 2: "R"}
        for resp, color in choice_colors.items():
            mask = response == float(resp)
            if mask.sum() == 0:
                continue
            ax.scatter(
                x[mask],
                np.ones(int(mask.sum())) * 1.03,
                c=color,
                s=5,
                marker="|",
                label=choice_labels.get(resp, str(resp)),
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    ax.set_ylim(0, 1)
    ax.set_ylabel("State probability")
    ax.set_title(payload.get("title", "Session deepdive"), pad=25)

    if "correct_bool" in trial_df.columns:
        rolling_window = int(payload.get("rolling_accuracy_window", 20))
        hits = pd.to_numeric(trial_df["correct_bool"], errors="coerce")
        rolling_acc = hits.rolling(rolling_window, min_periods=1).mean().to_numpy(dtype=float) * 100.0
        axr = ax.twinx()
        axr.plot(
            x,
            rolling_acc,
            color="black",
            lw=1.8,
            linestyle="-",
            alpha=0.7,
            label=f"Rolling accuracy ({rolling_window} trials)",
        )
        chance = payload.get("chance_level")
        if chance is not None:
            chance_pct = float(chance) * 100.0 if float(chance) <= 1.0 else float(chance)
            axr.axhline(chance_pct, color="grey", lw=0.9, linestyle="--", alpha=0.5)
        axr.set_ylim(0, 105)
        axr.set_ylabel("Accuracy (%)", color="black")
        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = axr.get_legend_handles_labels()
        ax.legend(lines + lines_r, labels + labels_r, bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)
    else:
        ax.legend(bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)

    if trace_cols:
        ax2 = axes[1]
        trace_colors = {
            "A_plus": "royalblue",
            "A_minus": "gold",
            "A_L": "royalblue",
            "A_C": "gold",
            "A_R": "tomato",
        }
        for col in trace_cols:
            if col in trial_df.columns:
                ax2.plot(
                    x,
                    pd.to_numeric(trial_df[col], errors="coerce"),
                    label=str(col),
                    color=trace_colors.get(str(col), "gray"),
                    lw=1.5,
                    alpha=0.85,
                )
        ax2.set_ylabel("Action trace")
        ax2.set_ylim(0, None)
        ax2.set_xlabel("Trial within session")
        ax2.legend(bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)
    else:
        ax.set_xlabel("Trial within session")

    sns.despine(fig=fig)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    return fig
