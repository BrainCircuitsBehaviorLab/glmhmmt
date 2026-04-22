from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from glmhmmt.plots_common import custom_boxplot
from glmhmmt.model_plotting.utils import require_columns, resolve_state_order, state_palette, to_pandas_df


def _line_sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def _grouped_values(df: pd.DataFrame, *, label_col: str, value_col: str, labels: list[str]) -> list[np.ndarray]:
    return [
        pd.to_numeric(df.loc[df[label_col].astype(str) == label, value_col], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for label in labels
    ]


def _plot_state_distribution(ax: plt.Axes, grouped_vals: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    custom_boxplot(
        ax,
        grouped_vals,
        positions=np.arange(len(labels)),
        widths=0.5,
        median_colors=colors,
        showfliers=False,
        zorder=1,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)


def _plot_session_violin(ax: plt.Axes, grouped_vals: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    valid = [
        (pos, np.asarray(vals, dtype=float), color)
        for pos, (vals, color) in enumerate(zip(grouped_vals, colors, strict=False))
        if np.asarray(vals, dtype=float).size > 0
    ]
    if valid:
        positions, values, violin_colors = zip(*valid, strict=False)
        violin = ax.violinplot(
            list(values),
            positions=list(positions),
            widths=0.5,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(violin["bodies"], violin_colors, strict=False):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)
            body.set_linewidth(1.0)
        violin["cmedians"].set_color("#666666")
        violin["cmedians"].set_linewidth(1.1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)


def _plot_switch_hist(ax: plt.Axes, switches: np.ndarray, title: str) -> None:
    xlabel = "# state switches / session"
    switches = np.asarray(switches, dtype=float)
    switches = switches[np.isfinite(switches)]
    if switches.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return
    max_switches = int(np.nanmax(switches))
    ax.hist(
        switches,
        bins=np.arange(-0.5, max_switches + 1.5, 1.0),
        color="#888888",
        alpha=0.75,
        edgecolor="white",
    )
    ax.set_xlim(-0.5, max_switches + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# sessions")
    ax.set_title(title)


def plot_state_accuracy(payload: dict) -> tuple[plt.Figure, pd.DataFrame]:
    accuracy_df = to_pandas_df(payload.get("accuracy_df"), name="accuracy_df")
    require_columns(accuracy_df, ["subject", "state_label", "accuracy", "n_trials"], name="accuracy_df")
    if accuracy_df.empty:
        raise ValueError("accuracy_df is empty.")

    states = payload.get("state_order") or resolve_state_order(accuracy_df)
    if "All" in set(accuracy_df["state_label"].astype(str)):
        states = ["All", *[state for state in states if state != "All"]]
    palette = {"All": "#999999", **state_palette([state for state in states if state != "All"])}

    work = accuracy_df.copy()
    work["accuracy_pct"] = pd.to_numeric(work["accuracy"], errors="coerce")
    if work["accuracy_pct"].max(skipna=True) <= 1.0:
        work["accuracy_pct"] *= 100.0

    grouped_acc = _grouped_values(work, label_col="state_label", value_col="accuracy_pct", labels=states)
    subject_lines = (
        work.pivot_table(index="subject", columns="state_label", values="accuracy_pct", aggfunc="first")
        .reindex(columns=states)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(figsize=payload.get("figsize", (4, 4)))
    _plot_state_distribution(ax, grouped_acc, states, [palette.get(state, "k") for state in states])
    if subject_lines.size:
        for row in subject_lines:
            ax.plot(
                np.arange(len(states)),
                row,
                color="#B0B0B0",
                alpha=0.15,
                linewidth=1.25,
                zorder=2,
            )
    chance = payload.get("chance_level")
    if chance is not None:
        chance_pct = float(chance) * 100.0 if float(chance) <= 1.0 else float(chance)
        ax.axhline(chance_pct, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_ylim(max(0.0, chance_pct - 10.0), 105.0)
    else:
        ax.set_ylim(0.0, 105.0)
    ax.set_xticklabels(states, rotation=20, ha="right")
    ax.set_xlabel("State")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(payload.get("title", "Accuracy by state"))
    sns.despine(fig=fig)
    fig.tight_layout()

    summary = (
        work.groupby("state_label", observed=True)
        .agg(**{"mean_acc (%)": ("accuracy_pct", "mean"), "total_trials": ("n_trials", "sum")})
        .reindex(states)
        .round(1)
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
    _plot_state_distribution(
        ax,
        _grouped_values(occupancy_df, label_col="state_label", value_col="occupancy", labels=states),
        states,
        [palette[state] for state in states],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fractional occupancy")
    ax.set_title(payload.get("title", "All selected subjects - overall occupancy"))
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

    switches = to_pandas_df(payload.get("switches_df"), name="switches_df")
    require_columns(switches, ["subject", "session", "n_switches"], name="switches_df")

    states = payload.get("state_order") or resolve_state_order(occupancy_df)
    palette = state_palette(states)
    subjects = list(pd.unique(occupancy_df["subject"].astype(str)))
    n_rows = len(subjects) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=payload.get("figsize", (14, 3.8 * n_rows)), squeeze=False)
    colors = [palette[state] for state in states]

    ax_all_occ, ax_all_sess, ax_all_chg = axes[0]
    _plot_state_distribution(
        ax_all_occ,
        _grouped_values(occupancy_df, label_col="state_label", value_col="occupancy", labels=states),
        states,
        colors,
    )
    ax_all_occ.set_ylim(0, 1)
    ax_all_occ.set_ylabel("Fractional occupancy")
    ax_all_occ.set_title("All selected subjects - overall occupancy")

    _plot_session_violin(
        ax_all_sess,
        _grouped_values(session_df, label_col="state_label", value_col="occupancy", labels=states),
        states,
        colors,
    )
    ax_all_sess.set_ylabel("Session occupancy")
    ax_all_sess.set_title("All selected sessions - occupancy by session")
    _plot_switch_hist(ax_all_chg, switches["n_switches"].to_numpy(dtype=float), "All selected sessions - state switches")

    for row_idx, subject in enumerate(subjects, start=1):
        subj_occ = occupancy_df[occupancy_df["subject"].astype(str) == subject]
        subj_session = session_df[session_df["subject"].astype(str) == subject]
        subj_switch = switches[switches["subject"].astype(str) == subject]
        ax_occ, ax_box, ax_chg = axes[row_idx]
        occ = (
            subj_occ.set_index(subj_occ["state_label"].astype(str))["occupancy"]
            .reindex(states)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        ax_occ.bar(states, occ, color=colors, alpha=0.85)
        ax_occ.set_ylim(0, 1)
        ax_occ.set_ylabel("Fractional occupancy")
        ax_occ.set_title(f"Subject {subject} - overall occupancy")

        _plot_session_violin(
            ax_box,
            _grouped_values(subj_session, label_col="state_label", value_col="occupancy", labels=states),
            states,
            colors,
        )
        ax_box.set_ylabel("Session occupancy")
        ax_box.set_title(f"Subject {subject} - occupancy by session")
        _plot_switch_hist(ax_chg, subj_switch["n_switches"].to_numpy(dtype=float), f"Subject {subject} - state switches")

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
    summary_df = to_pandas_df(payload.get("dwell_summary_df"), name="dwell_summary_df")
    require_columns(summary_df, ["state_label", "bin_center", "empirical"], name="dwell_summary_df")
    if summary_df.empty:
        raise ValueError("dwell_summary_df is empty.")
    states = payload.get("state_order") or resolve_state_order(summary_df)
    palette = state_palette(states)
    max_dwell = int(payload.get("max_dwell", max(1.0, summary_df["bin_center"].max() * 2.0)))
    fig, axes = plt.subplots(
        1,
        len(states),
        figsize=payload.get("figsize", (4.1 * len(states), 3.3)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y_max = float(payload.get("y_max", max(0.01, summary_df["empirical"].max())))
    for idx, state in enumerate(states):
        ax = axes[0, idx]
        sub = summary_df[summary_df["state_label"].astype(str) == state].sort_values("bin_center")
        color = palette[state]
        has_predicted = "predicted" in sub.columns and sub["predicted"].notna().any()
        if has_predicted:
            ax.plot(sub["bin_center"], sub["predicted"], color=color, lw=2.6, marker="o", ms=5.5)
            y_max = max(y_max, float(sub["predicted"].max()))
        yerr = None
        if {"yerr_low", "yerr_high"}.issubset(sub.columns):
            yerr = np.vstack([sub["yerr_low"].to_numpy(dtype=float), sub["yerr_high"].to_numpy(dtype=float)])
            y_max = max(y_max, float((sub["empirical"] + sub["yerr_high"]).max()))
        ax.errorbar(
            sub["bin_center"],
            sub["empirical"],
            yerr=yerr,
            color=color,
            lw=2.4,
            linestyle="--" if has_predicted else "-",
            marker="o",
            ms=5.5,
            capsize=0,
            elinewidth=2.2,
        )
        ax.set_title(state, color=color, fontsize=18, pad=14)
        ax.set_xlim(0, max_dwell)
        ax.set_ylim(0.0, 1.08 * y_max)
        ax.set_xticks([0, max_dwell // 3, (2 * max_dwell) // 3, max_dwell])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.supylabel("frac. dwell times in state")
    fig.supxlabel("state dwell time\n# trials")
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.6, linestyle="-", label="predicted"),
        Line2D([0], [0], color="black", lw=2.4, linestyle="--", marker="o", ms=5.5, label="empirical"),
    ]
    if "predicted" not in summary_df.columns or summary_df["predicted"].isna().all():
        legend_handles = [legend_handles[1]]
    axes[0, min(1, len(states) - 1)].legend(handles=legend_handles, loc="upper right", frameon=False)
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_state_dwell_times(payload: dict) -> plt.Figure:
    return plot_state_dwell_times_by_subject(payload)


def plot_state_posterior_count_kde(payload: dict) -> plt.Figure:
    posterior_df = to_pandas_df(payload.get("posterior_df"), name="posterior_df")
    require_columns(posterior_df, ["state_label", "probability"], name="posterior_df")
    if posterior_df.empty:
        raise ValueError("posterior_df is empty.")
    states = payload.get("state_order") or resolve_state_order(posterior_df)
    palette = state_palette(states)
    bins = int(payload.get("bins", 20))
    edges = np.linspace(0.0, 1.0, bins + 1)
    fig, ax = plt.subplots(figsize=payload.get("figsize", (5, 4)))
    legend_handles = []
    for state in states:
        vals = pd.to_numeric(
            posterior_df.loc[posterior_df["state_label"].astype(str) == state, "probability"],
            errors="coerce",
        ).dropna().clip(0.0, 1.0).to_numpy(dtype=float)
        if vals.size <= 1:
            continue
        weights = np.full(vals.shape, 100.0 / float(vals.size))
        color = palette[state]
        ax.hist(vals, bins=edges, weights=weights, histtype="stepfilled", color=color, alpha=0.18, edgecolor="none")
        ax.hist(vals, bins=edges, weights=weights, histtype="step", color=color, linewidth=2.0)
        legend_handles.append(Line2D([0], [0], color=color, lw=2.0, label=state))
    thresh = payload.get("threshold")
    if thresh is not None:
        ax.axvline(float(thresh), color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Posterior probability")
    ax.set_ylabel("Trials (%)")
    ax.set_title(payload.get("title", "Posterior count distribution"))
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def _change_direction_title(direction: str, engaged_label: str, disengaged_label: str) -> str:
    if direction == "into_engaged":
        return f"{disengaged_label} -> {engaged_label}"
    if direction == "out_of_engaged":
        return f"{engaged_label} -> {disengaged_label}"
    return direction.replace("_", " ").title()


def _plot_change_triggered_mean(
    ax: plt.Axes,
    x: np.ndarray,
    means: dict[str, np.ndarray],
    sems: dict[str, np.ndarray],
    *,
    labels: list[str],
    colors: dict[str, str],
    title: str,
) -> None:
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    for label in labels:
        mean = means[label]
        sem = sems[label]
        color = colors[label]
        ax.plot(x, mean, color=color, lw=2.4, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25, edgecolor="none", linewidth=0)
    ax.grid(axis="x", which="major", color="0.5", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.0)
    if float(x[0]) == float(x[-1]):
        ax.set_xlim(float(x[0]) - 0.5, float(x[-1]) + 0.5)
    else:
        ax.set_xlim(x[0], x[-1])
    ax.set_title(title)
    ax.set_ylabel("Posterior probability")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_change_triggered_posteriors_summary(payload: dict) -> plt.Figure:
    change_df = to_pandas_df(payload.get("change_df"), name="change_df")
    require_columns(change_df, ["subject", "direction", "relative_trial", "state_label", "probability"], name="change_df")
    if change_df.empty:
        raise ValueError("change_df is empty.")
    labels = list(payload.get("state_order") or resolve_state_order(change_df))
    if len(labels) < 2:
        raise ValueError("change_df must contain at least two state labels.")
    labels = labels[:2]
    colors = state_palette(labels)
    directions = list(payload.get("directions") or pd.unique(change_df["direction"].astype(str)))
    if "window" in payload:
        window = int(payload["window"])
        x = np.arange(-window, window + 1, dtype=int)
    else:
        x = np.asarray(sorted(pd.unique(change_df["relative_trial"])), dtype=float)
    fig, axes = plt.subplots(1, len(directions), figsize=payload.get("figsize", (14.0, 4.4)), squeeze=False, sharex=True, sharey=True)

    subject_means = (
        change_df.groupby(["subject", "direction", "state_label", "relative_trial"], observed=True)
        .agg(probability=("probability", "mean"))
        .reset_index()
    )
    for ax, direction in zip(axes[0], directions, strict=False):
        sub = subject_means[subject_means["direction"].astype(str) == direction]
        direction_title = _change_direction_title(direction, labels[0], labels[1])
        if sub.empty:
            ax.text(0.5, 0.5, "No confident changes", ha="center", va="center")
            ax.set_title(direction_title)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(x[0], x[-1])
            continue
        means: dict[str, np.ndarray] = {}
        sems: dict[str, np.ndarray] = {}
        for label in labels:
            label_sub = sub[sub["state_label"].astype(str) == label]
            stats = (
                label_sub.groupby("relative_trial", observed=True)["probability"]
                .agg(mean="mean", sem=_line_sem)
                .reindex(x)
            )
            means[label] = stats["mean"].to_numpy(dtype=float)
            sems[label] = stats["sem"].to_numpy(dtype=float)
        n_subjects = int(sub["subject"].nunique())
        n_changes = int(
            change_df.loc[change_df["direction"].astype(str) == direction, ["subject", "change_id"]]
            .drop_duplicates()
            .shape[0]
        ) if "change_id" in change_df.columns else n_subjects
        _plot_change_triggered_mean(
            ax,
            x,
            means,
            sems,
            labels=labels,
            colors=colors,
            title=f"{direction_title}  (n={n_subjects} subjects, {n_changes} changes)",
        )
        ax.set_xlabel("Trials relative to state change")
    if len(directions) > 1:
        axes[0, 1].set_ylabel("")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_change_triggered_posteriors_by_subject(payload: dict) -> plt.Figure:
    change_df = to_pandas_df(payload.get("change_df"), name="change_df")
    require_columns(change_df, ["subject", "direction", "relative_trial", "state_label", "probability"], name="change_df")
    if change_df.empty:
        raise ValueError("change_df is empty.")
    labels = list(payload.get("state_order") or resolve_state_order(change_df))
    if len(labels) < 2:
        raise ValueError("change_df must contain at least two state labels.")
    labels = labels[:2]
    colors = state_palette(labels)
    directions = list(payload.get("directions") or pd.unique(change_df["direction"].astype(str)))
    subjects = list(pd.unique(change_df["subject"].astype(str)))
    if "window" in payload:
        window = int(payload["window"])
        x = np.arange(-window, window + 1, dtype=int)
    else:
        x = np.asarray(sorted(pd.unique(change_df["relative_trial"])), dtype=float)
    fig, axes = plt.subplots(
        len(subjects),
        len(directions),
        figsize=payload.get("figsize", (14.0, max(3.0, 2.8 * len(subjects)))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_idx, subject in enumerate(subjects):
        for col_idx, direction in enumerate(directions):
            ax = axes[row_idx, col_idx]
            sub = change_df[
                (change_df["subject"].astype(str) == subject)
                & (change_df["direction"].astype(str) == direction)
            ]
            direction_title = _change_direction_title(direction, labels[0], labels[1])
            if sub.empty:
                ax.text(0.5, 0.5, "No confident changes", ha="center", va="center")
                ax.set_title(f"Subject {subject} - {direction_title}")
                ax.set_ylim(0.0, 1.0)
                ax.set_xlim(x[0], x[-1])
                continue
            means: dict[str, np.ndarray] = {}
            sems: dict[str, np.ndarray] = {}
            for label in labels:
                stats = (
                    sub[sub["state_label"].astype(str) == label]
                    .groupby("relative_trial", observed=True)["probability"]
                    .agg(mean="mean", sem=_line_sem)
                    .reindex(x)
                )
                means[label] = stats["mean"].to_numpy(dtype=float)
                sems[label] = stats["sem"].to_numpy(dtype=float)
            n_changes = int(
                change_df.loc[
                    (change_df["subject"].astype(str) == subject)
                    & (change_df["direction"].astype(str) == direction),
                    "change_id",
                ].nunique()
            ) if "change_id" in change_df.columns else 1
            _plot_change_triggered_mean(
                ax,
                x,
                means,
                sems,
                labels=labels,
                colors=colors,
                title=f"Subject {subject} - {direction_title}  (n={n_changes} changes)",
            )
    for col_idx in range(len(directions)):
        axes[-1, col_idx].set_xlabel("Trials relative to confident state change")
    if len(directions) > 1:
        for row_idx in range(len(subjects)):
            axes[row_idx, 1].set_ylabel("")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig
