from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

from glmhmmt.views import get_state_palette

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


def _is_polars_df(df) -> bool:
    return pl is not None and isinstance(df, pl.DataFrame)


def _subject_df(trial_df, subj: str):
    if _is_polars_df(trial_df):
        return trial_df.filter(pl.col("subject") == subj)
    return trial_df[trial_df["subject"] == subj]


def _require_columns(df, required: Sequence[str], *, source: str = "trial_df") -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")


def _default_choice_meta(num_classes: int):
    if num_classes == 2:
        return {0: "royalblue", 1: "tomato"}, {0: "L", 1: "R"}
    return {0: "royalblue", 1: "gold", 2: "tomato"}, {0: "L", 1: "C", 2: "R"}


def _state_labels_and_colors(view):
    palette = get_state_palette(view.K)
    rank_order = view.state_idx_order
    labels = [view.state_name_by_idx.get(k, f"State {k}") for k in rank_order]
    colors = [palette[view.state_rank_by_idx.get(int(k), int(k)) % len(palette)] for k in rank_order]
    return rank_order, labels, colors


def _resolve_static_transition_matrix(view) -> np.ndarray | None:
    transition_matrix = getattr(view, "transition_matrix", None)
    if transition_matrix is not None:
        A = np.asarray(transition_matrix, dtype=float)
        return A if A.ndim == 2 else None

    transition_bias = getattr(view, "transition_bias", None)
    transition_weights = getattr(view, "transition_weights", None)
    if transition_bias is None or transition_weights is not None:
        return None

    bias = np.asarray(transition_bias, dtype=float)
    if bias.ndim != 2:
        return None

    shifted = bias - bias.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def _state_dwell_lengths(
    state_seq: np.ndarray,
    session_seq: np.ndarray,
) -> dict[int, list[int]]:
    dwell_by_state: dict[int, list[int]] = {}
    T = min(len(state_seq), len(session_seq))
    if T == 0:
        return dwell_by_state

    current_state = int(state_seq[0])
    current_session = session_seq[0]
    run_len = 1

    for idx in range(1, T):
        next_state = int(state_seq[idx])
        next_session = session_seq[idx]
        if next_state != current_state or next_session != current_session:
            dwell_by_state.setdefault(current_state, []).append(run_len)
            current_state = next_state
            current_session = next_session
            run_len = 1
        else:
            run_len += 1

    dwell_by_state.setdefault(current_state, []).append(run_len)
    return dwell_by_state


def _geometric_dwell_pmf(self_prob: float, max_dwell: int) -> np.ndarray:
    p = float(np.clip(self_prob, 0.0, np.nextafter(1.0, 0.0)))
    dwell = np.arange(1, max_dwell + 1, dtype=float)
    return (1.0 - p) * np.power(p, dwell - 1.0)


def _wilson_interval(
    counts: np.ndarray,
    total: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    if total <= 0:
        zeros = np.zeros_like(counts, dtype=float)
        return zeros, zeros

    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    total_f = float(total)
    phat = counts / total_f
    denom = 1.0 + (z**2) / total_f
    center = (phat + (z**2) / (2.0 * total_f)) / denom
    half_width = (
        z
        * np.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * total_f)) / total_f)
        / denom
    )
    return np.clip(center - half_width, 0.0, 1.0), np.clip(center + half_width, 0.0, 1.0)


def _empirical_dwell_pmf(
    dwell_lengths: np.ndarray,
    max_dwell: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dwell_lengths = np.asarray(dwell_lengths, dtype=int)
    if dwell_lengths.size == 0:
        zeros = np.zeros(max_dwell, dtype=float)
        return zeros, zeros, zeros

    in_range = dwell_lengths[(dwell_lengths >= 1) & (dwell_lengths <= max_dwell)]
    counts = np.bincount(in_range, minlength=max_dwell + 1)[1 : max_dwell + 1]
    probs = counts / float(dwell_lengths.size)
    low, high = _wilson_interval(counts, int(dwell_lengths.size), confidence_level)
    return probs, low, high


def _binned_dwell_summary(
    self_prob: float,
    dwell_lengths: np.ndarray,
    *,
    max_dwell: int,
    bin_width: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bins = int(np.ceil(max_dwell / float(bin_width)))
    centers = bin_width * np.arange(n_bins, dtype=float) + (bin_width / 2.0)

    predicted_full = _geometric_dwell_pmf(self_prob, max_dwell)
    predicted = np.zeros(n_bins, dtype=float)
    for bi in range(n_bins):
        lo = bi * bin_width
        hi = min((bi + 1) * bin_width, max_dwell)
        predicted[bi] = float(np.sum(predicted_full[lo:hi]))

    dwell_lengths = np.asarray(dwell_lengths, dtype=int)
    total_runs = int(dwell_lengths.size)
    if total_runs == 0:
        zeros = np.zeros(n_bins, dtype=float)
        return centers, predicted, zeros, zeros

    in_range = dwell_lengths[(dwell_lengths >= 1) & (dwell_lengths <= max_dwell)]
    if in_range.size == 0:
        zeros = np.zeros(n_bins, dtype=float)
        return centers, predicted, zeros, zeros

    bin_idx = (in_range - 1) // int(bin_width)
    counts = np.bincount(bin_idx, minlength=n_bins)[:n_bins]
    empirical = counts / float(total_runs)
    ci_low, ci_high = _wilson_interval(counts, total_runs, confidence_level)
    yerr = np.vstack(
        [
            np.clip(empirical - ci_low, 0.0, None),
            np.clip(ci_high - empirical, 0.0, None),
        ]
    )
    return centers, predicted, empirical, yerr


def _sort_subject_trials(df_sub, session_col: str, sort_col: str | Sequence[str] | None):
    if df_sub is None:
        return df_sub

    columns = df_sub.columns
    sort_keys: list[str] = []
    if session_col in columns:
        sort_keys.append(session_col)

    if sort_col is None:
        pass
    elif isinstance(sort_col, str):
        if sort_col in columns and sort_col not in sort_keys:
            sort_keys.append(sort_col)
    else:
        for col in sort_col:
            if col in columns and col not in sort_keys:
                sort_keys.append(col)

    if not sort_keys:
        return df_sub
    if _is_polars_df(df_sub):
        return df_sub.sort(sort_keys)
    return df_sub.sort_values(sort_keys, kind="stable")


def _resolve_boxplot_colors(colors, n_artists: int, name: str) -> list[str]:
    if isinstance(colors, str) or np.isscalar(colors):
        return [colors] * n_artists

    resolved = list(colors)
    if len(resolved) == 1 and n_artists > 1:
        resolved *= n_artists
    if len(resolved) != n_artists:
        raise ValueError(f"{name} must have length {n_artists}, got {len(resolved)}.")
    return resolved

def fig_size(n_cols=1, ratio=None):
    """
    Get figure size for A4 page with n_cols columns and specified ratio (width/height).
    :param n_cols: Number of columns (0 for full page)
    :param ratio: Width/height ratio (None for default)
    :return: tuple (width, height) in inches
    """

    sns.set_style('ticks')
    sns.set_context('notebook')

    if ratio is None:
        default_figsize = np.array(plt.rcParams['figure.figsize'])
        default_ratio = default_figsize[0] / default_figsize[1]
        ratio = default_ratio  # 4:3

    # All measurements are in inches
    A4_size = np.array((8.27, 11.69))  # A4 measurements
    margins = 2  # On both dimension
    size = A4_size - margins  # Effective size after margins removal (2 per dimension)
    width = size[0]
    height = size[1]

    # Full page (minus margins)
    if n_cols == 0:
        # Full A4 minus margins
        figsize = (width, height)
        if ratio == 1:  # Square
            figsize = (size[0], size[0])
        return figsize

    else:
        fig_width = width / n_cols
        fig_height = fig_width / ratio
        figsize = (fig_width, fig_height)
        return figsize

def custom_boxplot(
    ax: plt.Axes,
    values,
    *,
    positions,
    widths,
    median_colors,
    box_facecolor: str = "white",
    box_edgecolor: str = "#666666",
    box_alpha: float = 1.0,
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
    """Thin wrapper around ``ax.boxplot`` for the project's boxplot style."""
    positions = list(np.atleast_1d(np.asarray(positions, dtype=float)))
    resolved_median_colors = _resolve_boxplot_colors(
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
        (vals, pos, color)
        for vals, pos, color in zip(grouped_values, positions, resolved_median_colors, strict=False)
        if vals.size > 0
    ]

    if valid_triplets:
        valid_values, valid_positions, valid_median_colors = zip(*valid_triplets, strict=False)
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
        line_positions = np.asarray(positions, dtype=float)
        if line_values.shape[1] != len(line_positions):
            raise ValueError(
                "line_values must have one column per box position. "
                f"Expected {len(line_positions)}, got {line_values.shape[1]}."
            )

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
                    lw=line_linewidth,
                    zorder=line_zorder,
                )

    return box


def _confident_change_event_indices(
    p_s: np.ndarray,
    posterior_threshold: float | None = None,
) -> np.ndarray:
    if len(p_s) <= 1:
        return np.array([], dtype=int)

    if posterior_threshold is None:
        v = np.argmax(p_s, axis=1)
        return np.flatnonzero(np.diff(v) != 0) + 1

    keep_idx = np.flatnonzero(np.max(p_s, axis=1) >= posterior_threshold)
    if keep_idx.size <= 1:
        return np.array([], dtype=int)

    v_conf = np.argmax(p_s[keep_idx], axis=1)
    return keep_idx[1:][np.diff(v_conf) != 0]


def _confident_change_events_by_direction(
    p_s: np.ndarray,
    engaged_k: int,
    posterior_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    empty = {
        "into_engaged": np.array([], dtype=int),
        "out_of_engaged": np.array([], dtype=int),
    }
    if len(p_s) <= 1:
        return empty

    if posterior_threshold is None:
        keep_idx = np.arange(len(p_s), dtype=int)
        v_conf = np.argmax(p_s, axis=1)
    else:
        keep_idx = np.flatnonzero(np.max(p_s, axis=1) >= posterior_threshold)
        if keep_idx.size <= 1:
            return empty
        v_conf = np.argmax(p_s[keep_idx], axis=1)

    change_mask = np.diff(v_conf) != 0
    if not np.any(change_mask):
        return empty

    prev_states = v_conf[:-1][change_mask]
    next_states = v_conf[1:][change_mask]
    change_idx = keep_idx[1:][change_mask]

    into_mask = (prev_states != engaged_k) & (next_states == engaged_k)
    out_mask = (prev_states == engaged_k) & (next_states != engaged_k)
    return {
        "into_engaged": change_idx[into_mask],
        "out_of_engaged": change_idx[out_mask],
    }


def _nanmean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr, axis=0)
    n_obs = np.sum(np.isfinite(arr), axis=0)
    sem = np.zeros_like(mean, dtype=float)
    valid = n_obs > 1
    if np.any(valid):
        sem[valid] = np.nanstd(arr[:, valid], axis=0, ddof=1) / np.sqrt(n_obs[valid])
    return mean, sem


def _change_triggered_traces(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    switch_posterior_threshold: float | None = None,
    window: int = 50,
) -> dict[str, dict[str, np.ndarray]]:
    traces_by_subject: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for subj, view in views.items():
        df_sub = _sort_subject_trials(_subject_df(trial_df, subj), session_col, sort_col)
        if session_col not in df_sub.columns:
            continue

        P = np.asarray(view.smoothed_probs, dtype=float)
        sess_arr = np.asarray(df_sub[session_col])
        T = min(len(P), len(sess_arr))
        if T <= 1:
            continue

        P = P[:T]
        sess_arr = sess_arr[:T]
        engaged_k = int(view.engaged_k())
        subject_traces = {
            "into_engaged": {"engaged": [], "disengaged": []},
            "out_of_engaged": {"engaged": [], "disengaged": []},
        }

        for sess in np.unique(sess_arr):
            p_s = P[sess_arr == sess]
            change_idx_by_direction = _confident_change_events_by_direction(
                p_s,
                engaged_k,
                switch_posterior_threshold,
            )

            for direction, change_idx in change_idx_by_direction.items():
                for idx in change_idx:
                    trace_eng = np.full(2 * window + 1, np.nan, dtype=float)
                    trace_dis = np.full(2 * window + 1, np.nan, dtype=float)
                    lo = max(0, int(idx) - window)
                    hi = min(len(p_s), int(idx) + window + 1)
                    dst_lo = window - (int(idx) - lo)
                    dst_hi = dst_lo + (hi - lo)
                    trace_eng[dst_lo:dst_hi] = p_s[lo:hi, engaged_k]
                    trace_dis[dst_lo:dst_hi] = 1.0 - p_s[lo:hi, engaged_k]
                    subject_traces[direction]["engaged"].append(trace_eng)
                    subject_traces[direction]["disengaged"].append(trace_dis)

        subject_traces_arr = {}
        for direction, traces in subject_traces.items():
            if not traces["engaged"]:
                continue
            subject_traces_arr[direction] = {
                "engaged": np.vstack(traces["engaged"]),
                "disengaged": np.vstack(traces["disengaged"]),
            }
        if subject_traces_arr:
            traces_by_subject[subj] = subject_traces_arr

    return traces_by_subject


def _change_posterior_meta(view) -> tuple[str, str, str, str]:
    palette = get_state_palette(view.K)
    engaged_k = int(view.engaged_k())
    engaged_rank = view.state_rank_by_idx.get(engaged_k, 0)
    engaged_color = palette[engaged_rank % len(palette)]
    engaged_label = view.state_name_by_idx.get(engaged_k, "Engaged")

    other_states = [k for k in view.state_idx_order if int(k) != engaged_k]
    if len(other_states) == 1:
        other_k = int(other_states[0])
        disengaged_label = view.state_name_by_idx.get(other_k, "Disengaged")
        disengaged_rank = view.state_rank_by_idx.get(other_k, 1)
        disengaged_color = palette[disengaged_rank % len(palette)]
    else:
        disengaged_label = "Non-engaged"
        disengaged_color = "#7f7f7f"

    return engaged_label, disengaged_label, engaged_color, disengaged_color


def _plot_change_triggered_mean(
    ax,
    x: np.ndarray,
    engaged_mean: np.ndarray,
    engaged_sem: np.ndarray,
    disengaged_mean: np.ndarray,
    disengaged_sem: np.ndarray,
    *,
    engaged_label: str,
    disengaged_label: str,
    engaged_color: str,
    disengaged_color: str,
    title: str,
) -> None:
    x = np.asarray(x, dtype=float)
    ax.plot(x, engaged_mean, color=engaged_color, lw=2.2, label=f"P({engaged_label})")
    ax.fill_between(
        x,
        np.clip(engaged_mean - engaged_sem, 0.0, 1.0),
        np.clip(engaged_mean + engaged_sem, 0.0, 1.0),
        color=engaged_color,
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(x, disengaged_mean, color=disengaged_color, lw=2.0, label=f"P({disengaged_label})")
    ax.fill_between(
        x,
        np.clip(disengaged_mean - disengaged_sem, 0.0, 1.0),
        np.clip(disengaged_mean + disengaged_sem, 0.0, 1.0),
        color=disengaged_color,
        alpha=0.16,
        edgecolor="none",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    tick_start = int(np.ceil(x[0] / 5.0) * 5)
    tick_stop = int(np.floor(x[-1] / 5.0) * 5)
    xticks = np.arange(tick_start, tick_stop + 1, 5, dtype=int)
    if xticks.size:
        ax.set_xticks(xticks)
        ax.grid(axis="x", which="major", color="0.5", linestyle=":", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(x[0], x[-1])
    ax.set_title(title)
    ax.set_ylabel("Posterior probability")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def _change_direction_title(
    direction: str,
    engaged_label: str,
    disengaged_label: str,
) -> str:
    if direction == "into_engaged":
        return f"{disengaged_label} -> {engaged_label}"
    if direction == "out_of_engaged":
        return f"{engaged_label} -> {disengaged_label}"
    return direction.replace("_", " ").title()


def plot_state_accuracy(
    views: dict,
    trial_df,
    *,
    thresh: float = 0.5,
    performance_col: str = "correct_bool",
    chance_level: float | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    subjects = list(views.keys())
    if not subjects:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, pd.DataFrame()

    first_view = views[subjects[0]]
    palette = get_state_palette(first_view.K)
    state_labels = [first_view.state_name_by_idx.get(k, f"State {k}") for k in first_view.state_idx_order]
    cmap = {"All": "#999999"}
    for k in first_view.state_idx_order:
        lbl = first_view.state_name_by_idx.get(k, f"State {k}")
        rank = first_view.state_rank_by_idx.get(int(k), int(k))
        cmap[lbl] = palette[rank % len(palette)]
    x_labels = ["All"] + state_labels
    if chance_level is None:
        chance_level = 100.0 / float(first_view.num_classes)

    records = []
    for subj in subjects:
        view = views[subj]
        P = np.asarray(view.smoothed_probs)
        df_sub = _subject_df(trial_df, subj)
        _require_columns(df_sub, [performance_col], source=f"trial_df[{subj!r}]")
        hits = np.asarray(df_sub[performance_col]).astype(float)
        T = min(len(P), len(hits))
        P, hits = P[:T], hits[:T]
        valid = np.isfinite(hits)

        if valid.sum() > 0:
            records.append(
                {
                    "subject": subj,
                    "label": "All",
                    "acc": hits[valid].mean() * 100,
                    "n": int(valid.sum()),
                }
            )

        for k in view.state_idx_order:
            lbl = view.state_name_by_idx.get(k, f"State {k}")
            mask = valid & (P[:, k] >= thresh)
            if mask.sum() > 0:
                records.append(
                    {
                        "subject": subj,
                        "label": lbl,
                        "acc": hits[mask].mean() * 100,
                        "n": int(mask.sum()),
                    }
                )

    if not records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, pd.DataFrame()

    df_acc = pd.DataFrame(records)
    tbl = (
        df_acc.groupby("label")[["acc", "n"]]
        .agg({"acc": "mean", "n": "sum"})
        .reindex(x_labels)
        .rename(columns={"acc": "mean_acc (%)", "n": "total_trials"})
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    grouped_acc = [
        df_acc[df_acc["label"] == lbl]["acc"].dropna().to_numpy(dtype=float)
        for lbl in x_labels
    ]
    subject_lines = (
        df_acc.pivot_table(index="subject", columns="label", values="acc", aggfunc="first")
        .reindex(columns=x_labels)
        .to_numpy(dtype=float)
    )
    custom_boxplot(
        ax,
        grouped_acc,
        positions=np.arange(len(x_labels)),
        widths=0.5,
        median_colors=[cmap.get(lbl, "k") for lbl in x_labels],
        line_values=subject_lines,
        showfliers=False,
        zorder=1,
    )

    ax.axhline(chance_level, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_xlabel("State")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0.0, chance_level - 10), 105)
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig, tbl


def plot_state_posterior_count_kde(
    views: dict,
    *,
    bins: int = 20,
    thresh: float | None = None,
) -> plt.Figure:
    subjects = list(views.keys())
    if not subjects:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    first_view = views[subjects[0]]
    _, labels, colors = _state_labels_and_colors(first_view)
    posterior_by_state: dict[str, list[np.ndarray]] = {lbl: [] for lbl in labels}
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0

    for subj in subjects:
        view = views[subj]
        P = np.asarray(view.smoothed_probs, dtype=float)
        if P.ndim != 2:
            continue

        for k in view.state_idx_order:
            lbl = view.state_name_by_idx.get(k, f"State {k}")
            vals = np.clip(P[:, k], 0.0, 1.0)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            posterior_by_state.setdefault(lbl, []).append(vals)

    if not any(posterior_by_state.values()):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(5, 4))

    legend_handles = []
    for lbl, color in zip(labels, colors):
        posterior_vals = posterior_by_state.get(lbl, [])
        posterior_vals = np.concatenate(posterior_vals) if posterior_vals else np.array([], dtype=float)
        if posterior_vals.size <= 1 or np.unique(posterior_vals).size <= 1:
            continue

        weights = np.full(posterior_vals.shape, 100.0 / float(posterior_vals.size))
        ax.hist(
            posterior_vals,
            bins=edges,
            weights=weights,
            histtype="stepfilled",
            color=color,
            alpha=0.18,
            edgecolor="none",
        )
        ax.hist(
            posterior_vals,
            bins=edges,
            weights=weights,
            histtype="step",
            color=color,
            linewidth=2.0,
        )

        legend_handles.append(Line2D([0], [0], color=color, lw=2.0, label=lbl))

    if thresh is not None:
        ax.axvline(thresh, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Posterior probability")
    ax.set_ylabel("Trials (%)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_session_trajectories(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
) -> plt.Figure:
    subjects = list(views.keys())
    K = next(iter(views.values())).K if views else 2
    fig, axes = plt.subplots(len(subjects), 1, figsize=(10, 3.5 * len(subjects)), squeeze=False)

    for i, subj in enumerate(subjects):
        ax = axes[i, 0]
        palette = get_state_palette(views[subj].K)
        P = np.asarray(views[subj].smoothed_probs)
        df_sub = _subject_df(trial_df, subj)
        _require_columns(df_sub, [session_col], source=f"trial_df[{subj!r}]")
        sess_arr = np.asarray(df_sub[session_col])
        T = min(len(P), len(sess_arr))
        P, sess_arr = P[:T], sess_arr[:T]
        sess_ids = np.unique(sess_arr)
        if len(sess_ids) == 0:
            continue

        max_len = max(int(np.sum(sess_arr == s)) for s in sess_ids)
        traj = np.full((len(sess_ids), max_len, K), np.nan)
        for si, s in enumerate(sess_ids):
            idx = np.where(sess_arr == s)[0]
            traj[si, : len(idx), :] = P[idx, :]

        mean = np.nanmean(traj, axis=0)
        n_obs = np.sum(~np.isnan(traj[:, :, 0]), axis=0)
        sem = np.nanstd(traj, axis=0, ddof=1) / np.maximum(n_obs[:, None] ** 0.5, 1)
        x = np.arange(mean.shape[0])

        for k in views[subj].state_idx_order:
            rank = views[subj].state_rank_by_idx.get(int(k), int(k))
            color = palette[rank % len(palette)]
            valid_x = ~np.isnan(mean[:, k])
            ax.plot(
                x[valid_x],
                mean[valid_x, k],
                color=color,
                lw=2,
                label=views[subj].state_name_by_idx.get(k, f"State {k}"),
            )
            ax.fill_between(
                x[valid_x],
                (mean[:, k] - sem[:, k])[valid_x],
                (mean[:, k] + sem[:, k])[valid_x],
                color=color,
                alpha=0.25,
            )

        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial within session")
        ax.set_ylabel("State probability")
        ax.set_title(f"Subject {subj} — avg. state trajectory  (n={len(sess_ids)} sessions)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_change_triggered_posteriors_summary(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    switch_posterior_threshold: float | None = None,
    window: int = 15,
) -> plt.Figure:
    subjects = list(views.keys())
    directions = ("into_engaged", "out_of_engaged")
    if not subjects:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    traces_by_subject = _change_triggered_traces(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
    )
    if not traces_by_subject:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No confident changes", ha="center", va="center")
        ax.axis("off")
        return fig

    first_view = views[subjects[0]]
    engaged_label, disengaged_label, engaged_color, disengaged_color = _change_posterior_meta(first_view)
    x = np.arange(-window, window + 1)

    fig, axes = plt.subplots(1, len(directions), figsize=(14.0, 4.4), squeeze=False, sharex=True, sharey=True)
    for ax, direction in zip(axes[0], directions, strict=False):
        engaged_subject_means = []
        disengaged_subject_means = []
        total_changes = 0
        for subj in subjects:
            traces = traces_by_subject.get(subj, {}).get(direction)
            if traces is None:
                continue
            engaged_subject_means.append(np.nanmean(traces["engaged"], axis=0))
            disengaged_subject_means.append(np.nanmean(traces["disengaged"], axis=0))
            total_changes += int(traces["engaged"].shape[0])

        direction_title = _change_direction_title(direction, engaged_label, disengaged_label)
        if not engaged_subject_means:
            ax.text(0.5, 0.5, "No confident changes", ha="center", va="center")
            ax.set_title(direction_title)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(x[0], x[-1])
            continue

        engaged_arr = np.vstack(engaged_subject_means)
        disengaged_arr = np.vstack(disengaged_subject_means)
        engaged_mean, engaged_sem = _nanmean_sem(engaged_arr)
        disengaged_mean, disengaged_sem = _nanmean_sem(disengaged_arr)
        _plot_change_triggered_mean(
            ax,
            x,
            engaged_mean,
            engaged_sem,
            disengaged_mean,
            disengaged_sem,
            engaged_label=engaged_label,
            disengaged_label=disengaged_label,
            engaged_color=engaged_color,
            disengaged_color=disengaged_color,
            title=f"{direction_title}  (n={len(engaged_subject_means)} subjects, {total_changes} changes)",
        )
        ax.set_xlabel("Trials relative to state change")

    if len(directions) > 1:
        axes[0, 1].set_ylabel("")
    axes[0, 0].set_xlabel("Trials relative to state change")
    if len(directions) > 1:
        axes[0, 1].set_xlabel("Trials relative to state change")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_change_triggered_posteriors_by_subject(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    switch_posterior_threshold: float | None = None,
    window: int = 15,
) -> plt.Figure:
    subjects = list(views.keys())
    directions = ("into_engaged", "out_of_engaged")
    if not subjects:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    traces_by_subject = _change_triggered_traces(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
    )

    fig, axes = plt.subplots(
        len(subjects),
        len(directions),
        figsize=(14.0, max(3.0, 2.8 * len(subjects))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    x = np.arange(-window, window + 1)

    for i, subj in enumerate(subjects):
        view = views[subj]
        engaged_label, disengaged_label, engaged_color, disengaged_color = _change_posterior_meta(view)
        for j, direction in enumerate(directions):
            ax = axes[i, j]
            traces = traces_by_subject.get(subj, {}).get(direction)
            direction_title = _change_direction_title(direction, engaged_label, disengaged_label)
            if traces is None:
                ax.text(0.5, 0.5, "No confident changes", ha="center", va="center")
                ax.set_title(f"Subject {subj} — {direction_title}")
                ax.set_ylim(0.0, 1.0)
                ax.set_xlim(x[0], x[-1])
                continue

            engaged_mean, engaged_sem = _nanmean_sem(traces["engaged"])
            disengaged_mean, disengaged_sem = _nanmean_sem(traces["disengaged"])
            _plot_change_triggered_mean(
                ax,
                x,
                engaged_mean,
                engaged_sem,
                disengaged_mean,
                disengaged_sem,
                engaged_label=engaged_label,
                disengaged_label=disengaged_label,
                engaged_color=engaged_color,
                disengaged_color=disengaged_color,
                title=f"Subject {subj} — {direction_title}  (n={traces['engaged'].shape[0]} changes)",
            )

    for j in range(len(directions)):
        axes[-1, j].set_xlabel("Trials relative to confident state change")
    if len(directions) > 1:
        for i in range(len(subjects)):
            axes[i, 1].set_ylabel("")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def _grouped_state_occupancy_values(records: list[dict], labels: list[str]) -> list[np.ndarray]:
    if not records:
        return [np.array([], dtype=float) for _ in labels]
    df = pd.DataFrame(records)
    return [
        df.loc[df["state_label"] == lbl, "occupancy"].to_numpy(dtype=float)
        for lbl in labels
    ]


def _plot_state_occupancy_distribution(
    ax: plt.Axes,
    grouped_vals: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
    if not any(np.asarray(vals, dtype=float).size for vals in grouped_vals):
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        custom_boxplot(
            ax,
            grouped_vals,
            positions=np.arange(len(labels)),
            widths=0.5,
            median_colors=colors,
            showfliers=False,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)


def _plot_state_occupancy_session_violin(
    ax: plt.Axes,
    grouped_vals: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
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


def _plot_state_switch_hist(ax: plt.Axes, changes_per_sess: list[int], title: str) -> None:
    xlabel = "# state switches / session"
    if not changes_per_sess:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return
    max_chg = max(changes_per_sess)
    ax.hist(
        changes_per_sess,
        bins=np.arange(-0.5, max_chg + 1.5, 1.0),
        color="#888888",
        alpha=0.75,
        edgecolor="white",
    )
    ax.set_xlim(-0.5, max_chg + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# sessions")
    ax.set_title(title)


def _subject_state_occupancy_summary(
    subj: str,
    view,
    trial_df,
    *,
    session_col: str,
    sort_col: str | None,
    switch_posterior_threshold: float | None,
) -> dict | None:
    df_sub = _sort_subject_trials(_subject_df(trial_df, subj), session_col, sort_col)
    if df_sub is None:
        return None
    _require_columns(df_sub, [session_col], source=f"trial_df[{subj!r}]")

    P = np.asarray(view.smoothed_probs)
    sess_arr = np.asarray(df_sub[session_col])
    T = min(len(P), len(sess_arr))
    if T == 0:
        return None

    P, sess_arr = P[:T], sess_arr[:T]
    rank_order, labels, colors = _state_labels_and_colors(view)
    overall = np.mean(P[:, rank_order], axis=0)
    overall_rows = [
        {
            "subject": subj,
            "state_idx": int(k),
            "state_label": label,
            "occupancy": float(occ),
        }
        for k, label, occ in zip(rank_order, labels, overall, strict=False)
    ]

    session_rows: list[dict] = []
    switch_rows: list[dict] = []
    changes_per_sess: list[int] = []
    for sess in np.unique(sess_arr):
        p_s = P[sess_arr == sess]
        if len(p_s) == 0:
            continue
        n_changes = int(len(_confident_change_event_indices(p_s, switch_posterior_threshold)))
        changes_per_sess.append(n_changes)
        switch_rows.append({"subject": subj, "session": sess, "switches": n_changes})
        session_occ = np.mean(p_s[:, rank_order], axis=0)
        for k, label, occ_s in zip(rank_order, labels, session_occ, strict=False):
            session_rows.append(
                {
                    "subject": subj,
                    "session": sess,
                    "state_idx": int(k),
                    "state_label": label,
                    "occupancy": float(occ_s),
                }
            )

    return {
        "labels": labels,
        "colors": colors,
        "overall": overall,
        "overall_rows": overall_rows,
        "session_rows": session_rows,
        "switch_rows": switch_rows,
        "changes_per_sess": changes_per_sess,
    }


def _prepare_state_occupancy_summaries(
    views: dict,
    trial_df,
    *,
    session_col: str,
    sort_col: str | None,
    switch_posterior_threshold: float | None,
) -> dict:
    if switch_posterior_threshold is not None:
        switch_posterior_threshold = float(switch_posterior_threshold)
        if not 0.0 <= switch_posterior_threshold <= 1.0:
            raise ValueError("switch_posterior_threshold must be between 0 and 1.")

    subjects = list(views.keys())
    if not subjects:
        return {"subjects": []}

    _, labels_all, colors_all = _state_labels_and_colors(views[subjects[0]])
    overall_records: list[dict] = []
    session_records: list[dict] = []
    switch_records: list[dict] = []
    subject_summaries: dict[str, dict | None] = {}

    for subj in subjects:
        summary = _subject_state_occupancy_summary(
            subj,
            views[subj],
            trial_df,
            session_col=session_col,
            sort_col=sort_col,
            switch_posterior_threshold=switch_posterior_threshold,
        )
        subject_summaries[subj] = summary
        if summary is None:
            continue
        overall_records.extend(summary["overall_rows"])
        session_records.extend(summary["session_rows"])
        switch_records.extend(summary["switch_rows"])

    return {
        "subjects": subjects,
        "labels_all": labels_all,
        "colors_all": colors_all,
        "overall_records": overall_records,
        "session_records": session_records,
        "switch_records": switch_records,
        "subject_summaries": subject_summaries,
    }


def plot_state_occupancy_overall_boxplot(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | None = None,
    switch_posterior_threshold: float | None = None,
) -> plt.Figure:
    summary = _prepare_state_occupancy_summaries(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
    )
    if not summary["subjects"]:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    labels_all = summary["labels_all"]
    colors_all = summary["colors_all"]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(labels_all)), 4.2))
    _plot_state_occupancy_distribution(
        ax,
        _grouped_state_occupancy_values(summary["overall_records"], labels_all),
        labels_all,
        colors_all,
    )
    ax.set_ylabel("Fractional occupancy")
    ax.set_title("All selected subjects - overall occupancy")
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_state_occupancy(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | None = None,
    switch_posterior_threshold: float | None = None,
) -> plt.Figure:
    summary = _prepare_state_occupancy_summaries(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
    )
    subjects = summary["subjects"]
    if not subjects:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    n_rows = len(subjects) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.8 * n_rows), squeeze=False)

    for i, subj in enumerate(subjects, start=1):
        ax_occ, ax_box, ax_chg = axes[i, 0], axes[i, 1], axes[i, 2]
        subj_summary = summary["subject_summaries"][subj]
        if subj_summary is None:
            for ax in (ax_occ, ax_box, ax_chg):
                ax.set_visible(False)
            continue

        labels = subj_summary["labels"]
        colors = subj_summary["colors"]
        overall = subj_summary["overall"]

        ax_occ.bar(labels, overall, color=colors, alpha=0.85)
        ax_occ.set_ylim(0, 1)
        ax_occ.set_ylabel("Fractional occupancy")
        ax_occ.set_title(f"Subject {subj} - overall occupancy")

        _plot_state_occupancy_session_violin(
            ax_box,
            _grouped_state_occupancy_values(subj_summary["session_rows"], labels),
            labels,
            colors,
        )
        ax_box.set_ylabel("Session occupancy")
        ax_box.set_title(f"Subject {subj} - occupancy by session")

        _plot_state_switch_hist(
            ax_chg,
            subj_summary["changes_per_sess"],
            f"Subject {subj} - state switches",
        )

    labels_all = summary["labels_all"]
    colors_all = summary["colors_all"]
    ax_all_occ, ax_all_sess, ax_all_chg = axes[0, 0], axes[0, 1], axes[0, 2]
    _plot_state_occupancy_distribution(
        ax_all_occ,
        _grouped_state_occupancy_values(summary["overall_records"], labels_all),
        labels_all,
        colors_all,
    )
    ax_all_occ.set_ylabel("Fractional occupancy")
    ax_all_occ.set_title("All selected subjects - overall occupancy")

    _plot_state_occupancy_session_violin(
        ax_all_sess,
        _grouped_state_occupancy_values(summary["session_records"], labels_all),
        labels_all,
        colors_all,
    )
    ax_all_sess.set_ylabel("Session occupancy")
    ax_all_sess.set_title("All selected sessions - occupancy by session")

    _plot_state_switch_hist(
        ax_all_chg,
        [int(row["switches"]) for row in summary["switch_records"]],
        "All selected sessions - state switches",
    )

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def _prepare_state_dwell_entries(
    views: dict,
    trial_df,
    *,
    session_col: str,
    sort_col: str | Sequence[str] | None,
    max_dwell: int | None,
    ci_level: float,
) -> tuple[list[dict], int, float, int]:
    if not views:
        raise ValueError("No views provided.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1.")

    raw_entries: list[dict] = []
    max_states = 0
    bin_width = 10
    inferred_max_dwell = 0

    for subj, view in views.items():
        A = _resolve_static_transition_matrix(view)
        if A is None:
            continue

        df_sub = _sort_subject_trials(_subject_df(trial_df, subj), session_col, sort_col)
        if df_sub is None:
            continue
        _require_columns(df_sub, [session_col], source=f"trial_df[{subj!r}]")

        session_arr = np.asarray(df_sub[session_col])
        map_states = np.asarray(view.map_states(), dtype=int)
        T = min(len(session_arr), len(map_states))
        if T == 0:
            continue

        session_arr = session_arr[:T]
        map_states = map_states[:T]
        dwell_by_state = _state_dwell_lengths(map_states, session_arr)
        rank_order, _labels, colors = _state_labels_and_colors(view)
        _, session_counts = np.unique(session_arr, return_counts=True)
        if session_counts.size > 0:
            inferred_max_dwell = max(inferred_max_dwell, int(np.max(session_counts)))

        max_states = max(max_states, len(rank_order))
        raw_entries.append(
            {
                "subject": subj,
                "rank_order": [int(k) for k in rank_order],
                "colors": colors,
                "dwell_by_state": dwell_by_state,
                "A": A,
            }
        )

    if max_dwell is None:
        max_dwell = inferred_max_dwell
    max_dwell = int(max_dwell)
    if max_dwell < 1:
        raise ValueError("max_dwell must be at least 1.")

    subject_entries: list[dict] = []
    y_max = 0.0
    for entry in raw_entries:
        panels: list[dict] = []
        for col_idx, state_idx in enumerate(entry["rank_order"]):
            dwell_lengths = np.asarray(entry["dwell_by_state"].get(int(state_idx), []), dtype=int)
            centers, predicted, empirical, yerr = _binned_dwell_summary(
                float(entry["A"][int(state_idx), int(state_idx)]),
                dwell_lengths,
                max_dwell=max_dwell,
                bin_width=bin_width,
                confidence_level=ci_level,
            )
            panels.append(
                {
                    "title": f"state {col_idx + 1}",
                    "color": entry["colors"][col_idx],
                    "x": centers,
                    "self_prob": float(entry["A"][int(state_idx), int(state_idx)]),
                    "dwell_lengths": dwell_lengths,
                    "predicted": predicted,
                    "empirical": empirical,
                    "yerr": yerr,
                    "n_runs": int(dwell_lengths.size),
                }
            )
            y_max = max(
                y_max,
                float(np.max(predicted)),
                float(np.max(empirical)),
                float(np.max(empirical + yerr[1])),
            )
        subject_entries.append({"subject": entry["subject"], "panels": panels})

    if not subject_entries:
        raise ValueError(
            "No subjects with both trial data and a homogeneous transition matrix were found."
        )

    resolved_y_max = float(y_max) if y_max > 0 else 1.0
    return subject_entries, max_states, resolved_y_max, max_dwell


def _draw_state_dwell_panel(
    ax,
    panel: dict,
    *,
    max_dwell: int,
    y_max: float,
    show_title: bool,
) -> None:
    ax.plot(
        panel["x"],
        panel["predicted"],
        color=panel["color"],
        lw=2.6,
        marker="o",
        ms=5.5,
    )
    ax.errorbar(
        panel["x"],
        panel["empirical"],
        yerr=panel["yerr"],
        color=panel["color"],
        lw=2.4,
        linestyle="--",
        marker="o",
        ms=5.5,
        capsize=0,
        elinewidth=2.2,
    )
    if show_title:
        ax.set_title(panel["title"], color=panel["color"], fontsize=18, pad=14)
    ax.set_xlim(0, max_dwell)
    ax.set_ylim(0.0, 1.08 * y_max)
    ax.set_xticks([0, max_dwell // 3, (2 * max_dwell) // 3, max_dwell])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _attach_state_dwell_axis_labels(fig, axes, *, max_states: int, max_dwell: int) -> None:
    fig.supylabel("frac. dwell times in state")
    fig.supxlabel("state dwell time\n# trials")
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.6, linestyle="-", label="predicted"),
        Line2D([0], [0], color="black", lw=2.4, linestyle="--", marker="o", ms=5.5, label="empirical"),
    ]
    legend_ax = axes[0, min(1, max_states - 1)]
    legend_ax.legend(handles=legend_handles, loc="upper right", frameon=False)


def plot_state_dwell_times_by_subject(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    max_dwell: int | None = None,
    ci_level: float = 0.68,
) -> plt.Figure:
    subject_entries, max_states, y_max, resolved_max_dwell = _prepare_state_dwell_entries(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )

    fig, axes = plt.subplots(
        len(subject_entries),
        max_states,
        figsize=(4.1 * max_states, 3.1 * len(subject_entries)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, entry in enumerate(subject_entries):
        for col_idx in range(max_states):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(entry["panels"]):
                ax.set_visible(False)
                continue

            _draw_state_dwell_panel(
                ax,
                entry["panels"][col_idx],
                max_dwell=resolved_max_dwell,
                y_max=y_max,
                show_title=(len(subject_entries) == 1 or row_idx == 0),
            )
            if len(subject_entries) > 1 and col_idx == 0:
                ax.set_ylabel(entry["subject"])

    _attach_state_dwell_axis_labels(fig, axes, max_states=max_states, max_dwell=resolved_max_dwell)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    sns.despine(fig=fig)
    return fig


def plot_state_dwell_times_summary(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    max_dwell: int | None = None,
    ci_level: float = 0.68,
) -> plt.Figure:
    subject_entries, max_states, y_max, resolved_max_dwell = _prepare_state_dwell_entries(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )

    fig, axes = plt.subplots(
        1,
        max_states,
        figsize=(4.1 * max_states, 3.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for col_idx in range(max_states):
        ax = axes[0, col_idx]
        available_panels = [
            entry["panels"][col_idx]
            for entry in subject_entries
            if col_idx < len(entry["panels"])
        ]
        if not available_panels:
            ax.set_visible(False)
            continue

        predicted_stack = np.vstack([panel["predicted"] for panel in available_panels])
        pooled_dwell_lengths = np.concatenate(
            [np.asarray(panel["dwell_lengths"], dtype=int) for panel in available_panels],
            axis=0,
        )
        pooled_weights = np.asarray([panel["n_runs"] for panel in available_panels], dtype=float)
        if not np.any(pooled_weights > 0):
            pooled_weights = np.ones(len(available_panels), dtype=float)
        _, _, empirical_pooled, yerr_pooled = _binned_dwell_summary(
            float(np.average([panel["self_prob"] for panel in available_panels], weights=pooled_weights)),
            pooled_dwell_lengths,
            max_dwell=resolved_max_dwell,
            bin_width=10,
            confidence_level=ci_level,
        )
        summary_panel = {
            "title": available_panels[0]["title"],
            "color": available_panels[0]["color"],
            "x": available_panels[0]["x"],
            "predicted": np.average(predicted_stack, axis=0, weights=pooled_weights),
            "empirical": empirical_pooled,
            "yerr": yerr_pooled,
        }
        _draw_state_dwell_panel(
            ax,
            summary_panel,
            max_dwell=resolved_max_dwell,
            y_max=y_max,
            show_title=True,
        )

    _attach_state_dwell_axis_labels(fig, axes, max_states=max_states, max_dwell=resolved_max_dwell)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    sns.despine(fig=fig)
    return fig


def plot_state_dwell_times(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    max_dwell: int | None = None,
    ci_level: float = 0.68,
) -> plt.Figure:
    return plot_state_dwell_times_by_subject(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )


def plot_session_deepdive(
    views: dict,
    trial_df,
    subj: str,
    sess,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    switch_posterior_threshold: float | None = None,
    performance_col: str = "correct_bool",
    response_col: str = "response",
    trace_x_cols: Sequence[str] = ("A_R", "A_L", "A_C"),
    trace_u_cols: Sequence[str] = ("A_plus", "A_minus"),
    choice_colors: dict[int, str] | None = None,
    choice_labels: dict[int, str] | None = None,
    engaged_window: int = 20,
    engaged_trace_mode: str = "rolling",
) -> plt.Figure:
    try:
        sess = int(sess)
    except (TypeError, ValueError):
        pass
    engaged_window = int(engaged_window)
    if engaged_window < 1:
        raise ValueError("engaged_window must be >= 1.")
    engaged_trace_mode = str(engaged_trace_mode).strip().lower()
    if engaged_trace_mode not in {"rolling", "raw"}:
        raise ValueError("engaged_trace_mode must be 'rolling' or 'raw'.")
    if switch_posterior_threshold is not None:
        switch_posterior_threshold = float(switch_posterior_threshold)
        if not 0.0 <= switch_posterior_threshold <= 1.0:
            raise ValueError("switch_posterior_threshold must be between 0 and 1.")

    df_sub_all = _subject_df(trial_df, subj)
    df_sub_all = df_sub_all.with_row_index("_align_idx")
    df_sub_all = _sort_subject_trials(df_sub_all, session_col, sort_col)
    _require_columns(df_sub_all, [session_col], source=f"trial_df[{subj!r}]")
    sess_row_indices = (
        df_sub_all.filter(pl.col(session_col) == sess)["_align_idx"].to_numpy()
    )
    df_sess = df_sub_all.filter(pl.col(session_col) == sess).drop("_align_idx")

    _require_columns(df_sess, [performance_col, response_col], source=f"trial_df[{subj!r}]")
    hits = np.asarray(df_sess[performance_col]).astype(float)
    response = np.asarray(df_sess[response_col]).astype(int)

    probs_all = np.asarray(views[subj].smoothed_probs)
    probs = probs_all[sess_row_indices]
    T = min(probs.shape[0], len(hits), len(response))
    probs, hits, response = probs[:T], hits[:T], response[:T]
    change_idx = _confident_change_event_indices(probs, switch_posterior_threshold)
    n_changes = int(len(change_idx))
    x = np.arange(T)

    X_sess = np.asarray(views[subj].X)[sess_row_indices][:T]
    X_idx = {f: i for i, f in enumerate(views[subj].feat_names)}
    trace_sources = {}
    if views[subj].U is not None:
        U_sess = np.asarray(views[subj].U)[sess_row_indices][:T]
        U_idx = {f: i for i, f in enumerate(views[subj].U_cols)}
        for name in trace_u_cols:
            if name in U_idx:
                trace_sources[name] = (U_sess, U_idx[name])
    for name in trace_x_cols:
        if name in X_idx and name not in trace_sources:
            trace_sources[name] = (X_sess, X_idx[name])

    rolling_acc = np.full(T, np.nan)
    window = 20
    for ti in range(T):
        start = max(0, ti - window + 1)
        window_hits = hits[start : ti + 1]
        valid_hits = window_hits[np.isfinite(window_hits)]
        if valid_hits.size:
            rolling_acc[ti] = 100.0 * valid_hits.mean()

    if choice_colors is None or choice_labels is None:
        choice_colors, choice_labels = _default_choice_meta(views[subj].num_classes)

    rank_order, _labels, colors = _state_labels_and_colors(views[subj])
    engaged_k = views[subj].engaged_k()
    probs_roll = np.empty_like(probs, dtype=float)
    probs_cumsum = np.cumsum(probs, axis=0, dtype=float)
    for ti in range(T):
        start = max(0, ti - engaged_window + 1)
        totals = probs_cumsum[ti] - (probs_cumsum[start - 1] if start > 0 else 0.0)
        probs_roll[ti] = totals / float(ti - start + 1)
    engaged_prob = probs[:, engaged_k]
    engaged_roll = np.empty(T, dtype=float)
    engaged_cumsum = np.cumsum(engaged_prob, dtype=float)
    for ti in range(T):
        start = max(0, ti - engaged_window + 1)
        total = engaged_cumsum[ti] - (engaged_cumsum[start - 1] if start > 0 else 0.0)
        engaged_roll[ti] = total / float(ti - start + 1)
    probs_display = probs_roll if engaged_trace_mode == "rolling" else probs
    engaged_trace = engaged_roll if engaged_trace_mode == "rolling" else engaged_prob
    engaged_label = views[subj].state_name_by_idx.get(engaged_k, "Engaged")
    engaged_trace_label = (
        f"P({engaged_label}) rolling ({engaged_window} trials)"
        if engaged_trace_mode == "rolling"
        else f"P({engaged_label}) raw"
    )

    n_rows = 2 if trace_sources else 1
    height_ratios = [2, 1.5] if trace_sources else [1]
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, 5 + 2.5 * (n_rows - 1)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes)
    ax1 = axes[0]

    bottom = np.zeros(T)
    for pos, k in enumerate(rank_order):
        color = colors[pos]
        ax1.fill_between(
            x,
            bottom,
            bottom + probs_display[:, k],
            alpha=0.7,
            color=color,
            label=views[subj].state_name_by_idx.get(k, f"State {k}"),
        )
        bottom += probs_display[:, k]

    palette = get_state_palette(views[subj].K)
    ax1.fill_between(
        x,
        0.0,
        engaged_trace,
        color=palette[0],
        alpha=0.18,
    )
    ax1.plot(
        x,
        engaged_trace,
        color=palette[0],
        lw=2,
        label=engaged_trace_label,
    )
    if change_idx.size:
        ax1.scatter(
            x[change_idx],
            engaged_trace[change_idx],
            s=34,
            color="crimson",
            linewidths=0,
            zorder=6,
            label="State change",
        )

    for resp, color in choice_colors.items():
        mask = response == resp
        if mask.sum() == 0:
            continue
        ax1.scatter(
            x[mask],
            np.ones(mask.sum()) * 1.03,
            c=color,
            s=5,
            marker="|",
            label=choice_labels.get(resp, str(resp)),
            transform=ax1.get_xaxis_transform(),
            clip_on=False,
        )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("State probability")
    title = f"Subject {subj}  —  session {sess}  ({T} trials, {n_changes} state changes)"
    ax1.set_title(title, pad=25)

    ax1r = ax1.twinx()
    ax1r.plot(x, rolling_acc, color="black", lw=1.8, linestyle="-", alpha=0.7, label=f"Rolling accuracy ({window} trials)")
    ax1r.axhline(100.0 / float(views[subj].num_classes), color="grey", lw=0.9, linestyle="--", alpha=0.5)
    ax1r.set_ylim(0, 105)
    ax1r.set_ylabel("Accuracy (%)", color="black")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines1r, labs1r = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, labs1 + labs1r, bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)

    trace_colors = {
        "A_plus": "royalblue",
        "A_minus": "gold",
        "A_L": "royalblue",
        "A_C": "gold",
        "A_R": "tomato",
    }
    if trace_sources:
        ax2 = axes[1]
        for name, (arr, ci) in trace_sources.items():
            ax2.plot(x, arr[:, ci], label=name, color=trace_colors.get(name, "gray"), lw=1.5, alpha=0.85)
        ax2.set_ylabel("Action trace")
        ax2.set_ylim(0, None)
        ax2.set_xlabel("Trial within session")
        ax2.legend(bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)
    else:
        ax1.set_xlabel("Trial within session")

    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    sns.despine(fig=fig, right=False)
    return fig
