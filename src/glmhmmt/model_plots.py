import math
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t, ttest_1samp, ttest_rel
from scipy.special import softmax as _softmax
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from typing import Callable, Sequence, Tuple
from glmhmmt.runtime import load_app_config
from glmhmmt.views import build_state_palette, get_state_color, get_state_palette, get_state_rank
from glmhmmt.plots_common import (
    custom_boxplot,
    plot_state_accuracy as _plot_state_accuracy_common,
    plot_change_triggered_posteriors_by_subject as _plot_change_triggered_posteriors_by_subject_common,
    plot_change_triggered_posteriors_summary as _plot_change_triggered_posteriors_summary_common,
    plot_state_posterior_count_kde as _plot_state_posterior_count_kde_common,
    plot_session_trajectories as _plot_session_trajectories_common,
    plot_state_occupancy as _plot_state_occupancy_common,
    plot_state_occupancy_overall_boxplot as _plot_state_occupancy_overall_boxplot_common,
    plot_state_dwell_times_by_subject as _plot_state_dwell_times_by_subject_common,
    plot_state_dwell_times_summary as _plot_state_dwell_times_summary_common,
    plot_state_dwell_times as _plot_state_dwell_times_common,
    plot_session_deepdive as _plot_session_deepdive_common,
)
sns.set_style("ticks")

cfg = load_app_config()
CI_BAND_ERR_KWS = {"edgecolor": "none", "linewidth": 0}
_state_color = get_state_color

def truncate_colormap(cmap_name, minval=0.2, maxval=0.9, n=256):
    """Trunca un colormap a un subrango."""
    cmap = cm.get_cmap(cmap_name, n)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def get_plot_path(subfolder: str, fname: str, model_name: str) -> Path:
    out_dir = Path("results") / "plots" / model_name / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname

def prepare_predictions_df(df_pred: pl.DataFrame) -> pl.DataFrame:
    df = df_pred.clone()

    if "correct_bool" not in df.columns:
        if "performance" in df.columns:
            df = df.with_columns(
                pl.col("performance").cast(pl.Boolean).alias("correct_bool")
            )
        else:
            raise ValueError("No encuentro 'performance' ni 'correct_bool' en df.")

    for col in ["pL", "pC", "pR"]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en df (predicciones por trial).")

    if "response" not in df.columns:
        raise ValueError("Falta la columna 'response' (0/1/2) en df.")

    if "p_model_correct" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("stimulus") == 0).then(pl.col("pL"))
            .when(pl.col("stimulus") == 1).then(pl.col("pC"))
            .when(pl.col("stimulus") == 2).then(pl.col("pR"))
            .otherwise(None)
            .alias("p_model_correct")
        )

    if "stimd_c" not in df.columns:
        if "stimd_n" in df.columns:
            df = df.with_columns(
                pl.col("stimd_n")
                .replace(cfg["encoding"]["stimd"], default=None)
                .alias("stimd_c")
            )
        else:
            raise ValueError("Falta 'stimd_c' y no existe 'stimd_n' para mapear.")

    if "ttype_c" not in df.columns:
        if "ttype_n" in df.columns:
            df = df.with_columns(
                pl.col("ttype_n")
                .replace(cfg["encoding"]["ttype"], default=None)
                .alias("ttype_c")
            )
        else:
            raise ValueError("Falta 'ttype_c' y no existe 'ttype_n' para mapear.")

    return df

def plot_cat_panel(ax, df, group_col, order, title, xlabel, ylabel=None, palette=None, labels=None):
    
    subj = (df.filter(pl.col(group_col).is_in(order)).group_by([group_col, "subject"])
            .agg([
                pl.col("correct_bool").mean().alias("correct_mean"),
                pl.col("p_model_correct").mean().alias("model_mean"),
                ]))
    if subj.height == 0:
        ax.set_visible(False)
        return

    g = (
        subj.group_by(group_col)
            .agg([
                pl.col("correct_mean").mean().alias("md"),
                pl.col("correct_mean").std(ddof=1).alias("sd"),
                pl.col("correct_mean").count().alias("nd"),
                pl.col("model_mean").mean().alias("mm"),
                pl.col("model_mean").std(ddof=1).alias("sm"),
                pl.col("model_mean").count().alias("nm"),
            ])
    )

    g = g.with_columns([
    pl.col("nd").clip(lower_bound=1),
    pl.col("nm").clip(lower_bound=1),
    ])

    # reordenar
    g = g.with_columns(pl.col(group_col).cast(pl.Categorical).alias(group_col))

    rows = {r[group_col]: r for r in g.to_dicts()}
    cats = [c for c in order if c in rows]
    md = np.array([rows[c]["md"] for c in cats])
    sd = np.array([rows[c]["sd"] for c in cats])
    nd = np.array([rows[c]["nd"] for c in cats])
    mm = np.array([rows[c]["mm"] for c in cats])
    sm = np.array([rows[c]["sm"] for c in cats])
    nm = np.array([rows[c]["nm"] for c in cats])

    # Si quieres también poner el modelo como línea:
    ax.plot(np.arange(len(cats)), mm, "-", color="black", lw=2, label="Model")
    

    colors = palette if palette else ["black"] * np.arange(len(cats))
    if (df["subject"].unique().shape[0] > 1 ):
        ax.fill_between(np.arange(len(cats)), mm-sm, mm+sm, color="black", alpha=0.12)
        sem_d = sd / np.sqrt(nd)
        sem_m = sm / np.sqrt(nm)
        ci_d  = sem_d * t.ppf(0.975, nd-1)
        ci_m  = sem_m * t.ppf(0.975, nm-1)
        for i, (xpos, yval, err) in enumerate(zip(np.arange(len(cats)), md, sd)):
            ax.errorbar(xpos, yval, yerr=err, fmt="o",
                        color=colors[i], ms=7, capsize=3)
    else: 
        for i, (xpos, yval) in enumerate(zip(np.arange(len(cats)), md)):
            ax.errorbar(xpos, yval, fmt="o",
                        color=colors[i], ms=7, capsize=3)

    ax.set_xticks(np.arange(len(cats)))
    # align labels to the subset of categories actually present in this panel
    if labels:
        _label_map = dict(zip(order, labels))
        _tick_labels = [_label_map.get(c, c) for c in cats]
    else:
        _tick_labels = cats
    ax.set_xticklabels(_tick_labels)

    ax.set_ylim(0.2, 1.05)
    ax.axhspan(0, 1/3, color="gray", alpha=0.15)
    ax.set_xlim(left=-0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _plot_state_panel(ax, df_state, group_col, order, color, label):
    """
    Draw:
      - Data dots  : mean(correct_bool) ± sd per category, MAP-assigned trials
      - Model line : mean(p_model_correct) per category, same trial subset
    Both in `color`.  Returns (data_errorbar_container, model_line) for legend
    building, or (None, None) if the subset is empty.
    """
    subj = (
        df_state
        .filter(pl.col(group_col).is_in(order))
        .group_by([group_col, "subject"])
        .agg([
            pl.col("correct_bool").mean().alias("acc"),
            pl.col("p_model_correct").mean().alias("model"),
        ])
    )
    if subj.height == 0:
        return None, None

    agg = (
        subj.group_by(group_col)
        .agg([
            pl.col("acc").mean().alias("md"),
            pl.col("acc").std(ddof=1).alias("sd"),
            pl.col("model").mean().alias("mm"),
            pl.col("model").std(ddof=1).alias("sm"),
        ])
    )
    rows   = {r[group_col]: r for r in agg.to_dicts()}
    cats   = [c for c in order if c in rows]
    if not cats:
        return None, None
    xpos = np.array([order.index(c) for c in cats])
    md     = np.array([rows[c]["md"] for c in cats])
    sd     = np.array([rows[c]["sd"] for c in cats])
    mm     = np.array([rows[c]["mm"] for c in cats])
    sm     = np.array([rows[c]["sm"] for c in cats])
    n_subj = subj["subject"].n_unique()

    # ── data dots ─────────────────────────────────────────────────────────────
    data_h = None
    for i, (x, y) in enumerate(zip(xpos, md)):
        eb = ax.errorbar(
            x, y,
            yerr=sd[i] if n_subj > 1 else None,
            fmt="o", color=color, ms=7, capsize=3,
            alpha=0.55, zorder=5, label="_nolegend_",
        )
        if data_h is None:
            data_h = eb

    # ── model prediction line ─────────────────────────────────────────────────
    (model_h,) = ax.plot(
        xpos, mm, "-", color=color, lw=2.2, alpha=0.95,
        zorder=6, label="_nolegend_",
    )
    if n_subj > 1:
        ax.fill_between(xpos, mm - sm, mm + sm, color=color, alpha=0.10, zorder=3)

    return data_h, model_h


def plot_categorical_performance_by_state(
    df,
    views: dict,
    model_name: str,
):
    """
    Plot per-state categorical performance: dots + line per state,  no pooled
    overlay.  Supports multi-subject DataFrames when `state_assign` is provided.

    Parameters
    ----------
    df            : prepared predictions DataFrame (prepare_predictions_df output)
    smoothed_probs: (T, K) array, ignored when state_assign is given
    state_labels  : {rank_idx: label_str}  e.g. {0: "Engaged", 1: "Disengaged"}
                    For pooled multi-subject calls use normalised rank indices
                    (0=Engaged, 1=Disengaged, …).
    model_name    : string for figure suptitle
    state_assign  : optional pre-computed (T,) int array of normalised state
                    ranks (0=Engaged, 1=Disengaged, …).  If provided,
                    smoothed_probs is ignored.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    if "state_rank" not in df.columns:
        raise ValueError("df must contain 'state_rank' (from build_trial_df).")

    # resolve K
    K = next(iter(views.values())).K if views else int(df["state_rank"].max()) + 1

    # resolve labels by rank
    state_labels = {}
    for v in views.values():
        for raw_idx, lbl in v.state_name_by_idx.items():
            rank = v.state_rank_by_idx[int(raw_idx)]
            state_labels.setdefault(rank, lbl)

    df = df.with_columns(pl.col("state_rank").cast(pl.Int64).alias("_state_k"))

    _state_colors = {
        k: get_state_color(state_labels.get(k, f"State {k}"), k, K=K)
        for k in range(K)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    ax1, ax2, ax3 = axes

    panels = [
        (ax1, df,                                   "ttype_c", cfg["plots"]["ttype"]["order"],
         "a) Trial difficulty", "Trial difficulty",  cfg["plots"]["ttype"]["labels"]),
        (ax2, df.filter(pl.col("ttype_c") == "DS"), "stimd_c", cfg["plots"]["stimd"]["order"],
         "b) Stim duration",    "Stimulus type",     cfg["plots"]["stimd"]["labels"]),
        (ax3, df.filter(pl.col("stimd_c") == "SS"), "ttype_c", cfg["plots"]["delay"]["order"],
         "c) Delay duration",   "Delay type",        cfg["plots"]["delay"]["labels"]),
    ]

    # collect handles for legend (first panel that has data for each state)
    _data_handles  = {}   # k -> first data errorbar container
    _model_handles = {}   # k -> first model line

    for ax, df_panel, gcol, order, title, xlabel, labels in panels:
        # per-state dots + line (no pooled layer)
        for k in range(K):
            df_k = df_panel.filter(pl.col("_state_k") == k)
            d_h, m_h = _plot_state_panel(
                ax, df_k, gcol, order,
                color=_state_colors[k],
                label=state_labels.get(k, f"State {k}"),
            )
            if k not in _data_handles and d_h is not None:
                _data_handles[k]  = d_h
                _model_handles[k] = m_h

        # axis decoration based on categories present across all states
        _cats = [c for c in order
                 if df_panel.filter(pl.col(gcol) == c).height > 0]
        if labels:
            _lmap = dict(zip(order, labels))
            _tick_labels = [_lmap.get(c, c) for c in _cats]
        else:
            _tick_labels = _cats
        ax.set_xticks(np.arange(len(_cats)))
        ax.set_xticklabels(_tick_labels)
        ax.set_ylim(0.2, 1.05)
        ax.axhspan(0, 1 / 3, color="gray", alpha=0.15)
        ax.set_xlim(left=-0.4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ax is ax1:
            ax.set_ylabel("Accuracy")

    # ── shared legend: data dots then model lines, grouped by state ───────────
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    legend_handles = []
    legend_labels  = []
    for k in range(K):
        _lbl   = state_labels.get(k, f"State {k}")
        _color = _state_colors[k]
        legend_handles.append(
            mlines.Line2D([], [], marker="o", color=_color, linestyle="None",
                          ms=7, alpha=0.55, label=f"{_lbl} data")
        )
        legend_labels.append(f"{_lbl} data")
        legend_handles.append(
            mlines.Line2D([], [], color=_color, lw=2.2, alpha=0.95,
                          label=f"{_lbl} model")
        )
        legend_labels.append(f"{_lbl} model")

    ax3.legend(legend_handles, legend_labels, fontsize=8, frameon=False,
               bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle(model_name, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, axes


def plot_categorical_performance_all(df, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    ax1, ax2, ax3 = axes
    df = df.drop("p_model_correct").rename({"p_model_correct_marginal": "p_model_correct"})
    df_a = df.clone()
    
    plot_cat_panel(ax1, df_a, "ttype_c", cfg["plots"]["ttype"]["order"],
                    title="a) Trial difficulty",
                    xlabel="Trial difficulty",
                    ylabel="Accuracy",
                    palette=cfg["plots"]["ttype"]["palette"], labels=cfg["plots"]["ttype"]["labels"])
    # b) Stim duration (DS, SS/SM/SL)
    df_b = df.filter(pl.col("ttype_c") == "DS")
    plot_cat_panel(ax2, df_b, "stimd_c", cfg["plots"]["stimd"]["order"],
                    title="b) Stim duration",
                    xlabel="Stimulus type",
                    palette=cfg["plots"]["stimd"]["palette"], labels=cfg["plots"]["stimd"]["labels"])

    # c) Delay duration (SS)
    df_c = df.filter(pl.col("stimd_c") == "SS")
    plot_cat_panel(ax3, df_c, "ttype_c", cfg["plots"]["delay"]["order"],
                    title="c) Delay duration",
                    xlabel="Delay type",
                    palette=cfg["plots"]["delay"]["palette"], labels=cfg["plots"]["delay"]["labels"])
    sns.despine()
    fig.tight_layout()
    return fig, axes


def plot_delay_or_stim_1d_on_ax( ax, df, subject, n_bins, which):
    """
    Makes the delay or stim duration plot for a single subject on the given axis.
    - which: "delay" or "stim"
    returns True if it plotted something, False if no data for that subject/condition (in which case the panel is left blank with a title indicating no data).
    """
    df = df.to_pandas()
    df_delay = df[df["stimd_c"] == "SS"]
    df_stim  = df[df["ttype_c"] == "DS"].copy()
    df_stim = df.copy()

    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim  = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", "stim_d"]
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim  = df_stim.dropna(subset=needed_cols)

    if which == "delay":
        d = df_delay
        xcol = "delay_d"
        xlabel = "Delay duration"
        title_suffix = "Delay"
        band_floor = 1/3
        palette_data = truncate_colormap("Purples_r", 0, 0.7)
    elif which == "stim":
        d = df_stim
        xcol = "stim_d"
        xlabel = "Stimulus duration"
        title_suffix = "Stimulus"
        band_floor = 1/3
        palette_data = truncate_colormap("Oranges", 0.3, 1.0)
    else:
        raise ValueError("which must be 'delay' or 'stim'")

    if d.empty:
        ax.set_title(f"{subject} - {title_suffix}\n(no data)", fontsize=9)
        ax.axis("off")
        return False

    d = d.copy()
    d["x_bin"], edges = pd.qcut(d[xcol], q=n_bins, retbins=True, duplicates="drop")

    centers = (
        d.groupby("x_bin", observed=True)[xcol].median().rename("center").reset_index().sort_values("center")
    )
    order_bins = list(centers["x_bin"])

    subj = (
        d.groupby(["x_bin", "subject"], observed=True)
         .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
         )
         .reset_index()
         .merge(centers, on="x_bin", how="left")
    )

    plot_df = subj.melt(
        id_vars=["x_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_df["kind"] = plot_df["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    sns.lineplot(
        data=plot_df[plot_df["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        err_kws=CI_BAND_ERR_KWS,
        ax=ax,
    )

    sns.lineplot(data=plot_df[plot_df["kind"] == "Data"], x="center", y="acc", hue="center", palette=palette_data, marker="o", linewidth=0,errorbar=("ci", 95), err_style="bars",legend=False,ax=ax,zorder=10,)

    ax.axhspan(0, band_floor, color="gray", alpha=0.15, zorder=0)

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frac. correct responses", fontsize=12)
    ax.set_title(f"{subject}", fontsize=12)
    ax.tick_params(labelsize=12)
    sns.despine()

    return True


def plot_categorical_strat_by_side(df, subject, model_name, df_silent = None, cond_col="stimd_c",
                                   cond_order=['VG', 'SL', 'SM', 'SS', 'SIL'], cond_labels=['Visual', 'Easy', 'Medium', 'Hard', 'Silent']):
    df = df.to_pandas()
    df = df.copy()
    df["x_c"] = (df["x_c"].astype("string").str.strip().str.upper())

    if cond_order is None:
        cond_order = list(df[cond_col].dropna().unique())
        cond_order = sorted(cond_order)

    if cond_labels is None:
        cond_labels = cond_order

    g = (df.groupby([cond_col, "x_c"], observed=True).agg(data_mean=("correct_bool", "mean"), model_mean=("p_model_correct", "mean"), n=("correct_bool", "size")).reset_index())

    g["data_sem"] = np.sqrt(g["data_mean"] * (1.0 - g["data_mean"]) / g["n"].clip(lower=1))

    if df_silent is not None:
        df_s = df_silent.copy()
        p_silent = {"L": df_s["pL_mean"], "C": df_s["pC_mean"], "R": df_s["pR_mean"]}

    cond_to_x = {c: i for i, c in enumerate(cond_order)}
    g["x_pos"] = g[cond_col].map(cond_to_x)

    side_palette = {'L': '#e41a1c', 'C': '#4daf4a', 'R': '#377eb8'}

    fig, ax = plt.subplots(figsize=(4,4))

    for side in ["L", "C", "R"]:
        sub = g[g["x_c"] == side].dropna(subset=["x_pos"])
        if sub.empty:
            continue

        sub = sub.sort_values("x_pos")

        ax.plot( sub["x_pos"], sub["model_mean"], "-", lw=2, color=side_palette.get(side, "gray"), label=f"Model {side}", zorder=2)

        ax.errorbar( sub["x_pos"], sub["data_mean"], yerr=sub["data_sem"], fmt="o", ms=5, capsize=3, color=side_palette.get(side, "gray"), linestyle="none", label=f"Data {side}", zorder=3)

        if df_silent is not None:
            ax.plot(len(cond_order)-1, p_silent[side],marker="D", ms=7,color=side_palette[side],linestyle="none",zorder=4)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_labels)

    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Frac. correct responses")
    ax.set_xlabel("Trial difficulty")
    ax.set_title(f"{subject}")

    # ax.legend(frameon=False, fontsize=8, ncol=2)
    sns.despine()
    fig.tight_layout()

    fname = f"fig_categorical_strat_by_side_{subject}.pdf"
    out_path = get_plot_path("strat_by_side", fname, model_name)

    return fig, ax

def plot_delay_binned_1d(df, model_name, subject=None, n_bins=7):
    # n_bins=3
    # df_delay = df[df['onset']==0.0].copy()
    df = df.to_pandas()
    df_delay = df[df['stimd_c'] == 'SS']
    # df_stim = df[df['ttype_c']!='VG'].copy()
    df_stim = df[df['ttype_c']=='DS'].copy()
    
    
    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", 'stim_d']
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim = df_stim.dropna(subset=needed_cols)
    if df_delay.empty:
        print(f"  (sin datos válidos para delay 1D en {subject})")
        return
    elif df_stim.empty:
        print(f"  (sin datos válidos para stim 1D en {subject})")
        return
    
    # df_delay["delay_bin"], edges = pd.qcut(df_delay["delay_duration"], q=n_bins, retbins=True, duplicates="drop")
    # df_stim["stim_bin"], edges_stim = pd.qcut(df_stim["stim_duration"], q=n_bins, retbins=True, duplicates="drop")
    # centers_delay = (df_delay.groupby("delay_bin", observed=True)["delay_duration"].median().rename("center").reset_index().sort_values("center"))
    # centers_stim = (df_stim.groupby("stim_bin", observed=True)["stim_duration"].median().rename("center").reset_index().sort_values("center"))
    # order_bins_delay = list(centers_delay["delay_bin"])
    # order_bins_stim = list(centers_stim["stim_bin"])
    
    df_delay["delay_bin"] = (
    df_delay.groupby("ttype_c", observed=True)["delay_d"]
    .transform(lambda s: pd.qcut(s, q=n_bins, duplicates="drop"))
    )

    # centers por ttype_c y bin
    centers_delay = (
        df_delay.groupby(["ttype_c", "delay_bin"], observed=True)["delay_d"]
        .median()
        .rename("center")
        .reset_index()
    )

    # (opcional) order de bins dentro de cada ttype_c según center
    centers_delay["bin_order"] = centers_delay.groupby("ttype_c")["center"].rank(method="dense")
    order_bins_delay = list(centers_delay["delay_bin"])
    # agregación por bin+subject+ttype_c
    subj_delay = (
        df_delay.groupby(["ttype_c", "delay_bin", "subject"], observed=True)
        .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
        )
        .reset_index()
        .merge(centers_delay, on=["ttype_c", "delay_bin"], how="left")
    )
    df_stim["stim_bin"] = (
    df_stim.groupby("stimd_c", observed=True)["stim_d"]
    .transform(lambda s: pd.qcut(s, q=n_bins, duplicates="drop"))   
    )

    centers_stim = (
        df_stim.groupby(["stimd_c", "stim_bin"], observed=True)["stim_d"]
        .median()
        .rename("center")
        .reset_index()
    )
    order_bins_stim = list(centers_stim["stim_bin"])
    subj_stim = (
        df_stim.groupby(["stimd_c", "stim_bin", "subject"], observed=True)
        .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
        )
        .reset_index()
        .merge(centers_stim, on=["stimd_c", "stim_bin"], how="left")
    )

    plot_stim = subj_stim.melt(
        id_vars=["stimd_c", "stim_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data", "model_acc": "Model"})


    # subj_delay = (df_delay.groupby(["delay_bin", "subject", "ttype_c"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_delay, on="delay_bin", how="left"))
    plot_delay = subj_delay.melt(id_vars=["delay_bin", "subject", "ttype_c", "center"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_delay["kind"] = plot_delay["kind"].map({"data_acc": "Data","model_acc": "Model"})

    # subj_stim = (df_stim.groupby(["stim_bin", "subject", "stimd_c"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_stim, on="stim_bin", how="left"))
    plot_stim = subj_stim.melt(id_vars=["stim_bin", "subject", "center", "stimd_c"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data","model_acc": "Model"})


    fig, ax = plt.subplots(figsize=(6, 6))

    sns.lineplot(
        data=plot_delay[plot_delay["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        hue="ttype_c",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        err_kws=CI_BAND_ERR_KWS,
        ax=ax,
    )
    sns.lineplot(x="center", y="acc", hue="ttype_c",data=plot_delay[plot_delay["kind"] == "Data"], errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, legend=False)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Delay duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")

    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Delay (1D, {len(order_bins_delay)} bins)")

    sns.despine()
    fig.tight_layout()

    fname = f"fig_delay_1d_{title_subj}.pdf"
    out_path = get_plot_path("binning", fname, model_name)
    fig.savefig(out_path, dpi=300)

    plt.show

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(
        data=plot_stim[plot_stim["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        hue="stimd_c",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        err_kws=CI_BAND_ERR_KWS,
        ax=ax,
    )
    sns.lineplot(x="center", y="acc", hue="stimd_c",data=plot_stim[plot_stim["kind"] == "Data"],errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, legend=False)
    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Stimulus duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Stimulus (1D, {len(order_bins_stim)} bins)")
    sns.despine()
    fig.tight_layout()
    fname = f"fig_stim_1d_{title_subj}.pdf"
    out_path = get_plot_path("binning", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.show()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# GLM-HMM / GLM-HMM-t notebook analysis helpers
# Shared plotting functions extracted from glmhmm_analysis.py and
# glmhmmt_analysis.py.  All functions return matplotlib Figure objects;
# marimo UI wrappers (mo.vstack etc.) stay in the notebooks.
# ─────────────────────────────────────────────────────────────────────────────

_AG_GROUPS = [
    # neutral intercept bias (constant 1): overall lateral vs centre preference
    ("$bias$", [("bias", "mean")]),
    # bias: L/R context indicators (no C equivalent)
    ("$bias_{coh}$",  [("biasL", 0), ("biasR", 1)]),
    ("$bias_{incoh}$", [("biasL", 1), ("biasR", 0)]),
    # onset — C side merged in: coh=P(C)|onsetC, incoh=(P(L)+P(R))/2|onsetC
    ("$onset_{coh}$",  [("onsetL", 0), ("onsetR", 1), ("onsetC", "neg_mean")]),
    ("$onset_{incoh}$", [("onsetL", 1), ("onsetR", 0), ("onsetC", "mean")]),
    # delay (shared scalar)
    ("delay", [("delay", "mean")]),
    # delay × side
    ("$D_{coh}$",  [("DL", 0), ("DR", 1)]),
    ("$D_{incoh}$", [("DL", 1), ("DR", 0)]),
    ("DC", [("DC", "neg_mean")]),
    # stimulus — SC merged in
    ("$S_{coh}$",  [("SL", 0), ("SR", 1), ("SC", "neg_mean")]),
    ("$S_{incoh}$", [("SL", 1), ("SR", 0), ("SC", "mean")]),
    # stimulus × delay — SCxdelay merged in
    ("$Sxd_{coh}$",  [("SLxdelay", 0), ("SRxdelay", 1), ("SCxdelay", "neg_mean")]),
    ("$Sxd_{incoh}$", [("SLxdelay", 1), ("SRxdelay", 0), ("SCxdelay", "mean")]),
    # stim interval × side — stim1…4 collapsed like SL/SC/SR
    ("$S1_{coh}$",   [("stim1L", 0), ("stim1R", 1), ("stim1C", "neg_mean")]),
    ("$S1_{incoh}$", [("stim1L", 1), ("stim1R", 0), ("stim1C", "mean")]),
    ("$S2_{coh}$",   [("stim2L", 0), ("stim2R", 1), ("stim2C", "neg_mean")]),
    ("$S2_{incoh}$", [("stim2L", 1), ("stim2R", 0), ("stim2C", "mean")]),
    ("$S3_{coh}$",   [("stim3L", 0), ("stim3R", 1), ("stim3C", "neg_mean")]),
    ("$S3_{incoh}$", [("stim3L", 1), ("stim3R", 0), ("stim3C", "mean")]),
    ("$S4_{coh}$",   [("stim4L", 0), ("stim4R", 1), ("stim4C", "neg_mean")]),
    ("$S4_{incoh}$", [("stim4L", 1), ("stim4R", 0), ("stim4C", "mean")]),
    # action history (perseveration vs alternation)
    ("$A_{coh}$",  [("A_L", 0), ("A_R", 1)]),
    ("$A_{incoh}$", [("A_L", 1), ("A_R", 0)]),
]


def _collect_emission_weight_frames(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    subjects: list,
):
    _CLS_LABELS  = ["Left (vs C)", "Right (vs C)"]
    _records     = []
    _ag_records  = []
    _feat_names  = names.get("X_cols", [])

    for subject in subjects:
        if subject not in arrays_store:
            continue
        _W      = arrays_store[subject]["emission_weights"]   # (K, 2, n_feat)
        _n      = _W.shape[2]
        _fnames = (arrays_store[subject].get("X_cols") or names.get("X_cols", []))[:_n]
        _f2i    = {f: i for i, f in enumerate(_fnames)}
        _feat_names = _fnames   # keep last subject's list for per-class axis labels

        for _k in range(_W.shape[0]):
            _slbl = state_labels.get(subject, {}).get(_k, f"State {_k}")
            for _c in range(_W.shape[1]):
                for _fi, _fn in enumerate(_fnames):
                    _records.append({
                        "subject":     subject,
                        "state":       _slbl,
                        "class":       _c,
                        "class_label": _CLS_LABELS[_c] if _c < len(_CLS_LABELS) else f"Class {_c}",
                        "feature":     _fn,
                        "weight":      float(_W[_k, _c, _fi]),
                    })
            # ── agonist: softmax ΔP (same convention as fit_glm.py) ──────────
            # logits = [W[k,0,f], 0.0, W[k,1,f]]  →  p = [P(L), P(C), P(R)]
            # mode=0 (L-vs-C weight, coh L feat) → P(L) − 1/3
            # mode=1 (R-vs-C weight, coh R feat) → P(R) − 1/3
            # mode="neg_mean" (C-only)           → P(C) − 1/3
            # mode="mean"    (neutral)           → (P(L)+P(R))/2 − 1/3
            _BASE = 1.0 / 3.0
            for _grp_label, _members in _AG_GROUPS:
                _vals = []
                for _fn, _mode in _members:
                    if _fn not in _f2i:
                        continue
                    _fi = _f2i[_fn]
                    _logits = [float(_W[_k, 0, _fi]), 0.0, float(_W[_k, 1, _fi])]
                    _p = _softmax(_logits)
                    if _mode == 0:
                        _vals.append(_p[0] - _BASE)                # P(L) − baseline
                    elif _mode == 1:
                        _vals.append(_p[2] - _BASE)                # P(R) − baseline
                    elif _mode == "neg_mean":
                        _vals.append(_p[1] - _BASE)                # P(C) − baseline
                    else:  # "mean" — neutral
                        _vals.append((_p[0] + _p[2]) / 2 - _BASE) # avg lateral
                if _vals:
                    _ag_records.append({
                        "subject": subject,
                        "state":   _slbl,
                        "feature": _grp_label,
                        "weight":  float(np.mean(_vals)),
                    })

    if not _records:
        raise ValueError("No emission weights found for the selected subjects.")

    _df_w     = pd.DataFrame(_records)
    _df_ag    = pd.DataFrame(_ag_records)
    _ag_order = [g for g, _ in _AG_GROUPS if g in _df_ag["feature"].values]
    _state_pal, _state_hue_order = build_state_palette(state_labels)
    return _df_w, _df_ag, _feat_names, _ag_order, _state_pal, _state_hue_order, _CLS_LABELS


def _pvalue_star(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def _draw_emission_summary_lineplot(
    ax,
    *,
    df_ag: pd.DataFrame,
    ag_order: list[str],
    state_pal: dict[str, str],
    state_hue_order: list[str],
    title: str,
) -> None:
    if df_ag.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return

    sns.lineplot(
        data=df_ag,
        x="feature",
        y="weight",
        hue="state",
        ax=ax,
        markers=True,
        marker="o",
        markersize=8,
        markeredgewidth=0,
        alpha=0.85,
        errorbar="se",
        palette=state_pal,
        hue_order=state_hue_order,
        err_kws={"edgecolor": "none", "linewidth": 0},
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel("ΔP (from 1/3 baseline)")
    ax.set_xlabel("")
    ax.set_title(title)
    if ag_order:
        ax.set_xticks(range(len(ag_order)))
        ax.set_xticklabels(ag_order)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles,
            labels,
            frameon=False,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
        )
        legend.set_title("")


def _draw_emission_summary_boxplot(
    ax,
    *,
    df_ag: pd.DataFrame,
    ag_order: list[str],
    state_pal: dict[str, str],
    state_hue_order: list[str],
    title: str | None = None,
) -> None:
    from scipy.stats import ttest_rel
    import itertools

    if df_ag.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        if title:
            ax.set_title(title)
        return

    sns.boxplot(
        data=df_ag,
        x="feature",
        y="weight",
        hue="state",
        ax=ax,
        palette=state_pal,
        hue_order=state_hue_order,
        width=0.8,
        showfliers=False,
        boxprops={"alpha": 0.7},
    )
    sns.stripplot(
        data=df_ag,
        x="feature",
        y="weight",
        hue="state",
        ax=ax,
        palette=state_pal,
        hue_order=state_hue_order,
        dodge=True,
        alpha=0.5,
        zorder=1,
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    k_states = len(state_hue_order)
    if k_states > 1:
        state_pairs = list(itertools.combinations(range(k_states), 2))
        hue_width = 0.8 / k_states
        y_range = df_ag["weight"].max() - df_ag["weight"].min()
        if pd.isna(y_range) or y_range == 0:
            y_range = 1

        for feat_idx, feat in enumerate(ag_order):
            feat_df = df_ag[df_ag["feature"] == feat]
            if feat_df.empty:
                continue

            y_max = feat_df["weight"].max()
            y_offset_step = y_range * 0.05
            current_y_offset = y_max + y_offset_step

            for p1, p2 in state_pairs:
                s1 = state_hue_order[p1]
                s2 = state_hue_order[p2]
                df1 = feat_df[feat_df["state"] == s1].set_index("subject")["weight"]
                df2 = feat_df[feat_df["state"] == s2].set_index("subject")["weight"]
                common_subjs = df1.index.intersection(df2.index)
                if len(common_subjs) < 2:
                    continue

                w1 = df1.loc[common_subjs].values
                w2 = df2.loc[common_subjs].values

                try:
                    _stat, pval = ttest_rel(w1, w2)
                    star = _pvalue_star(pval)
                except Exception:
                    star = ""

                if not star:
                    continue

                offset_1 = (p1 - (k_states - 1) / 2) * hue_width
                offset_2 = (p2 - (k_states - 1) / 2) * hue_width
                x1 = feat_idx + offset_1
                x2 = feat_idx + offset_2
                h = y_range * 0.02
                ax.plot(
                    [x1, x1, x2, x2],
                    [current_y_offset, current_y_offset + h, current_y_offset + h, current_y_offset],
                    lw=1,
                    c="k",
                )
                ax.text((x1 + x2) / 2, current_y_offset + h, star, ha="center", va="bottom", color="k")
                current_y_offset += y_offset_step * 1.5

    ax.set_xticks(range(len(ag_order)))
    ax.set_xticklabels(ag_order)
    ax.set_xlabel("")
    ax.set_ylabel("ΔP (from 1/3 baseline)")
    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= k_states:
        ax.legend(
            handles[:k_states],
            labels[:k_states],
            frameon=False,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
        )
    elif handles:
        ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")


def plot_emission_weights_by_subject(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
):
    """Per-subject barplots of emission weights."""
    _df_w, _, _feat_names, _, _state_pal, _state_hue_order, _cls_labels = _collect_emission_weight_frames(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )

    _selected = [s for s in subjects if s in arrays_store and "emission_weights" in arrays_store[s]]
    _n_classes = max(1, _df_w["class"].nunique())
    _n_cols = min(3, max(1, len(_selected)))
    _n_panels = max(1, len(_selected) * _n_classes)
    _n_rows = int(math.ceil(_n_panels / _n_cols))
    _fig_w = max(6, len(_feat_names) * 0.8) * _n_cols
    _fig_h = max(3.4, 3.2 * _n_rows)
    fig_bar, axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(_fig_w, _fig_h),
        sharey=True,
        squeeze=False,
    )

    _x = np.arange(len(_feat_names))
    _bar_w = 0.8 / max(1, len(_state_hue_order))

    for subj_idx, subj in enumerate(_selected):
        for class_idx in range(_n_classes):
            panel_idx = subj_idx * _n_classes + class_idx
            ax = axes[panel_idx // _n_cols, panel_idx % _n_cols]
            _sub = _df_w[(_df_w["subject"] == subj) & (_df_w["class"] == class_idx)]
            for state_pos, state_name in enumerate(_state_hue_order):
                _state_sub = (
                    _sub[_sub["state"] == state_name]
                    .set_index("feature")
                    .reindex(_feat_names)
                    .reset_index()
                )
                _offset = (state_pos - (len(_state_hue_order) - 1) / 2) * _bar_w
                ax.bar(
                    _x + _offset,
                    _state_sub["weight"].to_numpy(dtype=float),
                    _bar_w,
                    label=state_name,
                    color=_state_pal[state_name],
                    alpha=0.85,
                )
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_xticks(_x)
            ax.set_xticklabels(_feat_names, rotation=35, ha="right")
            ax.set_title(
                f"Subject {subj} — {_cls_labels[class_idx] if class_idx < len(_cls_labels) else f'Class {class_idx}'}"
            )
            if panel_idx % _n_cols == 0:
                ax.set_ylabel("Weight")

    for panel_idx in range(_n_panels, _n_rows * _n_cols):
        axes[panel_idx // _n_cols, panel_idx % _n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig_bar.suptitle(f"Emission weights by subject  (K={K})", y=1.01)
    fig_bar.tight_layout()
    sns.despine(fig=fig_bar)
    if save_path is not None:
        fig_bar.savefig(save_path, dpi=300)
    return fig_bar


def plot_emission_weights_summary_lineplot(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
):
    _ = save_path
    _df_w, _df_ag, _feat_names, _ag_order, _state_pal, _state_hue_order, _CLS_LABELS = _collect_emission_weight_frames(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )
    fig, ax = plt.subplots(figsize=(max(6.0, 2.1 * max(1, len(_ag_order))), 4.2))
    _draw_emission_summary_lineplot(
        ax,
        df_ag=_df_ag,
        ag_order=_ag_order,
        state_pal=_state_pal,
        state_hue_order=_state_hue_order,
        title=f"Emission weights - collapsed line summary  (K={K})",
    )
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_emission_weights_summary_boxplot(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
):
    _ = save_path
    _df_w, _df_ag, _feat_names, _ag_order, _state_pal, _state_hue_order, _CLS_LABELS = _collect_emission_weight_frames(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )
    fig, ax = plt.subplots(figsize=(max(6.0, 2.1 * max(1, len(_ag_order))), 4.2))
    _draw_emission_summary_boxplot(
        ax,
        df_ag=_df_ag,
        ag_order=_ag_order,
        state_pal=_state_pal,
        state_hue_order=_state_hue_order,
        title=f"Emission weights - collapsed box summary  (K={K})",
    )
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_emission_weights(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
):
    """
    Emission-weight summaries: collapsed agonist view + per-choice-class panels.

    The per-subject barplots live in ``plot_emission_weights_by_subject`` so
    notebooks can render or comment them independently.
    """
    _df_w, _df_ag, _feat_names, _ag_order, _state_pal, _state_hue_order, _CLS_LABELS = _collect_emission_weight_frames(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )

    # ── 1. Agonist (collapsed) figure ─────────────────────────────────────────
    fig_ag, axes_ag = plt.subplots(1, 2, figsize=(max(8.0, len(_ag_order) * 3.0), 4.2), sharex=True)
    ax_ag_line, ax_ag_box = axes_ag

    _draw_emission_summary_lineplot(
        ax_ag_line,
        df_ag=_df_ag,
        ag_order=_ag_order,
        state_pal=_state_pal,
        state_hue_order=_state_hue_order,
        title=f"Emission weights - collapsed view  (K={K})",
    )
    _draw_emission_summary_boxplot(
        ax_ag_box,
        df_ag=_df_ag,
        ag_order=_ag_order,
        state_pal=_state_pal,
        state_hue_order=_state_hue_order,
    )

    K_ag = len(_state_hue_order)
    state_pairs = []
    hue_width = 0.8 / K_ag if K_ag else 0.8
    fig_ag.tight_layout()
    sns.despine(fig=fig_ag)

    # ── 2. Per-class figure ────────────────────────────────────────────────────
    _n_classes = _df_w["class"].nunique()
    fig_cls, axes_cls_grid = plt.subplots(
        2, _n_classes, figsize=(6 * _n_classes, 8), sharex=True, squeeze=False
    )
    
    y_range_cls = _df_w["weight"].max() - _df_w["weight"].min()
    if pd.isna(y_range_cls) or y_range_cls == 0:
        y_range_cls = 1
        
    for _c in range(_n_classes):
        _ax_line = axes_cls_grid[0, _c]
        _ax_box = axes_cls_grid[1, _c]
        _sub = _df_w[_df_w["class"] == _c]
        
        sns.lineplot(
            data=_sub, x="feature", y="weight", hue="state", ax=_ax_line,
            markers=True, marker="o", markersize=8, markeredgewidth=0,
            alpha=0.8, errorbar="se",
            palette=_state_pal, hue_order=_state_hue_order,
            legend=False
        )
        _ax_line.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        _ax_line.set_title(_CLS_LABELS[_c] if _c < len(_CLS_LABELS) else f"Class {_c}")
        _ax_line.set_xlabel("")
        _ax_line.set_ylabel("Weight" if _c == 0 else "")

        sns.boxplot(
            data=_sub, x="feature", y="weight", hue="state", ax=_ax_box,
            palette=_state_pal, hue_order=_state_hue_order,
            width=0.8, showfliers=False, boxprops={'alpha': 0.7}
        )
        sns.stripplot(
            data=_sub, x="feature", y="weight", hue="state", ax=_ax_box,
            palette=_state_pal, hue_order=_state_hue_order,
            dodge=True, alpha=0.5, zorder=1, legend=False
        )
        _ax_box.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        
        # Annotations
        for m, feat in enumerate(_feat_names):
            feat_df = _sub[_sub["feature"] == feat]
            if feat_df.empty: continue
            
            y_max = feat_df["weight"].max()
            y_offset_step = y_range_cls * 0.05
            current_y_offset = y_max + y_offset_step
            
            for p1, p2 in state_pairs:
                s1 = _state_hue_order[p1]
                s2 = _state_hue_order[p2]
                
                df1 = feat_df[feat_df["state"] == s1].set_index("subject")["weight"]
                df2 = feat_df[feat_df["state"] == s2].set_index("subject")["weight"]
                
                common_subjs = df1.index.intersection(df2.index)
                if len(common_subjs) < 2: continue
                
                w1 = df1.loc[common_subjs].values
                w2 = df2.loc[common_subjs].values
    
                try:
                    stat, pval = ttest_rel(w1, w2)
                    star = get_star(pval)
                except Exception:
                    star = ""
    
                if star:
                    offset_1 = (p1 - (K_ag - 1) / 2) * hue_width
                    offset_2 = (p2 - (K_ag - 1) / 2) * hue_width
                    x1 = m + offset_1
                    x2 = m + offset_2
                    
                    h = y_range_cls * 0.02
                    _ax_box.plot([x1, x1, x2, x2], [current_y_offset, current_y_offset+h, current_y_offset+h, current_y_offset], lw=1, c='k')
                    _ax_box.text((x1+x2)/2, current_y_offset+h, star, ha='center', va='bottom', color='k')
                    current_y_offset += y_offset_step * 1.5

        _ax_box.set_xticks(range(len(_feat_names)))
        _ax_box.set_xticklabels(_feat_names, rotation=35, ha="right")
        _ax_box.set_xlabel("")
        _ax_box.set_ylabel("Weight" if _c == 0 else "")
        
        handles, labels_lgd = _ax_box.get_legend_handles_labels()
        if len(handles) >= K_ag:
            _ax_box.legend(handles[:K_ag], labels_lgd[:K_ag], frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        else:
            _ax_box.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig_cls.suptitle(f"Emission weights per choice  (K={K})", y=1.02)
    fig_cls.tight_layout()
    sns.despine(fig=fig_cls)

    return fig_ag, fig_cls


def _resolve_transition_matrix_from_arrays(arrays_store: dict, subj: str) -> np.ndarray | None:
    arr = arrays_store.get(subj, {})
    if "transition_matrix" in arr:
        return np.asarray(arr["transition_matrix"])
    if "transition_bias" in arr:
        bias = np.asarray(arr["transition_bias"])
        exp_bias = np.exp(bias - bias.max(axis=-1, keepdims=True))
        return exp_bias / exp_bias.sum(axis=-1, keepdims=True)
    return None


def _apply_label_priority(labels: Sequence[str], priority: Sequence[str] | None) -> list[int]:
    if not priority:
        return list(range(len(labels)))

    remaining = list(range(len(labels)))
    ordered: list[int] = []
    lower_labels = [label.lower() for label in labels]
    for desired in priority:
        desired_lower = desired.lower()
        matched = [idx for idx in remaining if lower_labels[idx] == desired_lower]
        ordered.extend(matched)
        remaining = [idx for idx in remaining if idx not in matched]
    ordered.extend(remaining)
    return ordered


def _reorder_binary_state_weights(
    weights: np.ndarray,
    state_labels: Sequence[str],
    *,
    state_label_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    labels = list(state_labels)
    if not labels:
        return np.asarray(weights), labels

    order = _apply_label_priority(labels, state_label_order)
    ordered_weights = np.take(np.asarray(weights), order, axis=0)
    ordered_labels = [labels[idx] for idx in order]
    return ordered_weights, ordered_labels


def _reorder_binary_feature_weights(
    weights: np.ndarray,
    feature_names: Sequence[str],
    *,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
) -> tuple[np.ndarray, list[str]]:
    names = list(feature_names)
    if not names:
        return np.asarray(weights), names

    desired = [name for name in (feature_order or ()) if name in names]
    ordered_names = desired + [name for name in names if name not in desired]
    order = [names.index(name) for name in ordered_names]
    ordered_weights = np.take(np.asarray(weights), order, axis=-1).copy()

    abs_feature_set = set(abs_features)
    for idx, name in enumerate(ordered_names):
        if name in abs_feature_set:
            ordered_weights[..., idx] = np.abs(ordered_weights[..., idx])

    return ordered_weights, ordered_names


def _prepare_binary_emission_subject(
    view,
    *,
    weight_sign: float,
    state_label_order: Sequence[str] | None,
    feature_order: Sequence[str] | None,
    abs_features: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    weights = np.asarray(view.emission_weights, dtype=float) * float(weight_sign)
    state_indices = list(view.state_idx_order)
    state_labels = [view.state_name_by_idx.get(k, f"State {k}") for k in state_indices]
    weights = weights[state_indices]
    weights, state_labels = _reorder_binary_state_weights(
        weights,
        state_labels,
        state_label_order=state_label_order,
    )
    weights, feature_names = _reorder_binary_feature_weights(
        weights,
        view.feat_names or [],
        feature_order=feature_order,
        abs_features=abs_features,
    )
    return weights, state_labels, feature_names


def _draw_binary_emission_weights_by_subject(
    ax,
    *,
    weights: np.ndarray,
    feature_names: Sequence[str],
    state_labels: Sequence[str],
    feature_labeler: Callable[[str], str] | None,
    K: int,
    title: str,
) -> None:
    plot_weights = np.asarray(weights, dtype=float)
    if plot_weights.ndim == 2:
        plot_weights = plot_weights[:, None, :]
    x = np.arange(plot_weights.shape[-1])
    bar_width = 0.8 / max(1, len(state_labels))
    colors = [get_state_color(label, idx, K=K) for idx, label in enumerate(state_labels)]
    tick_labels = [feature_labeler(name) if feature_labeler else name for name in feature_names]

    for state_idx, state_label in enumerate(state_labels):
        offset = (state_idx - (len(state_labels) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            plot_weights[state_idx].mean(axis=0),
            bar_width,
            label=state_label,
            color=colors[state_idx],
            alpha=0.85,
        )

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylabel("Weight")
    ax.set_title(title)


def _prepare_binary_emission_summary_arrays(
    views: dict,
    *,
    K: int,
    weight_sign: float,
    state_label_order: Sequence[str] | None,
    feature_order: Sequence[str] | None,
    abs_features: Sequence[str],
) -> tuple[np.ndarray | None, list[str], list[str], list[str]]:
    if not views:
        return None, [], [], []

    all_weights: list[np.ndarray] = []
    hue_order: list[str] | None = None
    feature_names: list[str] = []

    for view in views.values():
        weights, state_labels, subject_feature_names = _prepare_binary_emission_subject(
            view,
            weight_sign=weight_sign,
            state_label_order=state_label_order,
            feature_order=feature_order,
            abs_features=abs_features,
        )
        if weights.ndim == 2:
            weights = weights[:, None, :]
        all_weights.append(weights)
        feature_names = list(subject_feature_names)
        if hue_order is None:
            hue_order = list(state_labels)

    if not all_weights:
        return None, [], [], []

    stacked = np.stack(all_weights, axis=0)
    resolved_hue_order = hue_order or [f"State {idx}" for idx in range(K)]
    state_colors = [get_state_color(label, idx, K=K) for idx, label in enumerate(resolved_hue_order)]
    return stacked, feature_names, resolved_hue_order, state_colors


def _binary_emission_plot_df(
    views=None,
    *,
    weights_df=None,
    K: int | None = None,
    weight_sign: float,
    state_label_order: Sequence[str] | None,
    feature_order: Sequence[str] | None,
    abs_features: Sequence[str],
    feature_labeler: Callable[[str], str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    if weights_df is None:
        if views is None:
            return pd.DataFrame(), [], [], {}
        from glmhmmt.postprocess import build_emission_weights_df

        weights_df = build_emission_weights_df(views)

    df = weights_df.to_pandas() if isinstance(weights_df, pl.DataFrame) else pd.DataFrame(weights_df).copy()
    if df.empty:
        return df, [], [], {}

    if "class_idx" in df.columns:
        df = df[df["class_idx"] == 0].copy()
    if df.empty:
        return df, [], [], {}

    df["feature"] = df["feature"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce") * float(weight_sign)
    if "subject" not in df.columns:
        df["subject"] = "subject-0"
    else:
        df["subject"] = df["subject"].astype(str)
    if "state_label" not in df.columns:
        df["state_label"] = "State 0"
    else:
        df["state_label"] = df["state_label"].astype(str)
    if "state_rank" not in df.columns:
        df["state_rank"] = [get_state_rank(label, fallback_idx=0) for label in df["state_label"]]
    df = df.dropna(subset=["weight"]).copy()
    if df.empty:
        return df, [], [], {}

    abs_feature_set = set(abs_features)
    if abs_feature_set:
        abs_mask = df["feature"].isin(abs_feature_set)
        df.loc[abs_mask, "weight"] = df.loc[abs_mask, "weight"].abs()

    feature_names = list(pd.unique(df["feature"]))
    desired = [name for name in (feature_order or ()) if name in feature_names]
    ordered_features = desired + [name for name in feature_names if name not in desired]
    df["feature"] = pd.Categorical(df["feature"], categories=ordered_features, ordered=True)
    df = df[df["feature"].notna()].copy()
    if df.empty:
        return df, [], [], {}

    state_meta = (
        df[["state_rank", "state_label"]]
        .drop_duplicates()
        .sort_values(["state_rank", "state_label"])
        .reset_index(drop=True)
    )
    base_state_order = state_meta["state_label"].tolist()
    state_order = [base_state_order[idx] for idx in _apply_label_priority(base_state_order, state_label_order)]
    palette = {
        label: get_state_color(label, idx, K=K or len(state_order) or None)
        for idx, label in enumerate(state_order)
    }

    display_features = [feature_labeler(name) if feature_labeler else name for name in ordered_features]
    label_map = dict(zip(ordered_features, display_features, strict=False))
    df["Feature"] = pd.Categorical(
        df["feature"].map(label_map),
        categories=display_features,
        ordered=True,
    )
    df["State"] = pd.Categorical(
        df["state_label"],
        categories=state_order,
        ordered=True,
    )
    return df, display_features, state_order, palette


def plot_binary_emission_weights_by_subject(
    views: dict,
    K: int,
    *,
    weights_df=None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    save_path=None,
) -> plt.Figure:
    df, display_features, state_order, palette = _binary_emission_plot_df(
        views,
        weights_df=weights_df,
        K=K,
        weight_sign=weight_sign,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    valid_subjs = sorted(pd.unique(df["subject"]).tolist()) if not df.empty else []
    if not valid_subjs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    n_cols = min(3, len(valid_subjs))
    n_rows = int(math.ceil(len(valid_subjs) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5, 0.7 * max(1, len(display_features))) * n_cols, 3.5 * n_rows),
        sharey=True,
        squeeze=False,
    )

    for idx, subj in enumerate(valid_subjs):
        ax = axes[idx // n_cols, idx % n_cols]
        sns.barplot(
            data=df[df["subject"] == subj],
            x="Feature",
            y="weight",
            hue="State",
            hue_order=state_order,
            palette=palette,
            errorbar=None,
            dodge=True,
            ax=ax,
        )
        ax.axhline(0, color="k", lw=0.8, ls="--")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_ylabel("Weight")
        ax.set_xlabel("")
        ax.set_title(f"Subject {subj}")

    for idx in range(len(valid_subjs), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle(f"Emission weights  (K={K})", y=1.01)
    fig.tight_layout()
    sns.despine(fig=fig)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_binary_emission_weights_summary_lineplot(
    views: dict,
    K: int,
    *,
    weights_df=None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
) -> plt.Figure:
    df, display_features, state_order, palette = _binary_emission_plot_df(
        views,
        weights_df=weights_df,
        K=K,
        weight_sign=weight_sign,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * max(1, len(display_features))), 4.2))
    sns.lineplot(
        data=df,
        x="Feature",
        y="weight",
        hue="State",
        hue_order=state_order,
        palette=palette,
        ax=ax,
        markers=True,
        marker="o",
        markersize=8,
        markeredgewidth=0,
        alpha=0.85,
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
    )
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Weight")
    ax.set_xlabel("")
    ax.set_title(f"Emission weights - line summary  (K={K})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_binary_emission_weights_summary_boxplot(
    views: dict,
    K: int,
    *,
    weights_df=None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
) -> plt.Figure:
    df, display_features, state_order, palette = _binary_emission_plot_df(
        views,
        weights_df=weights_df,
        K=K,
        weight_sign=weight_sign,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

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
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_ylabel("Weight")
    ax.set_xlabel("")
    ax.set_title(f"Emission weights - box summary  (K={K})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[: len(state_order)], labels[: len(state_order)], frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def plot_binary_emission_weights_summary(
    views: dict,
    K: int,
    *,
    weights_df=None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
) -> plt.Figure:
    if weights_df is not None:
        df, _, _, _ = _binary_emission_plot_df(
            views,
            weights_df=weights_df,
            K=K,
            weight_sign=weight_sign,
            state_label_order=state_label_order,
            feature_order=feature_order,
            abs_features=abs_features,
            feature_labeler=feature_labeler,
        )
        n_subjects = df["subject"].nunique() if not df.empty else 0
    else:
        n_subjects = len(views)

    if n_subjects > 1:
        return plot_binary_emission_weights_summary_boxplot(
            views,
            K,
            weights_df=weights_df,
            weight_sign=weight_sign,
            state_label_order=state_label_order,
            feature_order=feature_order,
            abs_features=abs_features,
            feature_labeler=feature_labeler,
        )
    return plot_binary_emission_weights_by_subject(
        views,
        K,
        weights_df=weights_df,
        weight_sign=weight_sign,
        state_label_order=state_label_order,
        feature_order=feature_order,
        abs_features=abs_features,
        feature_labeler=feature_labeler,
    )


def plot_binary_emission_weights(
    views: dict,
    K: int,
    *,
    weights_df=None,
    weight_sign: float = 1.0,
    state_label_order: Sequence[str] | None = None,
    feature_order: Sequence[str] | None = None,
    abs_features: Sequence[str] = (),
    feature_labeler: Callable[[str], str] | None = None,
    save_path=None,
) -> tuple[plt.Figure, plt.Figure]:
    return (
        plot_binary_emission_weights_by_subject(
            views,
            K,
            weights_df=weights_df,
            weight_sign=weight_sign,
            state_label_order=state_label_order,
            feature_order=feature_order,
            abs_features=abs_features,
            feature_labeler=feature_labeler,
            save_path=save_path,
        ),
        plot_binary_emission_weights_summary(
            views,
            K,
            weights_df=weights_df,
            weight_sign=weight_sign,
            state_label_order=state_label_order,
            feature_order=feature_order,
            abs_features=abs_features,
            feature_labeler=feature_labeler,
        ),
    )


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


def plot_lapse_rates_boxplot(
    views: dict,
    K: int | None = None,
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
        df = (
            df.groupby(["Subject", label_col], as_index=False)["Rate"]
            .mean()
        )
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(order)), 4.2))
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

        paired = (
            df.pivot_table(index="Subject", columns=label_col, values="Rate", aggfunc="first")
            .reindex(columns=order)
        )
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
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def _default_state_labels(K: int) -> list[str]:
    if K == 1:
        return ["State 0"]
    if K == 2:
        return ["Disengaged", "Engaged"]
    if K == 3:
        return ["Engaged", "Biased L", "Biased R"]
    return [f"State {k}" for k in range(K)]


def plot_trans_mat(
    trans_mat: np.ndarray,
    state_labels: Sequence[str] | None = None,
    title: str = "Transition matrix",
    ax: plt.Axes | None = None,
    figsize: Tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of a transition matrix."""
    A = np.asarray(trans_mat, dtype=float)
    if A.ndim == 3:
        A = A.mean(axis=0)
    K = A.shape[0]
    labels = list(state_labels) if state_labels else _default_state_labels(K)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (3 + 0.4 * K, 3 + 0.4 * K))
    else:
        fig = ax.figure

    im = ax.imshow(A, vmin=0, vmax=1, cmap="bone", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(K):
        for j in range(K):
            ax.text(
                j,
                i,
                f"{A[i, j]:.2f}",
                ha="center",
                va="center",
                color="w" if A[i, j] < 0.5 else "k",
                fontsize=9,
            )
    ticks = range(K)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylabel("State $t$")
    ax.set_xlabel("State $t+1$")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    return fig


def plot_trans_mat_boxplots(
    all_trans_mats: np.ndarray,
    state_labels: Sequence[str] | None = None,
    title: str = "Diagonal transitions",
    figsize: Tuple[float, float] | None = None,
) -> plt.Figure:
    """Boxplot of diagonal transition probabilities across subjects."""
    A = np.asarray(all_trans_mats, dtype=float)
    K = A.shape[1]
    labels = list(state_labels) if state_labels else _default_state_labels(K)
    colors = get_state_palette(K)[:K]
    diag = np.stack([A[:, k, k] for k in range(K)], axis=1)

    fig, ax = plt.subplots(figsize=figsize or (2 + 0.8 * K, 3.5))
    for idx, color in enumerate(colors):
        custom_boxplot(
            ax,
            diag[:, idx],
            positions=[idx],
            widths=0.5,
            median_colors=color,
            box_edgecolor=color,
            whisker_color=color,
            showfliers=False,
            showcaps=False,
        )
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(stay)")
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_occupancy(
    smoothed_probs: np.ndarray,
    state_labels: Sequence[str] | None = None,
    trial_range: Tuple[int, int] | None = None,
    title: str = "Posterior state probabilities",
    figsize: Tuple[float, float] | None = None,
) -> plt.Figure:
    """Line plot of posterior state occupancy over trials."""
    P = np.asarray(smoothed_probs, dtype=float)
    if trial_range is not None:
        P = P[trial_range[0] : trial_range[1]]
    _, K = P.shape
    labels = list(state_labels) if state_labels else _default_state_labels(K)
    colors = get_state_palette(K)[:K]

    fig, ax = plt.subplots(figsize=figsize or (14, 2.5))
    for k in range(K):
        ax.plot(P[:, k], color=colors[k], label=labels[k], lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(state)")
    ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_occupancy_boxplot(
    all_smoothed_probs: Sequence[np.ndarray],
    state_labels: Sequence[str] | None = None,
    title: str = "State occupancy",
    figsize: Tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Boxplot of fractional state occupancy across subjects."""
    P_list = [np.asarray(p, dtype=float) for p in all_smoothed_probs]
    K = P_list[0].shape[1]
    labels = list(state_labels) if state_labels else _default_state_labels(K)
    colors = get_state_palette(K)[:K]
    occupancies = np.array([p.mean(axis=0) for p in P_list], dtype=float)

    fig, ax = plt.subplots(figsize=figsize or (2 + 0.8 * K, 3.5))
    for idx, color in enumerate(colors):
        custom_boxplot(
            ax,
            occupancies[:, idx],
            positions=[idx],
            widths=0.5,
            median_colors=color,
            box_edgecolor=color,
            whisker_color=color,
            showfliers=False,
            showcaps=False,
        )
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Occupancy")
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, occupancies


def norm_ll(
    lls: Sequence[float],
    n_trials: Sequence[int],
    ll_null: Sequence[float] | None = None,
    to_bits: bool = True,
) -> np.ndarray:
    """Normalise log-likelihoods to bits or nats per trial."""
    lls_arr = np.asarray(lls, dtype=float)
    n_arr = np.asarray(n_trials, dtype=float)
    if ll_null is not None:
        lls_arr = lls_arr - np.asarray(ll_null, dtype=float)
    else:
        lls_arr = lls_arr - np.log(0.5) * n_arr
    ll_norm = lls_arr / n_arr
    if to_bits:
        ll_norm = ll_norm / np.log(2)
    return ll_norm


def plot_ll(
    lls: Sequence[float],
    n_trials: Sequence[int],
    ll_null: Sequence[float] | None = None,
    to_bits: bool = True,
    ax: plt.Axes | None = None,
    color: str = "k",
    label: str | None = None,
    figsize: Tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Boxplot of normalised log-likelihoods."""
    ll_norm = norm_ll(lls, n_trials, ll_null, to_bits)
    ylabel = f"LL ({'bits' if to_bits else 'nats'}/Trial)"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (2, 3.5))
    else:
        fig = ax.figure

    custom_boxplot(
        ax,
        ll_norm,
        positions=[0],
        widths=0.5,
        median_colors=color,
        box_edgecolor=color,
        whisker_color=color,
        showfliers=False,
        showcaps=False,
    )
    jitter = np.linspace(-0.06, 0.06, len(ll_norm)) if len(ll_norm) > 1 else np.array([0.0])
    ax.scatter(jitter, ll_norm, color=color, alpha=0.45, s=20, zorder=3, linewidths=0)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels([label or "model"])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ll_norm


def plot_model_comparison(
    all_lls: dict,
    all_n_trials: Sequence[int],
    ll_null: Sequence[float] | None = None,
    to_bits: bool = True,
    title: str = "Model comparison",
    figsize: Tuple[float, float] | None = None,
) -> plt.Figure:
    """Line plot of mean +/- SEM normalised LL vs number of states."""
    ns_list = sorted(all_lls.keys())
    means = []
    sems = []
    for ns in ns_list:
        ll_n = norm_ll(all_lls[ns], all_n_trials, ll_null, to_bits)
        means.append(ll_n.mean())
        sems.append(sem(ll_n))

    fig, ax = plt.subplots(figsize=figsize or (4, 3))
    ax.errorbar(ns_list, means, yerr=sems, fmt="o-", color="k", capsize=4)
    ax.set_xlabel("N states")
    ax.set_ylabel(f"LL ({'bits' if to_bits else 'nats'}/Trial)")
    ax.set_title(title)
    ax.set_xticks(ns_list)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_model_comparison_diffs(
    all_lls: dict,
    all_n_trials: Sequence[int],
    ll_null: Sequence[float] | None = None,
    to_bits: bool = True,
    title: str = "ΔLL per state step",
    figsize: Tuple[float, float] | None = None,
) -> plt.Figure:
    """Strip plot of consecutive ΔLL values."""
    ns_list = sorted(all_lls.keys())
    norms = {ns: norm_ll(all_lls[ns], all_n_trials, ll_null, to_bits) for ns in ns_list}
    diffs = [norms[ns_list[i + 1]] - norms[ns_list[i]] for i in range(len(ns_list) - 1)]
    labels = [f"{ns_list[i + 1]}s-{ns_list[i]}s" for i in range(len(ns_list) - 1)]

    df = pd.DataFrame(
        {
            "ΔLL": np.concatenate(diffs),
            "comparison": np.repeat(labels, [len(diff) for diff in diffs]),
        }
    )

    fig, ax = plt.subplots(figsize=figsize or (2.5 * len(diffs), 3.5))
    sns.stripplot(x="comparison", y="ΔLL", data=df, color="k", alpha=0.5, jitter=True, ax=ax)
    ax.axhline(0, color="k", ls="--", lw=0.8)

    for i, diff in enumerate(diffs):
        _, pval = ttest_1samp(diff, 0)
        ymax = ax.get_ylim()[1]
        star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        ax.text(i, ymax * 0.95, star, ha="center", va="top", fontsize=11)

    ax.set_xlabel("N states comparison")
    ax.set_ylabel(f"ΔLL ({'bits' if to_bits else 'nats'}/Trial)")
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_transition_matrix_by_subject(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
):
    """Per-subject transition-matrix heatmaps."""
    _selected = [s for s in subjects if _resolve_transition_matrix_from_arrays(arrays_store, s) is not None]
    if not _selected:
        raise ValueError("No transition matrices found for selected subjects.")

    _n_cols = min(3, len(_selected))
    _n_rows = int(math.ceil(len(_selected) / _n_cols))
    fig, axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(4.2 * _n_cols, 3.4 * _n_rows),
        squeeze=False,
    )

    for idx, subj in enumerate(_selected):
        ax = axes[idx // _n_cols, idx % _n_cols]
        _A = _resolve_transition_matrix_from_arrays(arrays_store, subj)
        _slbl = state_labels.get(subj, {k: f"S{k}" for k in range(K)})
        _tick_labels = [_slbl.get(k, f"S{k}") for k in range(K)]
        sns.heatmap(
            _A,
            ax=ax,
            cmap="bone",
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=_tick_labels,
            yticklabels=_tick_labels,
            cbar=idx == 0,
            cbar_kws={"shrink": 0.8, "label": "probability"},
        )
        ax.set_title(f"Subject {subj}")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")

    for idx in range(len(_selected), _n_rows * _n_cols):
        axes[idx // _n_cols, idx % _n_cols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_transition_matrix(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
):
    """Mean transition-matrix heatmap across selected subjects."""
    _selected = [s for s in subjects if _resolve_transition_matrix_from_arrays(arrays_store, s) is not None]
    if not _selected:
        raise ValueError("No transition matrices found for selected subjects.")

    _A_mean = np.mean([_resolve_transition_matrix_from_arrays(arrays_store, s) for s in _selected], axis=0)
    _first_labels = state_labels.get(_selected[0], {k: f"S{k}" for k in range(K)})
    _tick_labels = [_first_labels.get(k, f"S{k}") for k in range(K)]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(
        _A_mean,
        ax=ax,
        cmap="bone",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=_tick_labels,
        yticklabels=_tick_labels,
        cbar_kws={"shrink": 0.8, "label": "probability"},
    )
    ax.set_title(f"Mean transition matrix  (n={len(_selected)} subjects)")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    fig.tight_layout()
    return fig


def plot_posterior_probs(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
    t0: int = 0,
    t1: int = 199,
):
    """
    Stacked-area posterior state probability plot with choice tick marks.

    Returns
    -------
    fig
    """
    _selected = [s for s in subjects if s in arrays_store]
    if not _selected:
        raise ValueError("No fitted arrays for selected subjects.")

    _colors        = get_state_palette(K)
    _choice_colors = {0: "royalblue", 1: "gold", 2: "tomato"}
    _choice_labels = {0: "L", 1: "C", 2: "R"}

    fig, axes = plt.subplots(len(_selected), 1,
                             figsize=(14, 3 * len(_selected)), squeeze=False)

    for _i, _subj in enumerate(_selected):
        _ax    = axes[_i, 0]
        _probs = arrays_store[_subj]["smoothed_probs"][t0: t1 + 1]
        _y     = arrays_store[_subj]["y"].astype(int)[t0: t1 + 1]
        _T_w   = _probs.shape[0]
        _x     = np.arange(t0, t0 + _T_w)

        _bottom = np.zeros(_T_w)
        _slbl   = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})
        for _k in range(K):
            _rank = get_state_rank(_slbl.get(_k, ""), _k)
            _ax.fill_between(
                _x, _bottom, _bottom + _probs[:, _k],
                alpha=0.7, color=_colors[_rank % len(_colors)],
                label=_slbl.get(_k, f"State {_k}"),
            )
            _bottom += _probs[:, _k]

        for _resp, _col in _choice_colors.items():
            _mask = _y == _resp
            _ax.scatter(
                _x[_mask], np.ones(_mask.sum()) * 1.03, c=_col, s=4, marker="|",
                label=_choice_labels[_resp],
                transform=_ax.get_xaxis_transform(), clip_on=False,
            )

        _ax.set_xlim(t0, t0 + _T_w - 1)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("State probability")
        _ax.set_title(f"Subject {_subj}")
        _ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                   fontsize=8, ncol=1, frameon=False)

    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    sns.despine(fig=fig)
    return fig


from matplotlib.collections import PathCollection
def strip_darken(ax, factor=0.7, lw=1.5):
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            fc = coll.get_facecolors()
            if len(fc) == 0:
                continue

            new_ec = fc.copy()
            new_ec[:, :3] *= factor

            coll.set_edgecolors(new_ec)
            coll.set_linewidth(lw)

def plot_state_accuracy(
    views: dict,
    trial_df,
    thresh: float = 0.5,
    performance_col: str = "correct_bool",
    chance_level: float | None = None,
    **kwargs,
) -> Tuple[plt.Figure, pd.DataFrame]:
    return _plot_state_accuracy_common(
        views,
        trial_df,
        thresh=thresh,
        performance_col=performance_col,
        chance_level=chance_level,
    )


def plot_session_trajectories(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    **kwargs,
):
    return _plot_session_trajectories_common(
        views,
        trial_df,
        session_col=session_col,
    )


def plot_state_posterior_count_kde(
    views: dict,
    thresh: float | None = None,
    bins: int = 40,
    **kwargs,
):
    return _plot_state_posterior_count_kde_common(
        views,
        thresh=thresh,
        bins=bins,
    )


def plot_change_triggered_posteriors_summary(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    switch_posterior_threshold: float | None = None,
    window: int = 15,
    **kwargs,
):
    return _plot_change_triggered_posteriors_summary_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
    )


def plot_change_triggered_posteriors_by_subject(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    switch_posterior_threshold: float | None = None,
    window: int = 15,
    **kwargs,
):
    return _plot_change_triggered_posteriors_by_subject_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
    )


def plot_state_occupancy(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    **kwargs,
):
    return _plot_state_occupancy_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        **kwargs,
    )


def plot_state_occupancy_overall_boxplot(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    **kwargs,
):
    return _plot_state_occupancy_overall_boxplot_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        **kwargs,
    )


def plot_state_dwell_times_by_subject(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
):
    return _plot_state_dwell_times_by_subject_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )


def plot_state_dwell_times_summary(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
):
    return _plot_state_dwell_times_summary_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )


def plot_state_dwell_times(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
):
    return _plot_state_dwell_times_common(
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
    sess: int,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    switch_posterior_threshold: float | None = None,
    performance_col: str = "correct_bool",
    response_col: str = "response",
    trace_x_cols: Tuple[str, ...] = ("A_R", "A_L", "A_C"),
    trace_u_cols: Tuple[str, ...] = ("A_plus", "A_minus"),
    **kwargs,
):
    return _plot_session_deepdive_common(
        views,
        trial_df,
        subj,
        sess,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        performance_col=performance_col,
        response_col=response_col,
        trace_x_cols=trace_x_cols,
        trace_u_cols=trace_u_cols,
        **kwargs,
    )


def plot_tau_sweep(sweep_path, subjects: list, K: int):
    """
    BIC vs τ and LL/trial vs τ plots from a tau-sweep parquet file.

    Parameters
    ----------
    sweep_path : path-like pointing to tau_sweep_summary.parquet
    subjects   : list of subject IDs to plot
    K          : number of states (used to filter the sweep dataframe)

    Returns
    -------
    fig, best_df (polars DataFrame: best τ per subject sorted by subject & K)
    """
    _df_sweep = pl.read_parquet(sweep_path)
    _subjects = [s for s in subjects
                 if s in _df_sweep["subject"].unique().to_list()]
    if not _subjects:
        raise ValueError("No sweep data found for the selected subjects.")

    fig, (_ax_bic, _ax_ll) = plt.subplots(1, 2, figsize=(12, 4))
    _palette = sns.color_palette("tab10", n_colors=len(_subjects))

    for _i, _subj in enumerate(_subjects):
        _d    = _df_sweep.filter(
            (pl.col("subject") == _subj) & (pl.col("K") == K)
        ).sort("tau")
        _tau  = _d["tau"].to_numpy()
        _bic  = _d["bic"].to_numpy()
        _ll   = _d["ll_per_trial"].to_numpy()
        _c    = _palette[_i]
        _ax_bic.plot(_tau, _bic, "-o", ms=3, color=_c, label=_subj)
        _ax_ll .plot(_tau, _ll,  "-o", ms=3, color=_c, label=_subj)
        _best  = int(np.argmin(_bic))
        _ax_bic.axvline(_tau[_best], color=_c, lw=0.8, linestyle="--", alpha=0.6)

    for _ax, _ylabel, _title in [
        (_ax_bic, "BIC",        "BIC vs τ  (lower is better)"),
        (_ax_ll,  "LL / trial", "Log-likelihood per trial vs τ"),
    ]:
        _ax.set_xlabel("τ (action-trace half-life)")
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        _ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=_ax)

    fig.tight_layout()

    best_df = (
        _df_sweep
        .filter(pl.col("subject").is_in(_subjects) & (pl.col("K") == K))
        .sort("bic")
        .group_by(["subject", "K"])
        .first()
        .select(["subject", "K", "tau", "bic", "ll_per_trial", "acc"])
        .sort(["subject", "K"])
    )
    return fig, best_df


def _sig_label(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def _transition_feature_names(feature_names, D: int) -> list[str]:
    cols = list(feature_names or [])
    if cols:
        return cols[:D]
    return [f"u{idx}" for idx in range(D)]


def _standardize_transition_weights(weights: np.ndarray) -> np.ndarray:
    weights_avg = np.asarray(weights, dtype=float).mean(axis=0)
    baseline = -np.mean(
        np.vstack([weights_avg, np.zeros((1, weights_avg.shape[1]), dtype=float)]),
        axis=0,
    )
    return weights_avg + baseline


def _resolve_transition_weight_inputs(
    arrays_store: dict | None,
    names: dict | None,
    K: int | None,
    subjects: list | None,
    state_labels: dict | None,
    views: dict | None,
):
    selected = list(subjects or [])

    if views is not None:
        if not selected:
            selected = list(views.keys())
        selected = [
            subj
            for subj in selected
            if subj in views and getattr(views[subj], "transition_weights", None) is not None
        ]
        if not selected:
            raise ValueError("No transition weights found for selected subjects.")
        if K is None:
            K = int(views[selected[0]].K)

        label_map = {subj: dict(views[subj].state_name_by_idx) for subj in selected}

        def get_tw(subj: str) -> np.ndarray:
            return np.asarray(views[subj].transition_weights, dtype=float)

        def get_ucols(subj: str, D: int) -> list[str]:
            return _transition_feature_names(getattr(views[subj], "U_cols", []), D)

    else:
        if arrays_store is None:
            raise ValueError("Provide either views or arrays_store.")
        if not selected:
            selected = list(arrays_store.keys())
        selected = [
            subj
            for subj in selected
            if subj in arrays_store and "transition_weights" in arrays_store[subj]
        ]
        if not selected:
            raise ValueError("No transition weights found for selected subjects.")
        if K is None:
            K = int(np.asarray(arrays_store[selected[0]]["transition_weights"]).shape[0])
        names = names or {}
        state_labels = state_labels or {}
        label_map = {
            subj: dict(state_labels.get(subj) or {k: f"State {k}" for k in range(K)})
            for subj in selected
        }

        def get_tw(subj: str) -> np.ndarray:
            return np.asarray(arrays_store[subj]["transition_weights"], dtype=float)

        def get_ucols(subj: str, D: int) -> list[str]:
            return _transition_feature_names(
                arrays_store[subj].get("U_cols") or names.get("U_cols"),
                D,
            )

    state_palette, states_order = build_state_palette(label_map, K=K)
    return selected, int(K), label_map, state_palette, states_order, get_tw, get_ucols


def _build_transition_weight_df(
    selected: list[str],
    K: int,
    label_map: dict,
    get_tw,
    get_ucols,
) -> tuple[pd.DataFrame, list[str]]:
    feature_order: list[str] | None = None
    records: list[dict[str, object]] = []

    for subj in selected:
        weights_std = _standardize_transition_weights(get_tw(subj))
        D = weights_std.shape[1]
        feature_names = get_ucols(subj, D)
        if feature_order is None:
            feature_order = list(feature_names)
        elif list(feature_names) != feature_order:
            raise ValueError("Transition feature columns differ across subjects.")

        for state_idx in range(min(K, weights_std.shape[0])):
            state_name = label_map[subj].get(state_idx, f"State {state_idx}")
            for feature_idx, feature_name in enumerate(feature_names):
                records.append(
                    {
                        "subject": subj,
                        "state": state_name,
                        "feature": feature_name,
                        "weight": float(weights_std[state_idx, feature_idx]),
                    }
                )

    feature_order = feature_order or []
    df_std = pd.DataFrame.from_records(records)
    if not feature_order:
        raise ValueError("No transition feature columns found.")
    df_std["feature"] = pd.Categorical(
        df_std["feature"],
        categories=feature_order,
        ordered=True,
    )
    return df_std, feature_order


def _paired_transition_weight_pvalues(
    df_std: pd.DataFrame,
    feature_order: Sequence[str],
    states_order: Sequence[str],
) -> dict[tuple[str, str, str], float]:
    pvalues: dict[tuple[str, str, str], float] = {}
    state_pairs = [
        (state_a, state_b)
        for idx, state_a in enumerate(states_order)
        for state_b in states_order[idx + 1 :]
    ]

    for feature_name in feature_order:
        feat_df = df_std[df_std["feature"] == feature_name]
        pivot = feat_df.pivot(index="subject", columns="state", values="weight").reindex(
            columns=list(states_order)
        )
        for state_a, state_b in state_pairs:
            paired = pivot[[state_a, state_b]].dropna()
            if len(paired) >= 2:
                pvalue = float(ttest_rel(paired[state_a], paired[state_b]).pvalue)
            else:
                pvalue = float("nan")
            pvalues[(feature_name, state_a, state_b)] = pvalue

    return pvalues


def _transition_weight_y_range(df_std: pd.DataFrame) -> float:
    y_range = float(df_std["weight"].max() - df_std["weight"].min())
    return y_range if np.isfinite(y_range) and y_range > 0 else 1.0


def _plot_transition_weight_lineplot(
    df_std: pd.DataFrame,
    selected: Sequence[str],
    K: int,
    feature_order: Sequence[str],
    states_order: Sequence[str],
    state_palette: dict[str, str],
    pvalues: dict[tuple[str, str, str], float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(4, len(feature_order) * 1.1), max(3, K * 1.0)))
    sns.lineplot(
        data=df_std,
        x="feature",
        y="weight",
        hue="state",
        ax=ax,
        markers=True,
        marker="o",
        markersize=9,
        markeredgewidth=0,
        alpha=0.85,
        errorbar="se",
        palette=state_palette,
        hue_order=states_order,
        sort=False,
    )

    for subj in selected:
        subj_df = df_std[df_std["subject"] == subj]
        for state_name in states_order:
            state_df = subj_df[subj_df["state"] == state_name].sort_values("feature")
            if state_df.empty:
                continue
            ax.plot(
                state_df["feature"].tolist(),
                state_df["weight"].to_numpy(),
                color=state_palette[state_name],
                alpha=0.25,
                linewidth=0.8,
            )

    y_range = _transition_weight_y_range(df_std)
    state_pairs = [
        (state_a, state_b)
        for idx, state_a in enumerate(states_order)
        for state_b in states_order[idx + 1 :]
    ]
    feature_xpos = {feature_name: idx for idx, feature_name in enumerate(feature_order)}
    for pair_idx, (state_a, state_b) in enumerate(state_pairs):
        for feature_name in feature_order:
            star = _sig_label(pvalues[(feature_name, state_a, state_b)])
            if star == "ns":
                continue
            feat_df = df_std[df_std["feature"] == feature_name]
            means = feat_df.groupby("state", observed=False)["weight"].mean()
            y_base = max(means.get(state_a, float("nan")), means.get(state_b, float("nan")))
            if not np.isfinite(y_base):
                continue
            ax.text(
                feature_xpos[feature_name],
                y_base + y_range * 0.06 * (pair_idx + 1),
                star,
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(feature_order)))
    ax.set_xticklabels(feature_order, rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Transition weight")
    ax.set_title(f"glmhmm-t K={K} — transition weights by state ({' / '.join(states_order)})")
    if ax.get_legend() is not None:
        ax.get_legend().set_title("")
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def _plot_transition_weight_boxplot(
    df_std: pd.DataFrame,
    K: int,
    feature_order: Sequence[str],
    states_order: Sequence[str],
    state_palette: dict[str, str],
    pvalues: dict[tuple[str, str, str], float],
) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=(max(5, len(feature_order) * 1.4), max(3, K * 1.0))
    )
    n_states = max(1, len(states_order))
    group_width = 0.8
    hue_width = group_width / n_states

    for state_idx, state_name in enumerate(states_order):
        color = state_palette[state_name]
        offset = (state_idx - (n_states - 1) / 2) * hue_width
        positions = [feat_idx + offset for feat_idx in range(len(feature_order))]
        values = [
            df_std[
                (df_std["feature"] == feature_name) & (df_std["state"] == state_name)
            ]["weight"].to_numpy(dtype=float)
            for feature_name in feature_order
        ]
        custom_boxplot(
            ax,
            values,
            positions=positions,
            widths=hue_width * 0.78,
            median_colors=color,
            box_edgecolor=color,
            whisker_color=color,
            box_facecolor="white",
            box_alpha=0.85,
            showfliers=False,
            showcaps=False,
        )
        for pos, vals in zip(positions, values, strict=False):
            if len(vals) == 0:
                continue
            jitter = (
                np.linspace(-hue_width * 0.12, hue_width * 0.12, len(vals))
                if len(vals) > 1
                else np.array([0.0])
            )
            ax.scatter(
                pos + jitter,
                vals,
                color=color,
                alpha=0.35,
                s=18,
                zorder=3,
                linewidths=0,
            )

    y_range = _transition_weight_y_range(df_std)
    state_pairs = [
        (state_a, state_b)
        for idx, state_a in enumerate(states_order)
        for state_b in states_order[idx + 1 :]
    ]
    for feat_idx, feature_name in enumerate(feature_order):
        feat_df = df_std[df_std["feature"] == feature_name]
        if feat_df.empty:
            continue
        current_y = float(feat_df["weight"].max()) + y_range * 0.05
        for state_a, state_b in state_pairs:
            star = _sig_label(pvalues[(feature_name, state_a, state_b)])
            if star == "ns":
                continue
            idx_a = states_order.index(state_a)
            idx_b = states_order.index(state_b)
            x1 = feat_idx + (idx_a - (n_states - 1) / 2) * hue_width
            x2 = feat_idx + (idx_b - (n_states - 1) / 2) * hue_width
            h = y_range * 0.02
            ax.plot(
                [x1, x1, x2, x2],
                [current_y, current_y + h, current_y + h, current_y],
                lw=1,
                c="k",
            )
            ax.text(
                (x1 + x2) / 2,
                current_y + h,
                star,
                ha="center",
                va="bottom",
                color="k",
            )
            current_y += y_range * 0.075

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(feature_order)))
    ax.set_xticklabels(feature_order, rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Transition weight")
    ax.set_title(f"glmhmm-t K={K} — transition weight boxplots ({' / '.join(states_order)})")
    ax.legend(
        [
            Line2D([0], [0], color=state_palette[state_name], lw=2, label=state_name)
            for state_name in states_order
        ],
        list(states_order),
        title="State",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_transition_weights(
    arrays_store: dict | None = None,
    names: dict | None = None,
    K: int | None = None,
    subjects: list | None = None,
    state_labels: dict | None = None,
    views: dict | None = None,
):
    """Plot standardized glmhmm-t transition weights as summary line and box plots."""
    selected, K, label_map, state_palette, states_order, get_tw, get_ucols = (
        _resolve_transition_weight_inputs(
            arrays_store=arrays_store,
            names=names,
            K=K,
            subjects=subjects,
            state_labels=state_labels,
            views=views,
        )
    )
    df_std, feature_order = _build_transition_weight_df(
        selected=selected,
        K=K,
        label_map=label_map,
        get_tw=get_tw,
        get_ucols=get_ucols,
    )
    pvalues = _paired_transition_weight_pvalues(
        df_std=df_std,
        feature_order=feature_order,
        states_order=states_order,
    )
    fig_line = _plot_transition_weight_lineplot(
        df_std=df_std,
        selected=selected,
        K=K,
        feature_order=feature_order,
        states_order=states_order,
        state_palette=state_palette,
        pvalues=pvalues,
    )
    fig_box = _plot_transition_weight_boxplot(
        df_std=df_std,
        K=K,
        feature_order=feature_order,
        states_order=states_order,
        state_palette=state_palette,
        pvalues=pvalues,
    )
    return fig_line, fig_box
