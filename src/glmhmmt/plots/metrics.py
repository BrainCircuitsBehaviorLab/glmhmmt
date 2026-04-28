from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glmhmmt.plots.utils import require_columns, resolve_state_order, state_palette, to_pandas_df
from glmhmmt.plots.common import resolve_single_axis, custom_boxplot


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


def ll_boxplot(
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

    fig, ax, created_fig = resolve_single_axis(ax=ax, figsize=figsize)

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
    if created_fig:
        fig.tight_layout()
    return fig, ll_norm
