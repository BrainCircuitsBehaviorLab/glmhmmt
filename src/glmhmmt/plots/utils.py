from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import polars as pl

from glmhmmt.views import get_state_color


def to_pandas_df(df_like, *, name: str = "data") -> pd.DataFrame:
    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()
    if isinstance(df_like, pl.DataFrame):
        return df_like.to_pandas()
    raise TypeError(f"{name} must be a pandas or Polars DataFrame.")


def require_columns(df: pd.DataFrame, columns: Sequence[str], *, name: str = "data") -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")


def ordered_unique(values: Iterable) -> list:
    return list(pd.unique(pd.Series(list(values)).dropna()))


def resolve_state_order(df: pd.DataFrame) -> list[str]:
    require_columns(df, ["state_label"], name="state data")
    if "state_rank" in df.columns:
        state_meta = (
            df[["state_rank", "state_label"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["state_rank", "state_label"])
        )
        labels = state_meta["state_label"].astype(str).tolist()
    else:
        labels = ordered_unique(df["state_label"].astype(str))
    if not labels:
        raise ValueError("state data does not contain any state labels.")
    return labels


def state_palette(state_order: Sequence[str]) -> dict[str, str]:
    return {
        label: get_state_color(label, idx, K=len(state_order))
        for idx, label in enumerate(state_order)
    }


def posterior_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        col for col in df.columns
        if isinstance(col, str) and col.startswith("p_state_") and col.removeprefix("p_state_").isdigit()
    ]
    return sorted(cols, key=lambda col: int(col.rsplit("_", 1)[1]))


def finite_or_raise(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr
