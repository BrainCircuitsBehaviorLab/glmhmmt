from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.optimize import minimize_scalar

from glmhmmt.runtime import get_results_dir


_CHOICE_LAG_PREFIX = "choice_lag_"
_CHOICE_LAG_GROUP_ORDER = {"all": -1, "L": 0, "C": 1, "R": 2}


@dataclass(frozen=True)
class ChoiceLagTauFit:
    tau_decay: float
    tau_ewma_half_life: float
    scale: float
    scales_by_group: dict[str, float]
    rss: float
    rmse: float
    r2: float
    success: bool
    message: str
    n_lags: int
    predicted_weights: np.ndarray

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["predicted_weights"] = self.predicted_weights.tolist()
        return row


@dataclass(frozen=True)
class ChoiceLagPlotRecord:
    subject: str
    group: str
    lags: np.ndarray
    weights: np.ndarray
    predicted_weights: np.ndarray


def _fit_kernel(lags: np.ndarray, tau_decay: float) -> np.ndarray:
    kernel = np.exp(-lags / float(tau_decay))
    kernel_sum = float(kernel.sum())
    if not np.isfinite(kernel_sum) or kernel_sum <= 0.0:
        raise ValueError("Exponential kernel could not be normalized.")
    return kernel / kernel_sum


def _fit_scale_for_tau(lags: np.ndarray, weights: np.ndarray, tau_decay: float) -> tuple[float, np.ndarray]:
    kernel = _fit_kernel(lags, tau_decay)
    denom = float(np.dot(kernel, kernel))
    if denom <= 0.0:
        raise ValueError("Degenerate exponential kernel during scale fit.")
    scale = float(np.dot(weights, kernel) / denom)
    return scale, scale * kernel


def _fit_scales_for_tau(
    lags: np.ndarray,
    weights: np.ndarray,
    group_labels: np.ndarray,
    tau_decay: float,
) -> tuple[float, dict[str, float], np.ndarray]:
    predicted = np.empty_like(weights, dtype=float)
    group_scales: dict[str, float] = {}
    unique_groups = list(dict.fromkeys(group_labels.tolist()))
    for group in unique_groups:
        mask = group_labels == group
        group_lags = lags[mask]
        group_weights = weights[mask]
        scale, group_predicted = _fit_scale_for_tau(group_lags, group_weights, tau_decay)
        predicted[mask] = group_predicted
        group_scales[str(group)] = scale
    overall_scale = next(iter(group_scales.values())) if len(group_scales) == 1 else np.nan
    return overall_scale, group_scales, predicted


def fit_choice_lag_exponential(
    lags: Iterable[float],
    weights: Iterable[float],
    *,
    group_labels: Iterable[object] | None = None,
    min_tau_decay: float = 1e-3,
    max_tau_decay: float = 1e3,
    flat_weight_tol: float = 1e-12,
) -> ChoiceLagTauFit:
    lags_np = np.asarray(list(lags), dtype=float).reshape(-1)
    weights_np = np.asarray(list(weights), dtype=float).reshape(-1)
    groups_np = (
        np.asarray(["all"] * lags_np.size, dtype=object)
        if group_labels is None
        else np.asarray(list(group_labels), dtype=object).reshape(-1)
    )

    if lags_np.size != weights_np.size:
        raise ValueError(
            f"lags and weights must have the same length; got {lags_np.size} and {weights_np.size}."
        )
    if groups_np.size != weights_np.size:
        raise ValueError(
            f"group_labels and weights must have the same length; got {groups_np.size} and {weights_np.size}."
        )
    if lags_np.size < 2:
        return ChoiceLagTauFit(
            tau_decay=np.nan,
            tau_ewma_half_life=np.nan,
            scale=np.nan,
            scales_by_group={},
            rss=np.nan,
            rmse=np.nan,
            r2=np.nan,
            success=False,
            message="need at least two choice-lag weights",
            n_lags=int(lags_np.size),
            predicted_weights=np.full(lags_np.shape, np.nan, dtype=float),
        )
    if not np.all(np.isfinite(lags_np)) or not np.all(np.isfinite(weights_np)):
        return ChoiceLagTauFit(
            tau_decay=np.nan,
            tau_ewma_half_life=np.nan,
            scale=np.nan,
            scales_by_group={},
            rss=np.nan,
            rmse=np.nan,
            r2=np.nan,
            success=False,
            message="non-finite lags or weights",
            n_lags=int(lags_np.size),
            predicted_weights=np.full(lags_np.shape, np.nan, dtype=float),
        )
    if float(np.linalg.norm(weights_np)) <= flat_weight_tol:
        return ChoiceLagTauFit(
            tau_decay=np.nan,
            tau_ewma_half_life=np.nan,
            scale=0.0,
            scales_by_group={group: 0.0 for group in dict.fromkeys(groups_np.tolist())},
            rss=0.0,
            rmse=0.0,
            r2=np.nan,
            success=False,
            message="choice-lag weights are flat",
            n_lags=int(lags_np.size),
            predicted_weights=np.zeros_like(weights_np),
        )

    def objective(tau_decay: float) -> float:
        _, _, predicted = _fit_scales_for_tau(lags_np, weights_np, groups_np, tau_decay)
        residual = weights_np - predicted
        return float(np.dot(residual, residual))

    result = minimize_scalar(
        objective,
        bounds=(float(min_tau_decay), float(max_tau_decay)),
        method="bounded",
    )
    tau_decay = float(result.x)
    scale, scales_by_group, predicted = _fit_scales_for_tau(lags_np, weights_np, groups_np, tau_decay)
    residual = weights_np - predicted
    rss = float(np.dot(residual, residual))
    rmse = float(np.sqrt(rss / lags_np.size))
    weight_mean = float(weights_np.mean())
    centered = weights_np - weight_mean
    tss = float(np.dot(centered, centered))
    r2 = float(1.0 - (rss / tss)) if tss > 0.0 else np.nan

    return ChoiceLagTauFit(
        tau_decay=tau_decay,
        tau_ewma_half_life=float(tau_decay * np.log(2.0)),
        scale=scale,
        scales_by_group=scales_by_group,
        rss=rss,
        rmse=rmse,
        r2=r2,
        success=bool(result.success),
        message="ok" if result.success else str(result.message),
        n_lags=int(lags_np.size),
        predicted_weights=predicted,
    )


def _choice_lag_items(x_cols: Iterable[object]) -> list[tuple[int, str, str]]:
    items: list[tuple[int, str, str]] = []
    for raw_col in x_cols:
        col = str(raw_col)
        if not col.startswith(_CHOICE_LAG_PREFIX):
            continue
        suffix = col.removeprefix(_CHOICE_LAG_PREFIX)
        if suffix.isdigit():
            items.append((int(suffix), "all", col))
            continue
        if len(suffix) >= 2 and suffix[:-1].isdigit() and suffix[-1].isalpha():
            items.append((int(suffix[:-1]), suffix[-1], col))
            continue
    return sorted(
        items,
        key=lambda item: (item[0], _CHOICE_LAG_GROUP_ORDER.get(item[1], 10**9), item[1], item[2]),
    )


def _fit_metadata(fit_dir: Path) -> dict[str, object]:
    config_path = fit_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_choice_lag_weights(
    arrays_path: Path,
    *,
    state_idx: int = 0,
    class_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(arrays_path, allow_pickle=True) as arrays:
        x_cols = [str(col) for col in arrays["X_cols"].tolist()]
        emission_weights = np.asarray(arrays["emission_weights"], dtype=float)

    if emission_weights.ndim != 3:
        raise ValueError(f"Unexpected emission weight shape in {arrays_path}.")
    if state_idx >= emission_weights.shape[0] or class_idx >= emission_weights.shape[1]:
        raise ValueError(
            f"Requested state/class slice ({state_idx}, {class_idx}) is unavailable in {arrays_path}."
        )

    weight_row = emission_weights[state_idx, class_idx, :]
    lag_items = _choice_lag_items(x_cols)
    if not lag_items:
        raise ValueError(f"No choice-lag columns found in {arrays_path.name}.")

    col_to_idx = {col: idx for idx, col in enumerate(x_cols)}
    lags = np.asarray([lag for lag, _, _ in lag_items], dtype=float)
    group_labels = np.asarray([group for _, group, _ in lag_items], dtype=object)
    cols = [col for _, _, col in lag_items]
    weights = np.asarray([weight_row[col_to_idx[col]] for col in cols], dtype=float)
    return lags, weights, group_labels, cols


def resolve_fit_dir(
    *,
    fit_dir: str | Path | None = None,
    task: str | None = None,
    model_kind: str = "glm",
    model_id: str | None = None,
) -> Path:
    if fit_dir is not None:
        return Path(fit_dir).expanduser().resolve()
    if task is None or model_id is None:
        raise ValueError("Provide either fit_dir or both task and model_id.")
    return (get_results_dir() / "fits" / task / model_kind / model_id).resolve()


def _plot_dir(*, fit_task: str, fit_model_kind: str, fit_model_id: str) -> Path:
    return (get_results_dir() / "plots" / fit_task / fit_model_kind / fit_model_id).resolve()


def _plot_display_weights(*, fit_task: str, values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=float)
    if str(fit_task) == "2AFC":
        return -values_np
    return values_np


def _save_choice_lag_summary_plot(
    *,
    records: list[ChoiceLagPlotRecord],
    fit_task: str,
    fit_model_kind: str,
    fit_model_id: str,
) -> list[Path]:
    if not records:
        return []

    group_order = {"all": -1, "L": 0, "C": 1, "R": 2}
    groups = sorted(
        {record.group for record in records},
        key=lambda group: (group_order.get(group, 10**9), str(group)),
    )
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(max(6.0, 5.0 * len(groups)), 4.5),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()
    empirical_color = "#374151"
    fitted_color = "#2563eb"
    mean_empirical_color = "#111827"
    mean_fitted_color = "#dc2626"

    for ax, group in zip(axes_flat, groups):
        group_records = [record for record in records if record.group == group]
        group_records = sorted(group_records, key=lambda record: record.subject)
        lag_grid = sorted({int(lag) for record in group_records for lag in record.lags.tolist()})
        lag_to_idx = {lag: idx for idx, lag in enumerate(lag_grid)}
        mean_weights = np.full(len(lag_grid), np.nan, dtype=float)
        mean_predicted = np.full(len(lag_grid), np.nan, dtype=float)

        empirical_stack = np.full((len(group_records), len(lag_grid)), np.nan, dtype=float)
        predicted_stack = np.full((len(group_records), len(lag_grid)), np.nan, dtype=float)
        for row_idx, record in enumerate(group_records):
            x = record.lags.astype(float)
            display_weights = _plot_display_weights(fit_task=fit_task, values=record.weights)
            display_predicted = _plot_display_weights(fit_task=fit_task, values=record.predicted_weights)
            # ax.plot(x, display_weights, color=empirical_color, alpha=0.15, linewidth=1.0)
            # ax.scatter(x, display_weights, color=empirical_color, alpha=0.15, s=12)
            ax.plot(x, display_predicted, color=fitted_color, alpha=0.15, linewidth=1.0)
            for lag, weight, predicted in zip(record.lags.tolist(), record.weights.tolist(), record.predicted_weights.tolist()):
                grid_idx = lag_to_idx[int(lag)]
                empirical_stack[row_idx, grid_idx] = float(_plot_display_weights(fit_task=fit_task, values=np.asarray([weight]))[0])
                predicted_stack[row_idx, grid_idx] = float(_plot_display_weights(fit_task=fit_task, values=np.asarray([predicted]))[0])

        if empirical_stack.size:
            with np.errstate(invalid="ignore"):
                mean_weights = np.nanmean(empirical_stack, axis=0)
                mean_predicted = np.nanmean(predicted_stack, axis=0)

        lag_grid_np = np.asarray(lag_grid, dtype=float)
        valid_mean_weights = np.isfinite(mean_weights)
        valid_mean_predicted = np.isfinite(mean_predicted)
        if np.any(valid_mean_weights):
            ax.plot(
                lag_grid_np[valid_mean_weights],
                mean_weights[valid_mean_weights],
                color=mean_empirical_color,
                linewidth=2.2,
                marker="o",
                markersize=4,
                label="Mean coefficients",
            )
        if np.any(valid_mean_predicted):
            ax.plot(
                lag_grid_np[valid_mean_predicted],
                mean_predicted[valid_mean_predicted],
                color=mean_fitted_color,
                linewidth=2.4,
                label="Mean fitted",
            )

        ax.set_xlabel("Lag")
        if group == groups[0]:
            ax.set_ylabel("Weight")
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    plot_dir = _plot_dir(
        fit_task=fit_task,
        fit_model_kind=fit_model_kind,
        fit_model_id=fit_model_id,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    sns.despine(fig=fig)
    for suffix in (".png", ".pdf"):
        out_path = plot_dir / f"choice_lag_tau_summary{suffix}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        outputs.append(out_path)
    plt.close(fig)
    return outputs


def export_choice_lag_tau_table(
    *,
    fit_dir: str | Path,
    out_path: str | Path | None = None,
    arrays_suffix: str = "glm_arrays.npz",
    state_idx: int = 0,
    class_idx: int = 0,
) -> pl.DataFrame:
    fit_dir_path = Path(fit_dir).expanduser().resolve()
    if not fit_dir_path.exists():
        raise FileNotFoundError(f"Missing fit directory: {fit_dir_path}")

    fit_meta = _fit_metadata(fit_dir_path)
    fit_model_id = str(fit_meta.get("model_id", fit_dir_path.name))
    fit_task = str(fit_meta.get("task", fit_dir_path.parent.parent.name if len(fit_dir_path.parents) >= 2 else ""))
    fit_model_kind = str(fit_dir_path.parent.name)

    rows: list[dict[str, object]] = []
    plot_records: list[ChoiceLagPlotRecord] = []
    for arrays_path in sorted(fit_dir_path.glob(f"*_{arrays_suffix}")):
        subject = arrays_path.name.removesuffix(f"_{arrays_suffix}").split("_", 1)[0]
        try:
            lags, weights, group_labels, lag_cols = _extract_choice_lag_weights(
                arrays_path,
                state_idx=state_idx,
                class_idx=class_idx,
            )
            fit = fit_choice_lag_exponential(lags, weights, group_labels=group_labels)
        except Exception as exc:
            fit = ChoiceLagTauFit(
                tau_decay=np.nan,
                tau_ewma_half_life=np.nan,
                scale=np.nan,
                scales_by_group={},
                rss=np.nan,
                rmse=np.nan,
                r2=np.nan,
                success=False,
                message=str(exc),
                n_lags=0,
                predicted_weights=np.asarray([], dtype=float),
            )
            lags = np.asarray([], dtype=float)
            weights = np.asarray([], dtype=float)
            group_labels = np.asarray([], dtype=object)
            lag_cols = []
        else:
            for group in dict.fromkeys(group_labels.tolist()):
                group_mask = group_labels == group
                plot_records.append(
                    ChoiceLagPlotRecord(
                        subject=subject,
                        group=str(group),
                        lags=lags[group_mask].astype(float),
                        weights=weights[group_mask].astype(float),
                        predicted_weights=fit.predicted_weights[group_mask].astype(float),
                    )
                )

        rows.append(
            {
                "subject": subject,
                "fit_task": fit_task,
                "fit_model_kind": fit_model_kind,
                "fit_model_id": fit_model_id,
                "fit_dir": str(fit_dir_path),
                "arrays_path": str(arrays_path),
                "choice_lag_cols": ",".join(lag_cols),
                "choice_lag_groups": ",".join(str(group) for group in group_labels.tolist()),
                "choice_lags": ",".join(str(int(lag)) for lag in lags.tolist()),
                "choice_lag_weights": ",".join(f"{weight:.12g}" for weight in weights.tolist()),
                "n_choice_lags": int(lags.size),
                "tau_decay": fit.tau_decay,
                "tau_ewma_half_life": fit.tau_ewma_half_life,
                "scale": fit.scale,
                "scales_by_group": json.dumps(fit.scales_by_group, sort_keys=True),
                "rss": fit.rss,
                "rmse": fit.rmse,
                "r2": fit.r2,
                "success": fit.success,
                "message": fit.message,
            }
        )

    table = pl.DataFrame(rows).sort("subject") if rows else pl.DataFrame(
        {
            "subject": pl.Series([], dtype=pl.Utf8),
            "fit_task": pl.Series([], dtype=pl.Utf8),
            "fit_model_kind": pl.Series([], dtype=pl.Utf8),
            "fit_model_id": pl.Series([], dtype=pl.Utf8),
            "fit_dir": pl.Series([], dtype=pl.Utf8),
            "arrays_path": pl.Series([], dtype=pl.Utf8),
            "choice_lag_cols": pl.Series([], dtype=pl.Utf8),
            "choice_lag_groups": pl.Series([], dtype=pl.Utf8),
            "choice_lags": pl.Series([], dtype=pl.Utf8),
            "choice_lag_weights": pl.Series([], dtype=pl.Utf8),
            "n_choice_lags": pl.Series([], dtype=pl.Int64),
            "tau_decay": pl.Series([], dtype=pl.Float64),
            "tau_ewma_half_life": pl.Series([], dtype=pl.Float64),
            "scale": pl.Series([], dtype=pl.Float64),
            "scales_by_group": pl.Series([], dtype=pl.Utf8),
            "rss": pl.Series([], dtype=pl.Float64),
            "rmse": pl.Series([], dtype=pl.Float64),
            "r2": pl.Series([], dtype=pl.Float64),
            "success": pl.Series([], dtype=pl.Boolean),
            "message": pl.Series([], dtype=pl.Utf8),
        }
    )

    target = (
        Path(out_path).expanduser().resolve()
        if out_path is not None
        else (fit_dir_path / "choice_lag_tau.parquet").resolve()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".csv":
        table.write_csv(target)
    else:
        table.write_parquet(target)
    _save_choice_lag_summary_plot(
        records=plot_records,
        fit_task=fit_task,
        fit_model_kind=fit_model_kind,
        fit_model_id=fit_model_id,
    )
    return table
