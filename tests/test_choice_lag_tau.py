from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE_ROOT / "src"))

from glmhmmt.choice_lag_tau import (
    export_choice_lag_tau_table,
    fit_choice_lag_exponential,
    _plot_display_weights,
)
from glmhmmt.runtime import configure_paths, get_runtime_paths


@pytest.fixture
def isolated_runtime(tmp_path):
    original = get_runtime_paths()
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()
    configure_paths(
        data_dir=data_dir,
        results_dir=results_dir,
        config_path=original.config_path,
    )
    try:
        yield results_dir
    finally:
        configure_paths(
            data_dir=original.data_dir,
            results_dir=original.results_dir,
            config_path=original.config_path,
        )


def _normalized_kernel(lags: np.ndarray, tau_decay: float) -> np.ndarray:
    kernel = np.exp(-lags / float(tau_decay))
    return kernel / kernel.sum()


def _write_subject_fit(
    fit_dir: Path,
    *,
    subject: str,
    x_cols: list[str],
    weights: np.ndarray,
) -> None:
    np.savez(
        fit_dir / f"{subject}_glm_arrays.npz",
        X_cols=np.asarray(x_cols, dtype=object),
        emission_weights=np.asarray([[[*weights]]], dtype=float),
    )


def test_fit_choice_lag_exponential_recovers_tau_and_half_life():
    lags = np.arange(1, 16, dtype=float)
    tau_decay = 2.75
    scale = -1.8
    weights = scale * _normalized_kernel(lags, tau_decay)

    fit = fit_choice_lag_exponential(lags, weights)

    assert fit.success is True
    assert fit.message == "ok"
    assert fit.n_lags == 15
    assert fit.tau_decay == pytest.approx(tau_decay, rel=1e-3)
    assert fit.tau_ewma_half_life == pytest.approx(tau_decay * np.log(2.0), rel=1e-3)
    assert fit.scale == pytest.approx(scale, rel=1e-3)
    np.testing.assert_allclose(fit.predicted_weights, weights, atol=1e-6)
    assert fit.rmse == pytest.approx(0.0, abs=1e-8)
    assert fit.r2 == pytest.approx(1.0, abs=1e-8)


def test_plot_display_weights_flips_only_two_afc():
    values = np.asarray([1.5, -0.25, 0.0], dtype=float)
    np.testing.assert_allclose(
        _plot_display_weights(fit_task="2AFC", values=values),
        np.asarray([-1.5, 0.25, -0.0], dtype=float),
    )
    np.testing.assert_allclose(
        _plot_display_weights(fit_task="2AFC_delay", values=values),
        values,
    )
    np.testing.assert_allclose(
        _plot_display_weights(fit_task="MCDR", values=values),
        values,
    )


def test_export_choice_lag_tau_table_writes_subject_rows(isolated_runtime, tmp_path):
    fit_dir = isolated_runtime / "fits" / "2AFC_delay" / "glm" / "one hot sessions delays lags"
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "config.json").write_text(
        json.dumps(
            {
                "task": "2AFC_delay",
                "model_id": "one hot sessions delays lags",
                "emission_cols": ["bias", "choice_lag_01", "choice_lag_02", "choice_lag_03"],
            }
        ),
        encoding="utf-8",
    )

    x_cols = ["bias", "choice_lag_01", "choice_lag_02", "choice_lag_03"]
    lags = np.arange(1, 4, dtype=float)

    mouse_a_weights = np.array([0.25, 0.0, 0.0, 0.0], dtype=float)
    mouse_a_weights[1:] = 1.2 * _normalized_kernel(lags, 1.5)
    _write_subject_fit(fit_dir, subject="mouse-a", x_cols=x_cols, weights=mouse_a_weights)

    mouse_b_weights = np.array([0.1, 0.0, 0.0, 0.0], dtype=float)
    mouse_b_weights[1:] = -0.7 * _normalized_kernel(lags, 4.0)
    _write_subject_fit(fit_dir, subject="mouse-b", x_cols=x_cols, weights=mouse_b_weights)

    out_path = tmp_path / "choice_lag_tau.parquet"
    table = export_choice_lag_tau_table(fit_dir=fit_dir, out_path=out_path)

    assert out_path.exists()
    assert isinstance(table, pl.DataFrame)
    assert table["subject"].to_list() == ["mouse-a", "mouse-b"]
    assert table["success"].to_list() == [True, True]
    assert table["n_choice_lags"].to_list() == [3, 3]
    assert table["tau_decay"].to_list() == pytest.approx([1.5, 4.0], rel=1e-3)
    assert table["tau_ewma_half_life"].to_list() == pytest.approx(
        [1.5 * np.log(2.0), 4.0 * np.log(2.0)],
        rel=1e-3,
    )

    written = pl.read_parquet(out_path).sort("subject")
    assert written["fit_model_id"].to_list() == ["one hot sessions delays lags", "one hot sessions delays lags"]
    assert written["choice_lag_cols"].to_list() == [
        "choice_lag_01,choice_lag_02,choice_lag_03",
        "choice_lag_01,choice_lag_02,choice_lag_03",
    ]
    plot_dir = isolated_runtime / "plots" / "2AFC_delay" / "glm" / "one hot sessions delays lags"
    assert (plot_dir / "choice_lag_tau_summary.png").exists()
    assert (plot_dir / "choice_lag_tau_summary.pdf").exists()


def test_export_choice_lag_tau_table_supports_grouped_mcdr_lags(isolated_runtime, tmp_path):
    fit_dir = isolated_runtime / "fits" / "MCDR" / "glm" / "one hot sessions lags"
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "config.json").write_text(
        json.dumps(
            {
                "task": "MCDR",
                "model_id": "one hot sessions lags",
                "emission_cols": [
                    "bias",
                    "choice_lag_01L",
                    "choice_lag_01C",
                    "choice_lag_01R",
                    "choice_lag_02L",
                    "choice_lag_02C",
                    "choice_lag_02R",
                    "choice_lag_03L",
                    "choice_lag_03C",
                    "choice_lag_03R",
                ],
            }
        ),
        encoding="utf-8",
    )

    x_cols = [
        "bias",
        "choice_lag_01L",
        "choice_lag_01C",
        "choice_lag_01R",
        "choice_lag_02L",
        "choice_lag_02C",
        "choice_lag_02R",
        "choice_lag_03L",
        "choice_lag_03C",
        "choice_lag_03R",
    ]
    lags = np.arange(1, 4, dtype=float)
    kernel = _normalized_kernel(lags, 2.5)
    grouped_weights = np.array(
        [
            1.2 * kernel[0],
            -0.5 * kernel[0],
            0.25 * kernel[0],
            1.2 * kernel[1],
            -0.5 * kernel[1],
            0.25 * kernel[1],
            1.2 * kernel[2],
            -0.5 * kernel[2],
            0.25 * kernel[2],
        ],
        dtype=float,
    )
    weights = np.concatenate(([0.1], grouped_weights))
    _write_subject_fit(fit_dir, subject="rat-a", x_cols=x_cols, weights=weights)

    out_path = tmp_path / "mcdr_choice_lag_tau.parquet"
    table = export_choice_lag_tau_table(fit_dir=fit_dir, out_path=out_path)

    assert out_path.exists()
    row = table.row(0, named=True)
    assert row["subject"] == "rat-a"
    assert row["success"] is True
    assert row["tau_decay"] == pytest.approx(2.5, rel=1e-3)
    assert row["tau_ewma_half_life"] == pytest.approx(2.5 * np.log(2.0), rel=1e-3)
    assert row["choice_lag_groups"] == "L,C,R,L,C,R,L,C,R"
    assert json.loads(row["scales_by_group"]) == pytest.approx({"C": -0.5, "L": 1.2, "R": 0.25}, rel=1e-3)
