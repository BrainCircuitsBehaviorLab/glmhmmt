from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def two_afc_module(monkeypatch):
    workspace_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(workspace_root / "glmhmmt" / "src"))
    monkeypatch.syspath_prepend(str(workspace_root / "NMDAR_paper"))
    module = importlib.import_module("src.process.two_afc")
    monkeypatch.setattr(module, "_max_subject_sessions", lambda: 3)
    monkeypatch.setattr(
        module,
        "_build_stim_param",
        lambda part, stim_abs_levels: np.zeros(len(part), dtype=np.float32),
    )
    return module


def test_two_afc_build_feature_df_adds_session_bias_and_choice_lags(two_afc_module):
    adapter = two_afc_module.TwoAFCAdapter()
    df_sub = pl.DataFrame(
        {
            "subject": ["mouse-1"] * 4,
            "Experiment": ["2AFC_2"] * 4,
            "Session": ["sess_a", "sess_a", "sess_b", "sess_b"],
            "Trial": [1, 2, 1, 2],
            "ILD": [2, -4, 0, 8],
            "Choice": [0, 1, 1, 0],
            "Hit": [0, 1, 1, 0],
            "Punish": [1, 0, 0, 1],
            "Side": [0, 1, 1, 0],
        }
    )

    feature_df = adapter.build_feature_df(df_sub)

    np.testing.assert_array_equal(feature_df["bias_0"].to_numpy(), np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["bias_1"].to_numpy(), np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
    assert "bias_2" not in feature_df.columns

    np.testing.assert_array_equal(feature_df["choice_lag_01"].to_numpy(), np.asarray([0.0, -1.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_02"].to_numpy(), np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_15"].to_numpy(), np.zeros(4, dtype=np.float32))

    np.testing.assert_array_equal(feature_df["stim_0"].to_numpy(), np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["stim_2"].to_numpy(), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["stim_4"].to_numpy(), np.asarray([0.0, -1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["stim_8"].to_numpy(), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    available_cols = adapter.available_emission_cols(feature_df)
    assert "bias_param" in available_cols
    assert "at_choice_param" in available_cols
    assert "bias_0" in available_cols
    assert "choice_lag_15" in available_cols


def test_two_afc_default_emission_cols_expand_bias_stim_and_choice_lag_families(two_afc_module):
    adapter = two_afc_module.TwoAFCAdapter()
    df_sub = pl.DataFrame(
        {
            "subject": ["mouse-1"] * 4,
            "Experiment": ["2AFC_2"] * 4,
            "Session": ["sess_a", "sess_a", "sess_b", "sess_b"],
            "Trial": [1, 2, 1, 2],
            "ILD": [2, -4, 0, 8],
            "Choice": [0, 1, 1, 0],
            "Hit": [0, 1, 1, 0],
            "Punish": [1, 0, 0, 1],
            "Side": [0, 1, 1, 0],
        }
    )

    feature_df = adapter.build_feature_df(df_sub)
    default_cols = adapter.default_emission_cols(feature_df)

    assert default_cols == [
        "bias_0",
        "bias_1",
        "stim_2",
        "stim_4",
        "stim_8",
        *[f"choice_lag_{lag_idx:02d}" for lag_idx in range(1, 16)],
    ]
