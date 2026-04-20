from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def mcdr_module(monkeypatch):
    workspace_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(workspace_root / "glmhmmt" / "src"))
    monkeypatch.syspath_prepend(str(workspace_root / "NMDAR_paper"))
    module = importlib.import_module("src.process.MCDR")
    monkeypatch.setattr(module, "_max_subject_sessions", lambda: 3)
    return module


def test_mcdr_build_feature_df_adds_session_bias_choice_lags_and_params(mcdr_module):
    adapter = mcdr_module.MCDRAdapter()
    df_sub = pl.DataFrame(
        {
            "subject": ["A83"] * 4,
            "session": [101, 101, 102, 102],
            "trial": [1, 2, 1, 2],
            "trial_idx": [0, 1, 2, 3],
            "x_c": ["L", "C", "R", "L"],
            "r_c": ["L", "C", "R", "L"],
            "ttype_n": [0.0, 1.0, 0.0, 1.0],
            "stimd_n": [1.0, 2.0, 3.0, 4.0],
            "performance": [1, 0, 1, 0],
            "timepoint_1": [0.1, 0.2, 0.3, 0.4],
            "timepoint_2": [0.4, 0.6, 0.8, 1.0],
            "timepoint_3": [0.8, 1.1, 1.4, 1.8],
            "timepoint_4": [1.2, 1.6, 2.0, 2.5],
            "onset": [0.0, 0.2, 0.1, 0.3],
            "offset": [1.2, 1.6, 2.0, 2.5],
            "stim_d": [1.2, 1.4, 1.9, 2.2],
            "delay_d": [0.0, 0.5, 1.0, 1.5],
            "response": [0, 1, 2, 0],
            "stimulus": [0, 1, 2, 0],
        }
    )

    feature_df = adapter.build_feature_df(df_sub)

    np.testing.assert_array_equal(feature_df["bias_0"].to_numpy(), np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["bias_1"].to_numpy(), np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["bias_2"].to_numpy(), np.zeros(4, dtype=np.float32))

    np.testing.assert_array_equal(feature_df["choice_lag_01L"].to_numpy(), np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_01C"].to_numpy(), np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_01R"].to_numpy(), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_02L"].to_numpy(), np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_15R"].to_numpy(), np.zeros(4, dtype=np.float32))
    assert feature_df["stim_param"].dtype == pl.Float32
    np.testing.assert_array_equal(feature_df["bias_param"].to_numpy(), np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_param"].to_numpy(), np.zeros(4, dtype=np.float32))

    available_cols = adapter.available_emission_cols(feature_df)
    assert "bias_param" in available_cols
    assert "choice_lag_param" in available_cols
    assert "stim_param" in available_cols
    assert "bias_0" in available_cols
    assert "choice_lag_15L" in available_cols
    assert "stim4R" in available_cols


def test_mcdr_default_emission_cols_expand_stim_hot_and_choice_lag_families(mcdr_module):
    adapter = mcdr_module.MCDRAdapter()
    df_sub = pl.DataFrame(
        {
            "subject": ["A83"] * 4,
            "session": [101, 101, 102, 102],
            "trial": [1, 2, 1, 2],
            "trial_idx": [0, 1, 2, 3],
            "x_c": ["L", "C", "R", "L"],
            "r_c": ["L", "C", "R", "L"],
            "ttype_n": [0.0, 1.0, 0.0, 1.0],
            "stimd_n": [1.0, 2.0, 3.0, 4.0],
            "performance": [1, 0, 1, 0],
            "timepoint_1": [0.1, 0.2, 0.3, 0.4],
            "timepoint_2": [0.4, 0.6, 0.8, 1.0],
            "timepoint_3": [0.8, 1.1, 1.4, 1.8],
            "timepoint_4": [1.2, 1.6, 2.0, 2.5],
            "onset": [0.0, 0.2, 0.1, 0.3],
            "offset": [1.2, 1.6, 2.0, 2.5],
            "stim_d": [1.2, 1.4, 1.9, 2.2],
            "delay_d": [0.0, 0.5, 1.0, 1.5],
            "response": [0, 1, 2, 0],
            "stimulus": [0, 1, 2, 0],
        }
    )

    feature_df = adapter.build_feature_df(df_sub)
    default_cols = adapter.default_emission_cols(feature_df)
    expected_stim_hot = [
        f"stim{stim_idx}{side}"
        for stim_idx in range(1, 5)
        for side in ("L", "C", "R")
    ]
    expected_choice_lags = [
        f"choice_lag_{lag_idx:02d}{side}"
        for lag_idx in range(1, 16)
        for side in ("L", "C", "R")
    ]

    assert default_cols == ["bias", *expected_stim_hot, *expected_choice_lags]


def test_mcdr_build_emission_groups_groups_side_specific_stimulus_and_choice_lags(mcdr_module):
    groups = mcdr_module._build_emission_groups(
        [
            "bias",
            "biasL",
            "biasC",
            "biasR",
            "stim1L",
            "stim1C",
            "stim1R",
            "stim2L",
            "stim2C",
            "stim2R",
            "stim3L",
            "stim3C",
            "stim3R",
            "stim4L",
            "stim4C",
            "stim4R",
            "A_L",
            "A_C",
            "A_R",
            "choice_lag_01L",
            "choice_lag_01C",
            "choice_lag_01R",
            "choice_lag_02L",
            "choice_lag_02C",
            "choice_lag_02R",
        ]
    )
    by_key = {group["key"]: group for group in groups}

    assert by_key["bias_side"]["members"] == {"L": "biasL", "C": "biasC", "R": "biasR"}
    assert by_key["stim1"]["members"] == {"L": "stim1L", "C": "stim1C", "R": "stim1R"}
    assert by_key["stim2"]["members"] == {"L": "stim2L", "C": "stim2C", "R": "stim2R"}
    assert by_key["stim3"]["members"] == {"L": "stim3L", "C": "stim3C", "R": "stim3R"}
    assert by_key["stim4"]["members"] == {"L": "stim4L", "C": "stim4C", "R": "stim4R"}
    assert by_key["stim_hot"]["toggle_members"] == [
        "stim1L",
        "stim1C",
        "stim1R",
        "stim2L",
        "stim2C",
        "stim2R",
        "stim3L",
        "stim3C",
        "stim3R",
        "stim4L",
        "stim4C",
        "stim4R",
    ]
    assert by_key["choice_lag"]["toggle_members"] == [
        "choice_lag_01L",
        "choice_lag_01C",
        "choice_lag_01R",
        "choice_lag_02L",
        "choice_lag_02C",
        "choice_lag_02R",
    ]
