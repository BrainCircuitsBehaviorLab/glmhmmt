from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def two_afc_delay_module(monkeypatch):
    workspace_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(workspace_root / "glmhmmt" / "src"))
    monkeypatch.syspath_prepend(str(workspace_root / "NMDAR_paper"))
    module = importlib.import_module("src.process.two_afc_delay")
    monkeypatch.setattr(module, "_max_subject_sessions", lambda: 3)
    return module


def test_two_afc_delay_build_feature_df_adds_session_bias_stim_hot_and_choice_lags(two_afc_delay_module):
    adapter = two_afc_delay_module.TwoAFCDelayAdapter()
    df_sub = pl.DataFrame(
        {
            "subject": ["mouse-1"] * 4,
            "drug": ["Rest"] * 4,
            "session": ["sess_a", "sess_a", "sess_b", "sess_b"],
            "trial": [1, 2, 1, 2],
            "stim": [2.0, -4.0, 0.0, 8.0],
            "choices": [-1.0, 1.0, 1.0, -1.0],
            "hit": [0.0, 1.0, 1.0, 0.0],
            "delays": [0.0, 1.0, 2.0, 3.0],
            "after_correct": [0.0, 1.0, 0.0, 1.0],
            "repeat": [0.0, 1.0, 1.0, 0.0],
            "repeat_choice_side": [0.0, 1.0, 1.0, 0.0],
            "WM": [0.0, 1.0, 0.0, 1.0],
            "RL": [1.0, 0.0, 1.0, 0.0],
        }
    )

    feature_df = adapter.build_feature_df(df_sub)

    np.testing.assert_array_equal(feature_df["bias_0"].to_numpy(), np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["bias_1"].to_numpy(), np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["bias_2"].to_numpy(), np.zeros(4, dtype=np.float32))

    np.testing.assert_array_equal(feature_df["stim_0"].to_numpy(), np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["stim_2"].to_numpy(), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["stim_4"].to_numpy(), np.asarray([0.0, -1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["stim_8"].to_numpy(), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    np.testing.assert_array_equal(feature_df["choice_lag_01"].to_numpy(), np.asarray([0.0, -1.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_02"].to_numpy(), np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(feature_df["choice_lag_15"].to_numpy(), np.zeros(4, dtype=np.float32))

    available_cols = adapter.available_emission_cols(feature_df)
    assert "bias_0" in available_cols
    assert "stim_8" in available_cols
    assert "choice_lag_15" in available_cols
