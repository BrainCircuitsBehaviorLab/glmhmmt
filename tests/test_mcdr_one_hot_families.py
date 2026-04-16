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


def test_mcdr_build_feature_df_adds_session_bias_and_choice_lags(mcdr_module):
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

    available_cols = adapter.available_emission_cols(feature_df)
    assert "bias_0" in available_cols
    assert "choice_lag_15L" in available_cols
