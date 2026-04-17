from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest


_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WORKSPACE_ROOT / "glmhmmt" / "src"))
sys.path.insert(0, str(_WORKSPACE_ROOT / "NMDAR_paper"))

from glmhmmt.cli.alexis_functions import get_action_trace
from glmhmmt.runtime import configure_paths, get_runtime_paths


def _expected_ewma(values: np.ndarray, *, half_life: float) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_np.size == 0:
        return values_np.copy()
    alpha = np.float32(1.0 - np.exp(np.log(0.5) / float(half_life)))
    decay = np.float32(1.0 - alpha)
    trace = np.empty_like(values_np, dtype=np.float32)
    trace[0] = values_np[0]
    for trial_idx in range(1, len(values_np)):
        trace[trial_idx] = (decay * trace[trial_idx - 1]) + (alpha * values_np[trial_idx])
    return trace


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


def test_two_afc_uses_subject_specific_choice_tau(isolated_runtime, monkeypatch):
    choice_tau = importlib.import_module("src.process._choice_tau")
    two_afc = importlib.import_module("src.process.two_afc")
    monkeypatch.setattr(two_afc, "_max_subject_sessions", lambda: 1)
    monkeypatch.setattr(
        two_afc,
        "_build_stim_param",
        lambda part, stim_abs_levels: np.zeros(len(part), dtype=np.float32),
    )
    choice_tau.clear_choice_tau_cache()

    tau_path = isolated_runtime / "fits" / "2AFC" / "glm" / "one hot sessions lags" / "choice_lag_tau.parquet"
    tau_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject": ["mouse-fast", "mouse-slow"],
            "tau_decay": [1.5, 4.0],
            "tau_ewma_half_life": [1.5 * np.log(2.0), 4.0 * np.log(2.0)],
        }
    ).write_parquet(tau_path)

    adapter = two_afc.TwoAFCAdapter()
    base_df = {
        "subject": ["mouse-fast"] * 5,
        "Experiment": ["2AFC_2"] * 5,
        "Session": ["sess-1"] * 5,
        "Trial": [1, 2, 3, 4, 5],
        "ILD": [2, -2, 2, -2, 2],
        "Choice": [0, 1, 1, 0, 1],
        "Hit": [1, 0, 1, 1, 0],
        "Punish": [0, 1, 0, 0, 1],
        "Side": [0, 1, 1, 0, 1],
    }
    fast_df = pl.DataFrame(base_df)
    slow_df = pl.DataFrame({**base_df, "subject": ["mouse-slow"] * 5})
    missing_df = pl.DataFrame({**base_df, "subject": ["mouse-missing"] * 5})

    fast_features = adapter.build_feature_df(fast_df)
    slow_features = adapter.build_feature_df(slow_df)
    missing_features = adapter.build_feature_df(missing_df)

    signed_choice = (2.0 * np.asarray(base_df["Choice"], dtype=np.float32)) - 1.0
    previous_choice = np.concatenate(([0.0], signed_choice[:-1])).astype(np.float32)
    expected_fast = _expected_ewma(previous_choice, half_life=1.5 * np.log(2.0))
    expected_slow = _expected_ewma(previous_choice, half_life=4.0 * np.log(2.0))

    np.testing.assert_allclose(fast_features["at_choice"].to_numpy(), expected_fast, atol=1e-6)
    np.testing.assert_allclose(slow_features["at_choice"].to_numpy(), expected_slow, atol=1e-6)
    assert not np.allclose(
        fast_features["at_choice"].to_numpy(),
        slow_features["at_choice"].to_numpy(),
    )

    default_at_choice, _, _, _ = get_action_trace(pd.DataFrame(missing_df.to_dict(as_series=False)))
    np.testing.assert_allclose(
        missing_features["at_choice"].to_numpy(),
        np.asarray(default_at_choice, dtype=np.float32),
        atol=1e-6,
    )


def test_two_afc_delay_uses_subject_specific_choice_tau(isolated_runtime, monkeypatch):
    choice_tau = importlib.import_module("src.process._choice_tau")
    two_afc_delay = importlib.import_module("src.process.two_afc_delay")
    monkeypatch.setattr(two_afc_delay, "_max_subject_sessions", lambda: 1)
    monkeypatch.setattr(two_afc_delay, "_all_delay_levels", lambda: (0.0, 1.0))
    choice_tau.clear_choice_tau_cache()

    tau_path = isolated_runtime / "fits" / "2AFC_delay" / "glm" / "one hot sessions delays lags" / "choice_lag_tau.parquet"
    tau_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject": ["mouse-fast", "mouse-slow"],
            "tau_decay": [1.25, 3.5],
            "tau_ewma_half_life": [1.25 * np.log(2.0), 3.5 * np.log(2.0)],
        }
    ).write_parquet(tau_path)

    adapter = two_afc_delay.TwoAFCDelayAdapter()
    base_df = {
        "subject": ["mouse-fast"] * 5,
        "drug": ["Rest"] * 5,
        "session": ["sess-1"] * 5,
        "trial": [1, 2, 3, 4, 5],
        "stim": [2.0, -2.0, 2.0, -2.0, 2.0],
        "choices": [-1.0, 1.0, 1.0, -1.0, 1.0],
        "hit": [1.0, 0.0, 1.0, 1.0, 0.0],
        "delays": [0.0, 1.0, 0.0, 1.0, 0.0],
        "after_correct": [0.0, 1.0, 1.0, 0.0, 1.0],
        "repeat": [0.0, 1.0, 1.0, 0.0, 1.0],
        "repeat_choice_side": [0.0, 1.0, 1.0, 0.0, 1.0],
        "WM": [0.0, 1.0, 0.0, 1.0, 0.0],
        "RL": [1.0, 0.0, 1.0, 0.0, 1.0],
    }
    fast_df = pl.DataFrame(base_df)
    slow_df = pl.DataFrame({**base_df, "subject": ["mouse-slow"] * 5})
    missing_df = pl.DataFrame({**base_df, "subject": ["mouse-missing"] * 5})

    fast_features = adapter.build_feature_df(fast_df)
    slow_features = adapter.build_feature_df(slow_df)
    missing_features = adapter.build_feature_df(missing_df)

    signed_choice = np.asarray(base_df["choices"], dtype=np.float32)
    previous_choice = np.concatenate(([0.0], signed_choice[:-1])).astype(np.float32)
    expected_fast = _expected_ewma(previous_choice, half_life=1.25 * np.log(2.0))
    expected_slow = _expected_ewma(previous_choice, half_life=3.5 * np.log(2.0))

    np.testing.assert_allclose(fast_features["at_choice"].to_numpy(), expected_fast, atol=1e-6)
    np.testing.assert_allclose(slow_features["at_choice"].to_numpy(), expected_slow, atol=1e-6)
    assert not np.allclose(
        fast_features["at_choice"].to_numpy(),
        slow_features["at_choice"].to_numpy(),
    )

    trace_input = pl.DataFrame(missing_df).to_pandas()[["choices", "hit"]].rename(columns={"choices": "Choice", "hit": "Hit"})
    trace_input["Choice"] = ((trace_input["Choice"] > 0).astype(np.int32)).to_numpy()
    trace_input["Punish"] = (1.0 - trace_input["Hit"]).to_numpy()
    default_at_choice, _, _, _ = get_action_trace(trace_input)
    np.testing.assert_allclose(
        missing_features["at_choice"].to_numpy(),
        np.asarray(default_at_choice, dtype=np.float32),
        atol=1e-6,
    )


def test_mcdr_uses_subject_specific_choice_tau(isolated_runtime):
    choice_tau = importlib.import_module("src.process._choice_tau")
    mcdr = importlib.import_module("src.process.mcdr")
    choice_tau.clear_choice_tau_cache()

    tau_path = isolated_runtime / "fits" / "MCDR" / "glm" / "one hot sessions lags" / "choice_lag_tau.parquet"
    tau_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject": ["rat-fast", "rat-slow"],
            "tau_decay": [1.5, 4.0],
            "tau_ewma_half_life": [1.5 * np.log(2.0), 4.0 * np.log(2.0)],
        }
    ).write_parquet(tau_path)

    adapter = mcdr.MCDRAdapter()
    base_df = {
        "subject": ["rat-fast"] * 5,
        "session": ["sess-1"] * 5,
        "trial": [1, 2, 3, 4, 5],
        "trial_idx": [0, 1, 2, 3, 4],
        "x_c": ["L", "C", "R", "L", "C"],
        "ttype_n": [0.0, 1.0, 0.0, 1.0, 0.0],
        "stimd_n": [1.0, 2.0, 3.0, 4.0, 5.0],
        "performance": [1, 0, 1, 1, 0],
        "timepoint_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "timepoint_2": [0.4, 0.6, 0.8, 1.0, 1.2],
        "timepoint_3": [0.8, 1.1, 1.4, 1.8, 2.1],
        "timepoint_4": [1.2, 1.6, 2.0, 2.5, 2.9],
        "onset": [0.0, 0.2, 0.1, 0.3, 0.2],
        "offset": [1.2, 1.6, 2.0, 2.5, 2.9],
        "delay_d": [0.0, 0.5, 1.0, 1.5, 2.0],
        "response": [0, 1, 2, 0, 1],
        "stimulus": [0, 1, 2, 0, 1],
    }
    fast_df = pl.DataFrame(base_df)
    slow_df = pl.DataFrame({**base_df, "subject": ["rat-slow"] * 5})
    missing_df = pl.DataFrame({**base_df, "subject": ["rat-missing"] * 5})

    fast_features = adapter.build_feature_df(fast_df)
    slow_features = adapter.build_feature_df(slow_df)
    missing_features = adapter.build_feature_df(missing_df)

    responses = np.asarray(base_df["response"], dtype=np.int32)
    previous_response = np.concatenate(([0], responses[:-1]))
    expected_fast_l = _expected_ewma((previous_response == 0).astype(np.float32), half_life=1.5 * np.log(2.0))
    expected_fast_c = _expected_ewma((previous_response == 1).astype(np.float32), half_life=1.5 * np.log(2.0))
    expected_fast_r = _expected_ewma((previous_response == 2).astype(np.float32), half_life=1.5 * np.log(2.0))
    expected_slow_l = _expected_ewma((previous_response == 0).astype(np.float32), half_life=4.0 * np.log(2.0))
    expected_default_l = _expected_ewma((previous_response == 0).astype(np.float32), half_life=50.0)

    np.testing.assert_allclose(fast_features["A_L"].to_numpy(), expected_fast_l, atol=1e-6)
    np.testing.assert_allclose(fast_features["A_C"].to_numpy(), expected_fast_c, atol=1e-6)
    np.testing.assert_allclose(fast_features["A_R"].to_numpy(), expected_fast_r, atol=1e-6)
    np.testing.assert_allclose(slow_features["A_L"].to_numpy(), expected_slow_l, atol=1e-6)
    assert not np.allclose(fast_features["A_L"].to_numpy(), slow_features["A_L"].to_numpy())
    np.testing.assert_allclose(missing_features["A_L"].to_numpy(), expected_default_l, atol=1e-6)
