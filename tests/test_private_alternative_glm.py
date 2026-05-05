from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import polars as pl

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from glmhmmt.glm import fit_private_alternative_glm, private_alternative_probs
from glmhmmt.postprocess import build_emission_weights_df, build_trial_df
from glmhmmt.views import build_views


def _mcdr_module(monkeypatch):
    workspace_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(workspace_root / "glmhmmt" / "src"))
    monkeypatch.syspath_prepend(str(workspace_root / "NMDAR_paper"))
    module = importlib.import_module("src.process.MCDR")
    monkeypatch.setattr(module, "_max_subject_sessions", lambda: 3)
    return module


def _mcdr_df() -> pl.DataFrame:
    return pl.DataFrame(
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


def test_mcdr_private_design_shape_and_side_alignment(monkeypatch):
    module = _mcdr_module(monkeypatch)
    adapter = module.MCDRAdapter()
    feature_df = adapter.build_feature_df(_mcdr_df())

    y, X_private, _, names = adapter.build_private_design_matrices(feature_df)
    X_np = np.asarray(X_private)
    cols = list(names["X_private_cols"])

    assert list(y) == [0, 1, 2, 0]
    assert X_np.shape == (4, 3, len(cols))
    assert names["emission_model"] == "private_alternative"
    assert names["alternative_order"] == ["L", "C", "R"]

    stim_idx = cols.index("stimd_z")
    lag_idx = cols.index("choice_lag_01")
    action_idx = cols.index("A")

    np.testing.assert_allclose(X_np[:, 0, stim_idx], feature_df["SL"].to_numpy())
    np.testing.assert_allclose(X_np[:, 1, stim_idx], feature_df["SC"].to_numpy())
    np.testing.assert_allclose(X_np[:, 2, stim_idx], feature_df["SR"].to_numpy())
    np.testing.assert_allclose(X_np[:, 0, lag_idx], feature_df["choice_lag_01L"].to_numpy())
    np.testing.assert_allclose(X_np[:, 1, lag_idx], feature_df["choice_lag_01C"].to_numpy())
    np.testing.assert_allclose(X_np[:, 2, lag_idx], feature_df["choice_lag_01R"].to_numpy())
    np.testing.assert_allclose(X_np[:, 0, action_idx], feature_df["A_L"].to_numpy())
    np.testing.assert_allclose(X_np[:, 1, action_idx], feature_df["A_C"].to_numpy())
    np.testing.assert_allclose(X_np[:, 2, action_idx], feature_df["A_R"].to_numpy())


def test_private_alternative_probs_sum_to_one_and_fit_decreases_nll():
    rng = np.random.default_rng(0)
    T = 250
    C = 3
    M = 2
    X_private = rng.normal(size=(T, C, M))
    true_w = np.asarray([1.2, -0.8])
    true_b = np.asarray([0.4, 0.0, -0.2])
    probs = private_alternative_probs(X_private, true_w, true_b)
    y = np.asarray([rng.choice(C, p=p) for p in probs], dtype=int)

    fit = fit_private_alternative_glm(X_private, y, num_classes=C, bias_reference_class_idx=1)
    start_probs = np.full((T, C), 1.0 / C)
    start_nll = -float(np.sum(np.log(start_probs[np.arange(T), y])))

    np.testing.assert_allclose(fit.predictive_probs.sum(axis=1), np.ones(T), rtol=1e-10, atol=1e-10)
    assert fit.negative_log_likelihood < start_nll
    assert fit.nll_history[-1] <= fit.nll_history[0]
    assert fit.private_weights.shape == (M,)
    assert fit.private_bias[1] == 0.0


class _DummyAdapter:
    probability_columns = ["pL", "pC", "pR"]

    def label_states(self, arrays_store, names, K, subjects):
        return (
            {subject: {0: "State 0"} for subject in subjects},
            {subject: [0] for subject in subjects},
        )

    def get_correct_class(self, df):
        return df["stimulus"].to_numpy().astype(int)


def test_private_alternative_views_postprocess_and_weight_summary():
    X_private = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=float,
    )
    private_weights = np.asarray([1.0, -0.5], dtype=float)
    private_bias = np.asarray([0.2, 0.0, -0.1], dtype=float)
    p_pred = private_alternative_probs(X_private, private_weights, private_bias)
    arrays = {
        "emission_model": np.asarray("private_alternative"),
        "smoothed_probs": np.ones((2, 1)),
        "predictive_state_probs": np.ones((2, 1)),
        "emission_weights": np.zeros((1, 2, 2), dtype=float),
        "X": X_private.mean(axis=1),
        "X_cols": np.asarray(["stimd_z", "A"], dtype=object),
        "X_private": X_private,
        "X_private_cols": np.asarray(["stimd_z", "A"], dtype=object),
        "private_weights": private_weights,
        "private_bias": private_bias,
        "y": np.asarray([0, 1], dtype=int),
        "p_pred": p_pred,
        "baseline_class_idx": np.asarray(1),
    }
    view = build_views({"s1": arrays}, _DummyAdapter(), K=1, subjects=["s1"])["s1"]

    np.testing.assert_allclose(view.state_conditional_probs()[:, 0, :], p_pred, rtol=1e-10)

    trial_df = build_trial_df(
        view,
        _DummyAdapter(),
        pl.DataFrame(
            {
                "trial_idx": [0, 1],
                "session": [1, 1],
                "stimulus": [0, 1],
                "response": [0, 1],
                "performance": [1, 1],
            }
        ),
        {
            "trial_idx": "trial_idx",
            "session": "session",
            "stimulus": "stimulus",
            "response": "response",
            "performance": "performance",
        },
    )
    np.testing.assert_allclose(trial_df["pL"].to_numpy(), p_pred[:, 0], rtol=1e-10)
    np.testing.assert_allclose(trial_df["p_model_correct"].to_numpy(), [p_pred[0, 0], p_pred[1, 1]], rtol=1e-10)

    weights_df = build_emission_weights_df({"s1": view})
    assert set(weights_df["emission_model"].to_list()) == {"private_alternative"}
    assert {"stimd_z", "A", "bias_0", "bias_1", "bias_2"}.issubset(set(weights_df["feature"].to_list()))
