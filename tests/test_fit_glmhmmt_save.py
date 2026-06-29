from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE_ROOT / "src"))

from glmhmmt.cli.fit_glmhmmt import save_cv_results, save_results


def _result(transitions: SimpleNamespace) -> dict:
    K = 2
    y = np.asarray([0, 1], dtype=int)
    p_pred = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    probs = np.ones((2, K), dtype=float) / K
    return {
        "subject": "s1",
        "K": K,
        "num_classes": 2,
        "fitted_params": SimpleNamespace(
            initial=SimpleNamespace(probs=np.asarray([0.5, 0.5], dtype=float)),
            emissions=SimpleNamespace(weights=np.zeros((K, 1, 1), dtype=float)),
            transitions=transitions,
        ),
        "lps": np.asarray([-3.0, -2.0], dtype=float),
        "smoothed_probs": probs,
        "p_pred": p_pred,
        "predictive_state_probs": probs,
        "raw_ll": -2.0,
        "T": int(y.shape[0]),
        "names": {"X_cols": ["bias"], "U_cols": []},
        "y": y,
        "X": np.ones((2, 1), dtype=float),
        "U": np.zeros((2, 0), dtype=float),
        "frozen_emissions": {},
        "baseline_class_idx": 0,
        "transition_weight_baseline_idx": -1,
        "best_restart": 1,
        "cv_mode": "balanced_session_holdout",
        "cv_repeats": 1,
        "repeats": [
            {
                "repeat_index": 1,
                "train_ll_per_trial": -1.0,
                "train_acc": 0.5,
                "test_ll_per_trial": -1.1,
                "test_acc": 0.5,
            }
        ],
        "best_repeat_index": 1,
        "best_repeat_test_ll_per_trial": -1.1,
    }


def test_save_results_persists_standard_transition_matrix(tmp_path):
    transition_matrix = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)

    save_results(_result(SimpleNamespace(transition_matrix=transition_matrix)), tmp_path)

    arrays = np.load(tmp_path / "s1_K2_glmhmmt_arrays.npz", allow_pickle=True)
    np.testing.assert_allclose(arrays["transition_matrix"], transition_matrix)
    assert "transition_bias" not in arrays.files
    assert "transition_weights" not in arrays.files


def test_save_cv_results_persists_standard_transition_matrix(tmp_path):
    transition_matrix = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)

    save_cv_results(_result(SimpleNamespace(transition_matrix=transition_matrix)), tmp_path)

    arrays = np.load(tmp_path / "s1_K2_glmhmmt_arrays.npz", allow_pickle=True)
    np.testing.assert_allclose(arrays["transition_matrix"], transition_matrix)
    assert "transition_bias" not in arrays.files
    assert "transition_weights" not in arrays.files

