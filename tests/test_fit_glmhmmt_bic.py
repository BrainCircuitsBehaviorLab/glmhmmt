from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE_ROOT / "src"))

from glmhmmt.cli.fit_glmhmmt import _count_free_params


def _result(*, K: int = 3, num_classes: int = 3, x_dim: int = 4, u_dim: int = 5, frozen=None) -> dict:
    return {
        "K": K,
        "num_classes": num_classes,
        "X": np.zeros((10, x_dim), dtype=float),
        "U": np.zeros((10, u_dim), dtype=float),
        "names": {"X_cols": ["bias", "stim", "choice_lag", "reward"][:x_dim]},
        "frozen_emissions": frozen or {},
    }


def test_glmhmmt_bic_counts_target_specific_transition_input_weights():
    # Input-driven transitions store one contrast vector per non-baseline target,
    # not one vector per source-target pair.
    assert _count_free_params(_result()) == 40


def test_glmhmmt_bic_counts_standard_transition_case():
    assert _count_free_params(_result(u_dim=0)) == 30


def test_glmhmmt_bic_excludes_frozen_emission_weights():
    result = _result(frozen={"0": {"stim": 0.0}, "2": {"reward": 0.0}})

    assert _count_free_params(result) == 36
