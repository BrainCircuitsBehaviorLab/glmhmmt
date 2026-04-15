from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from glmhmmt.runtime import configure_paths, get_runtime_paths
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    clear_fitted_regressor_cache,
    resolved_source_features,
    weighted_sum_regressor,
)


@pytest.fixture
def isolated_runtime(tmp_path):
    original = get_runtime_paths()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    configure_paths(
        data_dir=data_dir,
        results_dir=tmp_path,
        config_path=original.config_path,
    )
    clear_fitted_regressor_cache()
    try:
        yield tmp_path
    finally:
        configure_paths(
            data_dir=original.data_dir,
            results_dir=original.results_dir,
            config_path=original.config_path,
        )
        clear_fitted_regressor_cache()


def _write_fit(
    results_root,
    *,
    task: str,
    model_id: str,
    emission_cols: list[str],
    weights: list[float],
    subject: str = "mouse-1",
) -> None:
    fit_dir = results_root / "fits" / task / "glm" / model_id
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "config.json").write_text(
        json.dumps({"emission_cols": emission_cols, "model_id": model_id}),
        encoding="utf-8",
    )
    np.savez(
        fit_dir / f"{subject}_glm_arrays.npz",
        X_cols=np.asarray(emission_cols, dtype=object),
        emission_weights=np.asarray([[[*weights]]], dtype=float),
    )


def test_resolves_exact_prefix_and_legacy_feature_selection(isolated_runtime):
    emission_cols = [
        "bias",
        "stim_0",
        "stim_2",
        "stim_4",
        "bias_0",
        "bias_1",
        "choice_lag_01",
        "choice_lag_02",
    ]
    _write_fit(
        isolated_runtime,
        task="2AFC",
        model_id="one hot sessions lags",
        emission_cols=emission_cols,
        weights=[0.0, -1.0, -2.0, -4.0, 0.3, 0.7, 1.5, -0.5],
    )

    exact_spec = FittedWeightRegressorSpec(
        target_name="one lag",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        source_features=("choice_lag_02",),
        source_feature_prefixes=("choice_lag_",),
    )
    prefix_spec = FittedWeightRegressorSpec(
        target_name="stim_param",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        source_feature_prefixes=("stim_",),
        exclude_features=("stim_0",),
        sign=-1.0,
    )
    legacy_spec = FittedWeightRegressorSpec(
        target_name="legacy_stim",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        exclude_features=("bias", "stim_0", "bias_0", "bias_1", "choice_lag_01", "choice_lag_02"),
        sign=-1.0,
    )

    assert resolved_source_features(exact_spec) == ("choice_lag_02",)
    assert resolved_source_features(prefix_spec) == ("stim_2", "stim_4")
    assert resolved_source_features(legacy_spec) == ("stim_2", "stim_4")


def test_weighted_sum_regressor_uses_only_its_selected_family(isolated_runtime):
    emission_cols = [
        "bias",
        "stim_0",
        "stim_2",
        "stim_4",
        "bias_0",
        "bias_1",
        "choice_lag_01",
        "choice_lag_02",
    ]
    _write_fit(
        isolated_runtime,
        task="2AFC",
        model_id="one hot sessions lags",
        emission_cols=emission_cols,
        weights=[0.0, -1.0, -2.0, -4.0, 0.3, 0.7, 1.5, -0.5],
    )

    stim_spec = FittedWeightRegressorSpec(
        target_name="stim_param",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        source_feature_prefixes=("stim_",),
        exclude_features=("stim_0",),
        sign=-1.0,
    )
    bias_spec = FittedWeightRegressorSpec(
        target_name="bias_param",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        source_feature_prefixes=("bias_",),
    )
    choice_spec = FittedWeightRegressorSpec(
        target_name="at_choice_param",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        source_feature_prefixes=("choice_lag_",),
    )

    part = pd.DataFrame(
        {
            "stim_2": [1.0, 0.0],
            "stim_4": [0.0, -1.0],
            "bias_0": [1.0, 0.0],
            "bias_1": [0.0, 1.0],
            "choice_lag_01": [1.0, -1.0],
            "choice_lag_02": [0.5, 0.0],
        }
    )

    np.testing.assert_allclose(
        weighted_sum_regressor(part, stim_spec),
        np.asarray([2.0, -4.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        weighted_sum_regressor(part, bias_spec),
        np.asarray([0.3, 0.7], dtype=np.float32),
    )
    np.testing.assert_allclose(
        weighted_sum_regressor(part, choice_spec),
        np.asarray([1.25, -1.5], dtype=np.float32),
    )


def test_weighted_sum_regressor_prefers_subject_specific_fit_weights(isolated_runtime):
    emission_cols = ["stim_2", "stim_4", "bias_0", "choice_lag_01"]
    _write_fit(
        isolated_runtime,
        task="2AFC",
        model_id="one hot sessions lags",
        emission_cols=emission_cols,
        weights=[2.0, 4.0, 0.5, 1.0],
        subject="mouse-1",
    )
    _write_fit(
        isolated_runtime,
        task="2AFC",
        model_id="one hot sessions lags",
        emission_cols=emission_cols,
        weights=[10.0, 20.0, 5.0, 7.0],
        subject="mouse-2",
    )

    stim_spec = FittedWeightRegressorSpec(
        target_name="stim_param",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        source_feature_prefixes=("stim_",),
    )

    part = pd.DataFrame(
        {
            "subject": ["mouse-2", "mouse-2"],
            "stim_2": [1.0, 0.0],
            "stim_4": [0.0, -1.0],
        }
    )

    np.testing.assert_allclose(
        weighted_sum_regressor(part, stim_spec),
        np.asarray([10.0, -20.0], dtype=np.float32),
    )


def test_weighted_sum_regressor_falls_back_to_pooled_weights_when_subject_fit_missing(isolated_runtime):
    emission_cols = ["stim_2", "stim_4"]
    _write_fit(
        isolated_runtime,
        task="2AFC",
        model_id="one hot sessions lags",
        emission_cols=emission_cols,
        weights=[2.0, 4.0],
        subject="mouse-1",
    )
    _write_fit(
        isolated_runtime,
        task="2AFC",
        model_id="one hot sessions lags",
        emission_cols=emission_cols,
        weights=[6.0, 8.0],
        subject="mouse-2",
    )

    stim_spec = FittedWeightRegressorSpec(
        target_name="stim_param",
        fit_task="2AFC",
        fit_model_kind="glm",
        fit_model_id="one hot sessions lags",
        arrays_suffix="glm_arrays.npz",
        source_feature_prefixes=("stim_",),
    )

    part = pd.DataFrame(
        {
            "subject": ["mouse-3", "mouse-3"],
            "stim_2": [1.0, 0.0],
            "stim_4": [0.0, -1.0],
        }
    )

    np.testing.assert_allclose(
        weighted_sum_regressor(part, stim_spec),
        np.asarray([4.0, -6.0], dtype=np.float32),
    )
