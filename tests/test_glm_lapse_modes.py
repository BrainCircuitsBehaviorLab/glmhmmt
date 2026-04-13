from __future__ import annotations

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from glmhmmt.cli.fit_glm import generate_model_id, save_results
from glmhmmt.glm import fit_glm
from glmhmmt.model_plots import plot_lapse_rates_boxplot
from glmhmmt.notebook_support.model_manager.widget import model_cfg


def _mock_minimize_factory(responses: list[np.ndarray]):
    queue = [np.asarray(response, dtype=float) for response in responses]

    def _mock_minimize(fun, x0, method=None, bounds=None, **kwargs):
        x = queue.pop(0) if queue else np.asarray(x0, dtype=float)
        return SimpleNamespace(x=x, fun=float(fun(x)), success=True)

    return _mock_minimize


def test_fit_glm_class_lapse_binary_returns_two_rates(monkeypatch):
    monkeypatch.setattr(
        "glmhmmt.glm.minimize",
        _mock_minimize_factory(
            [
                np.array([0.0]),
                np.array([0.0, 0.10, 0.20]),
            ]
        ),
    )

    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    fit = fit_glm(X, y, num_classes=2, lapse_mode="class", n_restarts=1, restart_noise_scale=0.0)

    assert fit.lapse_mode == "class"
    assert fit.lapse_rates.shape == (2,)
    np.testing.assert_allclose(fit.lapse_rates, np.array([0.10, 0.20]))
    np.testing.assert_allclose(fit.predictive_probs.sum(axis=1), np.ones(4))


def test_fit_glm_class_lapse_three_class_returns_three_rates(monkeypatch):
    monkeypatch.setattr(
        "glmhmmt.glm.minimize",
        _mock_minimize_factory(
            [
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0, 0.10, 0.20, 0.30]),
            ]
        ),
    )

    X = np.zeros((3, 1), dtype=float)
    y = np.array([0, 1, 2], dtype=int)
    fit = fit_glm(X, y, num_classes=3, lapse_mode="class", n_restarts=1, restart_noise_scale=0.0)

    assert fit.lapse_mode == "class"
    assert fit.lapse_rates.shape == (3,)
    assert fit.lapse_labels == ("class_0", "class_1", "class_2")
    assert float(fit.lapse_rates.sum()) <= 1.0
    np.testing.assert_allclose(fit.predictive_probs.sum(axis=1), np.ones(3))


def test_fit_glm_history_mode_uses_shared_repeat_and_skips_first_trial(monkeypatch):
    monkeypatch.setattr(
        "glmhmmt.glm.minimize",
        _mock_minimize_factory(
            [
                np.array([0.0]),
                np.array([0.0, 0.30, 0.10]),
            ]
        ),
    )

    X = np.zeros((4, 1), dtype=float)
    y = np.array([1, 1, 0, 1], dtype=int)
    fit = fit_glm(
        X,
        y,
        num_classes=2,
        lapse_mode="history",
        lapse_max=1.0,
        n_restarts=1,
        restart_noise_scale=0.0,
    )

    np.testing.assert_allclose(fit.predictive_probs[0], np.array([0.5, 0.5]))
    assert fit.predictive_probs[1, 1] > fit.predictive_probs[1, 0]
    assert fit.predictive_probs[2, 1] > fit.predictive_probs[2, 0]
    assert fit.predictive_probs[3, 0] > fit.predictive_probs[3, 1]
    assert fit.lapse_labels == ("repeat", "alternate")
    np.testing.assert_allclose(fit.lapse_rates, np.array([0.30, 0.10]))


def test_fit_glm_history_mode_uses_single_shared_constraint(monkeypatch):
    captured: dict[str, object] = {}

    def _mock_minimize(fun, x0, method=None, bounds=None, constraints=(), **kwargs):
        captured["method"] = method
        captured["constraints"] = constraints
        x = np.asarray(x0, dtype=float)
        return SimpleNamespace(x=x, fun=float(fun(x)), success=True)

    monkeypatch.setattr("glmhmmt.glm.minimize", _mock_minimize)

    X = np.zeros((4, 1), dtype=float)
    y = np.array([1, 1, 0, 1], dtype=int)
    fit_glm(
        X,
        y,
        num_classes=2,
        lapse_mode="history",
        lapse_max=1.0,
        n_restarts=1,
        restart_noise_scale=0.0,
    )

    constraints = captured["constraints"]
    assert captured["method"] == "SLSQP"
    assert len(constraints) == 1

    constraint_fun = constraints[0]["fun"]
    np.testing.assert_allclose(
        constraint_fun(np.array([0.0, 0.20, 0.30], dtype=float)),
        0.5,
    )


def test_fit_glm_history_conditioned_alternate_component_splits_across_non_previous_classes(monkeypatch):
    monkeypatch.setattr(
        "glmhmmt.glm.minimize",
        _mock_minimize_factory(
                [
                    np.array([0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60, 0.0]),
                ]
            ),
        )

    X = np.zeros((3, 1), dtype=float)
    y = np.array([1, 2, 0], dtype=int)
    fit = fit_glm(
        X,
        y,
        num_classes=3,
        lapse_mode="history_conditioned",
        lapse_max=1.0,
        n_restarts=1,
        restart_noise_scale=0.0,
    )

    np.testing.assert_allclose(fit.predictive_probs[0], np.full(3, 1 / 3))
    np.testing.assert_allclose(fit.predictive_probs[1], np.array([0.43333333, 0.13333333, 0.43333333]))
    assert fit.lapse_labels == (
        "repeat_prev_0",
        "repeat_prev_1",
        "repeat_prev_2",
        "alternate_prev_0",
        "alternate_prev_1",
        "alternate_prev_2",
    )


def test_fit_glm_rejects_unknown_lapse_mode():
    X = np.zeros((2, 1), dtype=float)
    y = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="Unsupported lapse_mode"):
        fit_glm(X, y, num_classes=2, lapse_mode="bad-mode")


def test_fit_glm_empty_dataset_returns_zero_lapse_shape():
    X = np.zeros((0, 2), dtype=float)
    y = np.zeros(0, dtype=int)

    fit = fit_glm(X, y, num_classes=3, lapse_mode="class")

    assert fit.lapse_rates.shape == (3,)
    np.testing.assert_allclose(fit.lapse_rates, np.zeros(3))
    assert fit.predictive_probs.shape == (0, 3)


def test_save_results_persists_lapse_metadata_and_counts_params(tmp_path):
    result = {
        "subject": "mouse-1",
        "W": np.zeros((3, 2), dtype=float),
        "p_pred": np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], dtype=float),
        "lapse_rates": np.array([0.1, 0.2, 0.3], dtype=float),
        "lapse_mode": "class",
        "lapse_labels": ("class_0", "class_1", "class_2"),
        "nll": 1.5,
        "success": True,
        "y": np.array([2, 1], dtype=int),
        "X": np.zeros((2, 2), dtype=float),
        "names": {"X_cols": ["bias", "stim"]},
        "T": 2,
    }

    save_results(result, tmp_path, tau=5.0)

    arrays = np.load(tmp_path / "mouse-1_glm_arrays.npz", allow_pickle=True)
    assert arrays["lapse_mode"].item() == "class"
    np.testing.assert_allclose(arrays["lapse_rates"], np.array([0.1, 0.2, 0.3]))
    assert arrays["lapse_labels"].tolist() == ["class_0", "class_1", "class_2"]

    metrics = pl.read_parquet(tmp_path / "mouse-1_glm_metrics.parquet")
    assert metrics["k"].item() == 7


def test_generate_model_id_changes_with_lapse_mode():
    args = ("MCDR", 5.0, ["bias", "stim"])
    no_lapse = generate_model_id(*args, lapse_mode="none")
    class_lapse = generate_model_id(*args, lapse_mode="class")
    history_lapse = generate_model_id(*args, lapse_mode="history")
    conditioned_lapse = generate_model_id(*args, lapse_mode="history_conditioned")

    assert no_lapse != class_lapse
    assert class_lapse != history_lapse
    assert history_lapse != conditioned_lapse


def test_model_cfg_accepts_new_and_legacy_lapse_fields():
    new_cfg = model_cfg.from_value({"lapse_mode": "history", "lapse_max": 0.4})

    assert new_cfg.lapse_mode == "history"
    assert new_cfg.lapse is True


def test_plot_lapse_rates_boxplot_uses_lapse_labels():
    views = {
        "s1": SimpleNamespace(
            subject="s1",
            lapse_rates=np.array([0.1, 0.2, 0.05, 0.15]),
            lapse_mode="history_conditioned",
            lapse_labels=("repeat_prev_0", "repeat_prev_1", "alternate_prev_0", "alternate_prev_1"),
        ),
        "s2": SimpleNamespace(
            subject="s2",
            lapse_rates=np.array([0.05, 0.15, 0.02, 0.08]),
            lapse_mode="history_conditioned",
            lapse_labels=("repeat_prev_0", "repeat_prev_1", "alternate_prev_0", "alternate_prev_1"),
        ),
    }

    fig = plot_lapse_rates_boxplot(views, choice_labels=("Left", "Right"))

    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert labels == [
        "Repeat after Left",
        "Repeat after Right",
        "Alternate after Left",
        "Alternate after Right",
    ]
    plt.close(fig)


def test_plot_lapse_rates_boxplot_can_collapse_history_choices():
    views = {
        "s1": SimpleNamespace(
            subject="s1",
            lapse_rates=np.array([0.1, 0.2, 0.05, 0.15]),
            lapse_mode="history_conditioned",
            lapse_labels=("repeat_prev_0", "repeat_prev_1", "alternate_prev_0", "alternate_prev_1"),
        ),
        "s2": SimpleNamespace(
            subject="s2",
            lapse_rates=np.array([0.05, 0.15, 0.02, 0.08]),
            lapse_mode="history_conditioned",
            lapse_labels=("repeat_prev_0", "repeat_prev_1", "alternate_prev_0", "alternate_prev_1"),
        ),
    }

    fig = plot_lapse_rates_boxplot(
        views,
        choice_labels=("Left", "Right"),
        collapse_history_choices=True,
        annotate_significance=False,
    )

    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert labels == ["Repeat", "Alternate"]
    plt.close(fig)


def test_plot_lapse_rates_boxplot_shared_history_uses_repeat_and_alternate_labels():
    views = {
        "s1": SimpleNamespace(
            subject="s1",
            lapse_rates=np.array([0.1, 0.05]),
            lapse_mode="history",
            lapse_labels=("repeat", "alternate"),
        ),
        "s2": SimpleNamespace(
            subject="s2",
            lapse_rates=np.array([0.2, 0.08]),
            lapse_mode="history",
            lapse_labels=("repeat", "alternate"),
        ),
    }

    fig = plot_lapse_rates_boxplot(
        views,
        choice_labels=("Left", "Right"),
        annotate_significance=False,
    )

    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert labels == ["Repeat", "Alternate"]
    plt.close(fig)
