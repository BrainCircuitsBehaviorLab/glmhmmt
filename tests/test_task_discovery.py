from __future__ import annotations

import importlib
import types

import numpy as np
import polars as pl
import pytest

task_registry = importlib.import_module("glmhmmt.tasks")


class _BaseAdapter(task_registry.TaskAdapter):
    task_key = "BASE"
    task_label = "Base"
    num_classes = 2
    data_file = "dummy.parquet"
    sort_col = "trial"
    session_col = "session"

    def subject_filter(self, df):
        return df

    def build_feature_df(self, df_sub, tau: float = 50.0):
        return df_sub

    def load_subject(self, df_sub, tau: float = 50.0, emission_cols=None, transition_cols=None):
        return np.empty((0,)), np.empty((0, 0)), np.empty((0, 0)), {"X_cols": [], "U_cols": []}

    def build_design_matrices(self, feature_df, emission_cols=None, transition_cols=None):
        return np.empty((0,)), np.empty((0, 0)), np.empty((0, 0)), {"X_cols": [], "U_cols": []}

    def default_emission_cols(self, df=None):
        return []

    def default_transition_cols(self):
        return []

    def get_plots(self) -> types.ModuleType:
        return types.ModuleType("dummy_plots")

    @property
    def behavioral_cols(self):
        return {
            "trial_idx": "trial",
            "trial": "trial",
            "session": "session",
            "stimulus": "stimulus",
            "response": "response",
            "performance": "performance",
        }

    def label_states(self, arrays_store: dict, names: dict, K: int, subjects: list):
        return {}, {}

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)


class _LocalAdapter(_BaseAdapter):
    task_key = "LOCAL"
    task_label = "Local"


class _ExternalAdapter(_BaseAdapter):
    task_key = "EXTERNAL"
    task_label = "External"


@pytest.fixture(autouse=True)
def reset_task_registry(monkeypatch):
    monkeypatch.setattr(task_registry, "_REGISTRY", {}, raising=False)
    monkeypatch.setattr(task_registry, "_ENTRY_POINTS_LOADED", False, raising=False)
    monkeypatch.setattr(task_registry, "_LOCAL_TASK_PATHS_LOADED", set(), raising=False)
    monkeypatch.setattr(task_registry, "_LOCAL_PLOT_MODULES_LOADED", {}, raising=False)


def test_get_task_options_filters_to_local_scope(monkeypatch):
    monkeypatch.setattr(task_registry, "_load_entry_point_adapters", lambda: None)
    monkeypatch.setattr(task_registry, "_load_local_task_packages", lambda: None)
    monkeypatch.setattr(task_registry, "_task_scope_is_local", lambda: True)
    monkeypatch.setattr(task_registry, "_active_local_task_dirs", lambda: [])
    monkeypatch.setattr(
        task_registry,
        "_adapter_defined_in_dirs",
        lambda cls, roots: cls is _LocalAdapter,
    )
    monkeypatch.setattr(
        task_registry,
        "_REGISTRY",
        {
            "local": _LocalAdapter,
            "external": _ExternalAdapter,
        },
        raising=False,
    )

    assert task_registry.get_task_options() == [{"value": "LOCAL", "label": "Local"}]
    adapter = task_registry.get_adapter("local")
    assert isinstance(adapter, _LocalAdapter)

    with pytest.raises(ValueError, match='Unknown task "external"|Unknown task \'external\''):
        task_registry.get_adapter("external")


def test_get_task_options_returns_empty_when_no_local_adapters(monkeypatch):
    monkeypatch.setattr(task_registry, "_load_entry_point_adapters", lambda: None)
    monkeypatch.setattr(task_registry, "_load_local_task_packages", lambda: None)
    monkeypatch.setattr(task_registry, "_task_scope_is_local", lambda: True)
    monkeypatch.setattr(task_registry, "_active_local_task_dirs", lambda: [])
    monkeypatch.setattr(task_registry, "_configured_task_dir_labels", lambda: ["/tmp/missing-adapters"])
    monkeypatch.setattr(task_registry, "_REGISTRY", {"external": _ExternalAdapter}, raising=False)
    monkeypatch.setattr(task_registry, "_adapter_defined_in_dirs", lambda cls, roots: False)

    assert task_registry.get_task_options() == []
    with pytest.raises(ValueError, match="No task adapters were found in the configured adapter directories"):
        task_registry.get_adapter("external")


def test_resolve_plots_module_uses_configured_plot_dirs(monkeypatch, tmp_path):
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    plot_file = plot_dir / "2AFC.py"
    plot_file.write_text("VALUE = 7\n", encoding="utf-8")

    monkeypatch.setattr(task_registry, "_active_local_plot_dirs", lambda: [plot_dir])
    monkeypatch.setattr(task_registry, "_configured_plot_dirs", lambda: [plot_dir])
    monkeypatch.setattr(task_registry, "_LOCAL_PLOT_MODULES_LOADED", {}, raising=False)

    plots = task_registry.resolve_plots_module(
        adapter_module_name="glmhmmt_local_tasks.process.two_afc",
        task_key="2AFC",
    )

    assert plots.VALUE == 7
