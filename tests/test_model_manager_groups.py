from __future__ import annotations

from pathlib import Path
import sys
import importlib

workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root / "glmhmmt" / "src"))
sys.path.insert(0, str(workspace_root / "NMDAR_paper"))
importlib.import_module("src.process.two_afc")
importlib.import_module("src.process.two_afc_delay")
importlib.import_module("src.process.MCDR")

from glmhmmt.notebook_support.model_manager import widget as widget_module
from glmhmmt.notebook_support.model_manager.widget import _build_2afc_emission_groups


def test_build_2afc_emission_groups_hides_one_hot_members():
    groups = _build_2afc_emission_groups(
        [
            "bias",
            "bias_param",
            "bias_0",
            "bias_1",
            "stim_vals",
            "stim_param",
            "stim_0",
            "stim_2",
            "stim_4",
            "stim_8",
            "stim_20",
            "at_choice",
            "at_choice_param",
            "choice_lag_01",
            "choice_lag_02",
            "wsls",
        ]
    )
    by_key = {group["key"]: group for group in groups}

    assert by_key["stim_hot"]["toggle_members"] == ["stim_2", "stim_4", "stim_8", "stim_20"]
    assert by_key["stim_hot"]["hide_members"] is True
    assert by_key["bias_hot"]["toggle_members"] == ["bias_0", "bias_1"]
    assert by_key["bias_hot"]["hide_members"] is True
    assert by_key["at_choice_lag"]["toggle_members"] == ["choice_lag_01", "choice_lag_02"]
    assert by_key["at_choice_lag"]["hide_members"] is True

    assert "stim_0" not in by_key
    assert "stim_2" not in by_key
    assert "bias_0" not in by_key
    assert "choice_lag_01" not in by_key
    assert by_key["bias"]["members"] == {"N": "bias"}
    assert by_key["stim_param"]["members"] == {"N": "stim_param"}
    assert by_key["at_choice_param"]["members"] == {"N": "at_choice_param"}


def test_refresh_groups_uses_hidden_families_for_2afc_delay(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_update_options", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_apply_default_state", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_fits_path", lambda self: tmp_path)

    widget = widget_module.ModelManagerWidget()
    widget.task = "2AFC_delay"
    widget.emission_cols_options = [
        "bias",
        "stim",
        "delay",
        "delay_param",
        "delay_0p1",
        "delay_3",
        "bias_0",
        "bias_1",
        "at_choice",
        "choice_lag_01",
        "choice_lag_02",
    ]
    widget.transition_cols_options = []

    widget._refresh_groups()

    by_key = {group["key"]: group for group in widget.emission_groups}
    assert by_key["delay_hot"]["toggle_members"] == ["delay_0p1", "delay_3"]
    assert by_key["bias_hot"]["toggle_members"] == ["bias_0", "bias_1"]
    assert by_key["at_choice_lag"]["toggle_members"] == ["choice_lag_01", "choice_lag_02"]
    assert "delay_3" not in by_key
    assert "bias_0" not in by_key
    assert "choice_lag_01" not in by_key


def test_build_mcdr_emission_groups_keeps_bulk_stim_and_choice_lag_toggles():
    groups = widget_module._build_mcdr_emission_groups(
        [
            "bias",
            "bias_param",
            "bias_0",
            "bias_1",
            "A_L",
            "A_C",
            "A_R",
            "choice_lag_param",
            "choice_lag_01L",
            "choice_lag_01C",
            "choice_lag_01R",
            "stim_param",
            "stim1L",
            "stim1C",
            "stim1R",
            "stim4L",
            "stim4C",
            "stim4R",
        ]
    )
    by_key = {group["key"]: group for group in groups}

    assert by_key["bias_hot"]["toggle_members"] == ["bias_0", "bias_1"]
    assert by_key["stim_one_hot"]["toggle_members"] == ["stim1L", "stim1C", "stim1R", "stim4L", "stim4C", "stim4R"]
    assert by_key["stim_one_hot"]["hide_members"] is True
    assert by_key["choice_lag"]["toggle_members"] == ["choice_lag_01L", "choice_lag_01R"]
    assert by_key["choice_lag"]["exclude_members"] == ["choice_lag_01C"]
    assert by_key["choice_lag"]["hide_members"] is True
    assert by_key["stim_param"]["members"] == {"N": "stim_param"}
    assert by_key["choice_lag_param"]["members"] == {"N": "choice_lag_param"}
    assert "stim1" not in by_key
    assert "stim4" not in by_key
    assert "choice_lag_01" not in by_key
