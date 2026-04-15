from __future__ import annotations

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
