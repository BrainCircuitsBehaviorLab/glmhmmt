import json

from glmhmmt.notebook_support.model_manager import widget as widget_module


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_update_options", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_apply_default_state", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_fits_path", lambda self: tmp_path)
    return widget_module.ModelManagerWidget()


def test_run_fit_command_increments_clicks(monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)

    widget.command = "run_fit"
    widget.command_payload = {}
    widget.command_nonce = 1

    assert widget.run_fit_clicks == 1
    assert widget.command == ""
    assert widget.command_payload == {}


def test_save_alias_command_writes_config(monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    widget.task = "MCDR"
    widget.model_type = "glmhmm"
    widget.subjects = ["s1", "s2"]
    widget.emission_cols = ["bias"]
    widget.K = 3

    widget.command = "save_alias"
    widget.command_payload = {"alias": "named_model"}
    widget.command_nonce = 1

    cfg_path = tmp_path / "named_model" / "config.json"
    assert cfg_path.exists()
    saved = json.loads(cfg_path.read_text())
    assert saved["model_id"] == "named_model"
    assert saved["alias"] == "named_model"
    assert saved["subjects"] == ["s1", "s2"]
    assert saved["K_list"] == [3]
    assert widget.saved_model_name == "named_model"
    assert widget.alias_status == "Saved as named_model"
    assert widget.command == ""
    assert widget.command_payload == {}


def test_delete_model_command_removes_directory(monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    model_dir = tmp_path / "obsolete"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "task": "MCDR",
                "model_id": "obsolete",
                "alias": "obsolete",
                "subjects": ["s1"],
                "emission_cols": ["bias"],
                "K_list": [2],
            }
        )
    )

    widget.command = "delete_model"
    widget.command_payload = {"name": "obsolete"}
    widget.command_nonce = 1

    assert not model_dir.exists()
    assert widget.alias_status == "Deleted obsolete"
    assert widget.command == ""
    assert widget.command_payload == {}
