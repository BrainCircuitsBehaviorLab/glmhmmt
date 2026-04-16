import json
from glmhmmt.notebook_support.model_manager import widget as widget_module


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_update_options", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_apply_default_state", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_fits_path", lambda self: tmp_path)
    return widget_module.ModelManagerWidget()


def test_model_manager_assets_are_file_backed():
    assert widget_module.ModelManagerWidget._esm._path.name == "widget.js"
    assert widget_module.ModelManagerWidget._css._path.name == "widget.css"
    assert widget_module.ModelManagerWidget._esm._path.is_file()
    assert widget_module.ModelManagerWidget._css._path.is_file()


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


def test_task_change_clears_load_selection_state(monkeypatch, tmp_path):
    calls = {"update": 0, "apply": 0}

    def fake_update(self):
        calls["update"] += 1

    def fake_apply(self):
        calls["apply"] += 1

    monkeypatch.setattr(widget_module.ModelManagerWidget, "_update_options", fake_update)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_apply_default_state", fake_apply)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_fits_path", lambda self: tmp_path)

    widget = widget_module.ModelManagerWidget()
    widget.existing_model = "saved_fit"
    widget.subjects = ["s1"]
    widget.emission_cols = ["bias"]
    widget.transition_cols = ["prev_state"]
    widget.frozen_emissions = {"0": {"bias": 0.5}}
    widget.alias = "saved_fit"
    widget.saved_model_name = "saved_fit"
    widget.alias_error = "old error"
    widget.alias_status = "old status"

    widget.task = "OTHER"

    assert widget.existing_model == ""
    assert widget.subjects == []
    assert widget.emission_cols == []
    assert widget.transition_cols == []
    assert widget.frozen_emissions == {}
    assert widget.alias == ""
    assert widget.saved_model_name == ""
    assert widget.alias_error == ""
    assert widget.alias_status == ""
    assert calls["apply"] == 1
    assert calls["update"] == 2


def test_existing_model_selection_survives_refresh_when_still_available(monkeypatch, tmp_path):
    def fake_update(self):
        if self.model_type == "glmhmm":
            names = ["__default__", "fit_a"]
        else:
            names = ["__default__"]
        self.existing_models = names
        self.existing_models_info = [{"id": name, "name": name} for name in names]
        if self.existing_model and self.existing_model not in names:
            self.existing_model = ""

    monkeypatch.setattr(widget_module.ModelManagerWidget, "_update_options", fake_update)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_apply_default_state", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_fits_path", lambda self: tmp_path)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_find_model_dir", lambda self, name: None)

    widget = widget_module.ModelManagerWidget()
    widget.existing_model = "fit_a"

    widget._update_options()

    assert widget.existing_models == ["__default__", "fit_a"]
    assert widget.existing_model == "fit_a"


def test_model_type_change_clears_stale_existing_model_selection(monkeypatch, tmp_path):
    def fake_update(self):
        if self.model_type == "glmhmm":
            names = ["__default__", "fit_a"]
        else:
            names = ["__default__"]
        self.existing_models = names
        self.existing_models_info = [{"id": name, "name": name} for name in names]
        if self.existing_model and self.existing_model not in names:
            self.existing_model = ""

    monkeypatch.setattr(widget_module.ModelManagerWidget, "_update_options", fake_update)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_apply_default_state", lambda self: None)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_fits_path", lambda self: tmp_path)
    monkeypatch.setattr(widget_module.ModelManagerWidget, "_find_model_dir", lambda self, name: None)

    widget = widget_module.ModelManagerWidget()
    widget.existing_model = "fit_a"

    widget.model_type = "glmhmmt"

    assert widget.existing_models == ["__default__"]
    assert widget.existing_model == ""
