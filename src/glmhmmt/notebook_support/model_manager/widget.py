"""ModelManagerWidget — anywidget-backed UI for selecting and loading GLM-HMM fits.

Python layer handles all semantics: regressor grouping, named-model detection,
fitted-subject counting, and config-to-state application.  The JS layer
(widget.js) handles rendering and user events only.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anywidget
import polars as pl
import traitlets

from glmhmmt.model import prune_frozen_emissions, serialize_frozen_emissions
from glmhmmt.notebook_support.analysis_common import (
    load_model_config,
    model_aliases_for_kind,
    remote_fits_enabled,
)
from glmhmmt.runtime import get_runtime_paths
from glmhmmt.tasks import get_adapter, get_task_options

try:
    from glmhmmt.tasks import build_selector_groups
except ImportError:
    def build_selector_groups(available_cols: list[str], registry: list[dict]) -> list[dict]:
        available = set(available_cols)
        registered: set[str] = set()
        result: list[dict] = []
        for group in registry:
            filtered = {k: v for k, v in group["members"].items() if v in available}
            if filtered:
                result.append({**group, "members": filtered})
                registered.update(filtered.values())
        for col in available_cols:
            if col not in registered:
                result.append({"key": col, "label": col, "members": {"N": col}})
        return result


_ASSET_DIR = Path(__file__).parent


def _paths():
    return get_runtime_paths()

# ── Private helpers ───────────────────────────────────────────────────────────

_HASH_RE   = re.compile(r"^[0-9a-f]{8}$")
_ARRAYS_RE = re.compile(r"^(.+?)(?:_K\d+)?_(glm|glmhmm|glmhmmt)_arrays\.npz$")


def _normalize_cv_mode(cv_mode: str) -> str:
    cv_mode = str(cv_mode)
    if cv_mode == "balanced_holdout":
        return "balanced_session_holdout"
    return cv_mode


def _build_regressor_groups(available_cols: list[str], registry: list[dict]) -> list[dict]:
    return build_selector_groups(available_cols, registry)


def _build_2afc_emission_groups(available_cols: list[str]) -> list[dict]:
    return get_adapter("2AFC").build_emission_groups(list(available_cols))


def _build_mcdr_emission_groups(available_cols: list[str]) -> list[dict]:
    groups = get_adapter("MCDR").build_emission_groups(list(available_cols))
    return [group for group in groups if group.get("key") not in {"stim1", "stim2", "stim3", "stim4", "stim_hot"}]


def _count_fitted_subjects(model_dir: Path) -> int:
    """Count unique subjects from *_arrays.npz files in *model_dir*.

    Expected filename format (new, without _K{N}): {subject}_{modelname}_arrays.npz
    Falls back gracefully to 0 for old _K{N}-format files.
    """
    subjects: set[str] = set()
    for f in model_dir.glob("*_arrays.npz"):
        m = _ARRAYS_RE.match(f.name)
        if m:
            subjects.add(m.group(1))
    return len(subjects)


def _is_displayable(cfg: dict) -> bool:
    """Return True if *cfg* represents a human-named model.

    A model is displayable when either:
    - it has a non-empty ``alias`` field, or
    - its ``model_id`` does not look like an 8-character hex hash.
    """
    alias = cfg.get("alias", "")
    if alias and alias.strip():
        return True
    model_id = cfg.get("model_id", "")
    return bool(model_id) and not _HASH_RE.match(model_id)


def _get_display_name(cfg: dict) -> str:
    """Return the user-facing display name for a model config."""
    alias = cfg.get("alias", "")
    if alias and alias.strip():
        return alias.strip()
    return cfg.get("model_id", "")


def _summarize_selected_regressors(selected_cols: list[str], groups: list[dict]) -> list[str]:
    """Collapse fully-selected hidden families into a single group key."""
    selected_set = set(selected_cols)
    consumed: set[str] = set()
    summary: list[str] = []

    for group in groups:
        members = list(group.get("toggle_members") or group.get("members", {}).values())
        if not members:
            continue
        if all(member in consumed for member in members):
            continue
        if all(member in selected_set for member in members):
            summary.append(str(group.get("key", group.get("label", ""))))
            consumed.update(members)

    for col in selected_cols:
        if col not in consumed:
            summary.append(col)

    return summary


def _format_emission_regressor_summary(task: str, emission_cols: list[str]) -> str:
    try:
        groups = get_adapter(task).build_emission_groups(list(emission_cols))
    except Exception:
        groups = build_selector_groups(list(emission_cols), [])
    return ", ".join(_summarize_selected_regressors(emission_cols, groups))


def _get_K_from_config(cfg: dict) -> Any:
    """Extract a concise K representation from a config dict."""
    if cfg.get("K"):
        return cfg["K"]
    if "K_list" in cfg and cfg["K_list"]:
        klist = cfg["K_list"]
        return klist[0] if len(klist) == 1 else f"{klist[0]}–{klist[-1]}"
    return "?"


@dataclass
class model_cfg:
    task: str = "MCDR"
    model_type: str = "glmhmm"
    existing: str | None = None
    alias: str = ""
    subjects: list[str] = field(default_factory=list)
    K: int = 2
    tau: int = 50
    cv_mode: str = "none"
    cv_repeats: int = 5
    condition_filter: str = "all"
    lapse_mode: str = "none"
    lapse_max: float = 0.2
    emission_cols: list[str] = field(default_factory=list)
    transition_cols: list[str] = field(default_factory=list)
    frozen_emissions: dict[str, Any] = field(default_factory=dict)
    run_fit_clicks: int = 0

    @property
    def lapse(self) -> bool:
        return str(self.lapse_mode) != "none"

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> "model_cfg":
        existing = value.get("existing_model")
        return cls(
            task=str(value.get("task", "MCDR")),
            model_type=str(value.get("model_type", "glmhmm")),
            existing=None if existing in ("", "__default__") else existing,
            alias=str(value.get("alias", "")),
            subjects=list(value.get("subjects", [])),
            K=int(value.get("K", 2)),
            tau=int(value.get("tau", 50)),
            cv_mode=_normalize_cv_mode(value.get("cv_mode", "none")),
            cv_repeats=int(value.get("cv_repeats", 5)),
            condition_filter=str(value.get("condition_filter", "all")),
            lapse_mode=str(value.get("lapse_mode", "none")),
            lapse_max=float(value.get("lapse_max", 0.2)),
            emission_cols=list(value.get("emission_cols", [])),
            transition_cols=list(value.get("transition_cols", [])),
            frozen_emissions=serialize_frozen_emissions(value.get("frozen_emissions", {})),
            run_fit_clicks=int(value.get("run_fit_clicks", 0)),
        )


# ── Widget ────────────────────────────────────────────────────────────────────

class ModelManagerWidget(anywidget.AnyWidget):
    _esm = _ASSET_DIR / "widget.js"
    _css = _ASSET_DIR / "widget.css"

    # ── traitlets ─────────────────────────────────────────────────────────────
    ui_mode    = traitlets.Unicode("new").tag(sync=True)      # "new" | "load"
    model_type = traitlets.Unicode("glmhmm").tag(sync=True)  # "glm" | "glmhmm" | "glmhmmt"
    task       = traitlets.Unicode("MCDR").tag(sync=True)
    task_options = traitlets.List(traitlets.Dict()).tag(sync=True)
    task_discovery_message = traitlets.Unicode("").tag(sync=True)
    is_2afc    = traitlets.Bool(False).tag(sync=True)

    existing_models      = traitlets.List(traitlets.Unicode()).tag(sync=True)
    existing_models_info = traitlets.List(traitlets.Dict()).tag(sync=True)
    existing_model       = traitlets.Unicode("").tag(sync=True)

    alias = traitlets.Unicode("").tag(sync=True)
    alias_error = traitlets.Unicode("").tag(sync=True)
    alias_status = traitlets.Unicode("").tag(sync=True)
    saved_model_name = traitlets.Unicode("").tag(sync=True)

    subjects_list = traitlets.List(traitlets.Unicode()).tag(sync=True)
    subjects      = traitlets.List(traitlets.Unicode()).tag(sync=True)

    k_options = traitlets.List(traitlets.Int(), default_value=[2, 3, 4, 5, 6]).tag(sync=True)
    K         = traitlets.Int(2).tag(sync=True)

    tau       = traitlets.Int(50).tag(sync=True)
    cv_mode   = traitlets.Unicode("none").tag(sync=True)
    cv_repeats = traitlets.Int(5).tag(sync=True)
    condition_filter_options = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    condition_filter = traitlets.Unicode("all").tag(sync=True)
    show_condition_filter = traitlets.Bool(False).tag(sync=True)

    lapse_mode = traitlets.Unicode("none").tag(sync=True)
    lapse_max = traitlets.Float(0.2).tag(sync=True)

    emission_cols_options = traitlets.List(traitlets.Unicode()).tag(sync=True)
    emission_cols         = traitlets.List(traitlets.Unicode()).tag(sync=True)
    emission_groups       = traitlets.List(traitlets.Dict()).tag(sync=True)
    frozen_emissions      = traitlets.Dict(default_value={}).tag(sync=True)

    transition_cols_options = traitlets.List(traitlets.Unicode()).tag(sync=True)
    transition_cols         = traitlets.List(traitlets.Unicode()).tag(sync=True)
    transition_groups       = traitlets.List(traitlets.Dict()).tag(sync=True)

    is_running = traitlets.Bool(False).tag(sync=True)

    run_fit_clicks   = traitlets.Int(0).tag(sync=True)
    save_alias_clicks = traitlets.Int(0).tag(sync=True)
    delete_model_name = traitlets.Unicode("").tag(sync=True)
    delete_model_clicks = traitlets.Int(0).tag(sync=True)
    command = traitlets.Unicode("").tag(sync=True)
    command_payload = traitlets.Dict(default_value={}).tag(sync=True)
    command_nonce = traitlets.Int(0).tag(sync=True)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_options()

    @property
    def model_cfg(self) -> model_cfg:
        return model_cfg.from_value(
            {
                "task": self.task,
                "model_type": self.model_type,
                "existing_model": self.existing_model,
                "alias": self.alias,
                "subjects": list(self.subjects),
                "K": int(self.K),
                "tau": int(self.tau),
                "cv_mode": self.cv_mode,
                "cv_repeats": int(self.cv_repeats),
                "condition_filter": self.condition_filter,
                "lapse_mode": str(self.lapse_mode),
                "lapse_max": float(self.lapse_max),
                "emission_cols": list(self.emission_cols),
                "transition_cols": list(self.transition_cols),
                "frozen_emissions": serialize_frozen_emissions(self.frozen_emissions),
                "run_fit_clicks": int(self.run_fit_clicks),
            }
        )

    # ── observers ─────────────────────────────────────────────────────────────

    @traitlets.observe("task")
    def _on_task_change(self, change):
        # Different task → reset all task-dependent selections to the new defaults.
        self.existing_model = ""
        self.subjects = []
        self.emission_cols = []
        self.transition_cols = []
        self.frozen_emissions = {}
        self.alias_error = ""
        self.alias_status = ""
        self.saved_model_name = ""
        self.alias = ""
        self.condition_filter = "all"
        self._apply_default_state()
        self._update_options()

    @traitlets.observe("model_type")
    def _on_model_type_change(self, change):
        # Same task, different model type; regressors are shared but refresh groups
        self.alias_error = ""
        self.alias_status = ""
        self.saved_model_name = ""
        self.alias = ""
        self.frozen_emissions = {}
        if self.model_type == "glmhmmt":
            self.condition_filter = "all"
        self._update_options()

    @traitlets.observe("alias")
    def _on_alias_change(self, change):
        if change["old"] != change["new"]:
            self.alias_error = ""
            self.alias_status = ""

    @traitlets.observe("K", "emission_cols", "frozen_emissions")
    def _on_frozen_inputs_change(self, change):
        if self.model_type == "glm":
            if self.frozen_emissions:
                self.frozen_emissions = {}
            return

        pruned = prune_frozen_emissions(self.frozen_emissions, int(self.K), list(self.emission_cols))
        if pruned != self.frozen_emissions:
            self.frozen_emissions = pruned

    @traitlets.observe("existing_model")
    def _on_existing_model_change(self, change):
        selected = change["new"]
        if not selected:
            return
        if selected == "__default__":
            self._apply_default_state()
            return
        found = self._find_model_dir(selected)
        if found is None:
            return
        _, cfg = found
        self._apply_config_to_state(cfg)

    @traitlets.observe("save_alias_clicks")
    def _on_save_alias_click(self, change):
        if change["new"] <= change["old"]:
            return
        self._save_named_model()

    @traitlets.observe("delete_model_clicks")
    def _on_delete_model_click(self, change):
        if change["new"] <= change["old"]:
            return
        self._delete_named_model()

    @traitlets.observe("command_nonce")
    def _on_command(self, change):
        cmd = self.command
        payload = self.command_payload or {}

        try:
            if cmd == "save_alias":
                alias = str(payload.get("alias", self.alias))
                self.alias = alias
                self._save_named_model()
            elif cmd == "delete_model":
                self.delete_model_name = str(payload.get("name", self.delete_model_name))
                self._delete_named_model()
            elif cmd == "run_fit":
                self.run_fit_clicks += 1
            elif cmd:
                self.alias_error = f"Unknown command: {cmd}"
        finally:
            self.command = ""
            self.command_payload = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_config_to_state(self, cfg: dict) -> None:
        """Write a config dict's values into the widget traitlets."""
        display_name = _get_display_name(cfg)
        if "emission_cols" in cfg:
            self.emission_cols = cfg["emission_cols"]
        if "transition_cols" in cfg:
            self.transition_cols = cfg["transition_cols"]
        if "subjects" in cfg:
            self.subjects = sorted(cfg["subjects"], key=str)
        if "tau" in cfg:
            self.tau = int(cfg["tau"])
        self.cv_mode = _normalize_cv_mode(cfg.get("cv_mode", "none"))
        self.cv_repeats = int(cfg.get("cv_repeats", 5))
        self.condition_filter = str(cfg.get("condition_filter", "all"))
        k = _get_K_from_config(cfg)
        if isinstance(k, int):
            self.K = k
        self.frozen_emissions = serialize_frozen_emissions(cfg.get("frozen_emissions", {}))
        if "lapse_mode" in cfg:
            self.lapse_mode = str(cfg["lapse_mode"])
        if "lapse_max" in cfg:
            self.lapse_max = float(cfg["lapse_max"])
        self.alias = display_name
        self.saved_model_name = display_name
        self.alias_error = ""
        self.alias_status = ""
        self._refresh_groups()

    def _apply_default_state(self) -> None:
        """Reset widget traitlets to the adapter's default configuration."""
        try:
            adapter = get_adapter(self.task)
            df_all  = adapter.read_dataset()
            df_all  = adapter.subject_filter(df_all)
            subjects = sorted(df_all["subject"].unique().to_list(), key=str)
            self.subjects  = subjects
            self.K         = 2
            self.tau       = 50
            self.cv_mode   = "none"
            self.cv_repeats = 5
            self.condition_filter = "all"
            self.lapse_mode = "none"
            self.lapse_max = 0.2
            is_2afc = adapter.num_classes == 2
            available_ecols = adapter.available_emission_cols(df_all)
            default_ecols = adapter.default_emission_cols(df_all)
            self.emission_cols_options = available_ecols
            self.emission_cols = list(default_ecols)
            self.transition_cols_options = adapter.available_transition_cols()
            self.transition_cols = adapter.default_transition_cols()
            self.frozen_emissions = {}
            self.alias = ""
            self.saved_model_name = ""
            self.alias_error = ""
            self.alias_status = ""
            self._refresh_groups()
        except Exception as e:
            print(f"Error applying default state for task {self.task}: {e}")

    def _fits_path(self) -> Path:
        return _paths().RESULTS / "fits" / self.task / self.model_type

    def _find_model_dir(self, display_name: str) -> tuple[Path, dict[str, Any]] | None:
        fits_path = self._fits_path()
        if remote_fits_enabled():
            cfg = load_model_config(
                task_name=self.task,
                model_kind=self.model_type,
                alias=display_name,
                local_root=fits_path,
            )
            if cfg:
                return fits_path / display_name, cfg
            return None
        if not fits_path.exists():
            return None
        for d in fits_path.iterdir():
            if not (d.is_dir() and (d / "config.json").exists()):
                continue
            try:
                cfg = json.loads((d / "config.json").read_text())
            except Exception:
                continue
            if _is_displayable(cfg) and _get_display_name(cfg) == display_name:
                return d, cfg
        return None

    def _is_valid_model_name(self, name: str) -> bool:
        if not name or name == "__default__":
            return False
        return Path(name).name == name and "/" not in name and "\\" not in name

    def _build_current_config(self, model_name: str) -> dict[str, Any]:
        cv_mode = _normalize_cv_mode(self.cv_mode)
        cfg: dict[str, Any] = {
            "task": self.task,
            "model_id": model_name,
            "alias": model_name,
            "subjects": list(self.subjects),
            "tau": int(self.tau),
            "cv_mode": cv_mode,
            "cv_repeats": int(self.cv_repeats) if cv_mode != "none" else 0,
            "emission_cols": list(self.emission_cols),
        }
        if self._supports_condition_filter():
            cfg["condition_filter"] = self.condition_filter
        if self.model_type == "glm":
            cfg["lapse_mode"] = str(self.lapse_mode)
            cfg["lapse_max"] = float(self.lapse_max)
        else:
            cfg["K_list"] = [int(self.K)]
            cfg["frozen_emissions"] = serialize_frozen_emissions(self.frozen_emissions)
        if self.model_type == "glmhmmt":
            cfg["transition_cols"] = list(self.transition_cols)
        return cfg

    def _saved_config_matches_current(self, model_dir: Path) -> bool:
        cfg_path = model_dir / "config.json"
        if not cfg_path.exists():
            return False
        try:
            saved = json.loads(cfg_path.read_text())
        except Exception:
            return False

        if saved.get("task") != self.task:
            return False
        if saved.get("subjects", []) != list(self.subjects):
            return False
        if int(saved.get("tau", -1)) != int(self.tau):
            return False
        if _normalize_cv_mode(saved.get("cv_mode", "none")) != _normalize_cv_mode(self.cv_mode):
            return False
        if _normalize_cv_mode(self.cv_mode) != "none" and int(saved.get("cv_repeats", 0)) != int(self.cv_repeats):
            return False
        if self._supports_condition_filter() and str(saved.get("condition_filter", "all")) != self.condition_filter:
            return False
        if saved.get("emission_cols", []) != list(self.emission_cols):
            return False

        if self.model_type == "glm":
            return (
                str(saved.get("lapse_mode", "none")) == str(self.lapse_mode)
                and float(saved.get("lapse_max", 0.2)) == float(self.lapse_max)
            )

        saved_k = _get_K_from_config(saved)
        if not isinstance(saved_k, int) or saved_k != int(self.K):
            return False

        if serialize_frozen_emissions(saved.get("frozen_emissions", {})) != serialize_frozen_emissions(self.frozen_emissions):
            return False

        if self.model_type == "glmhmmt":
            return saved.get("transition_cols", []) == list(self.transition_cols)

        return True

    def _save_named_model(self) -> None:
        target_name = self.alias.strip()
        self.alias_error = ""
        self.alias_status = ""

        if not self._is_valid_model_name(target_name):
            self.alias_error = "Enter a valid model name."
            return

        fits_path = self._fits_path()
        fits_path.mkdir(parents=True, exist_ok=True)

        current_name = self.saved_model_name.strip()
        current_dir = fits_path / current_name if current_name else None
        target_dir = fits_path / target_name

        if target_name != current_name and target_dir.exists():
            self.alias_error = "Duplicate name."
            return

        try:
            can_rename_current = (
                current_dir is not None
                and current_dir.exists()
                and current_dir != target_dir
                and self._saved_config_matches_current(current_dir)
            )
            if can_rename_current:
                current_dir.rename(target_dir)
            else:
                target_dir.mkdir(parents=True, exist_ok=True)

            cfg = self._build_current_config(target_name)
            (target_dir / "config.json").write_text(json.dumps(cfg, indent=4))
        except Exception as e:
            self.alias_error = f"Could not save model: {e}"
            return

        self.saved_model_name = target_name
        self.alias = target_name
        self._update_options()
        if target_name in self.existing_models:
            self.existing_model = target_name
        self.alias_status = f"Saved as {target_name}"

    def _delete_named_model(self) -> None:
        target_name = self.delete_model_name.strip()
        self.alias_error = ""
        self.alias_status = ""

        if not target_name or target_name == "__default__":
            self.alias_error = "Default cannot be deleted."
            return

        found = self._find_model_dir(target_name)
        if found is None:
            self.alias_error = f"Model {target_name} was not found."
            self._update_options()
            return

        model_dir, _ = found
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            self.alias_error = f"Could not delete model: {e}"
            return

        deleted_current = self.saved_model_name == target_name or self.existing_model == target_name
        self._update_options()
        if deleted_current:
            self._apply_default_state()
            self.existing_model = ""
        self.alias_status = f"Deleted {target_name}"

    def _refresh_groups(self) -> None:
        """Rebuild emission_groups / transition_groups from current *_options traits."""
        try:
            adapter = get_adapter(self.task)
        except Exception:
            self.emission_groups = build_selector_groups(list(self.emission_cols_options), [])
            self.transition_groups = build_selector_groups(list(self.transition_cols_options), [])
            return

        self.emission_groups = adapter.build_emission_groups(list(self.emission_cols_options))
        self.transition_groups = adapter.build_transition_groups(list(self.transition_cols_options))

    def _clear_adapter_state(self) -> None:
        self.is_2afc = False
        self.show_condition_filter = False
        self.condition_filter_options = []
        self.subjects_list = []
        self.subjects = []
        self.emission_cols_options = []
        self.emission_cols = []
        self.transition_cols_options = []
        self.transition_cols = []
        self.existing_models = []
        self.existing_models_info = []
        self.existing_model = ""
        self._refresh_groups()

    def _refresh_task_options(self) -> str:
        self.task_options = get_task_options()
        if not self.task_options:
            self.task_discovery_message = (
                "No task adapters were found in the configured adapter folders."
            )
            return "empty"

        self.task_discovery_message = ""
        valid_tasks = {opt["value"] for opt in self.task_options}
        if self.task not in valid_tasks:
            self.task = self.task_options[0]["value"]
            return "changed"
        return "ready"

    def _supports_condition_filter(self) -> bool:
        if self.model_type == "glmhmmt":
            return False
        try:
            return bool(get_adapter(self.task).condition_filter_options())
        except Exception:
            return False

    def _build_model_info_list(
        self, fits_path: Path, default_info: dict
    ) -> tuple[list[str], list[dict]]:
        """Scan *fits_path* for displayable named models.

        Returns ``(display_name_list, info_dict_list)`` sorted by name,
        always prepended with the synthetic Default row.
        """
        names: list[str]      = []
        info_list: list[dict] = []

        aliases = model_aliases_for_kind(
            task_name=self.task,
            model_kind=self.model_type,
            local_root=fits_path,
        )
        for display_name in aliases:
            cfg = load_model_config(
                task_name=self.task,
                model_kind=self.model_type,
                alias=display_name,
                local_root=fits_path,
            )
            if not cfg or not _is_displayable(cfg):
                continue
            model_dir = fits_path / display_name
            n_subjects = _count_fitted_subjects(model_dir) if model_dir.exists() else 0
            cv_mode = _normalize_cv_mode(cfg.get("cv_mode", "none"))
            cv_repeats = int(cfg.get("cv_repeats", 0))
            cv_label = f"{cv_mode}×{cv_repeats}" if cv_mode != "none" and cv_repeats > 0 else cv_mode
            info_list.append({
                "id":       display_name,
                "name":     display_name,
                "subjects": n_subjects if n_subjects > 0 else len(cfg.get("subjects", [])),
                "K":        _get_K_from_config(cfg),
                "tau":      cfg.get("tau", ""),
                "cv":       cv_label,
                "regressors": _format_emission_regressor_summary(
                    self.task,
                    list(cfg.get("emission_cols", [])),
                ),
                "transition_regressors": ", ".join(cfg.get("transition_cols", [])),
            })
            names.append(display_name)

        info_list.sort(key=lambda x: x["name"])
        names.sort()
        return (["__default__"] + names, [default_info] + info_list)

    def _update_options(self) -> None:
        """Refresh all dynamic options from the adapter and the fits directory."""
        refresh_status = self._refresh_task_options()
        if refresh_status == "empty":
            self._clear_adapter_state()
            return
        if refresh_status == "changed":
            return

        # ── adapter-derived options ───────────────────────────────────────────
        default_info: dict = {
            "id": "__default__", "name": "Default",
            "subjects": "?", "K": 2, "tau": 50, "cv": "none", "regressors": "", "transition_regressors": "",
        }
        try:
            adapter = get_adapter(self.task)
            self.is_2afc = adapter.num_classes == 2
            self.show_condition_filter = self._supports_condition_filter()
            adapter_filter_options = list(adapter.condition_filter_options())
            self.condition_filter_options = adapter_filter_options if self.show_condition_filter else []
            if not self.show_condition_filter:
                self.condition_filter = "all"
            elif self.condition_filter not in adapter_filter_options:
                self.condition_filter = "all"

            df_all   = adapter.read_dataset()
            df_all   = adapter.subject_filter(df_all)
            subjects = sorted(df_all["subject"].unique().to_list(), key=str)

            self.subjects_list = subjects
            if not self.subjects:
                self.subjects = subjects

            available_ecols = adapter.available_emission_cols(df_all)
            default_ecols = adapter.default_emission_cols(df_all)
            self.emission_cols_options = available_ecols
            if not self.emission_cols:
                self.emission_cols = list(default_ecols)

            tcols = adapter.default_transition_cols()
            self.transition_cols_options = adapter.available_transition_cols()
            if not self.transition_cols:
                self.transition_cols = tcols

            default_ecols = list(default_ecols)
            default_info = {
                "id":         "__default__",
                "name":       "Default",
                "subjects":   len(subjects),
                "K":          2,
                "tau":        50,
                "cv":         "none",
                "regressors": _format_emission_regressor_summary(
                    self.task,
                    list(default_ecols),
                ),
                "transition_regressors": ", ".join(tcols),
            }
        except Exception as e:
            self.task_discovery_message = str(e)
            print(f"Error loading adapter for task {self.task}: {e}")

        # ── saved-model list ──────────────────────────────────────────────────
        fits_path = _paths().RESULTS / "fits" / self.task / self.model_type
        display_names, info_list = self._build_model_info_list(fits_path, default_info)
        self.existing_models      = display_names
        self.existing_models_info = info_list

        # Drop stale selection if the selected model no longer exists
        if self.existing_model and self.existing_model not in display_names:
            self.existing_model = ""

        self._refresh_groups()
