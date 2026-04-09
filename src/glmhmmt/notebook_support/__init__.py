"""Shared helpers for repository marimo notebooks."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "wrap_anywidget",
    "apply_state_tweak_to_trial_df",
    "apply_state_tweak_to_view",
    "build_editor_payload",
    "CoefficientEditorWidget",
    "make_plot_saver",
    "ModelManagerWidget",
    "model_cfg",
]

_LAZY_IMPORTS = {
    "wrap_anywidget": ("glmhmmt.notebook_support.anywidget_compat", "wrap_anywidget"),
    "apply_state_tweak_to_trial_df": (
        "glmhmmt.notebook_support.coefficient_editor_utils",
        "apply_state_tweak_to_trial_df",
    ),
    "apply_state_tweak_to_view": (
        "glmhmmt.notebook_support.coefficient_editor_utils",
        "apply_state_tweak_to_view",
    ),
    "build_editor_payload": (
        "glmhmmt.notebook_support.coefficient_editor_utils",
        "build_editor_payload",
    ),
    "CoefficientEditorWidget": (
        "glmhmmt.notebook_support.coefficient_editor_widget",
        "CoefficientEditorWidget",
    ),
    "make_plot_saver": ("glmhmmt.notebook_support.figure_save_utils", "make_plot_saver"),
    "ModelManagerWidget": ("glmhmmt.notebook_support.model_manager", "ModelManagerWidget"),
    "model_cfg": ("glmhmmt.notebook_support.model_manager", "model_cfg"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
