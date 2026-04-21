from __future__ import annotations

from importlib import import_module

__all__ = [
    "SoftmaxGLMHMM",
    "GLMFitResult",
    "fit_glm",
    "SubjectFitView",
    "build_views",
    "_LABEL_RANK",
    "_STATE_HEX",
    "build_trial_df",
    "build_emission_weights_df",
    "build_posterior_df",
    "build_session_deepdive_payload",
    "build_session_trajectories_payload",
    "build_state_accuracy_payload",
    "build_state_dwell_times_payload",
    "build_state_occupancy_payload",
    "build_state_posterior_count_payload",
    "TaskAdapter",
    "get_adapter",
    "get_task_options",
    "configure_paths",
    "get_runtime_paths",
    "init_project_config",
    "load_app_config",
]

_LAZY_IMPORTS = {
    "SoftmaxGLMHMM": ("glmhmmt.model", "SoftmaxGLMHMM"),
    "GLMFitResult": ("glmhmmt.glm", "GLMFitResult"),
    "fit_glm": ("glmhmmt.glm", "fit_glm"),
    "SubjectFitView": ("glmhmmt.views", "SubjectFitView"),
    "build_views": ("glmhmmt.views", "build_views"),
    "_LABEL_RANK": ("glmhmmt.views", "_LABEL_RANK"),
    "_STATE_HEX": ("glmhmmt.views", "_STATE_HEX"),
    "build_trial_df": ("glmhmmt.postprocess", "build_trial_df"),
    "build_emission_weights_df": ("glmhmmt.postprocess", "build_emission_weights_df"),
    "build_posterior_df": ("glmhmmt.postprocess", "build_posterior_df"),
    "build_session_deepdive_payload": ("glmhmmt.postprocess", "build_session_deepdive_payload"),
    "build_session_trajectories_payload": ("glmhmmt.postprocess", "build_session_trajectories_payload"),
    "build_state_accuracy_payload": ("glmhmmt.postprocess", "build_state_accuracy_payload"),
    "build_state_dwell_times_payload": ("glmhmmt.postprocess", "build_state_dwell_times_payload"),
    "build_state_occupancy_payload": ("glmhmmt.postprocess", "build_state_occupancy_payload"),
    "build_state_posterior_count_payload": ("glmhmmt.postprocess", "build_state_posterior_count_payload"),
    "TaskAdapter": ("glmhmmt.tasks", "TaskAdapter"),
    "get_adapter": ("glmhmmt.tasks", "get_adapter"),
    "get_task_options": ("glmhmmt.tasks", "get_task_options"),
    "configure_paths": ("glmhmmt.runtime", "configure_paths"),
    "get_runtime_paths": ("glmhmmt.runtime", "get_runtime_paths"),
    "init_project_config": ("glmhmmt.runtime", "init_project_config"),
    "load_app_config": ("glmhmmt.runtime", "load_app_config"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
