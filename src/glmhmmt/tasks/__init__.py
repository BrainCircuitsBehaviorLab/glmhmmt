"""Task adapters for the GLM-HMM pipeline.

Each adapter encapsulates all task-specific knowledge so that fit scripts
and analysis notebooks can be written once and work for any task.

Usage
-----
    from glmhmmt.tasks import get_adapter

    adapter = get_adapter("mcdr")   # or "two_afc" / "2AFC" / "MCDR"
    df      = adapter.read_dataset()
    df      = adapter.subject_filter(df)
    y, X, U, names = adapter.load_subject(df_sub, tau=50.0)
    plots   = adapter.get_plots()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from importlib.metadata import entry_points
import os
from pathlib import Path
import pkgutil
import sys
from typing import List, Tuple, Dict, Any
import warnings

import types
import polars as pl

from glmhmmt.runtime import get_config_path, get_data_dir, load_app_config


ENV_TASK_PATHS = "GLMHMMT_TASK_PATHS"


class TaskAdapter(ABC):
    """Abstract base for task-specific configuration & data loading."""

    # ── class-level attributes (override in subclass) ──────────────────────
    task_key: str = NotImplemented      # canonical UI / CLI task name
    task_label: str = NotImplemented    # human-readable task label
    num_classes: int = NotImplemented   # 2 or 3
    data_file: str = NotImplemented     # filename under the runtime dataset root
    sort_col: str = NotImplemented      # trial ordering column
    session_col: str = NotImplemented   # session identifier column

    # ── data preparation ────────────────────────────────────────────────────

    @abstractmethod
    def subject_filter(self, df: Any) -> Any:
        """Apply task-specific subject/session filtering to the full DataFrame."""

    @abstractmethod
    def build_feature_df(self, df_sub: Any, tau: float = 50.0) -> Any:
        """Return a task-owned trial dataframe with design-matrix columns.

        This dataframe is the explicit contract between each task and the
        shared fitting code. It should contain the task's raw behavioral
        columns plus any derived emission / transition regressors needed to
        assemble ``X`` and ``U``.
        """

    @abstractmethod
    def load_subject(
        self,
        df_sub: Any,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` for one subject's DataFrame slice.

        ``U`` must always be returned (use an empty array for tasks that lack
        transition features — shape ``(T, 0)``).
        ``names`` must contain ``"X_cols"`` and ``"U_cols"``.
        """

    @abstractmethod
    def build_design_matrices(
        self,
        feature_df: Any,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` from a task-owned feature dataframe."""

    # ── column defaults  ────────────────────────────────────────────────────

    @abstractmethod
    def default_emission_cols(self, df: Any = None) -> List[str]:
        """Ordered list of default emission regressors.

        Adapters may inspect ``df`` to include dynamic task-owned columns such
        as one-hot stimulus bins or frame-level regressors.
        """

    @abstractmethod
    def default_transition_cols(self) -> List[str]:
        """Ordered list of transition regressor names for UI initialisation."""

    def available_emission_cols(self, df: Any = None) -> List[str]:
        """Ordered list of selectable emission regressors."""
        return self.default_emission_cols(df)

    def available_transition_cols(self) -> List[str]:
        """Ordered list of selectable transition regressors."""
        return self.default_transition_cols()

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df: Any = None,
    ) -> Dict[str, List[str]]:
        """Resolve the selected design variable names without rebuilding arrays.

        This is a cheap metadata-only helper for notebook loading paths that
        need fallback ``X_cols`` / ``U_cols`` for older fit artifacts.
        Adapters with dynamic expansion can override it.
        """
        x_cols = (
            list(emission_cols)
            if emission_cols is not None
            else list(self.default_emission_cols(df))
        )
        u_cols = list(transition_cols) if transition_cols is not None else list(self.default_transition_cols())

        available_ecols = self.available_emission_cols(df)
        bad_e = [c for c in x_cols if c not in available_ecols]
        bad_u = [c for c in u_cols if c not in self.available_transition_cols()]
        if bad_e:
            raise ValueError(
                f"Unknown emission_cols: {bad_e}. Available: {available_ecols}"
            )
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {self.available_transition_cols()}"
            )
        return {"X_cols": x_cols, "U_cols": u_cols}

    def sf_cols(self, df: Any) -> List[str]:
        """Optional dynamic stimulus-frame columns for binary tasks."""
        return []

    def cv_balance_labels(self, feature_df: Any):
        """Return per-trial labels used for CV balancing, or ``None`` if unsupported."""
        return None

    def dataset_path(self) -> Path:
        """Return the on-disk dataset path for this task."""
        return get_data_dir() / self.data_file

    def read_dataset(self) -> pl.DataFrame:
        """Load and return the full task dataset."""
        return pl.read_parquet(self.dataset_path())

    # ── plot module ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_plots(self) -> types.ModuleType:
        """Return the task-specific plots module."""

    @property
    def choice_labels(self) -> list[str]:
        """Ordered human-readable labels for choice classes."""
        return [f"Choice {idx}" for idx in range(self.num_classes)]

    @property
    def probability_columns(self) -> list[str]:
        """Trial-level probability column names aligned with choice_labels."""
        if self.num_classes == 2:
            return ["pL", "pR"]
        if self.num_classes == 3:
            return ["pL", "pC", "pR"]
        return [f"p_{idx}" for idx in range(self.num_classes)]

    # ── column mapping  ──────────────────────────────────────────────────────

    @property
    @abstractmethod
    def behavioral_cols(self) -> Dict[str, str]:
        """Mapping from canonical column names to actual column names.

        Required canonical keys and their semantics:
            ``"trial_idx"``   — global, monotonically increasing trial index
            ``"trial"``       — within-session trial number (may equal trial_idx)
            ``"session"``     — session identifier
            ``"stimulus"``    — integer correct-class index (0/1/2 for L/C/R)
            ``"response"``    — integer chosen class
            ``"performance"`` — 0/1 trial outcome
        """

    # ── state labelling ──────────────────────────────────────────────────────

    @abstractmethod
    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """Return ``(state_labels, state_order)`` for all subjects.

        state_labels : {subj: {state_idx: label_str}}
        state_order  : {subj: [state_idx, ...]}  sorted by engagement rank desc
        """
        ...

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        """Return correct class index per trial as int array of shape (T,).

        Must return values in {0, ..., C-1}. Invalid/ambiguous trials may be -1.
        """
        raise NotImplementedError


# ── registry & factory ─────────────────────────────────────────────────────

_REGISTRY: dict[str, type[TaskAdapter]] = {}
_ENTRY_POINTS_LOADED = False
_LOCAL_TASK_PATHS_LOADED: set[Path] = set()

def _register(keys: list[str]):
    """Class decorator that registers an adapter under one or more keys."""
    def decorator(cls: type[TaskAdapter]) -> type[TaskAdapter]:
        for k in keys:
            key = k.lower()
            existing = _REGISTRY.get(key)
            if existing is not None and existing is not cls:
                warnings.warn(
                    f"Task key {k!r} already registered by "
                    f"{existing.__module__}.{existing.__name__}; "
                    f"overriding with {cls.__module__}.{cls.__name__}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            _REGISTRY[key] = cls
        return cls
    return decorator


def _resolve_task_dir(raw: object, *, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _configured_task_dirs() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    config_path = get_config_path()
    config_base = config_path.parent if config_path is not None else Path.cwd()

    env_value = os.environ.get(ENV_TASK_PATHS)
    if env_value:
        for raw in env_value.split(os.pathsep):
            task_dir = _resolve_task_dir(raw, base_dir=Path.cwd())
            if task_dir is not None and task_dir not in seen:
                seen.add(task_dir)
                candidates.append(task_dir)

    try:
        app_cfg = load_app_config()
    except Exception as exc:
        warnings.warn(
            f"Failed to load task-path configuration: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        app_cfg = {}

    plugins_cfg = app_cfg.get("plugins", {})
    raw_task_paths = plugins_cfg.get("task_paths", []) if isinstance(plugins_cfg, dict) else []
    if isinstance(raw_task_paths, (str, Path)):
        raw_task_paths = [raw_task_paths]
    if isinstance(raw_task_paths, list):
        for raw in raw_task_paths:
            task_dir = _resolve_task_dir(raw, base_dir=config_base)
            if task_dir is not None and task_dir not in seen:
                seen.add(task_dir)
                candidates.append(task_dir)

    cwd_task_dir = (Path.cwd() / "tasks").resolve()
    if cwd_task_dir not in seen:
        seen.add(cwd_task_dir)
        candidates.append(cwd_task_dir)

    return candidates


def _load_local_task_packages() -> None:
    for package_dir in _configured_task_dirs():
        if package_dir in _LOCAL_TASK_PATHS_LOADED:
            continue

        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            continue

        package_parent = str(package_dir.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)

        try:
            pkg = importlib.import_module("tasks")
        except Exception as exc:
            warnings.warn(
                f"Failed to import task package from {package_dir}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if not getattr(pkg, "GLMHMMT_TASK_PACKAGE", False):
            continue

        pkg_file = getattr(pkg, "__file__", None)
        if pkg_file is not None and Path(pkg_file).resolve() != init_file.resolve():
            warnings.warn(
                f"Skipping task directory {package_dir}: Python already loaded "
                f"'tasks' from {Path(pkg_file).resolve().parent}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        _LOCAL_TASK_PATHS_LOADED.add(package_dir)

        for _, module_name, ispkg in pkgutil.iter_modules(pkg.__path__):
            if ispkg or module_name.startswith("_") or module_name == "plots":
                continue
            full_name = f"tasks.{module_name}"
            try:
                importlib.import_module(full_name)
            except Exception as exc:
                warnings.warn(
                    f"Failed to import task module {full_name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def _load_entry_point_adapters() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return

    _ENTRY_POINTS_LOADED = True
    for ep in entry_points(group="glmhmmt.tasks"):
        try:
            ep.load()
        except Exception as exc:
            warnings.warn(
                f"Failed to load glmhmmt task plugin {ep.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def get_adapter(task: str) -> TaskAdapter:
    """Return an instantiated TaskAdapter for *task*.

    Accepted values (case-insensitive):
        ``"mcdr"``, ``"MCDR"``          → MCDRAdapter
        ``"two_afc"``, ``"2afc"``,
        ``"2AFC"``, ``"two_AFC"``        → TwoAFCAdapter
        ``"nuo_auditory"``,
        ``"auditory_2afc"``              → NuoAuditoryAdapter
    """
    _load_entry_point_adapters()
    _load_local_task_packages()
    key = task.lower().replace("-", "_")
    cls = _REGISTRY.get(key)
    if cls is None:
        known = ", ".join(f'"{k}"' for k in _REGISTRY)
        raise ValueError(f"Unknown task {task!r}. Known tasks: {known}")
    return cls()


def get_task_options() -> list[dict[str, str]]:
    """Return unique task options for UI selectors.

    Each option is ``{"value": <task_key>, "label": <task_label>}``.
    """
    _load_entry_point_adapters()
    _load_local_task_packages()
    seen: dict[str, str] = {}
    for cls in dict.fromkeys(_REGISTRY.values()):
        task_key = getattr(cls, "task_key", None)
        task_label = getattr(cls, "task_label", None)
        if task_key and task_label and task_key not in seen:
            seen[task_key] = task_label
    return [{"value": key, "label": seen[key]} for key in seen]
