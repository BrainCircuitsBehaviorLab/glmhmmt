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
import hashlib
import importlib
from importlib.util import module_from_spec, spec_from_file_location
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
ENV_PLOT_PATHS = "GLMHMMT_PLOT_PATHS"


def build_selector_groups(available_cols: list[str], registry: list[dict]) -> list[dict]:
    """Filter *registry* to columns present in *available_cols*.

    Columns not covered by any registry entry are appended as solo N-only groups
    so they're always accessible.
    """
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


class TaskAdapter(ABC):
    """Abstract base for task-specific configuration & data loading."""

    # ── class-level attributes (override in subclass) ──────────────────────
    task_key: str = NotImplemented      # canonical UI / CLI task name
    task_label: str = NotImplemented    # human-readable task label
    num_classes: int = NotImplemented   # 2 or 3
    baseline_class_idx: int = 0          # implicit softmax reference class
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

    def build_emission_groups(self, available_cols: List[str]) -> list[dict]:
        """Return widget emission groups for the task selector."""
        return build_selector_groups(list(available_cols), [])

    def build_transition_groups(self, available_cols: List[str]) -> list[dict]:
        """Return widget transition groups for the task selector."""
        return build_selector_groups(list(available_cols), [])

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
_LOCAL_PLOT_MODULES_LOADED: dict[Path, tuple[types.ModuleType, tuple[int, int]]] = {}
_LOCAL_TASK_IMPORT_ROOT = "glmhmmt_local_tasks"
_LOCAL_PLOT_IMPORT_ROOT = "glmhmmt_local_plots"

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


def _normalize_path_list(raw: object) -> list[object]:
    if isinstance(raw, (str, Path)):
        return [raw]
    if isinstance(raw, list):
        return raw
    return []


def _configured_path_values(
    app_cfg: dict,
    *,
    keys: tuple[str, ...],
) -> tuple[list[object], bool]:
    values: list[object] = []
    explicit = False

    for section_name in ("plugins", "paths"):
        section = app_cfg.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key in section:
                explicit = True
                values.extend(_normalize_path_list(section.get(key)))

    return values, explicit


def _configured_dirs(
    *,
    env_var: str,
    config_keys: tuple[str, ...],
    default_dirnames: tuple[str, ...],
) -> tuple[list[Path], bool]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    scope_active = False

    config_path = get_config_path()
    config_base = config_path.parent if config_path is not None else Path.cwd()

    env_value = os.environ.get(env_var)
    if env_value is not None:
        scope_active = True
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

    raw_paths, explicit_in_config = _configured_path_values(app_cfg, keys=config_keys)
    if explicit_in_config:
        scope_active = True
    for raw in raw_paths:
        task_dir = _resolve_task_dir(raw, base_dir=config_base)
        if task_dir is not None and task_dir not in seen:
            seen.add(task_dir)
            candidates.append(task_dir)

    for dirname in default_dirnames:
        cwd_task_dir = (Path.cwd() / dirname).resolve()
        if cwd_task_dir.exists() and cwd_task_dir.is_dir():
            scope_active = True
            if cwd_task_dir not in seen:
                seen.add(cwd_task_dir)
                candidates.append(cwd_task_dir)

    return candidates, scope_active


def _configured_task_dirs() -> list[Path]:
    candidates, _ = _configured_dirs(
        env_var=ENV_TASK_PATHS,
        config_keys=("adapter_paths",),
        default_dirnames=("adapters", "tasks"),
    )
    return candidates


def _task_scope_is_local() -> bool:
    _, scope_active = _configured_dirs(
        env_var=ENV_TASK_PATHS,
        config_keys=("adapter_paths",),
        default_dirnames=("adapters", "tasks"),
    )
    return scope_active


def _configured_plot_dirs() -> list[Path]:
    candidates, _ = _configured_dirs(
        env_var=ENV_PLOT_PATHS,
        config_keys=("plot_paths",),
        default_dirnames=("plots",),
    )
    return candidates


def _active_local_plot_dirs() -> list[Path]:
    dirs: list[Path] = []
    for package_dir in _configured_plot_dirs():
        if package_dir.exists() and package_dir.is_dir():
            dirs.append(package_dir)
    return dirs


def _configured_task_dir_labels() -> list[str]:
    return [str(path) for path in _configured_task_dirs()]


def _no_local_adapter_message() -> str:
    configured = _configured_task_dir_labels()
    if configured:
        return (
            "No task adapters were found in the configured adapter directories: "
            + ", ".join(configured)
        )
    return (
        "No task adapters were found. Configure [plugins].adapter_paths in config.toml."
    )



def _path_is_within(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _active_local_task_dirs() -> list[Path]:
    dirs: list[Path] = []
    for package_dir in _configured_task_dirs():
        if package_dir.exists() and package_dir.is_dir():
            dirs.append(package_dir)
    return dirs


def _adapter_defined_in_dirs(cls: type[TaskAdapter], roots: list[Path]) -> bool:
    module = sys.modules.get(cls.__module__)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        resolved = Path(module_file).resolve()
    except OSError:
        return False
    return _path_is_within(resolved, roots)


def _load_module_from_path(
    module_path: Path,
    *,
    import_root: str,
    loaded_modules: dict[Path, tuple[types.ModuleType, tuple[int, int]]],
) -> types.ModuleType:
    resolved = module_path.resolve()
    stat = resolved.stat()
    signature = (stat.st_mtime_ns, stat.st_size)

    cached = loaded_modules.get(resolved)
    if cached is not None:
        cached_module, cached_signature = cached
        if cached_signature == signature:
            return cached_module

    digest = hashlib.sha1(
        f"{resolved}:{signature[0]}:{signature[1]}".encode("utf-8")
    ).hexdigest()[:12]
    module_name = f"{import_root}.{resolved.stem}_{digest}"
    spec = spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {resolved}.")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    loaded_modules[resolved] = (module, signature)
    return module


def _plot_module_candidates(
    *,
    adapter_module_name: str | None,
    task_key: str | None,
) -> list[str]:
    candidates: list[str] = []
    for raw in (
        Path(adapter_module_name).name if adapter_module_name else None,
        adapter_module_name.rsplit(".", 1)[-1] if adapter_module_name else None,
        task_key,
        task_key.lower() if task_key else None,
        task_key.replace("-", "_") if task_key else None,
        task_key.lower().replace("-", "_") if task_key else None,
    ):
        if raw and raw not in candidates:
            candidates.append(raw)
    return candidates


def resolve_plots_module(
    *,
    adapter_module_name: str | None = None,
    task_key: str | None = None,
) -> types.ModuleType:
    """Resolve a task-owned plot module from configured plot directories."""
    candidates = _plot_module_candidates(
        adapter_module_name=adapter_module_name,
        task_key=task_key,
    )
    for plot_dir in _active_local_plot_dirs():
        for module_name in candidates:
            plot_file = plot_dir / f"{module_name}.py"
            if plot_file.exists():
                return _load_module_from_path(
                    plot_file,
                    import_root=_LOCAL_PLOT_IMPORT_ROOT,
                    loaded_modules=_LOCAL_PLOT_MODULES_LOADED,
                )

    configured_dirs = [str(path) for path in _configured_plot_dirs()]
    raise ImportError(
        "Could not resolve a plot module for "
        f"task_key={task_key!r}, adapter_module_name={adapter_module_name!r}. "
        f"Checked plot directories: {configured_dirs or ['<none configured>']}."
    )


def _load_local_task_packages() -> None:
    for package_dir in _configured_task_dirs():
        if package_dir in _LOCAL_TASK_PATHS_LOADED:
            continue

        package_parent = str(package_dir.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)

        init_file = package_dir / "__init__.py"
        if init_file.exists():
            package_name = f"{_LOCAL_TASK_IMPORT_ROOT}.{package_dir.name}"
            spec = spec_from_file_location(
                package_name,
                init_file,
                submodule_search_locations=[str(package_dir)],
            )
            if spec is None or spec.loader is None:
                warnings.warn(
                    f"Failed to build import spec for task package {package_dir}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            pkg = module_from_spec(spec)
            sys.modules[package_name] = pkg
            try:
                spec.loader.exec_module(pkg)
            except Exception as exc:
                sys.modules.pop(package_name, None)
                warnings.warn(
                    f"Failed to import task package from {package_dir}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            if not getattr(pkg, "GLMHMMT_TASK_PACKAGE", False):
                continue

            iter_paths = list(getattr(pkg, "__path__", [str(package_dir)]))
            prefix = f"{package_name}."
        else:
            iter_paths = [str(package_dir)]
            prefix = ""

        _LOCAL_TASK_PATHS_LOADED.add(package_dir)

        for _, module_name, ispkg in pkgutil.iter_modules(iter_paths):
            if ispkg or module_name.startswith("_") or module_name == "plots":
                continue
            full_name = f"{prefix}{module_name}" if prefix else module_name
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
    local_scope = _task_scope_is_local()
    local_dirs = _active_local_task_dirs()
    key = task.lower().replace("-", "_")
    cls = _REGISTRY.get(key)
    if cls is not None and local_scope and not _adapter_defined_in_dirs(cls, local_dirs):
        cls = None
    if cls is None:
        visible_keys = [
            key_name
            for key_name, adapter_cls in _REGISTRY.items()
            if not local_scope or _adapter_defined_in_dirs(adapter_cls, local_dirs)
        ]
        if not visible_keys and local_scope:
            raise ValueError(f"Unknown task {task!r}. {_no_local_adapter_message()}")
        known = ", ".join(f'"{k}"' for k in visible_keys)
        raise ValueError(f"Unknown task {task!r}. Known tasks: {known}")
    return cls()


def get_task_options() -> list[dict[str, str]]:
    """Return unique task options for UI selectors.

    Each option is ``{"value": <task_key>, "label": <task_label>}``.
    """
    _load_entry_point_adapters()
    _load_local_task_packages()
    local_scope = _task_scope_is_local()
    local_dirs = _active_local_task_dirs()
    seen: dict[str, str] = {}
    for cls in dict.fromkeys(_REGISTRY.values()):
        if local_scope and not _adapter_defined_in_dirs(cls, local_dirs):
            continue
        task_key = getattr(cls, "task_key", None)
        task_label = getattr(cls, "task_label", None)
        if task_key and task_label and task_key not in seen:
            seen[task_key] = task_label
    return [{"value": key, "label": seen[key]} for key in seen]
