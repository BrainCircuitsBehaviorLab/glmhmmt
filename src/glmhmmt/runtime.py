from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import os
from pathlib import Path
import tomllib


PROJECT_CONFIG_NAME = "config.toml"
DEFAULT_CONFIG_RESOURCE = "resources/default_config.toml"

ENV_DATA_DIR = "GLMHMMT_DATA_DIR"
ENV_RESULTS_DIR = "GLMHMMT_RESULTS_DIR"
ENV_CONFIG_PATH = "GLMHMMT_CONFIG_PATH"
@dataclass(frozen=True)
class RuntimePaths:
    data_dir: Path
    results_dir: Path
    config_path: Path | None
    package_root: Path
    code_root: Path
    workspace_root: Path

    @property
    def ROOT(self) -> Path:
        return self.workspace_root

    @property
    def CODE_DIR(self) -> Path:
        return self.code_root

    @property
    def RESULTS(self) -> Path:
        return self.results_dir

    @property
    def CONFIG(self) -> Path:
        if self.config_path is None:
            raise FileNotFoundError("No runtime config file is configured.")
        return self.config_path

    @property
    def DATA_PATH(self) -> Path:
        return self.data_dir

    def show_paths(self) -> None:
        print(f"ROOT      = {self.workspace_root}")
        print(f"CODE_DIR  = {self.code_root}")
        print(f"DATA_PATH = {self.data_dir}")
        print(f"RESULTS   = {self.results_dir}")
        print(f"CONFIG    = {self.config_path}")


_runtime_overrides: dict[str, Path | None] = {
    "data_dir": None,
    "results_dir": None,
    "config_path": None,
}


def _package_roots() -> tuple[Path, Path, Path]:
    package_root = Path(__file__).resolve().parents[2]
    code_root = package_root.parent if package_root.parent.name == "code" else package_root
    workspace_root = code_root.parent if code_root.name == "code" else package_root.parent
    return package_root, code_root, workspace_root


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_configured_path(value: object, *, config_path: Path) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (config_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


@lru_cache(maxsize=1)
def _load_default_config() -> dict:
    default_config = files("glmhmmt").joinpath(DEFAULT_CONFIG_RESOURCE)
    with default_config.open("rb") as f:
        return tomllib.load(f)


def _default_project_config_path() -> Path | None:
    package_root, _, _ = _package_roots()
    candidate = package_root / PROJECT_CONFIG_NAME
    return candidate if candidate.exists() else None


def _find_upward_config_path(start: str | Path | None = None) -> Path | None:
    current = Path(start).expanduser().resolve() if start is not None else Path.cwd().resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def _serialize_path(value: str | Path) -> str:
    return Path(value).expanduser().resolve().as_posix()


def _render_project_config(
    *,
    data_dir: str | Path,
    results_dir: str | Path,
    adapter_paths: list[str | Path] | None = None,
) -> str:
    lines = [
        "# Project-local overrides for glmhmmt.",
        "# Relative paths are resolved relative to this file.",
        "# Precedence: configure_paths(...) > GLMHMMT_CONFIG_PATH > nearest config.toml >",
        "# packaged defaults.",
        "",
        "[paths]",
        f'data_dir = "{_serialize_path(data_dir)}"',
        f'results_dir = "{_serialize_path(results_dir)}"',
    ]
    if adapter_paths:
        lines.extend(
            [
                "",
                "[plugins]",
                "adapter_paths = [",
                *[f'    "{_serialize_path(path)}",' for path in adapter_paths],
                "]",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "[plugins]",
                "# Optional local task packages. Prefer installed entry points when possible.",
                '# adapter_paths = ["/absolute/path/to/adapters"]',
            ]
        )
    lines.append("")
    return "\n".join(lines)


def init_project_config(
    *,
    path: str | Path | None = None,
    force: bool = False,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    adapter_paths: list[str | Path] | None = None,
) -> Path:
    target = Path(path).expanduser().resolve() if path is not None else (Path.cwd() / PROJECT_CONFIG_NAME).resolve()
    if target.exists() and not force:
        raise FileExistsError(
            f"Config already exists at {target}. Pass force=True to overwrite it."
        )

    current = get_runtime_paths()
    resolved_data_dir = data_dir or current.data_dir
    resolved_results_dir = results_dir or current.results_dir

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _render_project_config(
            data_dir=resolved_data_dir,
            results_dir=resolved_results_dir,
            adapter_paths=adapter_paths,
        ),
        encoding="utf-8",
    )
    return target


def _configured_paths_from_file(config_path: Path | None) -> dict[str, Path]:
    if config_path is None or not config_path.exists():
        return {}

    raw_cfg = _read_toml(config_path)
    raw_paths = raw_cfg.get("paths", {})
    if not isinstance(raw_paths, dict):
        return {}

    resolved: dict[str, Path] = {}
    for key in ("data_dir", "results_dir"):
        path_value = _resolve_configured_path(raw_paths.get(key), config_path=config_path)
        if path_value is not None:
            resolved[key] = path_value
    return resolved


def configure_paths(
    *,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> None:
    updates = {
        "data_dir": data_dir,
        "results_dir": results_dir,
        "config_path": config_path,
    }
    changed = False
    for key, value in updates.items():
        resolved = None if value is None else Path(value).expanduser().resolve()
        if _runtime_overrides.get(key) != resolved:
            _runtime_overrides[key] = resolved
            changed = True
    if changed:
        get_runtime_paths.cache_clear()
        _load_merged_app_config.cache_clear()


@lru_cache(maxsize=1)
def get_runtime_paths() -> RuntimePaths:
    package_root, code_root, workspace_root = _package_roots()

    repo_data_dir = workspace_root / "data"
    repo_results_dir = code_root / "results"
    config_path = (
        _runtime_overrides["config_path"]
        or _path_from_env(ENV_CONFIG_PATH)
        or _find_upward_config_path()
        or _default_project_config_path()
    )
    configured_paths = _configured_paths_from_file(config_path)

    data_dir = (
        _runtime_overrides["data_dir"]
        or _path_from_env(ENV_DATA_DIR)
        or configured_paths.get("data_dir")
        or (repo_data_dir if repo_data_dir.exists() else Path("data").resolve())
    )
    results_dir = (
        _runtime_overrides["results_dir"]
        or _path_from_env(ENV_RESULTS_DIR)
        or configured_paths.get("results_dir")
        or (repo_results_dir if repo_results_dir.exists() else Path("results").resolve())
    )
    return RuntimePaths(
        data_dir=data_dir,
        results_dir=results_dir,
        config_path=config_path,
        package_root=package_root,
        code_root=code_root,
        workspace_root=workspace_root,
    )


def get_data_dir() -> Path:
    return get_runtime_paths().data_dir


def get_results_dir() -> Path:
    return get_runtime_paths().results_dir


def get_config_path() -> Path | None:
    return get_runtime_paths().config_path


@lru_cache(maxsize=None)
def _load_merged_app_config(config_key: str | None) -> dict:
    cfg = deepcopy(_load_default_config())
    if config_key is None:
        return cfg

    config_path = Path(config_key)
    if config_path.exists():
        cfg = _merge_dicts(cfg, _read_toml(config_path))
    return cfg


def load_app_config(config_path: str | Path | None = None) -> dict:
    resolved = Path(config_path).expanduser().resolve() if config_path else get_config_path()
    config_key = str(resolved) if resolved is not None else None
    return deepcopy(_load_merged_app_config(config_key))


def add_runtime_path_args(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--data-dir", type=str, default=None, help=f"Override data directory (env: {ENV_DATA_DIR}).")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=f"Override results directory (env: {ENV_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Override the config TOML used for runtime paths and plotting/model settings "
            f"(env: {ENV_CONFIG_PATH})."
        ),
    )
def configure_paths_from_args(args: argparse.Namespace) -> None:
    configure_paths(
        data_dir=getattr(args, "data_dir", None),
        results_dir=getattr(args, "results_dir", None),
        config_path=getattr(args, "config_path", None),
    )
