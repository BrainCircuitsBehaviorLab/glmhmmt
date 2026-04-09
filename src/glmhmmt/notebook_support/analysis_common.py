from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import numpy as np
import polars as pl
import requests

try:
    import boto3
except ImportError:  # pragma: no cover - optional for local-only notebook usage
    boto3 = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in constrained runtimes
    load_dotenv = None


ENV_PUBLIC_FITS_BASE_URL = "GLMHMMT_PUBLIC_FITS_BASE_URL"
ENV_R2_ENDPOINT_URL = "GLMHMMT_R2_ENDPOINT_URL"
ENV_R2_REGION = "GLMHMMT_R2_REGION"
ENV_R2_BUCKET = "GLMHMMT_R2_BUCKET"
ENV_R2_PREFIX = "GLMHMMT_R2_PREFIX"
ENV_R2_ACCESS_KEY_ID = "GLMHMMT_R2_ACCESS_KEY_ID"
ENV_R2_SECRET_ACCESS_KEY = "GLMHMMT_R2_SECRET_ACCESS_KEY"
ENV_PUBLIC_FITS_MANIFEST_URL = "GLMHMMT_PUBLIC_FITS_MANIFEST_URL"
DEFAULT_PUBLIC_FITS_BASE_URL = "https://fits.javirm.com/public-fits"
DEFAULT_PUBLIC_FITS_MANIFEST_URL = f"{DEFAULT_PUBLIC_FITS_BASE_URL}/manifest.json"
DEFAULT_R2_PREFIX = "public-fits"
_ENV_LOADED = False
_MANIFEST_CACHE: dict[str, dict] = {}


def resolve_selected_model_id(
    current_hash: str,
    existing_model: str | None,
    alias: str | None,
) -> str:
    return existing_model or alias or current_hash


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    if load_dotenv is None:
        return

    here = Path(__file__).resolve()
    candidates = [
        here.parents[5] / ".env",
        here.parents[4] / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _env(name: str, default: str = "") -> str:
    _load_env_once()
    return os.environ.get(name, default).strip()


def _fit_rel_dir(task_name: str, model_kind: str, alias: str) -> str:
    return f"{task_name}/{model_kind}/{alias}"


def _remote_manifest_url() -> str:
    explicit = _env(ENV_PUBLIC_FITS_MANIFEST_URL)
    if explicit:
        return explicit

    base_url = _env(ENV_PUBLIC_FITS_BASE_URL)
    if base_url:
        return f"{base_url.rstrip('/')}/manifest.json"
    return DEFAULT_PUBLIC_FITS_MANIFEST_URL


def _should_use_remote(local_root: Path | None = None) -> bool:
    if local_root is not None and local_root.exists():
        return False
    return True


def remote_fits_enabled(local_root: Path | None = None) -> bool:
    return _should_use_remote(local_root)


def _get_manifest() -> dict:
    manifest_url = _remote_manifest_url()
    if not manifest_url:
        return {}
    cached = _MANIFEST_CACHE.get(manifest_url)
    if cached is not None:
        return cached
    response = requests.get(manifest_url, timeout=30)
    response.raise_for_status()
    manifest = response.json()
    _MANIFEST_CACHE[manifest_url] = manifest
    return manifest


def _remote_alias_node(task_name: str, model_kind: str, alias: str) -> dict | None:
    manifest = _get_manifest()
    return (
        manifest.get("tasks", {})
        .get(task_name, {})
        .get(model_kind, {})
        .get(alias)
    )


def model_aliases_for_kind(*, task_name: str, model_kind: str, local_root: Path | None = None) -> list[str]:
    if remote_fits_enabled(local_root):
        manifest = _get_manifest()
        aliases = (
            manifest.get("tasks", {})
            .get(task_name, {})
            .get(model_kind, {})
            .keys()
        )
        return sorted(str(alias) for alias in aliases)

    if local_root is None or not local_root.exists():
        return []
    return sorted(child.name for child in local_root.iterdir() if child.is_dir())


def _remote_file_list(task_name: str, model_kind: str, alias: str) -> list[str]:
    node = _remote_alias_node(task_name, model_kind, alias)
    if not node:
        return []
    return [str(path) for path in node.get("files", [])]


def _iter_matching_remote_files(
    *,
    task_name: str,
    model_kind: str,
    alias: str,
    suffix: str,
    k: int | None = None,
) -> list[str]:
    matches: list[str] = []
    for rel_path in _remote_file_list(task_name, model_kind, alias):
        name = Path(rel_path).name
        if not name.endswith(suffix):
            continue
        if k is not None and f"_K{k}_" not in name and not name.endswith(f"_K{k}{suffix}"):
            if f"_K{k}" not in name:
                continue
        matches.append(rel_path)
    return sorted(matches)


def _build_public_url(rel_path: str) -> str:
    base_url = _env(ENV_PUBLIC_FITS_BASE_URL) or DEFAULT_PUBLIC_FITS_BASE_URL
    return f"{base_url.rstrip('/')}/{quote(rel_path, safe='/')}"


def _read_remote_bytes(rel_path: str) -> bytes:
    base_url = _env(ENV_PUBLIC_FITS_BASE_URL)
    if base_url:
        response = requests.get(_build_public_url(rel_path), timeout=60)
        response.raise_for_status()
        return response.content

    bucket = _env(ENV_R2_BUCKET)
    endpoint = _env(ENV_R2_ENDPOINT_URL)
    access_key = _env(ENV_R2_ACCESS_KEY_ID)
    secret_key = _env(ENV_R2_SECRET_ACCESS_KEY)
    region = _env(ENV_R2_REGION, "auto")
    if not all([bucket, endpoint, access_key, secret_key]):
        raise RuntimeError(
            "Remote fits require either a public base URL or R2 credentials in the environment."
        )
    if boto3 is None:
        raise RuntimeError("boto3 is required for authenticated R2 access.")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    obj = client.get_object(Bucket=bucket, Key=rel_path)
    return obj["Body"].read()


def _read_remote_json(rel_path: str) -> dict:
    return json.loads(_read_remote_bytes(rel_path).decode("utf-8"))


def load_model_config(
    *,
    task_name: str,
    model_kind: str,
    alias: str | None,
    local_root: Path | None = None,
) -> dict:
    if not alias:
        return {}
    rel_path = f"{_fit_rel_dir(task_name, model_kind, alias)}/config.json"
    if remote_fits_enabled(local_root):
        try:
            return _read_remote_json(rel_path)
        except Exception:
            return {}

    if local_root is None:
        return {}
    cfg_path = local_root / alias / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def load_metrics_dir(
    *,
    task_name: str,
    model_kind: str,
    alias: str | None,
    local_root: Path | None = None,
    label_map: dict[str, str] | None = None,
) -> pl.DataFrame | None:
    if not alias:
        return None

    if remote_fits_enabled(local_root):
        metric_paths = _iter_matching_remote_files(
            task_name=task_name,
            model_kind=model_kind,
            alias=alias,
            suffix="_metrics.parquet",
        )
        if not metric_paths:
            return None
        frames = []
        for rel_path in metric_paths:
            try:
                frames.append(pl.read_parquet(BytesIO(_read_remote_bytes(rel_path))))
            except Exception:
                pass
    else:
        if local_root is None:
            return None
        fit_dir = local_root / alias
        if not fit_dir.exists():
            return None
        files = sorted(fit_dir.glob("*_metrics.parquet"))
        if not files:
            return None
        frames = []
        for path in files:
            try:
                frames.append(pl.read_parquet(path))
            except Exception:
                pass

    if not frames:
        return None

    df = pl.concat(frames, how="diagonal")
    if "nll" in df.columns and "ll_per_trial" not in df.columns:
        df = df.with_columns((-pl.col("nll") / pl.col("n_trials")).alias("ll_per_trial"))
    if "K" not in df.columns:
        df = df.with_columns(pl.lit(1, dtype=pl.Int64).alias("K"))
    else:
        df = df.with_columns(pl.col("K").cast(pl.Int64))
    if "model_kind" not in df.columns:
        df = df.with_columns(pl.lit(model_kind).alias("model_kind"))

    labels = label_map or {}
    model_label = f"{labels.get(model_kind, model_kind.upper())} ({alias})"
    return df.with_columns(
        [
            pl.lit(alias).alias("model_alias"),
            pl.lit(model_label).alias("model_label"),
        ]
    )


def load_fit_bundle(
    *,
    task_name: str,
    model_kind: str,
    alias: str,
    k: int,
    subjects: list[str],
    get_adapter,
    build_views,
    scoring_key: str | None = None,
    local_root: Path | None = None,
) -> tuple[object, dict, dict[str, list[str]], dict]:
    adapter = get_adapter(task_name)
    arrays_suffix = {
        "glm": "glm_arrays.npz",
        "glmhmm": "glmhmm_arrays.npz",
        "glmhmmt": "glmhmmt_arrays.npz",
        "glmhmm-t": "glmhmmt_arrays.npz",
    }[model_kind]

    arrays_store: dict[str, dict] = {}
    names: dict[str, list[str]] = {}
    for subject in subjects:
        candidate_names: list[str]
        if model_kind == "glm":
            candidate_names = [f"{subject}_glm_arrays.npz"]
        else:
            candidate_names = [
                f"{subject}_K{k}_{arrays_suffix}",
                f"{subject}_{arrays_suffix}",
            ]
        for name in candidate_names:
            rel_path = f"{_fit_rel_dir(task_name, model_kind, alias)}/{name}"
            try:
                if remote_fits_enabled(local_root):
                    data = dict(np.load(BytesIO(_read_remote_bytes(rel_path)), allow_pickle=True))
                else:
                    if local_root is None:
                        raise FileNotFoundError
                    local_path = local_root / alias / name
                    if not local_path.exists():
                        raise FileNotFoundError
                    data = dict(np.load(local_path, allow_pickle=True))
            except Exception:
                continue

            saved_names = _extract_saved_names(data)
            if "X_cols" in data:
                data["X_cols"] = [str(v) for v in data["X_cols"]]
            elif "X_cols" in saved_names:
                data["X_cols"] = list(saved_names["X_cols"])
            if "U_cols" in data:
                data["U_cols"] = [str(v) for v in data["U_cols"]]
            elif "U_cols" in saved_names:
                data["U_cols"] = list(saved_names["U_cols"])
            arrays_store[subject] = data
            names = {
                "X_cols": list(data.get("X_cols", [])),
                "U_cols": list(data.get("U_cols", [])),
            }
            break

    if scoring_key is not None and hasattr(adapter, "scoring_key"):
        adapter.scoring_key = scoring_key
    views = build_views(arrays_store, adapter, k, list(arrays_store.keys())) if arrays_store else {}
    return adapter, arrays_store, names, views


def _extract_saved_names(arrays: dict) -> dict[str, list[str]]:
    saved_names: dict[str, list[str]] = {}
    raw_names = arrays.get("names")
    if raw_names is None or getattr(raw_names, "shape", None) != ():
        return saved_names
    raw_dict = raw_names.item()
    if not isinstance(raw_dict, dict):
        return saved_names
    for key in ("X_cols", "U_cols"):
        values = raw_dict.get(key)
        if values is not None:
            saved_names[key] = [str(v) for v in values]
    return saved_names


def _first_matching_width(width: int, *candidates) -> list[str]:
    if width <= 0:
        return []

    longer: list[str] | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        cols = [str(v) for v in list(candidate)]
        if len(cols) == width:
            return cols
        if longer is None and len(cols) > width:
            longer = cols[:width]
    if longer is not None:
        return longer
    return [f"feature_{idx}" for idx in range(width)]


def _first_matching_u_width(width: int, *candidates) -> list[str]:
    if width <= 0:
        return []

    longer: list[str] | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        cols = [str(v) for v in list(candidate)]
        if len(cols) == width:
            return cols
        if longer is None and len(cols) > width:
            longer = cols[:width]
    if longer is not None:
        return longer
    return [f"transition_feature_{idx}" for idx in range(width)]


def load_fit_arrays(
    *,
    out_dir: Path,
    arrays_suffix: str,
    adapter,
    df_all: pl.DataFrame,
    subjects: list[str],
    emission_cols: list[str] | None,
    transition_cols: list[str] | None = None,
    k: int | None = None,
    postprocess_array: Callable[[dict], dict] | None = None,
) -> tuple[dict, dict[str, list[str]]]:
    df_for_names = df_all.filter(pl.col("subject").is_in(subjects)) if subjects else df_all.head(0)
    try:
        resolved_names = adapter.resolve_design_names(
            emission_cols=emission_cols,
            transition_cols=transition_cols,
            df=df_for_names,
        )
    except Exception:
        resolved_names = {"X_cols": [], "U_cols": []}

    remote_mode = remote_fits_enabled(out_dir)
    path_candidates: list[Path] = []
    rel_candidates: list[str] = []
    if remote_mode:
        parts = out_dir.parts
        try:
            fit_idx = parts.index("fits")
            task_name = parts[fit_idx + 1]
            model_kind = parts[fit_idx + 2]
            alias = parts[fit_idx + 3]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Could not infer fit location from {out_dir}") from exc
        rel_candidates = _iter_matching_remote_files(
            task_name=task_name,
            model_kind=model_kind,
            alias=alias,
            suffix=arrays_suffix,
            k=k,
        )
    else:
        path_candidates = list(sorted(out_dir.glob(f"*_{arrays_suffix}")))
        if k is not None:
            path_candidates += [
                f for f in sorted(out_dir.glob(f"*_K{k}_{arrays_suffix}")) if f not in path_candidates
            ]

    arrays_store = {}
    file_iter = rel_candidates if remote_mode else path_candidates
    for path in file_iter:
        path_name = Path(path).name
        subject = path_name.removesuffix(f"_{arrays_suffix}")
        if k is not None:
            subject = subject.removesuffix(f"_K{k}")

        if remote_mode:
            arrays = dict(np.load(BytesIO(_read_remote_bytes(str(path))), allow_pickle=True))
        else:
            arrays = dict(np.load(path, allow_pickle=True))
        if postprocess_array is not None:
            arrays = postprocess_array(arrays)

        saved_names = _extract_saved_names(arrays)
        emission_width = int(np.asarray(arrays["emission_weights"]).shape[2]) if "emission_weights" in arrays else 0
        arrays["X_cols"] = _first_matching_width(
            emission_width,
            arrays.get("X_cols"),
            saved_names.get("X_cols"),
            resolved_names.get("X_cols"),
        )

        transition_width = 0
        if "transition_weights" in arrays:
            transition_width = int(np.asarray(arrays["transition_weights"]).shape[2])
        elif "U" in arrays:
            transition_width = int(np.asarray(arrays["U"]).shape[1])
        arrays["U_cols"] = _first_matching_u_width(
            transition_width,
            arrays.get("U_cols"),
            saved_names.get("U_cols"),
            resolved_names.get("U_cols"),
        )

        arrays_store[subject] = arrays

    names = {
        "X_cols": list(resolved_names.get("X_cols", [])),
        "U_cols": list(resolved_names.get("U_cols", [])),
    }
    return arrays_store, names


def select_subject_behavior_df(
    df_all: pl.DataFrame,
    *,
    subject,
    sort_col,
    session_col: str,
    min_session_length: int = 1,
) -> pl.DataFrame:
    df_sub = df_all.filter(pl.col("subject") == subject).sort(sort_col)
    if min_session_length > 1:
        df_sub = df_sub.filter(pl.col(session_col).count().over(session_col) >= min_session_length)
    return df_sub


def build_trial_and_weights_df(
    df_all: pl.DataFrame,
    *,
    views: dict,
    adapter,
    min_session_length: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    from glmhmmt.postprocess import build_emission_weights_df, build_trial_df

    trial_frames = []
    for subject, view in views.items():
        df_sub = select_subject_behavior_df(
            df_all,
            subject=subject,
            sort_col=adapter.sort_col,
            session_col=adapter.session_col,
            min_session_length=min_session_length,
        )
        if df_sub.height != view.T:
            print(f"⚠️  {subject}: row mismatch ({df_sub.height} vs {view.T}), skipping")
            continue
        trial_frames.append(build_trial_df(view, adapter, df_sub, adapter.behavioral_cols))

    trial_df = pl.concat(trial_frames) if trial_frames else pl.DataFrame()
    weights_df = build_emission_weights_df(views)
    return trial_df, weights_df
