"""Utilities for regressors derived from fitted model weights."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json

import numpy as np

from glmhmmt.runtime import get_results_dir


@dataclass(frozen=True)
class FittedWeightRegressorSpec:
    """Describe a regressor built as a weighted sum of fitted feature columns."""

    target_name: str
    fit_task: str
    fit_model_kind: str
    fit_model_id: str
    arrays_suffix: str
    source_features: tuple[str, ...] = ()
    source_feature_prefixes: tuple[str, ...] = ()
    exclude_features: tuple[str, ...] = ()
    excluded_subjects: tuple[str, ...] = ()
    sign: float = 1.0
    state_idx: int = 0
    class_idx: int = 0


def _fit_dir(spec: FittedWeightRegressorSpec):
    return get_results_dir() / "fits" / spec.fit_task / spec.fit_model_kind / spec.fit_model_id


def _fit_signature(spec: FittedWeightRegressorSpec) -> tuple[tuple[str, int, int], ...]:
    """Return a cache key that changes when the underlying fit files change."""
    fit_dir = _fit_dir(spec)
    if not fit_dir.exists():
        raise FileNotFoundError(
            f"Missing fit directory for {spec.target_name!r}: {fit_dir}"
        )

    paths = [fit_dir / "config.json", *sorted(fit_dir.glob(f"*_{spec.arrays_suffix}"))]
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        signature.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def resolved_source_features(spec: FittedWeightRegressorSpec) -> tuple[str, ...]:
    return _resolved_source_features_cached(spec, _fit_signature(spec))


@lru_cache(maxsize=None)
def _resolved_source_features_cached(
    spec: FittedWeightRegressorSpec,
    _signature: tuple[tuple[str, int, int], ...],
) -> tuple[str, ...]:
    """Return the fitted feature names contributing to this derived regressor."""
    if spec.source_features:
        return tuple(spec.source_features)

    config_path = _fit_dir(spec) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not resolve source features for {spec.target_name!r}: "
            f"missing config at {config_path}."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    emission_cols = tuple(
        str(col)
        for col in (
            config.get("resolved_emission_cols")
            or config.get("emission_cols")
            or []
        )
    )
    exclude = set(spec.exclude_features)

    def _resolve_from_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
        if spec.source_feature_prefixes:
            prefixes = tuple(spec.source_feature_prefixes)
            return tuple(
                col
                for col in columns
                if any(col.startswith(prefix) for prefix in prefixes)
                and col not in exclude
            )
        return tuple(col for col in columns if col not in exclude)

    resolved = _resolve_from_columns(emission_cols)
    if not resolved and spec.source_feature_prefixes:
        fit_dir = _fit_dir(spec)
        for arr_path in sorted(fit_dir.glob(f"*_{spec.arrays_suffix}")):
            with np.load(arr_path, allow_pickle=True) as arr:
                x_cols_raw = arr.get("X_cols")
                if x_cols_raw is None:
                    continue
                array_cols = tuple(str(col) for col in x_cols_raw.tolist())
            resolved = _resolve_from_columns(array_cols)
            if resolved:
                break

    if not resolved:
        if spec.source_feature_prefixes:
            source_desc = (
                f"matching prefixes {list(spec.source_feature_prefixes)} from "
                f"{list(emission_cols)}"
            )
        else:
            source_desc = (
                f"excluding {sorted(exclude)} from {list(emission_cols)}"
            )
        raise ValueError(
            f"No fitted source features remained for {spec.target_name!r} after "
            f"{source_desc}."
        )
    return resolved


def mean_feature_weights_from_fit(spec: FittedWeightRegressorSpec) -> dict[str, float]:
    return _mean_feature_weights_from_fit_cached(spec, _fit_signature(spec))


def subject_feature_weights_from_fit(
    spec: FittedWeightRegressorSpec,
    subject: str,
) -> dict[str, float]:
    return _subject_feature_weights_from_fit_cached(spec, str(subject), _fit_signature(spec))


def _load_feature_weights_for_subject(
    spec: FittedWeightRegressorSpec,
    subject: str,
    *,
    signature: tuple[tuple[str, int, int], ...],
) -> dict[str, float]:
    _ = signature
    features = resolved_source_features(spec)
    excluded_subjects = set(spec.excluded_subjects)
    if subject in excluded_subjects:
        raise ValueError(
            f"Subject {subject!r} is excluded for {spec.target_name!r}."
        )

    fit_dir = _fit_dir(spec)
    arr_path = fit_dir / f"{subject}_{spec.arrays_suffix}"
    if not arr_path.exists():
        raise FileNotFoundError(
            f"Missing fitted arrays for {spec.target_name!r} subject {subject!r}: {arr_path}"
        )

    with np.load(arr_path, allow_pickle=True) as arr:
        x_cols = [str(col) for col in arr["X_cols"].tolist()]
        emission_weights = np.asarray(arr["emission_weights"], dtype=float)

    if emission_weights.ndim != 3:
        raise ValueError(
            f"Unexpected emission weight rank for {spec.target_name!r} subject {subject!r}."
        )
    if emission_weights.shape[0] <= spec.state_idx or emission_weights.shape[1] <= spec.class_idx:
        raise ValueError(
            f"Missing state/class slice for {spec.target_name!r} subject {subject!r}."
        )

    row = emission_weights[spec.state_idx, spec.class_idx, :]
    col_to_weight = {
        col: spec.sign * float(weight)
        for col, weight in zip(x_cols, row)
    }
    missing = [feat for feat in features if feat not in col_to_weight]
    if missing:
        raise ValueError(
            f"Missing fitted feature weights for {spec.target_name!r} subject {subject!r}: {missing}."
        )
    return {feat: col_to_weight[feat] for feat in features}


@lru_cache(maxsize=None)
def _subject_feature_weights_from_fit_cached(
    spec: FittedWeightRegressorSpec,
    subject: str,
    _signature: tuple[tuple[str, int, int], ...],
) -> dict[str, float]:
    """Return fitted weights for one subject keyed by feature name."""
    fit_dir = _fit_dir(spec)
    if not fit_dir.exists():
        raise FileNotFoundError(
            f"Missing fit directory for {spec.target_name!r}: {fit_dir}"
        )
    return _load_feature_weights_for_subject(spec, subject, signature=_signature)


@lru_cache(maxsize=None)
def _mean_feature_weights_from_fit_cached(
    spec: FittedWeightRegressorSpec,
    _signature: tuple[tuple[str, int, int], ...],
) -> dict[str, float]:
    """Return pooled fitted weights keyed by feature name."""
    fit_dir = _fit_dir(spec)
    if not fit_dir.exists():
        raise FileNotFoundError(
            f"Missing fit directory for {spec.target_name!r}: {fit_dir}"
        )

    features = resolved_source_features(spec)
    collected: dict[str, list[float]] = {feat: [] for feat in features}
    valid_subjects = 0
    excluded_subjects = set(spec.excluded_subjects)

    for arr_path in sorted(fit_dir.glob(f"*_{spec.arrays_suffix}")):
        subject = arr_path.name.removesuffix(f"_{spec.arrays_suffix}").split("_", 1)[0]
        if subject in excluded_subjects:
            continue

        try:
            col_to_weight = _load_feature_weights_for_subject(
                spec,
                subject,
                signature=_signature,
            )
        except Exception:
            continue

        for feat in features:
            collected[feat].append(col_to_weight[feat])
        valid_subjects += 1

    if valid_subjects == 0:
        raise ValueError(
            f"No valid fitted arrays found for {spec.target_name!r} in {fit_dir} "
            f"with required features {list(features)}."
        )

    missing = [feat for feat, vals in collected.items() if not vals]
    if missing:
        raise ValueError(
            f"Could not pool fitted weights for {spec.target_name!r}; "
            f"missing values for features {missing}."
        )

    return {
        feat: float(np.mean(vals))
        for feat, vals in collected.items()
    }


def clear_fitted_regressor_cache() -> None:
    """Clear cached fit-derived regressors after manual file updates."""
    _resolved_source_features_cached.cache_clear()
    _subject_feature_weights_from_fit_cached.cache_clear()
    _mean_feature_weights_from_fit_cached.cache_clear()


def _infer_single_subject(part) -> str | None:
    if "subject" not in part.columns:
        return None
    subject_series = np.asarray(part["subject"])
    unique = {
        str(value)
        for value in subject_series.tolist()
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    }
    if len(unique) != 1:
        return None
    return next(iter(unique))


def weighted_sum_regressor(
    part,
    spec: FittedWeightRegressorSpec,
    *,
    dtype=np.float32,
    subject: str | None = None,
) -> np.ndarray:
    """Build a regressor by summing feature columns weighted by fitted values."""
    features = resolved_source_features(spec)
    resolved_subject = str(subject) if subject is not None else _infer_single_subject(part)
    if resolved_subject is not None:
        try:
            weights = subject_feature_weights_from_fit(spec, resolved_subject)
        except (FileNotFoundError, ValueError):
            weights = mean_feature_weights_from_fit(spec)
    else:
        weights = mean_feature_weights_from_fit(spec)
    missing_cols = [feat for feat in features if feat not in part.columns]
    if missing_cols:
        raise ValueError(
            f"Cannot build {spec.target_name!r}; missing dataframe columns {missing_cols}."
        )

    out = np.zeros(len(part), dtype=dtype)
    for feat in features:
        out += np.asarray(part[feat], dtype=dtype) * dtype(weights[feat])
    return out
