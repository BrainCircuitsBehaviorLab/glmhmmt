from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import expit, softmax


LAPSE_MODES = {"none", "class", "history", "history_conditioned"}


@dataclass(frozen=True)
class GLMFitResult:
    """Result of fitting a single-state GLM model."""

    weights: np.ndarray
    predictive_probs: np.ndarray
    lapse_rates: np.ndarray
    lapse_mode: str
    lapse_labels: tuple[str, ...]
    negative_log_likelihood: float
    success: bool
    y: np.ndarray
    X: np.ndarray

    @property
    def num_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.weights.shape[0])


def _validate_glm_inputs(
    X: ArrayLike,
    y: ArrayLike,
    *,
    num_classes: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=int).reshape(-1)

    if X_np.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (T, M); got ndim={X_np.ndim}.")
    if y_np.ndim != 1:
        raise ValueError(f"y must be a 1D array of shape (T,); got ndim={y_np.ndim}.")
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError(
            "X and y must have the same number of trials; "
            f"got X.shape[0]={X_np.shape[0]} and y.shape[0]={y_np.shape[0]}."
        )
    if y_np.size and y_np.min() < 0:
        raise ValueError("y must contain non-negative class indices.")

    resolved_num_classes = int(num_classes) if num_classes is not None else None
    if resolved_num_classes is None:
        if y_np.size == 0:
            raise ValueError("num_classes must be provided when fitting an empty dataset.")
        resolved_num_classes = int(y_np.max()) + 1

    if resolved_num_classes < 2:
        raise ValueError(f"num_classes must be at least 2; got {resolved_num_classes}.")
    if y_np.size and y_np.max() >= resolved_num_classes:
        raise ValueError(
            f"y contains class index {int(y_np.max())}, but num_classes={resolved_num_classes}."
        )

    return X_np, y_np, resolved_num_classes


def _normalize_lapse_mode(lapse_mode: str | None, lapse: bool | None) -> str:
    if lapse_mode is None:
        return "class" if lapse else "none"
    normalized = str(lapse_mode).strip().lower()
    if normalized not in LAPSE_MODES:
        raise ValueError(
            f"Unsupported lapse_mode={lapse_mode!r}; expected one of {sorted(LAPSE_MODES)}."
        )
    return normalized


def _num_lapse_params(num_classes: int, lapse_mode: str) -> int:
    if lapse_mode == "none":
        return 0
    if lapse_mode == "history":
        return 2
    if lapse_mode == "history_conditioned":
        return 2 * num_classes
    return num_classes


def _lapse_labels(num_classes: int, lapse_mode: str) -> tuple[str, ...]:
    if lapse_mode == "none":
        return tuple(f"class_{idx}" for idx in range(num_classes))
    if lapse_mode == "class":
        return tuple(f"class_{idx}" for idx in range(num_classes))
    if lapse_mode == "history":
        return ("repeat", "alternate")
    return tuple(f"repeat_prev_{idx}" for idx in range(num_classes)) + tuple(
        f"alternate_prev_{idx}" for idx in range(num_classes)
    )


def _project_lapse_params(
    x0: np.ndarray,
    weight_dim: int,
    lapse_size: int,
    lapse_max: float,
    lapse_mode: str,
    num_classes: int,
) -> np.ndarray:
    projected = np.asarray(x0, dtype=float).copy()
    if lapse_size <= 0:
        return projected
    lapse_rates = np.clip(projected[weight_dim : weight_dim + lapse_size], 0.0, lapse_max)
    if lapse_mode == "class":
        lapse_sum = float(lapse_rates.sum())
        if lapse_sum > 1.0:
            lapse_rates /= lapse_sum
    elif lapse_mode == "history":
        pair_sum = float(lapse_rates[0] + lapse_rates[1])
        if pair_sum > 1.0:
            lapse_rates /= pair_sum
    elif lapse_mode == "history_conditioned":
        repeat_rates = lapse_rates[:num_classes]
        alternate_rates = lapse_rates[num_classes:]
        for idx in range(num_classes):
            pair_sum = float(repeat_rates[idx] + alternate_rates[idx])
            if pair_sum > 1.0:
                repeat_rates[idx] /= pair_sum
                alternate_rates[idx] /= pair_sum
        lapse_rates = np.concatenate([repeat_rates, alternate_rates])
    projected[weight_dim : weight_dim + lapse_size] = lapse_rates
    return projected


def _base_logits(X_np: np.ndarray, w_flat_w: np.ndarray, num_classes: int) -> np.ndarray:
    T, M = X_np.shape
    W_pair = w_flat_w.reshape(num_classes - 1, M)
    if num_classes == 3:
        return np.stack([X_np @ W_pair[0], np.zeros(T), X_np @ W_pair[1]], axis=1)
    if num_classes == 2:
        return np.stack([-(X_np @ W_pair[0]), np.zeros(T)], axis=1)
    return np.concatenate([X_np @ W_pair.T, np.zeros((T, 1))], axis=1)


def _base_predictive_probs(X_np: np.ndarray, w_flat_w: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 2:
        W_pair = w_flat_w.reshape(num_classes - 1, X_np.shape[1])
        p_right = expit(-(X_np @ W_pair[0]))
        return np.stack([1.0 - p_right, p_right], axis=1)
    return softmax(_base_logits(X_np, w_flat_w, num_classes), axis=1)


def _history_repeat_targets(
    y_np: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    T = y_np.shape[0]
    targets = np.zeros((T, num_classes), dtype=float)
    if T <= 1:
        return targets
    prev = y_np[:-1]
    targets[np.arange(1, T), prev] = 1.0
    return targets


def _history_alternate_targets(
    y_np: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    T = y_np.shape[0]
    targets = np.zeros((T, num_classes), dtype=float)
    if T <= 1:
        return targets
    prev = y_np[:-1]
    if num_classes == 2:
        targets[1:, 1 - prev] = 1.0
        return targets
    for t in range(1, T):
        prev_class = int(y_np[t - 1])
        other_classes = [cls for cls in range(num_classes) if cls != prev_class]
        targets[t, other_classes] = 1.0 / float(len(other_classes))
    return targets


def _apply_lapse_mode(
    base_probs: np.ndarray,
    y_np: np.ndarray,
    num_classes: int,
    lapse_mode: str,
    lapse_rates: np.ndarray,
) -> np.ndarray:
    probs = np.asarray(base_probs, dtype=float)
    if lapse_mode == "none":
        return probs

    lapse_rates = np.asarray(lapse_rates, dtype=float)
    if lapse_mode == "class":
        total_mass = float(lapse_rates.sum())
        return lapse_rates[None, :] + (1.0 - total_mass) * probs

    repeat_targets = _history_repeat_targets(y_np, num_classes)
    alternate_targets = _history_alternate_targets(y_np, num_classes)
    if probs.shape[0] > 0:
        trial_mass = np.zeros(probs.shape[0], dtype=float)
        if lapse_mode == "history":
            repeat_rate = float(lapse_rates[0])
            alternate_rate = float(lapse_rates[1])
            if probs.shape[0] > 1:
                trial_mass[1:] = repeat_rate + alternate_rate
            adjusted = probs * (1.0 - trial_mass[:, None])
            adjusted += repeat_targets * np.concatenate([[0.0], np.full(probs.shape[0] - 1, repeat_rate)])[:, None]
            adjusted += alternate_targets * np.concatenate([[0.0], np.full(probs.shape[0] - 1, alternate_rate)])[:, None]
        else:
            repeat_rates = lapse_rates[:num_classes]
            alternate_rates = lapse_rates[num_classes:]
            if probs.shape[0] > 1:
                prev = y_np[:-1]
                trial_mass[1:] = repeat_rates[prev] + alternate_rates[prev]
            adjusted = probs * (1.0 - trial_mass[:, None])
            adjusted += repeat_targets * np.concatenate([[0.0], repeat_rates[y_np[:-1]]])[:, None]
            adjusted += alternate_targets * np.concatenate([[0.0], alternate_rates[y_np[:-1]]])[:, None]
        adjusted[0] = probs[0]
        return adjusted
    return probs


def fit_glm(
    X: ArrayLike,
    y: ArrayLike,
    *,
    num_classes: int | None = None,
    lapse_mode: str | None = None,
    lapse: bool | None = None,
    lapse_max: float = 0.2,
    n_restarts: int = 5,
    restart_noise_scale: float = 0.05,
    seed: int | None = 0,
    maxiter: int = 2000,
    ftol: float = 1e-9,
) -> GLMFitResult:
    """Fit a single-state GLM model directly from arrays."""

    X_np, y_np, num_classes = _validate_glm_inputs(X, y, num_classes=num_classes)
    T, M = X_np.shape

    lapse_mode = _normalize_lapse_mode(lapse_mode, lapse)
    if lapse_max < 0:
        raise ValueError(f"lapse_max must be non-negative; got {lapse_max}.")
    if int(n_restarts) < 1:
        raise ValueError(f"n_restarts must be at least 1; got {n_restarts}.")
    if restart_noise_scale < 0:
        raise ValueError(
            f"restart_noise_scale must be non-negative; got {restart_noise_scale}."
        )

    fit_lapse = lapse_mode != "none"
    lapse_size = _num_lapse_params(num_classes, lapse_mode)
    lapse_labels = _lapse_labels(num_classes, lapse_mode)

    def neg_log_likelihood(w_flat: np.ndarray) -> float:
        if fit_lapse:
            projected = _project_lapse_params(
                w_flat, weight_dim, lapse_size, lapse_max, lapse_mode, num_classes
            )
            w_flat_w = projected[:weight_dim]
            lapse_rates_local = projected[weight_dim : weight_dim + lapse_size]
        else:
            w_flat_w = w_flat
            lapse_rates_local = np.zeros(lapse_size, dtype=float)
        base_probs = _base_predictive_probs(X_np, w_flat_w, num_classes)
        probs = _apply_lapse_mode(base_probs, y_np, num_classes, lapse_mode, lapse_rates_local)
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return -float(np.sum(np.log(probs[np.arange(T), y_np])))

    weight_dim = (num_classes - 1) * M
    n_params = weight_dim + lapse_size
    method = "L-BFGS-B"
    minimize_kwargs = {"options": {"maxiter": maxiter, "ftol": ftol}}

    if fit_lapse:
        bounds = [(-np.inf, np.inf)] * weight_dim + [(0.0, lapse_max)] * lapse_size
        if lapse_mode == "class":
            constraints = ({
                "type": "ineq",
                "fun": lambda w_flat, offset=weight_dim, size=lapse_size: 1.0
                - np.sum(w_flat[offset : offset + size]),
            },)
        else:
            constraints = tuple(
                {
                    "type": "ineq",
                    "fun": (
                        lambda w_flat, offset=weight_dim, cls_idx=cls_idx, ncls=num_classes: 1.0
                        - w_flat[offset + cls_idx]
                        - w_flat[offset + ncls + cls_idx]
                    ),
                }
                for cls_idx in range(num_classes)
            )
        method = "SLSQP"
        minimize_kwargs["constraints"] = constraints
        base_x0 = np.zeros(n_params, dtype=float)
        if T > 0:
            warm_start = fit_glm(
                X_np,
                y_np,
                num_classes=num_classes,
                lapse_mode="none",
                lapse_max=lapse_max,
                n_restarts=1,
                restart_noise_scale=0.0,
                seed=seed,
                maxiter=maxiter,
                ftol=ftol,
            )
            if weight_dim > 0:
                base_x0[:weight_dim] = warm_start.weights[:-1].reshape(-1)
    else:
        bounds = None
        x0 = np.zeros(n_params, dtype=float)

    if T > 0:
        if fit_lapse:
            rng = np.random.default_rng(seed)
            best_res = None
            best_fun = float("inf")

            for _ in range(int(n_restarts)):
                x0 = base_x0.copy()
                if restart_noise_scale > 0.0:
                    x0 += rng.normal(0.0, restart_noise_scale, size=n_params)
                x0 = _project_lapse_params(
                    x0, weight_dim, lapse_size, lapse_max, lapse_mode, num_classes
                )
                candidate_res = minimize(
                    neg_log_likelihood,
                    x0,
                    method=method,
                    bounds=bounds,
                    **minimize_kwargs,
                )
                candidate_x = _project_lapse_params(
                    np.asarray(candidate_res.x, dtype=float),
                    weight_dim,
                    lapse_size,
                    lapse_max,
                    lapse_mode,
                    num_classes,
                )
                candidate_fun = float(neg_log_likelihood(candidate_x))
                candidate_success = bool(candidate_res.success)
                best_success = False if best_res is None else bool(best_res.success)

                if (
                    best_res is None
                    or candidate_fun < best_fun
                    or (
                        np.isclose(candidate_fun, best_fun)
                        and candidate_success
                        and not best_success
                    )
                ):
                    best_res = candidate_res
                    best_fun = candidate_fun

            res = best_res
        else:
            res = minimize(
                neg_log_likelihood,
                x0,
                method=method,
                bounds=bounds,
                **minimize_kwargs,
            )
        success = bool(res.success)
        w_flat = np.asarray(res.x, dtype=float)
        nll = float(res.fun)
    else:
        success = False
        w_flat = np.zeros(n_params, dtype=float)
        nll = float("nan")

    if fit_lapse:
        w_flat = _project_lapse_params(
            w_flat, weight_dim, lapse_size, lapse_max, lapse_mode, num_classes
        )
        lapse_rates = w_flat[weight_dim : weight_dim + lapse_size].copy()
        nll = float(neg_log_likelihood(w_flat)) if T > 0 else float("nan")
        w_flat_w = w_flat[:weight_dim]
    else:
        lapse_rates = np.zeros(num_classes, dtype=float)
        w_flat_w = w_flat

    W_pair = w_flat_w.reshape(num_classes - 1, M)
    if num_classes == 3:
        weights = np.stack([W_pair[0], np.zeros(M), W_pair[1]], axis=0)
    elif num_classes == 2:
        weights = np.stack([W_pair[0], np.zeros(M)], axis=0)
    else:
        weights = np.concatenate([W_pair, np.zeros((1, M))], axis=0)

    base_probs = _base_predictive_probs(X_np, w_flat_w, num_classes)
    predictive_probs = _apply_lapse_mode(base_probs, y_np, num_classes, lapse_mode, lapse_rates)
    if T > 0:
        predictive_probs = np.clip(predictive_probs, 1e-10, 1.0)
        predictive_probs /= predictive_probs.sum(axis=1, keepdims=True)

    return GLMFitResult(
        weights=weights,
        predictive_probs=predictive_probs,
        lapse_rates=lapse_rates,
        lapse_mode=lapse_mode,
        lapse_labels=lapse_labels,
        negative_log_likelihood=nll,
        success=success,
        y=y_np,
        X=X_np,
    )
