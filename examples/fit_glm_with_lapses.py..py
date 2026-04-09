from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, softmax


@dataclass(frozen=True)
class GLMFitResult:
    """Result of fitting a single-state GLM model."""

    weights: np.ndarray
    predictive_probs: np.ndarray
    lapse_rates: np.ndarray
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


def fit_glm(
    X: ArrayLike,
    y: ArrayLike,
    *,
    num_classes: int | None = None,
    lapse: bool = False,
    lapse_max: float = 0.2,
    maxiter: int = 2000,
    ftol: float = 1e-9,
) -> GLMFitResult:
    """Fit a single-state GLM model directly from arrays.

    The binary case optionally supports two lapse parameters constrained to
    ``[0, lapse_max]`` with ``gamma_left + gamma_right <= 1``.
    """

    X_np, y_np, num_classes = _validate_glm_inputs(X, y, num_classes=num_classes)
    T, M = X_np.shape

    if lapse and num_classes != 2:
        raise ValueError("lapse=True is only supported for binary GLMs (num_classes=2).")
    if lapse_max < 0:
        raise ValueError(f"lapse_max must be non-negative; got {lapse_max}.")

    fit_lapse = lapse and num_classes == 2
    method = "L-BFGS-B"
    minimize_kwargs = {"options": {"maxiter": maxiter, "ftol": ftol}}

    if fit_lapse:
        def neg_log_likelihood(w_flat: np.ndarray) -> float:
            w = w_flat[:M]
            g_left = w_flat[M]
            g_right = w_flat[M + 1]
            p_right_base = expit(-(X_np @ w))
            p_right = g_left + (1.0 - g_left - g_right) * p_right_base
            p_right = np.clip(p_right, 1e-10, 1.0 - 1e-10)
            log_p_right = np.log(p_right)
            log_p_left = np.log(1.0 - p_right)
            return -float(np.sum(np.where(y_np == 1, log_p_right, log_p_left)))

        n_params = M + 2
        bounds = [(-np.inf, np.inf)] * M + [(0.0, lapse_max), (0.0, lapse_max)]
        constraints = ({
            "type": "ineq",
            "fun": lambda w_flat, m=M: 1.0 - w_flat[m] - w_flat[m + 1],
        },)
        method = "SLSQP"
        minimize_kwargs["constraints"] = constraints
        x0 = np.zeros(n_params)
    else:
        def neg_log_likelihood(w_flat: np.ndarray) -> float:
            W_pair = w_flat.reshape(num_classes - 1, M)
            if num_classes == 3:
                logits = np.stack([X_np @ W_pair[0], np.zeros(T), X_np @ W_pair[1]], axis=1)
            else:
                logits = np.concatenate([X_np @ W_pair.T, np.zeros((T, 1))], axis=1)
            log_p = log_softmax(logits, axis=1)
            return -float(np.sum(log_p[np.arange(T), y_np]))

        n_params = (num_classes - 1) * M
        bounds = None
        x0 = np.zeros(n_params)

    if T > 0:
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
        lapse_rates = np.asarray([w_flat[M], w_flat[M + 1]], dtype=float)
        lapse_rates = np.clip(lapse_rates, 0.0, lapse_max)
        lapse_sum = float(lapse_rates.sum())
        if lapse_sum > 1.0:
            lapse_rates /= lapse_sum
        w_flat[M:M + 2] = lapse_rates
        nll = float(neg_log_likelihood(w_flat))
        w_flat_w = w_flat[:M]
    else:
        lapse_rates = np.zeros(2, dtype=float)
        w_flat_w = w_flat

    W_pair = w_flat_w.reshape(num_classes - 1, M)
    if num_classes == 3:
        weights = np.stack([W_pair[0], np.zeros(M), W_pair[1]], axis=0)
        logits = np.stack([X_np @ W_pair[0], np.zeros(T), X_np @ W_pair[1]], axis=1)
        predictive_probs = softmax(logits, axis=1)
    elif num_classes == 2:
        weights = np.stack([W_pair[0], np.zeros(M)], axis=0)
        p_right_base = expit(-(X_np @ W_pair[0]))
        if fit_lapse:
            g_left, g_right = lapse_rates
            p_right = g_left + (1.0 - g_left - g_right) * p_right_base
        else:
            p_right = p_right_base
        predictive_probs = np.stack([1.0 - p_right, p_right], axis=1)
    else:
        weights = np.concatenate([W_pair, np.zeros((1, M))], axis=0)
        logits = np.concatenate([X_np @ W_pair.T, np.zeros((T, 1))], axis=1)
        predictive_probs = softmax(logits, axis=1)

    return GLMFitResult(
        weights=weights,
        predictive_probs=predictive_probs,
        lapse_rates=lapse_rates,
        negative_log_likelihood=nll,
        success=success,
        y=y_np,
        X=X_np,
    )
