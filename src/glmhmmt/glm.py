from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import softmax


LAPSE_MODES = {"none", "class", "history", "history_conditioned"}
ProgressCallback = Callable[[dict[str, Any]], None]


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

    def simulate(
        self,
        *,
        seed: int | None = None,
        n_simulations: int = 1,
    ) -> np.ndarray:
        """Simulate choice sequences from this fitted GLM on its stored design matrix."""
        return simulate_glm_choices(
            self.X,
            self.weights,
            baseline_class_idx=0,
            lapse_mode=self.lapse_mode,
            lapse_rates=self.lapse_rates,
            seed=seed,
            n_simulations=n_simulations,
        )


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


def _validate_baseline_class_idx(num_classes: int, baseline_class_idx: int) -> int:
    baseline = int(baseline_class_idx)
    if not 0 <= baseline < int(num_classes):
        raise ValueError(
            f"baseline_class_idx must be in [0, {int(num_classes) - 1}]; got {baseline}."
        )
    return baseline


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


def _base_logits(
    X_np: np.ndarray,
    w_flat_w: np.ndarray,
    num_classes: int,
    baseline_class_idx: int,
) -> np.ndarray:
    T, M = X_np.shape
    W_pair = w_flat_w.reshape(num_classes - 1, M)
    logits_without_baseline = X_np @ W_pair.T
    zero = np.zeros((T, 1), dtype=logits_without_baseline.dtype)
    baseline = int(baseline_class_idx)
    return np.concatenate(
        [
            logits_without_baseline[:, :baseline],
            zero,
            logits_without_baseline[:, baseline:],
        ],
        axis=1,
    )


def _base_predictive_probs(
    X_np: np.ndarray,
    w_flat_w: np.ndarray,
    num_classes: int,
    baseline_class_idx: int,
) -> np.ndarray:
    return softmax(_base_logits(X_np, w_flat_w, num_classes, baseline_class_idx), axis=1)


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
    targets[1:] = 1.0 / float(num_classes - 1)
    targets[np.arange(1, T), prev] = 0.0
    return targets


def _apply_lapse_mode(
    base_probs: np.ndarray,
    y_np: np.ndarray,
    num_classes: int,
    lapse_mode: str,
    lapse_rates: np.ndarray,
    *,
    repeat_targets: np.ndarray | None = None,
    alternate_targets: np.ndarray | None = None,
    prev_choices: np.ndarray | None = None,
) -> np.ndarray:
    probs = np.asarray(base_probs, dtype=float)
    if lapse_mode == "none":
        return probs

    lapse_rates = np.asarray(lapse_rates, dtype=float)
    if lapse_mode == "class":
        total_mass = float(lapse_rates.sum())
        return lapse_rates[None, :] + (1.0 - total_mass) * probs

    if repeat_targets is None:
        repeat_targets = _history_repeat_targets(y_np, num_classes)
    if alternate_targets is None:
        alternate_targets = _history_alternate_targets(y_np, num_classes)
    if prev_choices is None:
        prev_choices = y_np[:-1]
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
                trial_mass[1:] = repeat_rates[prev_choices] + alternate_rates[prev_choices]
            adjusted = probs * (1.0 - trial_mass[:, None])
            adjusted += repeat_targets * np.concatenate([[0.0], repeat_rates[prev_choices]])[:, None]
            adjusted += alternate_targets * np.concatenate([[0.0], alternate_rates[prev_choices]])[:, None]
        adjusted[0] = probs[0]
        return adjusted
    return probs


def _resolve_full_weight_matrix(
    weights: ArrayLike,
    *,
    baseline_class_idx: int,
    num_classes: int | None = None,
) -> np.ndarray:
    W = np.asarray(weights, dtype=float)
    if W.ndim != 2:
        raise ValueError(f"weights must be a 2D array; got shape {W.shape}.")

    baseline = int(baseline_class_idx)
    if num_classes is None:
        num_classes = W.shape[0] if baseline < W.shape[0] else W.shape[0] + 1
    num_classes = int(num_classes)
    if not 0 <= baseline < num_classes:
        raise ValueError(f"baseline_class_idx must be in [0, {num_classes - 1}].")

    if W.shape[0] == num_classes:
        return W
    if W.shape[0] != num_classes - 1:
        raise ValueError(
            "weights must have either num_classes rows or num_classes - 1 rows; "
            f"got {W.shape[0]} rows for num_classes={num_classes}."
        )

    zero_row = np.zeros((1, W.shape[1]), dtype=W.dtype)
    return np.concatenate([W[:baseline], zero_row, W[baseline:]], axis=0)


def glm_probs_from_weights(
    X: ArrayLike,
    weights: ArrayLike,
    *,
    baseline_class_idx: int = 0,
    num_classes: int | None = None,
) -> np.ndarray:
    """Compute base softmax GLM probabilities from a design matrix and weights."""
    X_np = np.asarray(X, dtype=float)
    if X_np.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X_np.shape}.")
    W = _resolve_full_weight_matrix(
        weights,
        baseline_class_idx=baseline_class_idx,
        num_classes=num_classes,
    )
    if X_np.shape[1] != W.shape[1]:
        raise ValueError(f"X has {X_np.shape[1]} columns but weights have {W.shape[1]}.")
    return softmax(X_np @ W.T, axis=1)


def simulate_glm_choices(
    X: ArrayLike,
    weights: ArrayLike,
    *,
    baseline_class_idx: int = 0,
    num_classes: int | None = None,
    lapse_mode: str | None = "none",
    lapse_rates: ArrayLike | None = None,
    seed: int | None = None,
    n_simulations: int = 1,
) -> np.ndarray:
    """Simulate categorical choices from a fitted GLM on a fixed design matrix.

    The design matrix is treated as fixed. History-conditioned lapse modes use
    previously simulated choices, but any history regressors already present in
    ``X`` are not recomputed after each simulated choice.
    """
    base_probs = glm_probs_from_weights(
        X,
        weights,
        baseline_class_idx=baseline_class_idx,
        num_classes=num_classes,
    )
    T, C = base_probs.shape
    lapse_mode = _normalize_lapse_mode(lapse_mode, None)
    lapse_rates_np = (
        np.zeros(_num_lapse_params(C, lapse_mode), dtype=float)
        if lapse_rates is None
        else np.asarray(lapse_rates, dtype=float).reshape(-1)
    )
    rng = np.random.default_rng(seed)
    n_simulations = int(n_simulations)
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    simulations = np.zeros((n_simulations, T), dtype=int)
    for sim_idx in range(n_simulations):
        for t in range(T):
            probs_t = base_probs[t].copy()
            if t > 0 and lapse_mode == "history":
                prev = int(simulations[sim_idx, t - 1])
                repeat_rate = float(lapse_rates_np[0]) if lapse_rates_np.size > 0 else 0.0
                alternate_rate = float(lapse_rates_np[1]) if lapse_rates_np.size > 1 else 0.0
                repeat_target = np.zeros(C, dtype=float)
                repeat_target[prev] = 1.0
                alternate_target = np.full(C, 1.0 / max(1, C - 1), dtype=float)
                alternate_target[prev] = 0.0
                probs_t = (1.0 - repeat_rate - alternate_rate) * probs_t
                probs_t += repeat_rate * repeat_target + alternate_rate * alternate_target
            elif t > 0 and lapse_mode == "history_conditioned":
                prev = int(simulations[sim_idx, t - 1])
                repeat_rates = lapse_rates_np[:C] if lapse_rates_np.size >= C else np.zeros(C, dtype=float)
                alternate_rates = (
                    lapse_rates_np[C : 2 * C]
                    if lapse_rates_np.size >= 2 * C
                    else np.zeros(C, dtype=float)
                )
                repeat_rate = float(repeat_rates[prev])
                alternate_rate = float(alternate_rates[prev])
                repeat_target = np.zeros(C, dtype=float)
                repeat_target[prev] = 1.0
                alternate_target = np.full(C, 1.0 / max(1, C - 1), dtype=float)
                alternate_target[prev] = 0.0
                probs_t = (1.0 - repeat_rate - alternate_rate) * probs_t
                probs_t += repeat_rate * repeat_target + alternate_rate * alternate_target
            elif lapse_mode == "class" and lapse_rates_np.size:
                total_mass = float(np.sum(lapse_rates_np))
                probs_t = lapse_rates_np + (1.0 - total_mass) * probs_t

            probs_t = np.clip(probs_t, 1e-12, 1.0)
            probs_t /= probs_t.sum()
            simulations[sim_idx, t] = int(rng.choice(C, p=probs_t))
    return simulations


@dataclass(frozen=True)
class PrivateAlternativeGLMFitResult:
    """Result of fitting a GLM with alternative-private regressors."""

    private_weights: np.ndarray
    private_bias: np.ndarray
    predictive_probs: np.ndarray
    negative_log_likelihood: float
    success: bool
    y: np.ndarray
    X_private: np.ndarray
    n_iter: int
    nll_history: tuple[float, ...]

    @property
    def num_trials(self) -> int:
        return int(self.X_private.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.X_private.shape[1])


def _validate_private_alternative_inputs(
    X_private: ArrayLike,
    y: ArrayLike,
    *,
    num_classes: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    X_np = np.asarray(X_private, dtype=float)
    y_np = np.asarray(y, dtype=int).reshape(-1)

    if X_np.ndim != 3:
        raise ValueError(f"X_private must be a 3D array of shape (T, C, M); got ndim={X_np.ndim}.")
    if not np.isfinite(X_np).all():
        raise ValueError("X_private must contain only finite values.")
    if y_np.ndim != 1:
        raise ValueError(f"y must be a 1D array of shape (T,); got ndim={y_np.ndim}.")
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError(
            "X_private and y must have the same number of trials; "
            f"got X_private.shape[0]={X_np.shape[0]} and y.shape[0]={y_np.shape[0]}."
        )

    resolved_num_classes = int(num_classes) if num_classes is not None else int(X_np.shape[1])
    if X_np.shape[1] != resolved_num_classes:
        raise ValueError(
            f"X_private has {X_np.shape[1]} alternatives but num_classes={resolved_num_classes}."
        )
    if resolved_num_classes < 2:
        raise ValueError(f"num_classes must be at least 2; got {resolved_num_classes}.")
    if y_np.size and y_np.min() < 0:
        raise ValueError("y must contain non-negative class indices.")
    if y_np.size and y_np.max() >= resolved_num_classes:
        raise ValueError(
            f"y contains class index {int(y_np.max())}, but num_classes={resolved_num_classes}."
        )
    return X_np, y_np, resolved_num_classes


def private_alternative_logits(
    X_private: ArrayLike,
    private_weights: ArrayLike,
    private_bias: ArrayLike | None = None,
) -> np.ndarray:
    """Compute logits for a shared-weight alternative-private GLM."""
    X_np = np.asarray(X_private, dtype=float)
    weights = np.asarray(private_weights, dtype=float).reshape(-1)
    if X_np.ndim != 3:
        raise ValueError(f"X_private must be a 3D array of shape (T, C, M); got shape {X_np.shape}.")
    if X_np.shape[2] != weights.shape[0]:
        raise ValueError(f"X_private has {X_np.shape[2]} features but private_weights has {weights.shape[0]}.")
    logits = np.einsum("tcm,m->tc", X_np, weights)
    if private_bias is not None:
        bias = np.asarray(private_bias, dtype=float).reshape(-1)
        if bias.shape[0] != X_np.shape[1]:
            raise ValueError(f"private_bias has {bias.shape[0]} classes but X_private has {X_np.shape[1]}.")
        logits = logits + bias[None, :]
    return logits


def private_alternative_probs(
    X_private: ArrayLike,
    private_weights: ArrayLike,
    private_bias: ArrayLike | None = None,
) -> np.ndarray:
    """Compute probabilities for a shared-weight alternative-private GLM."""
    return softmax(private_alternative_logits(X_private, private_weights, private_bias), axis=1)


def fit_private_alternative_glm(
    X_private: ArrayLike,
    y: ArrayLike,
    *,
    num_classes: int | None = None,
    include_bias: bool = True,
    bias_reference_class_idx: int = 1,
    maxiter: int = 100,
    tol: float = 1e-7,
    ridge: float = 1e-6,
    initial_weights: ArrayLike | None = None,
    initial_bias: ArrayLike | None = None,
) -> PrivateAlternativeGLMFitResult:
    """Fit a multinomial GLM with per-alternative regressors and shared weights.

    The model is ``softmax(b_i + w @ X_private[t, i])``.  When
    ``include_bias`` is true, all classes except ``bias_reference_class_idx``
    get a fitted fixed bias.
    """
    X_np, y_np, resolved_num_classes = _validate_private_alternative_inputs(
        X_private,
        y,
        num_classes=num_classes,
    )
    reference = _validate_baseline_class_idx(resolved_num_classes, bias_reference_class_idx)
    T, C, M = X_np.shape
    bias_classes = [idx for idx in range(C) if idx != reference] if include_bias else []
    n_bias = len(bias_classes)
    n_params = n_bias + M

    theta = np.zeros(n_params, dtype=float)
    if initial_bias is not None and include_bias:
        init_bias = np.asarray(initial_bias, dtype=float).reshape(-1)
        if init_bias.shape[0] != C:
            raise ValueError(f"initial_bias has {init_bias.shape[0]} classes but num_classes={C}.")
        theta[:n_bias] = init_bias[bias_classes]
    if initial_weights is not None:
        init_weights = np.asarray(initial_weights, dtype=float).reshape(-1)
        if init_weights.shape[0] != M:
            raise ValueError(f"initial_weights has {init_weights.shape[0]} entries but X_private has {M} features.")
        theta[n_bias:] = init_weights

    y_onehot = np.zeros((T, C), dtype=float)
    if T > 0:
        y_onehot[np.arange(T), y_np] = 1.0

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bias = np.zeros(C, dtype=float)
        if include_bias:
            bias[bias_classes] = params[:n_bias]
        weights = params[n_bias:]
        return bias, weights

    def objective(params: np.ndarray) -> float:
        bias, weights = unpack(params)
        probs = np.clip(private_alternative_probs(X_np, weights, bias), 1e-12, 1.0)
        return -float(np.sum(np.log(probs[np.arange(T), y_np]))) if T > 0 else float("nan")

    def design_tensor() -> np.ndarray:
        z = np.zeros((T, C, n_params), dtype=float)
        if include_bias:
            for param_idx, class_idx in enumerate(bias_classes):
                z[:, class_idx, param_idx] = 1.0
        z[:, :, n_bias:] = X_np
        return z

    Z = design_tensor()
    nll_history: list[float] = []
    success = False
    if T == 0:
        bias, weights = unpack(theta)
        return PrivateAlternativeGLMFitResult(
            private_weights=weights,
            private_bias=bias,
            predictive_probs=np.empty((0, C), dtype=float),
            negative_log_likelihood=float("nan"),
            success=False,
            y=y_np,
            X_private=X_np,
            n_iter=0,
            nll_history=tuple(),
        )

    current_nll = objective(theta)
    nll_history.append(current_nll)
    n_iter = 0
    damping = float(max(ridge, 1e-12))

    for iter_idx in range(1, int(maxiter) + 1):
        bias, weights = unpack(theta)
        probs = private_alternative_probs(X_np, weights, bias)
        residual = probs - y_onehot
        grad = np.einsum("tc,tcp->p", residual, Z)
        cov = np.eye(C, dtype=float)[None, :, :] * probs[:, :, None] - probs[:, :, None] * probs[:, None, :]
        hess = np.einsum("tcp,tcq,tqr->pr", Z, cov, Z)
        hess = hess + damping * np.eye(n_params, dtype=float)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]

        grad_norm = float(np.linalg.norm(grad, ord=2))
        if grad_norm < tol:
            success = True
            n_iter = iter_idx - 1
            break

        accepted = False
        step_scale = 1.0
        for _ in range(25):
            candidate = theta - step_scale * step
            candidate_nll = objective(candidate)
            if np.isfinite(candidate_nll) and candidate_nll <= current_nll + 1e-10:
                theta = candidate
                current_nll = candidate_nll
                nll_history.append(current_nll)
                accepted = True
                break
            step_scale *= 0.5

        n_iter = iter_idx
        if not accepted:
            break
        if len(nll_history) >= 2 and abs(nll_history[-2] - nll_history[-1]) < tol:
            success = True
            break
    else:
        success = True

    bias, weights = unpack(theta)
    probs = private_alternative_probs(X_np, weights, bias)
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    final_nll = objective(theta)
    return PrivateAlternativeGLMFitResult(
        private_weights=weights,
        private_bias=bias,
        predictive_probs=probs,
        negative_log_likelihood=final_nll,
        success=bool(success),
        y=y_np,
        X_private=X_np,
        n_iter=int(n_iter),
        nll_history=tuple(float(v) for v in nll_history),
    )


def fit_glm(
    X: ArrayLike,
    y: ArrayLike,
    *,
    num_classes: int | None = None,
    baseline_class_idx: int = 0,
    lapse_mode: str | None = None,
    lapse: bool | None = None,
    lapse_max: float = 0.2,
    n_restarts: int = 5,
    restart_noise_scale: float = 0.05,
    seed: int | None = 0,
    maxiter: int = 2000,
    ftol: float = 1e-9,
    progress_callback: ProgressCallback | None = None,
) -> GLMFitResult:
    """Fit a single-state GLM model directly from arrays."""

    X_np, y_np, num_classes = _validate_glm_inputs(X, y, num_classes=num_classes)
    baseline_class_idx = _validate_baseline_class_idx(num_classes, baseline_class_idx)
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
    repeat_targets = None
    alternate_targets = None
    prev_choices = None
    if lapse_mode in {"history", "history_conditioned"}:
        repeat_targets = _history_repeat_targets(y_np, num_classes)
        alternate_targets = _history_alternate_targets(y_np, num_classes)
        prev_choices = y_np[:-1]

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
        base_probs = _base_predictive_probs(
            X_np,
            w_flat_w,
            num_classes,
            baseline_class_idx,
        )
        probs = _apply_lapse_mode(
            base_probs,
            y_np,
            num_classes,
            lapse_mode,
            lapse_rates_local,
            repeat_targets=repeat_targets,
            alternate_targets=alternate_targets,
            prev_choices=prev_choices,
        )
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
        elif lapse_mode == "history":
            constraints = ({
                "type": "ineq",
                "fun": lambda w_flat, offset=weight_dim: 1.0
                - w_flat[offset]
                - w_flat[offset + 1],
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
                baseline_class_idx=baseline_class_idx,
                lapse_mode="none",
                lapse_max=lapse_max,
                n_restarts=1,
                restart_noise_scale=0.0,
                seed=seed,
                maxiter=maxiter,
                ftol=ftol,
                progress_callback=None,
            )
            if weight_dim > 0:
                explicit_weights = np.delete(
                    warm_start.weights,
                    baseline_class_idx,
                    axis=0,
                )
                base_x0[:weight_dim] = explicit_weights.reshape(-1)
    else:
        bounds = None
        x0 = np.zeros(n_params, dtype=float)

    if T > 0:
        if fit_lapse:
            rng = np.random.default_rng(seed)
            best_res = None
            best_fun = float("inf")

            for restart_idx in range(int(n_restarts)):
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "restart_start",
                            "restart_index": restart_idx + 1,
                            "restart_total": int(n_restarts),
                        }
                    )
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
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "restart_complete",
                            "restart_index": restart_idx + 1,
                            "restart_total": int(n_restarts),
                            "negative_log_likelihood": candidate_fun,
                            "success": candidate_success,
                        }
                    )

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
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "restart_start",
                        "restart_index": 1,
                        "restart_total": 1,
                    }
                )
            res = minimize(
                neg_log_likelihood,
                x0,
                method=method,
                bounds=bounds,
                **minimize_kwargs,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "restart_complete",
                        "restart_index": 1,
                        "restart_total": 1,
                        "negative_log_likelihood": float(res.fun),
                        "success": bool(res.success),
                    }
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
    zero_row = np.zeros((1, M), dtype=W_pair.dtype)
    weights = np.concatenate(
        [
            W_pair[:baseline_class_idx],
            zero_row,
            W_pair[baseline_class_idx:],
        ],
        axis=0,
    )

    base_probs = _base_predictive_probs(
        X_np,
        w_flat_w,
        num_classes,
        baseline_class_idx,
    )
    predictive_probs = _apply_lapse_mode(
        base_probs,
        y_np,
        num_classes,
        lapse_mode,
        lapse_rates,
        repeat_targets=repeat_targets,
        alternate_targets=alternate_targets,
        prev_choices=prev_choices,
    )
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
