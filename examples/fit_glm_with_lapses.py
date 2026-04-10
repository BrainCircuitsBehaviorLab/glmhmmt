from __future__ import annotations

import numpy as np

from glmhmmt.glm import fit_glm


def _simulate_class_lapse_example(
    rng: np.random.Generator,
    *,
    num_trials: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(num_trials, 4))
    true_w = np.array([1.4, -0.9, 0.5, 0.2])
    gamma_left = 0.06
    gamma_right = 0.03

    p_right_base = 1.0 / (1.0 + np.exp(X @ true_w))
    p_right = gamma_left + (1.0 - gamma_left - gamma_right) * p_right_base
    y = rng.binomial(1, p_right, size=num_trials)
    return X, y


def _simulate_history_lapse_example(
    rng: np.random.Generator,
    *,
    num_trials: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(num_trials, 4))
    true_w = np.array([1.1, -0.8, 0.35, 0.15])
    repeat_lapse = np.array([0.08, 0.16], dtype=float)
    alternate_lapse = np.array([0.06, 0.04], dtype=float)

    y = np.zeros(num_trials, dtype=int)
    base_probs = 1.0 / (1.0 + np.exp(X @ true_w))
    y[0] = rng.binomial(1, base_probs[0])

    for t in range(1, num_trials):
        prev = int(y[t - 1])
        p_right = float(base_probs[t])
        probs = np.array([1.0 - p_right, p_right], dtype=float)
        history_mass = repeat_lapse[prev] + alternate_lapse[prev]
        probs *= 1.0 - history_mass
        probs[prev] += repeat_lapse[prev]
        probs[1 - prev] += alternate_lapse[prev]
        y[t] = rng.choice(2, p=probs)

    return X, y


def _print_fit_summary(label: str, fit, y: np.ndarray) -> None:
    predicted_choice = np.argmax(fit.predictive_probs, axis=1)
    accuracy = np.mean(predicted_choice == y)

    print(f"\n{label}")
    print("fit success:", fit.success)
    print("lapse mode:", fit.lapse_mode)
    print("weights shape:", fit.weights.shape)
    print("lapse labels:", fit.lapse_labels)
    print("estimated lapse rates:", fit.lapse_rates)
    print("negative log likelihood:", fit.negative_log_likelihood)
    print("training accuracy:", accuracy)
    print("first 5 predicted probabilities:")
    print(fit.predictive_probs[:5])


def main() -> None:
    rng = np.random.default_rng(7)

    X_class, y_class = _simulate_class_lapse_example(rng)
    fit_class = fit_glm(X_class, y_class, num_classes=2, lapse_mode="class", lapse_max=0.2)
    _print_fit_summary("Class-lapse example", fit_class, y_class)

    X_history, y_history = _simulate_history_lapse_example(rng)
    fit_history = fit_glm(
        X_history,
        y_history,
        num_classes=2,
        lapse_mode="history",
        lapse_max=0.2,
    )
    _print_fit_summary("Combined repeat/alternate history-lapse example", fit_history, y_history)


if __name__ == "__main__":
    main()
