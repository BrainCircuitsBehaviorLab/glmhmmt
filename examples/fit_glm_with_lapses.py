from __future__ import annotations

import numpy as np

from glm import fit_glm


def main() -> None:
    rng = np.random.default_rng(7)
    num_trials = 500

    X = rng.normal(size=(num_trials, 4))

    true_w = np.array([1.4, -0.9, 0.5, 0.2])
    gamma_left = 0.06
    gamma_right = 0.03

    linear_score = X @ true_w
    p_right_base = 1.0 / (1.0 + np.exp(linear_score))
    p_right = gamma_left + (1.0 - gamma_left - gamma_right) * p_right_base
    y = rng.binomial(1, p_right, size=num_trials)

    fit = fit_glm(X, y, num_classes=2, lapse=True, lapse_max=0.2)

    predicted_choice = np.argmax(fit.predictive_probs, axis=1)
    accuracy = np.mean(predicted_choice == y)

    print("fit success:", fit.success)
    print("num trials:", fit.num_trials)
    print("weights shape:", fit.weights.shape)
    print("estimated weights for class 0:", fit.weights[0])
    print("estimated lapse rates [gamma_left, gamma_right]:", fit.lapse_rates)
    print("negative log likelihood:", fit.negative_log_likelihood)
    print("training accuracy:", accuracy)
    print("first 5 predicted probabilities:")
    print(fit.predictive_probs[:5])


if __name__ == "__main__":
    main()