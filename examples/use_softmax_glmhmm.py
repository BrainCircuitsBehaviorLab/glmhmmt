"""Minimal direct use of ``glmhmmt.SoftmaxGLMHMM``.

This example avoids task adapters completely and shows the core package API
that another lab member can import into their own project.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from glmhmmt import SoftmaxGLMHMM


def main() -> None:
    num_trials = 200
    num_states = 3
    num_classes = 2
    emission_input_dim = 4
    transition_input_dim = 0

    key = jr.PRNGKey(0)
    X = jr.normal(key, (num_trials, emission_input_dim))
    U = jnp.empty((num_trials, 0), dtype=X.dtype)
    inputs = jnp.concatenate([X, U], axis=1)

    y = jnp.asarray((X[:, 0] > 0).astype(jnp.int32))
    session_ids = jnp.zeros((num_trials,), dtype=jnp.int32)

    model = SoftmaxGLMHMM(
        num_states=num_states,
        num_classes=num_classes,
        emission_input_dim=emission_input_dim,
        transition_input_dim=transition_input_dim,
    )
    params, props = model.initialize(jr.PRNGKey(1))
    fitted_params, log_probs = model.fit_em_multisession(
        params,
        props,
        y,
        inputs,
        session_ids,
        num_iters=5,
        verbose=False,
    )

    pred = model.predict_choice_probs_multisession(fitted_params, y, inputs, session_ids)
    if model.transition_input_dim == 0:
        transition = fitted_params.transitions.transition_matrix
    else:
        transition = model.transition_component._compute_transition_matrices(
            fitted_params.transitions,
            inputs,
        )
    print("log_probs:", log_probs)
    print("pred shape:", pred.shape)
    print("transition shape:", transition.shape)


if __name__ == "__main__":
    main()
