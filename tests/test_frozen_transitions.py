"""Transition freezes are part of the fitted model and its optimization."""
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glmhmmt import fit_hmm
from glmhmmt.model import InputDrivenSoftmaxTransitions, SoftmaxGLMHMM
from glmhmmt.cli import fit_glmhmmt as cli

NAMES = {"X_cols": ["bias", "drug_x_choice_lag_param"],
         "U_cols": ["drug_x_cumulative_reward", "Drug", "reward"]}


def arrays(sessions=2, length=6):
    rng = np.random.default_rng(14)
    n = sessions * length
    drug = (np.arange(n) % 3 != 0).astype(np.float32)
    reward = rng.normal(size=n).astype(np.float32)
    U = np.column_stack([drug * reward, drug, reward])
    X = np.column_stack([np.ones(n), rng.normal(size=n)]).astype(np.float32)
    y = (X[:, 1] + rng.normal(size=n) > 0).astype(np.int32)
    return y, X, U, np.repeat(np.arange(sessions), length)


def model(fixed=0.75, **kwargs):
    return SoftmaxGLMHMM(
        2, 2, 2, 3, emission_feature_names=NAMES["X_cols"],
        transition_feature_names=NAMES["U_cols"],
        frozen_emissions={"1": {"drug_x_choice_lag_param": 0.0}},
        frozen_transitions={"1": {"Drug": 0.0, "drug_x_cumulative_reward": fixed}},
        m_step_num_iters=8, **kwargs,
    )


@pytest.mark.parametrize("fixed", [0.0, 0.75])
@pytest.mark.parametrize("multisession", [False, True])
def test_real_em_and_adam_keep_both_freezes_exact(fixed, multisession):
    y, X, U, sessions = arrays()
    hmm = model(fixed)
    params, props = hmm.initialize(jr.PRNGKey(4))
    np.testing.assert_array_equal(params.transitions.weights[1, :, 0], np.float32(fixed))
    np.testing.assert_array_equal(params.transitions.weights[1, :, 1], 0.0)
    # A supplied warm start may violate freezes. It must be projected before
    # the first E-step, rather than ablated after optimization.
    warm = params._replace(transitions=params.transitions._replace(
        weights=jnp.full((2, 1, 3), 1.25)))
    inputs = jnp.asarray(np.column_stack([X, U]))
    state = warm
    for _ in range(3):
        if multisession:
            state, lps = hmm.fit_em_multisession(
                state, props, y, inputs, sessions, num_iters=1, verbose=False)
        else:
            state, lps = hmm.fit_em(state, props, y, inputs, num_iters=1, verbose=False)
        np.testing.assert_array_equal(state.transitions.weights[1, :, 0], np.float32(fixed))
        np.testing.assert_array_equal(state.transitions.weights[1, :, 1], 0.0)
        np.testing.assert_array_equal(state.emissions.weights[1, :, 1], 0.0)
        assert np.all(np.isfinite(lps))
    assert not np.array_equal(state.transitions.weights[0], warm.transitions.weights[0])
    assert not np.array_equal(state.transitions.weights[1, :, 2], warm.transitions.weights[1, :, 2])
    assert not np.array_equal(state.transitions.bias, warm.transitions.bias)
    assert not np.array_equal(state.emissions.weights[0], warm.emissions.weights[0])


def test_m_step_gradients_and_every_adam_step_use_constrained_weights():
    seen = []
    adam = optax.adam(0.03)

    def update(grads, opt_state, params):
        jax.debug.callback(lambda g, p: seen.append((np.asarray(g), np.asarray(p))),
                           grads[1], params[1], ordered=True)
        return adam.update(grads, opt_state, params)

    transitions = InputDrivenSoftmaxTransitions(
        3, 2, 3, frozen_transitions={"1": {"Drug": 23.5}},
        transition_feature_names=NAMES["U_cols"],
        m_step_optimizer=optax.GradientTransformation(adam.init, update),
        m_step_num_iters=6, stickiness=0.0,
    )
    _, X, U, _ = arrays(sessions=1)
    params, props = transitions.initialize(weights=jnp.ones((3, 2, 3)) * 0.4)
    # Non-uniform edge counts make free gradients unambiguously nonzero.
    xi = jnp.asarray(np.random.default_rng(2).uniform(size=(1, 5, 3, 3)))
    inputs = jnp.asarray(np.column_stack([X, U]))[None, 1:]
    mask = jnp.zeros((3, 2, 3), dtype=bool).at[1, :, 1].set(True)

    def reference_loss(b, w):
        w = jnp.where(mask, 23.5, w)
        full_b = jnp.stack([jnp.insert(b[i], i, 0.0) for i in range(3)])
        full_w = jnp.stack([jnp.insert(w[i], i, 0.0, axis=0) for i in range(3)])
        logp = jax.nn.log_softmax(full_b + jnp.einsum("ijd,nd->nij", full_w, U[1:]), axis=-1)
        return -(xi[0] * logp).sum() + 1e-3 * ((b*b).sum() + (w*w).sum())

    expected_grad = jax.grad(reference_loss, argnums=1)(params.bias, params.weights)
    fitted, _ = transitions.m_step(params, props, (xi, inputs), None)
    jax.block_until_ready(fitted.weights)
    assert len(seen) == 6
    np.testing.assert_allclose(seen[0][0], expected_grad, atol=2e-6)
    for grad, weights in seen:
        np.testing.assert_array_equal(grad[1, :, 1], 0.0)
        np.testing.assert_array_equal(weights[1, :, 1], 23.5)
        assert np.any(grad[0] != 0.0)
    np.testing.assert_array_equal(fitted.weights[1, :, 1], 23.5)
    assert not np.array_equal(fitted.weights[0], params.weights[0])
    assert not np.array_equal(fitted.weights[1, :, 0], params.weights[1, :, 0])


@pytest.mark.parametrize("shape", [(2, 1, 3), (2, 2, 3), (2, 3), (1, 3)])
def test_supplied_and_legacy_weights_are_frozen_after_coercion(shape):
    transitions = model().transition_component
    supplied = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    params, _ = transitions.initialize(weights=supplied, bias=jnp.array([[-2.0], [-3.0]]))
    expected = transitions._coerce_weights(supplied).at[1, :, 0].set(0.75).at[1, :, 1].set(0.0)
    np.testing.assert_array_equal(params.weights, expected)
    np.testing.assert_array_equal(params.bias, [[-2.0], [-3.0]])


def test_likelihood_and_gradients_match_explicit_restricted_path_sum():
    y, X, U, _ = arrays(sessions=1, length=4)
    hmm = model()
    params, _ = hmm.initialize(transition_weights=jnp.ones((2, 1, 3)))
    warm_weights = params.transitions.weights.at[1, :, :2].set(7.0)
    inputs = jnp.asarray(np.column_stack([X, U]))

    def params_with(w):
        return params._replace(transitions=params.transitions._replace(weights=w))

    def reference_ll(w):
        # Independent enumeration, using U[t] on entry to trial t.
        w = w.at[1, :, 0].set(0.75).at[1, :, 1].set(0.0)
        logits = params.transitions.bias[:, 0] + jnp.einsum("kd,td->tk", w[:, 0], U)
        logits = jnp.stack([jnp.stack([jnp.zeros(len(y)), logits[:, 0]], axis=-1),
                            jnp.stack([logits[:, 1], jnp.zeros(len(y))], axis=-1)], axis=1)
        log_a = jax.nn.log_softmax(logits, axis=-1)
        eta = jnp.einsum("kcm,tm->tkc", params.emissions.weights, X)
        log_y = jax.nn.log_softmax(jnp.concatenate([jnp.zeros_like(eta), eta], axis=-1), axis=-1)
        path_ll = []
        for path in product(range(2), repeat=len(y)):
            lp = jnp.log(params.initial.probs[path[0]])
            for t, state in enumerate(path):
                lp += log_y[t, state, y[t]]
                if t:
                    lp += log_a[t, path[t-1], state]
            path_ll.append(lp)
        return jax.scipy.special.logsumexp(jnp.stack(path_ll))

    def actual_ll(w):
        return hmm.marginal_log_prob(params_with(w), y, inputs)

    np.testing.assert_allclose(actual_ll(warm_weights), reference_ll(warm_weights), atol=1e-6)
    grad = jax.grad(actual_ll)(warm_weights)
    np.testing.assert_allclose(grad, jax.grad(reference_ll)(warm_weights), atol=2e-6)
    np.testing.assert_array_equal(grad[1, :, :2], 0.0)
    assert np.any(np.asarray(grad[0]) != 0.0)
    projected = hmm._constrain_transition_params(params_with(warm_weights))
    np.testing.assert_allclose(hmm.log_prior(params_with(warm_weights)), hmm.log_prior(projected))
    matrices = hmm.transition_component._compute_transition_matrices(projected.transitions, inputs)
    changed = inputs.at[0, 2:].set(99.0)
    np.testing.assert_array_equal(matrices, hmm.transition_component._compute_transition_matrices(projected.transitions, changed))
    np.testing.assert_allclose(matrices[0, 1], hmm.transition_component.distribution(projected.transitions, 1, inputs[1]).probs_parameter())


def test_array_api_propagates_both_freezes_to_every_cv_fit(monkeypatch):
    import importlib
    fitting = importlib.import_module("glmhmmt.fit")
    fits = []
    original = fitting._fit_arrays

    def record(*args, **kwargs):
        fit = original(*args, **kwargs)
        fits.append(fit)
        return fit

    monkeypatch.setattr(fitting, "_fit_arrays", record)
    y, X, U, sessions = arrays()
    result = fit_hmm(
        y, X, sessions, U=U, num_states=2, num_classes=2,
        emission_names=NAMES["X_cols"], transition_names=NAMES["U_cols"],
        frozen_emissions={"1": {"drug_x_choice_lag_param": 0.0}},
        frozen_transitions={"1": {"drug_x_cumulative_reward": 0.75}},
        cv_folds=2, num_iters=2, m_step_num_iters=4, n_restarts=1,
        stickiness=0.0, verbose=False,
    )
    assert result.frozen_transitions == {1: {"drug_x_cumulative_reward": 0.75}}
    assert len(fits) == 3
    for fit in fits:
        np.testing.assert_array_equal(fit.fitted_params.transitions.weights[1, :, 0], 0.75)
        np.testing.assert_array_equal(fit.fitted_params.emissions.weights[1, :, 1], 0.0)


@pytest.mark.parametrize("frozen", [None, {}])
def test_no_freeze_matches_original_unconstrained_adam(frozen):
    trans = InputDrivenSoftmaxTransitions(2, 2, 3, stickiness=0.0, m_step_num_iters=5,
                                        frozen_transitions=frozen)
    params, props = trans.initialize(jr.PRNGKey(9))
    _, X, U, _ = arrays(sessions=1)
    xi = jnp.asarray(np.random.default_rng(7).uniform(size=(1, 5, 2, 2)))
    inputs = jnp.asarray(np.column_stack([X, U]))[None, 1:]

    def old_loss(b, w):
        full_b = jnp.stack([jnp.array([0.0, b[0, 0]]), jnp.array([b[1, 0], 0.0])])
        full_w = jnp.stack([jnp.stack([jnp.zeros(3), w[0, 0]]), jnp.stack([w[1, 0], jnp.zeros(3)])])
        logits = full_b + jnp.einsum("ijd,nd->nij", full_w, U[1:])
        return -(xi[0] * jax.nn.log_softmax(logits, axis=-1)).sum() + 1e-3 * ((b*b).sum() + (w*w).sum())

    pair = (params.bias, params.weights)
    opt_state = trans.m_step_optimizer.init(pair)
    for _ in range(5):
        grads = jax.grad(old_loss, argnums=(0, 1))(*pair)
        updates, opt_state = trans.m_step_optimizer.update(grads, opt_state, pair)
        pair = tuple(jnp.clip(p, -20.0, 20.0) for p in optax.apply_updates(pair, updates))
    fitted, _ = trans.m_step(params, props, (xi, inputs), None)
    np.testing.assert_allclose(fitted.bias, pair[0], atol=1e-7)
    np.testing.assert_allclose(fitted.weights, pair[1], atol=1e-7)
    assert props.weights.constrainer is None


@pytest.mark.parametrize(("spec", "message"), [
    ({"-1": {"Drug": 0}}, "state index"),
    ({2: {"Drug": 0}}, "state index"),
    ({1: {"missing": 0}}, "transition_feature_names"),
    ({1: {"Drug": np.nan}}, "finite"),
    ({"bad": {"Drug": 0}}, "integer"),
])
def test_invalid_transition_freezes_fail_before_fitting(spec, message):
    with pytest.raises(ValueError, match=message):
        SoftmaxGLMHMM(2, 2, 2, 3, frozen_transitions=spec,
                      transition_feature_names=NAMES["U_cols"])


def test_transition_freezes_require_u_and_names():
    with pytest.raises(ValueError, match="regressors"):
        SoftmaxGLMHMM(2, 2, 2, 0, frozen_transitions={1: {"Drug": 0}})
    with pytest.raises(ValueError, match="transition_feature_names"):
        SoftmaxGLMHMM(2, 2, 2, 3, frozen_transitions={1: {"Drug": 0}})


@pytest.mark.parametrize("cv", [False, True])
def test_legacy_cli_fits_and_saves_transition_freezes(monkeypatch, tmp_path, cv):
    y, X, U, sessions = arrays(sessions=5, length=4)
    df = pl.DataFrame({"session": sessions, "trial": np.arange(len(y))})
    adapter = SimpleNamespace(num_classes=2, session_col="session")
    monkeypatch.setattr(cli, "_load_subject_feature_df", lambda *a, **k: (adapter, df))

    def prepare(adapter, frame, *args, **kwargs):
        indices = frame["trial"].to_numpy()
        return (jnp.asarray(y[indices]), jnp.asarray(X[indices]), jnp.asarray(U[indices]),
                sessions[indices], NAMES.copy())

    monkeypatch.setattr(cli, "_prepare_arrays", prepare)
    fitted = []
    original_restart = cli.fit_best_restart

    def record(*args, **kwargs):
        result = original_restart(*args, **kwargs)
        fitted.append(result[0])
        return result

    monkeypatch.setattr(cli, "fit_best_restart", record)
    spec = {"1": {"Drug": 0.0, "drug_x_cumulative_reward": 0.75}}
    fit_fn = cli.fit_subject_cv if cv else cli.fit_subject
    result = fit_fn("s1", 2, num_iters=2, m_step_num_iters=4, n_restarts=1,
                    frozen_transitions=spec, stickiness=0.0, verbose=False)
    assert result["frozen_transitions"] == spec
    assert len(fitted) == (5 if cv else 1)
    for params in fitted:
        np.testing.assert_array_equal(params.transitions.weights[1, :, 0], 0.75)
        np.testing.assert_array_equal(params.transitions.weights[1, :, 1], 0.0)
    (cli.save_cv_results if cv else cli.save_results)(result, tmp_path)
    with np.load(tmp_path / "s1_K2_glmhmmt_arrays.npz", allow_pickle=True) as saved:
        assert json.loads(saved["frozen_transitions_json"].item()) == spec
        np.testing.assert_array_equal(saved["transition_weights"], result["fitted_params"].transitions.weights)


@pytest.mark.parametrize("cv_mode", ["none", "balanced_session_holdout"])
def test_cli_main_propagates_json_config_and_model_id(monkeypatch, tmp_path, cv_mode):
    spec = {1: {"Drug": 0.0}}
    serialized = {"1": {"Drug": 0.0}}
    received = []
    adapter = SimpleNamespace(default_transition_cols=lambda: NAMES["U_cols"])
    monkeypatch.setattr(cli, "get_adapter", lambda task: adapter)
    monkeypatch.setattr(cli, "get_results_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "fit_subject", lambda *a, **kw: received.append(kw) or {})
    monkeypatch.setattr(cli, "fit_subject_cv", lambda *a, **kw: received.append(kw) or {})
    monkeypatch.setattr(cli, "save_results", lambda *a: None)
    monkeypatch.setattr(cli, "save_cv_results", lambda *a: None)
    cli.main(subjects=["s1"], K_list=[2], emission_cols=NAMES["X_cols"],
             frozen_transitions=spec, cv_mode=cv_mode, verbose=False)
    assert received[0]["frozen_transitions"] == serialized
    config_path, = tmp_path.rglob("config.json")
    cfg = json.loads(config_path.read_text())
    assert cfg["frozen_transitions"] == serialized
    common = dict(task="MCDR", K=2, tau=50.0, emission_cols=NAMES["X_cols"],
                  transition_cols=NAMES["U_cols"], cv_mode=cv_mode)
    assert cli.generate_model_id(**common, frozen_transitions=spec) == cli.generate_model_id(**common, frozen_transitions=serialized)
    assert cli.generate_model_id(**common) == cli.generate_model_id(**common, frozen_transitions={})
    assert cli.generate_model_id(**common) != cli.generate_model_id(**common, frozen_transitions=spec)


def test_warm_start_with_base_model_properties_stays_constrained():
    hmm = SoftmaxGLMHMM(
        2, 2, 2, 3, m_step_num_iters=8,
        transition_feature_names=NAMES["U_cols"],
        frozen_transitions={1: {"drug_x_cumulative_reward": 0.75, "Drug": 0.0}},
    )
    base = SoftmaxGLMHMM(2, 2, 2, 3, m_step_num_iters=8)
    params, base_props = base.initialize()
    y, X, U, sessions = arrays()
    inputs = jnp.asarray(np.column_stack([X, U]))
    fitted, _ = hmm.fit_em_multisession(params, base_props, y, inputs, sessions,
                                        num_iters=2, verbose=False)
    np.testing.assert_array_equal(fitted.transitions.weights[1, :, 0], 0.75)
    np.testing.assert_array_equal(fitted.transitions.weights[1, :, 1], 0.0)


def test_cli_parser_accepts_both_json_freezes(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "configure_paths_from_args", lambda args: None)
    monkeypatch.setattr(cli, "main", lambda **kwargs: calls.append(kwargs))
    cli.cli_main([
        "--K", "2", "--subjects", "s1",
        "--frozen_transitions", '{"1":{"Drug":0.0}}',
        "--frozen_emissions", '{"1":{"drug_x_choice_lag_param":0.0}}',
    ])
    assert calls[0]["frozen_transitions"] == {"1": {"Drug": 0.0}}
    assert calls[0]["frozen_emissions"] == {"1": {"drug_x_choice_lag_param": 0.0}}
