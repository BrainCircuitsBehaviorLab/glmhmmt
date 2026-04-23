import math
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.types import IntScalar, Scalar
from dynamax.hidden_markov_model.inference import hmm_two_filter_smoother
from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jaxtyping import Float, Array
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet


def normalize_frozen_emissions(frozen: Any) -> dict[int, dict[str, float]]:
    """Normalize a raw freeze mapping into ``{int_state: {feature: float}}``."""
    if not isinstance(frozen, dict):
        return {}

    result: dict[int, dict[str, float]] = {}
    for raw_state, raw_features in frozen.items():
        try:
            state = int(raw_state)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid state index {raw_state!r} in frozen emissions; must be an integer.")
        if not isinstance(raw_features, dict):
            raise ValueError(f"Invalid feature map for state {state} in frozen emissions; must be a dict of {{feature_name: value}}.")

        feature_map: dict[str, float] = {}
        for feature_name, raw_value in raw_features.items():
            if not isinstance(feature_name, str):
                raise ValueError(f"Invalid feature name {feature_name!r} for state {state} in frozen emissions; must be a string.")
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid freeze value {raw_value!r} for feature '{feature_name}' in state {state}; must be a float.")
            if not math.isfinite(value):
                raise ValueError(f"Invalid freeze value {raw_value!r} for feature '{feature_name}' in state {state}; must be finite.")
            feature_map[feature_name] = value

        if feature_map:
            result[state] = feature_map

    return result


def serialize_frozen_emissions(frozen: Any) -> dict[str, dict[str, float]]:
    """Convert a freeze mapping to a JSON-safe, sorted representation."""
    normalized = normalize_frozen_emissions(frozen)
    return {
        str(state): {feature: normalized[state][feature] for feature in sorted(normalized[state])}
        for state in sorted(normalized)
    }


def prune_frozen_emissions(
    frozen: Any,
    num_states: int,
    allowed_features: list[str],
) -> dict[str, dict[str, float]]:
    """Drop freezes for removed states or deselected features."""
    normalized = normalize_frozen_emissions(frozen)
    allowed = set(allowed_features)
    pruned: dict[int, dict[str, float]] = {}

    for state, feature_map in normalized.items():
        if state < 0 or state >= num_states:
            continue
        kept = {feature: value for feature, value in feature_map.items() if feature in allowed}
        if kept:
            pruned[state] = kept

    return serialize_frozen_emissions(pruned)


def _resolve_frozen(
    frozen: Dict[int, Dict[str, float]],
    feature_names: List[str],
    num_states: int,
    num_classes: int,
    input_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a feature-name freeze spec into boolean mask and values arrays.

    Translates the human-readable `{state: {feature_name: value}}` dict into dense NumPy arrays suitable for use inside a JAX bijector.

    Args:
        frozen: Per-state feature freeze specification of the form
            `{state_idx: {feature_name: fixed_value}}`.
        feature_names: Ordered list of emission feature names matching the
            columns of `X` (i.e. `names["X_cols"]` from
            the adapter design-matrix build. The position of a name in this list
            is its column index in `W`.
        num_states: Number of HMM hidden states `K`.
        num_classes: Number of choice categories `C` (output classes).
        input_dim: Number of emission features `M` (columns of `X`).

    Returns:
        A tuple `(mask, values)` where *mask* is a `bool` array of shape `(K, C-1, M)` that is `True` where a weight is frozen, and *values* is a `float32` array of the same shape holding the fixed value for each frozen entry.

    Raises:
        ValueError: If a feature name in `frozen` is not found in
            `feature_names`.

    Example:

        ```python
        mask, vals = _resolve_frozen(
            {0: {"SL": 0.0, "SR": 0.0}},
            feature_names=["biasL", "biasR", "SL", "SR"],
            num_states=3, num_classes=3, input_dim=4,
        )
        ```
    """
    shape = (num_states, num_classes - 1, input_dim)
    mask = np.zeros(shape, dtype=bool)
    values = np.zeros(shape, dtype=np.float32)
    for state_idx, feat_dict in frozen.items():
        for feat_name, val in feat_dict.items():
            if feat_name not in feature_names:
                raise ValueError(
                    f"Feature '{feat_name}' not found in emission_feature_names.\nAvailable: {feature_names}"
                )
            col = feature_names.index(feat_name)
            mask[state_idx, :, col] = True
            values[state_idx, :, col] = val
    return mask, values


def make_freeze_bijector(
    frozen: Dict[int, Dict[str, float]],
    feature_names: List[str],
    num_states: int,
    num_classes: int,
    input_dim: int,
) -> tfb.Bijector:
    """Build a TFP bijector that freezes specific (state, feature) weight entries.

    Returns a `tfb.Bijector` acting on the full weight tensor `W` of shape `(K, C-1, M)`.  Frozen entries are clamped to their fixed value on every forward pass; gradients **cannot** flow through them because the log-det-Jacobian is `-inf` for those dimensions.

    The bijector is passed as `ParameterProperties(constrainer=...)` so that dynamax enforces the constraint automatically inside the EM M-step.

    Args:
        frozen: Per-state feature freeze spec, see `_resolve_frozen`.
            Example: `{0: {"SL": 0.0, "SC": 0.0, "SR": 0.0}}`.
        feature_names: Ordered emission feature names matching the columns of
            `X`. Pass `names["X_cols"]` from the adapter design-matrix build.
        num_states: Number of HMM hidden states `K`.
        num_classes: Number of choice categories `C`.
        input_dim: Number of emission features `M`.

    Returns:
        A `tfb.Bijector` whose `_forward` replaces frozen entries with
        their fixed values and whose `_forward_log_det_jacobian` returns
        `-inf` for frozen dimensions (blocking gradient flow).

    Note:
        The bijector operates on the **entire** `(K, C-1, M)` tensor in one
        shot.  `forward_min_event_ndims` is set to `3` so TFP treats the
        whole tensor as a single event for log-det purposes.

    Example:

        ```python
        bij = make_freeze_bijector(
            frozen={0: {"SL": 0.0, "SR": 0.0}},
            feature_names=["biasL", "biasR", "SL", "SR"],
            num_states=3, num_classes=3, input_dim=4,
        )
        props = ParameterProperties(constrainer=bij)
        ```
    """
    mask, values = _resolve_frozen(frozen, feature_names, num_states, num_classes, input_dim)
    mask_jnp = jnp.array(mask)
    values_jnp = jnp.array(values)
    ndims = len(mask.shape)

    class _FeatureFreezeBijector(tfb.Bijector):
        def __init__(self):
            super().__init__(forward_min_event_ndims=ndims, name="feature_freeze")

        def _forward(self, x):
            # unconstrained → constrained: frozen entries snap to fixed value
            return jnp.where(mask_jnp, values_jnp, x)

        def _inverse(self, y):
            # constrained → unconstrained: frozen entries map to 0
            return jnp.where(mask_jnp, jnp.zeros_like(y), y)

        def _forward_log_det_jacobian(self, x):
            # frozen dims contribute -inf (zero volume); free dims contribute 0
            return jnp.where(mask_jnp, -jnp.inf, 0.0).sum()

    return _FeatureFreezeBijector()


class ParamsSoftmaxGLMHMMEmissions(NamedTuple):
    # weights: (K, C-1, M)
    weights: Union[Float[Array, "num_states num_classes_minus1 input_dim"], ParameterProperties]


class SoftmaxGLMHMMEmissions(HMMEmissions):
    """Emission component for a multinomial GLM-HMM with softmax link.

    Models choice probabilities as a softmax function of a linear combination
    of emission features:

    ```
    P(y_t = c | z_t = k, x_t) = softmax(W[k] @ x_t)[c]
    ```

    where `W` has shape `(K, C-1, M)` with weight convention
    `W[state, output_contrast, feature]`.

    Specific `(state, feature)` entries can be **frozen** to fixed values via
    a TFP bijector constrainer, excluding them from gradient updates in the EM
    M-step.

    Args:
        num_states: Number of HMM hidden states `K`.
        num_classes: Number of choice categories `C` (e.g. 3 for L / C / R).
        input_dim: Total input dimension seen by the emission distribution.
        emission_input_dim: Number of columns in the emission design matrix `X`.
        weight_scale: Standard deviation of the random normal weight
            initialisation.  Default `1.0`.
        frozen: Per-state feature freeze specification of the form
            `{state_idx: {feature_name: fixed_value}}`.  Feature names must
            appear in `emission_feature_names`; the constraint is enforced by
            a bijector so gradients cannot flow through frozen entries.
            Example: `{0: {"SL": 0.0, "SC": 0.0, "SR": 0.0}}`.
        emission_feature_names: Ordered names of the emission features matching
            the columns of `X`.  Pass `names["X_cols"]` from
            the adapter design-matrix build. **Required** when `frozen` is set.
        baseline_class_idx: Class index used as the softmax reference
            category. Its logit is fixed to zero and no explicit weight row is
            learned for it. Default `0`.
        m_step_optimizer: Optimizer used in the gradient M-step.
            Default `optax.adam(1e-2)`.
        m_step_num_iters: Number of gradient steps per M-step call.
            Default `100`.

    Raises:
        ValueError: If `frozen` is set but `emission_feature_names` is
            not provided.
        ValueError: If a feature name in `frozen` is not found in
            `emission_feature_names` (raised by `_resolve_frozen` during
            `initialize`).

    Example:

        ```python
        y, X, U, names = adapter.load_subject(df_sub)

        emissions = SoftmaxGLMHMMEmissions(
            num_states=3, num_classes=3,
            input_dim=X.shape[1], emission_input_dim=X.shape[1],
            frozen={0: {"SL": 0.0, "SC": 0.0, "SR": 0.0}},
            emission_feature_names=names["X_cols"],
        )
        params, props = emissions.initialize(jr.PRNGKey(0))
        ```
    """

    def __init__(
        self,
        num_states: int,
        num_classes: int,
        input_dim: int,
        emission_input_dim: int,
        weight_scale: Scalar = 1.0,
        frozen: Optional[Dict[int, Dict[str, float]]] = None,
        emission_feature_names: Optional[List[str]] = None,
        baseline_class_idx: int = 0,
        m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
        m_step_num_iters: int = 100,
    ):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.emission_input_dim = emission_input_dim
        self.weight_scale = weight_scale
        self.frozen = frozen or {}
        self.emission_feature_names = emission_feature_names or []
        self.baseline_class_idx = int(baseline_class_idx)
        if not 0 <= self.baseline_class_idx < self.num_classes:
            raise ValueError(
                f"baseline_class_idx must be in [0, {self.num_classes - 1}] for num_classes={self.num_classes}; "
                f"got {baseline_class_idx}."
            )
        if self.frozen and not self.emission_feature_names:
            raise ValueError(
                "emission_feature_names must be provided when frozen is set. "
                "Pass names['X_cols'] from the adapter design-matrix build."
            )

    @property
    def emission_shape(self):
        # y_t is an int -> shape ()
        return ()

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self, key, method="random", emission_weights=None):
        if emission_weights is None:
            key, subkey = jr.split(key)
            W = self.weight_scale * jr.normal(subkey, (self.num_states, self.num_classes - 1, self.input_dim))
        else:
            W = emission_weights

        # Snap frozen entries to their fixed value in the initial weights so
        # the bijector _forward is a no-op at t=0.
        if self.frozen:
            mask, values = _resolve_frozen(
                self.frozen,
                self.emission_feature_names,
                self.num_states,
                self.num_classes,
                self.input_dim,
            )
            W = jnp.where(jnp.array(mask), jnp.array(values), W)
            constrainer = make_freeze_bijector(
                self.frozen,
                self.emission_feature_names,
                self.num_states,
                self.num_classes,
                self.input_dim,
            )
            w_props = ParameterProperties(constrainer=constrainer)
        else:
            w_props = ParameterProperties()

        params = ParamsSoftmaxGLMHMMEmissions(weights=W)
        props = ParamsSoftmaxGLMHMMEmissions(weights=w_props)
        return params, props

    def log_prior(self, params):
        l2 = 1e-4
        return -l2 * jnp.sum(params.weights**2)

    def distribution(self, params, state, inputs):
        x = inputs[: self.emission_input_dim]  # (M,)
        # eta: (C-1,) — one contrast per non-base class in original class order,
        # excluding the selected baseline class.
        eta = params.weights[state] @ x  # (C-1,)

        zero = jnp.zeros(1, dtype=jnp.float32)
        logits = jnp.concatenate(
            [
                eta[: self.baseline_class_idx],
                zero,
                eta[self.baseline_class_idx :],
            ]
        )
        return tfd.Categorical(logits=logits)

    def _compute_conditional_logliks(self, params, emissions, inputs=None):
        """Compute per-timestep log-likelihoods with padding-sentinel support.

        Timesteps where `emissions[t] == num_classes` are treated as padding
        and contribute log-likelihood `0` (uniform emission), so they do not
        affect the forward-backward probabilities.

        Args:
            params: Emission parameters (`ParamsSoftmaxGLMHMMEmissions`).
            emissions: Integer array of shape `(T,)` with values in
                `0 … num_classes`.  Value `num_classes` is the padding sentinel.
            inputs: Float array of shape `(T, D)`.

        Returns:
            Log-likelihood array of shape `(T, K)`; padded rows are `0`.
        """
        mask = emissions == self.num_classes  # (T,)  True = padding
        # sentinel → 0, safe for Categorical
        safe_emissions = jnp.where(mask, 0, emissions)

        def f(emission, inpt):
            return jax.vmap(lambda state: self.distribution(params, state, inpt).log_prob(emission))(
                jnp.arange(self.num_states)
            )

        lls = jax.vmap(f)(safe_emissions, inputs)  # (T, K)
        return jnp.where(mask[:, None], 0.0, lls)

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        """Collect emission sufficient statistics for the M-step.

        Replaces the padding sentinel `num_classes` with `0` before
        passing emissions to the gradient.  Padded positions are already
        zeroed in `smoothed_probs` by `SoftmaxGLMHMM.e_step`, so they
        contribute nothing to the gradient.

        Args:
            params: Current emission parameters.
            posterior: HMM posterior from the E-step.
            emissions: Integer array of shape `(T,)`.
            inputs: Float array of shape `(T, D)`.

        Returns:
            Tuple `(smoothed_probs, safe_emissions, inputs)` ready for the
            M-step gradient.
        """
        safe_emissions = jnp.where(emissions == self.num_classes, 0, emissions)
        return posterior.smoothed_probs, safe_emissions, inputs


class ParamsInputDrivenTransitions(NamedTuple):
    bias: Union[Float[Array, "num_states num_states"], ParameterProperties]
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]


class InputDrivenSoftmaxTransitions(HMMTransitions):
    def __init__(
        self,
        num_states: int,
        emission_input_dim: int,
        transition_input_dim: int,
        weight_scale=0.01,
        stickiness: float = 10.0,
        m_step_optimizer=optax.adam(1e-2),
        m_step_num_iters=50,
    ):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.emission_input_dim = emission_input_dim
        self.transition_input_dim = transition_input_dim
        self.weight_scale = weight_scale
        self.stickiness = stickiness

    def initialize(self, key: Optional[Array] = None, method: str = "prior", bias=None, weights=None):
        if key is None:
            key = jr.PRNGKey(0)
        key1, key2 = jr.split(key, 2)
        K, D = self.num_states, self.transition_input_dim

        if bias is None:
            b = jnp.zeros((K, K), dtype=jnp.float32)
            # Apply stickiness as an initial diagonal offset (learnable from here)
            b = b + self.stickiness * jnp.eye(K, dtype=jnp.float32)
        else:
            b = jnp.asarray(bias, jnp.float32)
        W = (
            self.weight_scale * jr.normal(key2, (K, K, D), dtype=jnp.float32)
            if weights is None
            else jnp.asarray(weights, jnp.float32)
        )

        params = ParamsInputDrivenTransitions(bias=b, weights=W)
        props = ParamsInputDrivenTransitions(bias=ParameterProperties(), weights=ParameterProperties())
        return params, props

    def log_prior(self, params):
        l2_bias = 1e-3
        l2_w = 1e-3
        return -(l2_bias * jnp.sum(params.bias**2) + l2_w * jnp.sum(params.weights**2))

    def distribution(self, params, state, inputs=None):
        u = inputs[self.emission_input_dim : self.emission_input_dim + self.transition_input_dim]
        i = jnp.asarray(state, jnp.int32)
        logits = params.bias[i] + jnp.einsum("kd,d->k", params.weights[i], u)
        return tfd.Categorical(logits=logits)

    def _compute_transition_matrices(self, params, inputs):
        """
        Return A[t,i,j] = P(z_{t+1}=j | z_t=i, inputs_t) for t=0..T-2
        Shape: (T-1, K, K)
        """

        # u_t used for transition t -> t+1, so only t=0..T-2
        u = inputs[:-1, self.emission_input_dim : self.emission_input_dim + self.transition_input_dim]  # (T-1, D)
        # logits[t, i, j] = bias[i,j] + sum_d W[i,j,d] * u[t,d]
        logits = params.bias[None, :, :] + jnp.einsum("ijd,td->tij", params.weights, u)  # (T-1,K,K)
        return jax.nn.softmax(logits, axis=-1)

    def collect_suff_stats(self, params, posterior, inputs=None):
        # posterior.trans_probs: (T-1, K, K) o con batch fuera
        # inputs: (T, Dall)
        # usamos inputs_t para transición t->t+1: t=0..T-2
        xi = posterior.trans_probs
        u_inputs = inputs[:-1]  # (T-1, Dall)
        return (xi, u_inputs)

    def m_step(self, params, props, batch_stats, m_step_state):
        # xi_b: (B,T-1,K,K), inputs_b: (B,T-1,Dall)
        xi_b, inputs_b = batch_stats

        B, Tm1, K, _ = xi_b.shape
        Dall = inputs_b.shape[-1]

        xi = xi_b.reshape((B * Tm1, K, K))  # (N,K,K)
        inputs_tr = inputs_b.reshape((B * Tm1, Dall))  # (N,Dall)

        u = inputs_tr[:, self.emission_input_dim : self.emission_input_dim + self.transition_input_dim]  # (N,D)

        b0, W0 = params.bias, params.weights
        l2 = 1e-3
        clip_val = 20.0

        def loss_fn(b, W):
            # logits[n,i,j] = b[i,j] + sum_d W[i,j,d] u[n,d]
            logits = b[None, :, :] + jnp.einsum("ijd,nd->nij", W, u)  # (N,K,K)
            logp = jax.nn.log_softmax(logits, axis=-1)  # (N,K,K)
            nll = -(xi * logp).sum()
            reg = l2 * (jnp.sum(b * b) + jnp.sum(W * W))
            return nll + reg

        opt = self.m_step_optimizer
        opt_state = opt.init((b0, W0))

        def step(carry, _):
            b, W, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(b, W)
            updates, opt_state = opt.update(grads, opt_state, (b, W))
            b, W = optax.apply_updates((b, W), updates)
            b = jnp.clip(b, -clip_val, clip_val)
            W = jnp.clip(W, -clip_val, clip_val)
            return (b, W, opt_state), loss

        (b, W, _), _ = jax.lax.scan(step, (b0, W0, opt_state), None, length=self.m_step_num_iters)
        return ParamsInputDrivenTransitions(bias=b, weights=W), m_step_state


class ParamsSoftmaxGLMHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: Union[ParamsStandardHMMTransitions, ParamsInputDrivenTransitions]
    emissions: ParamsSoftmaxGLMHMMEmissions


class SoftmaxGLMHMM(HMM):
    """Multinomial GLM Hidden Markov Model with optional input-driven transitions.

    Combines a softmax GLM emission model with either standard Dirichlet
    transitions or input-driven softmax transitions.  Supports per-state
    feature freezing via bijector-constrained `ParameterProperties`.

    Args:
        num_states: Number of hidden states `K`.
        num_classes: Number of choice categories `C` (e.g. 3 for L / C / R).
        emission_input_dim: Number of columns in the emission design matrix `X`.
        transition_input_dim: Number of columns in the transition design matrix `U`.  Pass `0` for standard (non-input-driven) transitions.
        initial_probs_concentration: Dirichlet concentration for the initial state distribution.  Default `1.1`.
        transition_matrix_concentration: Dirichlet concentration for transition matrix rows.  Only used when `transition_input_dim == 0`. Default `1.1`.
        transition_matrix_stickiness: Extra weight on the diagonal of the Dirichlet prior to encourage self-transitions.  Only used when `transition_input_dim == 0`.  Default `0.0`.
        weight_scale: Standard deviation of the random normal weight initialisation for emission and transition weights.  Default `1.0`.
        frozen_emissions: Per-state emission feature freeze specification of the form `{state_idx: {feature_name: fixed_value}}`.  Feature names must appear in `emission_feature_names`; frozen entries are enforced by a bijector so they receive no gradient updates. Example: `{0: {"SL": 0.0, "SC": 0.0, "SR": 0.0}}`.
        emission_feature_names: Ordered names of the `X` columns — pass `names["X_cols"]` from the adapter design-matrix build. **Required** when `frozen_emissions` is set.
        baseline_class_idx: Class index used as the softmax reference category. Its logit is fixed to zero and no explicit weight row is learned for it. Default `0`.
        m_step_optimizer: Optimizer shared by emission and transition M-steps. Default `optax.adam(1e-2)`.
        m_step_num_iters: Number of gradient steps per M-step call.Default `100`.

    Raises:
        ValueError: If `frozen_emissions` is set but
            `emission_feature_names` is not provided.

    Example:

        ```python
        import tomllib
        import jax.random as jr
        import jax.numpy as jnp
        y, X, U, names = adapter.load_subject(df_sub, tau=50.0)
        inputs = jnp.concatenate([X, U], axis=1)

        # --- plain fit ---
        model = SoftmaxGLMHMM(
            num_states=3, num_classes=3,
            emission_input_dim=X.shape[1],
            transition_input_dim=U.shape[1],
        )
        params, props = model.initialize(jr.PRNGKey(0))
        params, lps = model.fit_em_multisession(
            params, props, y, inputs, session_ids, num_iters=50
        )

        # --- with state-0 frozen as bias-only (spec from package config) ---
        from glmhmmt.runtime import load_app_config
        cfg = load_app_config()
        frozen = {int(k): v for k, v in cfg["glmhmm"]["frozen_emissions"].items()}

        model = SoftmaxGLMHMM(
            num_states=3, num_classes=3,
            emission_input_dim=X.shape[1],
            transition_input_dim=U.shape[1],
            frozen_emissions=frozen,
            emission_feature_names=names["X_cols"],
        )
        ```
    """

    def __init__(
        self,
        num_states: int,
        num_classes: int,
        emission_input_dim: int,
        transition_input_dim: int,
        initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
        transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
        transition_matrix_stickiness: Scalar = 0.0,
        weight_scale: Scalar = 1.0,
        frozen_emissions: Optional[Dict[int, Dict[str, float]]] = None,
        emission_feature_names: Optional[List[str]] = None,
        baseline_class_idx: int = 0,
        m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
        m_step_num_iters: int = 100,
    ):

        self.inputs_dim = emission_input_dim + transition_input_dim
        self.num_states = num_states
        self.num_classes = num_classes
        self.emission_input_dim = emission_input_dim
        self.transition_input_dim = transition_input_dim
        self.baseline_class_idx = int(baseline_class_idx)
        if not 0 <= self.baseline_class_idx < self.num_classes:
            raise ValueError(
                f"baseline_class_idx must be in [0, {self.num_classes - 1}] for num_classes={self.num_classes}; "
                f"got {baseline_class_idx}."
            )

        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)

        if transition_input_dim > 0:
            transition_component = InputDrivenSoftmaxTransitions(
                num_states=num_states,
                emission_input_dim=emission_input_dim,
                transition_input_dim=transition_input_dim,
                weight_scale=weight_scale,
                stickiness=transition_matrix_stickiness,
                m_step_optimizer=m_step_optimizer,
                m_step_num_iters=m_step_num_iters,
            )
        else:
            transition_component = StandardHMMTransitions(
                num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness
            )

        emission_component = SoftmaxGLMHMMEmissions(
            num_states=num_states,
            num_classes=num_classes,
            input_dim=self.emission_input_dim,
            emission_input_dim=emission_input_dim,
            weight_scale=weight_scale,
            frozen=frozen_emissions,
            emission_feature_names=emission_feature_names,
            baseline_class_idx=self.baseline_class_idx,
            m_step_optimizer=m_step_optimizer,
            m_step_num_iters=m_step_num_iters,
        )

        super().__init__(num_states, initial_component, transition_component, emission_component)

        # JIT-compile once at construction time so every call reuses the same
        self._e_step_jit = jit(self.e_step)
        self._m_step_jit = jit(self.m_step)
        self._smoother_jit = jit(self.smoother)
        self._predict_jit = jit(self.predict_choice_probs)
        self._predict_state_jit = jit(self.predict_state_probs)

        # Batched (vmap) versions used by the multisession helpers.
        self._batched_e_step_jit = jit(jax.vmap(self.e_step, in_axes=(None, 0, 0)))
        self._batched_smoother_jit = jit(jax.vmap(self.smoother, in_axes=(None, 0, 0)))
        self._batched_predict_jit = jit(jax.vmap(self.predict_choice_probs, in_axes=(None, 0, 0)))
        self._batched_predict_state_jit = jit(jax.vmap(self.predict_state_probs, in_axes=(None, 0, 0)))

    @property
    def inputs_shape(self) -> Tuple[int, ...]:
        return (self.inputs_dim,)

    def _validate_emissions(self, emissions, *, allow_padding: bool, name: str = "emissions") -> np.ndarray:
        """Validate categorical emission labels before entering JIT-compiled code."""
        emissions_np = np.asarray(emissions, dtype=int).reshape(-1)
        if emissions_np.size == 0:
            return emissions_np

        min_valid = 0
        max_valid = self.num_classes if allow_padding else (self.num_classes - 1)
        invalid_mask = (emissions_np < min_valid) | (emissions_np > max_valid)
        if np.any(invalid_mask):
            invalid_values = sorted({int(v) for v in emissions_np[invalid_mask].tolist()})
            allowed = (
                f"0..{self.num_classes} (with {self.num_classes} reserved as padding)"
                if allow_padding
                else f"0..{self.num_classes - 1}"
            )
            raise ValueError(
                f"{name} must contain categorical class indices in {allowed}; "
                f"found invalid values {invalid_values}."
            )
        return emissions_np

    def initialize(
        self,
        key: Array = jr.PRNGKey(0),
        method: str = "prior",
        initial_probs: Optional[Float[Array, " num_states"]] = None,
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
        emission_weights: Optional[Float[Array, "num_states num_classes input_dim"]] = None,
    ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialise all model parameters and their properties.

        Args:
            key: JAX PRNG key.
            method: Initialisation method forwarded to sub-components
                (`"prior"` or `"random"`).
            initial_probs: Optional fixed initial state distribution of shape
                `(K,)`.
            transition_matrix: Optional fixed transition matrix of shape
                `(K, K)`.  Ignored when using input-driven transitions.
            emission_weights: Optional fixed emission weights of shape
                `(K, C-1, M)`.

        Returns:
            Tuple `(params, props)` of `ParamsSoftmaxGLMHMM` NamedTuples.
        """
        key1, key2, key3 = jr.split(key, 3)
        params, props = dict(), dict()

        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs
        )

        if self.transition_input_dim > 0:
            params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method)
        else:
            params["transitions"], props["transitions"] = self.transition_component.initialize(
                key2, method=method, transition_matrix=transition_matrix
            )

        params["emissions"], props["emissions"] = self.emission_component.initialize(
            key3, method=method, emission_weights=emission_weights
        )

        return ParamsSoftmaxGLMHMM(**params), ParamsSoftmaxGLMHMM(**props)

    def predict_choice_probs(self, params, emissions, inputs):
        """Compute one-step-ahead predictive choice probabilities.

        Returns `p(y_t = c | y_{1:t-1}, x_{1:t})` for every timestep.

        Args:
            params: Fitted model parameters (`ParamsSoftmaxGLMHMM`).
            emissions: Integer array of shape `(T,)`.
            inputs: Float array of shape `(T, D)`.

        Returns:
            Float array of shape `(T, C)` with choice probabilities.
        """

        filt = self.filter(params=params, emissions=emissions, inputs=inputs)
        alpha = filt.filtered_probs

        T, K = alpha.shape
        A = self.transition_component._compute_transition_matrices(params.transitions, inputs)  # (T-1, K, K)

        if A.ndim == 2:  # homogeneous transitions
            A = jnp.broadcast_to(A[None, :, :], (T - 1, K, K))

        def emission_probs_t(t):
            return jax.vmap(
                lambda k: self.emission_component.distribution(params.emissions, k, inputs[t]).probs_parameter()
            )(jnp.arange(K))

        p_y_given_z = jax.vmap(emission_probs_t)(jnp.arange(T))  # (T,K,C)
        pi = params.initial.probs  # (K,)
        pred_z = jnp.vstack([pi, jnp.einsum("tk,tkj->tj", alpha[:-1], A)])

        return jnp.einsum("tk,tkc->tc", pred_z, p_y_given_z)

    def predict_state_probs(self, params, emissions, inputs):
        """Compute one-step-ahead predictive state probabilities.

        Returns ``p(z_t = k | y_{1:t-1}, x_{1:t})`` for every timestep.

        Args:
            params: Fitted model parameters (`ParamsSoftmaxGLMHMM`).
            emissions: Integer array of shape `(T,)`.
            inputs: Float array of shape `(T, D)`.

        Returns:
            Float array of shape `(T, K)` with one-step-ahead predictive
            state probabilities.
        """
        filt = self.filter(params=params, emissions=emissions, inputs=inputs)
        alpha = filt.filtered_probs

        T, K = alpha.shape
        A = self.transition_component._compute_transition_matrices(params.transitions, inputs)  # (T-1, K, K)

        if A.ndim == 2:  # homogeneous transitions
            A = jnp.broadcast_to(A[None, :, :], (T - 1, K, K))

        pi = params.initial.probs  # (K,)
        return jnp.vstack([pi, jnp.einsum("tk,tkj->tj", alpha[:-1], A)])

    def e_step(self, params, emissions, inputs=None):
        """Run the E-step with padding-sentinel masking.

        Timesteps where `emissions[t] == num_classes` are padding:
        their log-likelihoods are set to `0` (uniform) and the
        corresponding posterior marginals are zeroed so the M-step only
        accumulates real-data information.

        Args:
            params: Current model parameters (`ParamsSoftmaxGLMHMM`).
            emissions: Integer array of shape `(T,)`; value `num_classes`
                is the padding sentinel.
            inputs: Float array of shape `(T, D)`.

        Returns:
            Tuple `(batch_stats, marginal_loglik)` where *batch_stats* is a
            tuple of sufficient statistics for each component and
            *marginal_loglik* is the log-marginal likelihood scalar.
        """
        valid = emissions != self.num_classes  # (T,)  False = padding

        pi0 = self.initial_component._compute_initial_probs(params.initial, inputs)
        A = self.transition_component._compute_transition_matrices(params.transitions, inputs)
        # Broadcast time-homogeneous transition matrices to (T-1, K, K) so
        # hmm_two_filter_smoother doesn't auto-sum padded steps over time.
        if A.ndim == 2:
            T = emissions.shape[0]
            if T > 0:
                A = jnp.broadcast_to(A[None, :, :], (T - 1, self.num_states, self.num_states))
        lls = self.emission_component._compute_conditional_logliks(params.emissions, emissions, inputs)

        args = (pi0, A, lls)
        posterior = hmm_two_filter_smoother(*args)

        # Zero smoothed marginals at padded steps
        smoothed_probs = jnp.where(valid[:, None], posterior.smoothed_probs, 0.0)

        # Zero pairwise marginals
        if posterior.trans_probs.ndim == 2:  # (K, K) — pre-summed
            trans_probs = posterior.trans_probs  # cannot mask per-step
        else:  # (T-1, K, K)
            valid_trans = (valid[:-1] & valid[1:])[:, None, None]
            trans_probs = jnp.where(valid_trans, posterior.trans_probs, 0.0)

        # Re-sum to (K, K) if the transition component expects time-homogeneous stats
        if self.transition_input_dim == 0 and trans_probs.ndim == 3:
            trans_probs = trans_probs.sum(axis=0)

        masked_post = posterior._replace(smoothed_probs=smoothed_probs, trans_probs=trans_probs)

        initial_stats = self.initial_component.collect_suff_stats(params.initial, masked_post, inputs)
        transition_stats = self.transition_component.collect_suff_stats(params.transitions, masked_post, inputs)
        emission_stats = self.emission_component.collect_suff_stats(params.emissions, masked_post, emissions, inputs)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    # ------------------------------------------------------------------
    # Multi-session helpers
    # ------------------------------------------------------------------

    def _split_by_session(self, emissions, inputs, session_ids, min_length: int = 2):
        """Split (emissions, inputs) into per-session lists.

        Args:
            emissions: Integer array of shape `(T,)`.
            inputs: Float array of shape `(T, D)`.
            session_ids: Array of shape `(T,)` with a session identifier per
                trial.  Trials are grouped by unique values in original order.
            min_length: Sessions with fewer than this many trials are dropped
                (they cannot produce a transition step).  Default `2`.

        Returns:
            List of `(emissions_s, inputs_s)` tuples, one per valid session.
        """
        session_ids_np = np.asarray(session_ids)
        _, first_idx = np.unique(session_ids_np, return_index=True)
        unique_sessions = session_ids_np[np.sort(first_idx)]
        emissions_np = np.asarray(emissions)
        inputs_np = np.asarray(inputs)
        sessions = []
        for s in unique_sessions:
            mask = session_ids_np == s
            if mask.sum() < min_length:
                continue
            sessions.append(
                (
                    jnp.array(emissions_np[mask]),
                    jnp.array(inputs_np[mask]),
                )
            )
        return sessions

    def _pad_sessions(self, sessions):
        """Pad a list of per-session arrays to a common length for batched vmapping.

        Emissions are padded with `num_classes` (the sentinel value) and
        inputs with `0`.

        Args:
            sessions: List of `(emissions_s, inputs_s)` tuples as returned
                by `_split_by_session`.

        Returns:
            A tuple `(e_pad, i_pad, lengths)` where *e_pad* has shape
            `(S, T_max)` (int32), *i_pad* has shape `(S, T_max, D)`
            (float32), and *lengths* is a list of true per-session lengths.
        """
        lengths = [int(e.shape[0]) for e, _ in sessions]
        T_max = max(lengths)
        S = len(sessions)
        D = int(sessions[0][1].shape[-1])

        e_pad = np.full((S, T_max), self.num_classes, dtype=np.int32)
        i_pad = np.zeros((S, T_max, D), dtype=np.float32)
        for idx, (e_s, i_s) in enumerate(sessions):
            T_s = lengths[idx]
            e_pad[idx, :T_s] = np.asarray(e_s)
            i_pad[idx, :T_s] = np.asarray(i_s)
        return jnp.array(e_pad), jnp.array(i_pad), lengths

    def fit_em_multisession(self, params, props, emissions, inputs, session_ids, num_iters=50, verbose=True):
        """Fit the model via EM over multiple independent sessions.

        Each session resets to `pi0` (no probability leak across session
        boundaries).  Sessions are padded and processed in a single vmapped
        E-step per EM iteration, compiled once by XLA.

        Args:
            params: Initial parameters (`ParamsSoftmaxGLMHMM`).
            props: Parameter properties (`ParamsSoftmaxGLMHMM` of
                `ParameterProperties`).
            emissions: Integer array of shape `(T_total,)`.
            inputs: Float array of shape `(T_total, D)`.
            session_ids: Array of shape `(T_total,)` mapping each trial to a
                session.
            num_iters: Number of EM iterations.  Default `50`.
            verbose: Print a progress bar with log-prob.  Default `True`.

        Returns:
            Tuple `(fitted_params, log_probs)` where *log_probs* is a JAX
            array of length `num_iters`.
        """
        from tqdm.auto import trange

        self._validate_emissions(emissions, allow_padding=False, name="emissions")
        sessions = self._split_by_session(emissions, inputs, session_ids)
        e_pad, i_pad, _ = self._pad_sessions(sessions)
        m_step_state = self.initialize_m_step_state(params, props)

        log_probs = []
        pbar = trange(num_iters, desc="EM") if verbose else range(num_iters)
        for _ in pbar:
            # Single vmapped E-step over all padded sessions (one XLA call)
            batch_stats, ll_batch = self._batched_e_step_jit(params, e_pad, i_pad)
            total_ll = float(jnp.sum(ll_batch))

            lp = self.log_prior(params) + total_ll
            log_probs.append(lp)
            if verbose:
                pbar.set_postfix({"log prob": f"{lp:.1f}"})

            # M-step: batch_stats has a leading session axis added by vmap;
            # each component's m_step sums over that axis internally.
            params, m_step_state = self._m_step_jit(params, props, batch_stats, m_step_state)

        return params, jnp.array(log_probs)

    def smoother_multisession(self, params, emissions, inputs, session_ids):
        """Run the smoother per session and concatenate results.

        Args:
            params: Fitted model parameters (`ParamsSoftmaxGLMHMM`).
            emissions: Integer array of shape `(T_total,)`.
            inputs: Float array of shape `(T_total, D)`.
            session_ids: Array of shape `(T_total,)`.

        Returns:
            Float array of shape `(T_total, K)` with smoothed state
            probabilities.
        """
        self._validate_emissions(emissions, allow_padding=False, name="emissions")
        sessions = self._split_by_session(emissions, inputs, session_ids)
        e_pad, i_pad, lengths = self._pad_sessions(sessions)
        posterior_batch = self._batched_smoother_jit(params, e_pad, i_pad)
        smoothed = np.asarray(posterior_batch.smoothed_probs)  # (S, T_max, K)
        return np.concatenate([smoothed[i, :T_s] for i, T_s in enumerate(lengths)], axis=0)

    def predict_choice_probs_multisession(self, params, emissions, inputs, session_ids):
        """Compute one-step-ahead predictive choice probabilities for all sessions.

        Args:
            params: Fitted model parameters (`ParamsSoftmaxGLMHMM`).
            emissions: Integer array of shape `(T_total,)`.
            inputs: Float array of shape `(T_total, D)`.
            session_ids: Array of shape `(T_total,)`.

        Returns:
            Float array of shape `(T_total, C)` with predictive choice
            probabilities.
        """
        sessions = self._split_by_session(emissions, inputs, session_ids)
        e_pad, i_pad, lengths = self._pad_sessions(sessions)
        probs_batch = np.asarray(self._batched_predict_jit(params, e_pad, i_pad))  # (S, T_max, C)
        return np.concatenate([probs_batch[i, :T_s] for i, T_s in enumerate(lengths)], axis=0)

    def predict_state_probs_multisession(self, params, emissions, inputs, session_ids):
        """Compute one-step-ahead predictive state probabilities for all sessions.

        Args:
            params: Fitted model parameters (`ParamsSoftmaxGLMHMM`).
            emissions: Integer array of shape `(T_total,)`.
            inputs: Float array of shape `(T_total, D)`.
            session_ids: Array of shape `(T_total,)`.

        Returns:
            Float array of shape `(T_total, K)` with one-step-ahead predictive
            state probabilities.
        """
        sessions = self._split_by_session(emissions, inputs, session_ids)
        e_pad, i_pad, lengths = self._pad_sessions(sessions)
        probs_batch = np.asarray(self._batched_predict_state_jit(params, e_pad, i_pad))  # (S, T_max, K)
        return np.concatenate([probs_batch[i, :T_s] for i, T_s in enumerate(lengths)], axis=0)
