# glmhmmt

`glmhmmt` is the installable package for the Dynamax-based GLM-HMM / GLM-HMMT code.

If someone in the lab only wants to import the model class in their own code, this package is enough:

```bash
pip install "git+https://github.com/BrainCircuitsBehaviorLab/glmhmmt.git"
```

or with `uv`:

```bash
uv pip install "git+https://github.com/BrainCircuitsBehaviorLab/glmhmmt.git"
```

If they want to add it as a dependency in another `uv` project:

```bash
uv add "glmhmmt @ git+https://github.com/BrainCircuitsBehaviorLab/glmhmmt.git"
```

Then in Python:

```python
from glmhmmt import SoftmaxGLMHMM
```

If they want the baseline GLM fit directly in their own code, including binary lapses:

```python
from glmhmmt import fit_glm
```

## Array-first GLM-HMM fitting

Use `fit_glmhmm` when choices are already encoded and the emission and
transition design matrices have already been constructed by the calling
analysis. The function performs no dataset loading, task resolution, feature
construction, model naming, or file writing.

```python
from glmhmmt import fit_glmhmm

fit = fit_glmhmm(
    y=choices,
    X=emission_matrix,
    U=transition_matrix,
    session_ids=session_ids,
    num_states=2,
    num_classes=2,
    emission_names=["bias", "stimulus", "choice_history"],
    transition_names=["reward_history"],
    baseline_class_idx=0,
    cv_folds=5,
    seed=0,
)
```

Pass `U=None` for a standard GLM-HMM or a `(T, Q)` transition matrix for a
GLM-HMM-T. Cross-validation splits whole sessions and is followed by a separate
fit on all trials; fold metrics are available in `fit.cv_metrics`. The same
function is also exported as `fit_hmm`.

## Frozen emission and transition coefficients

Both options use `{internal_state_index: {feature_name: fixed_value}}`.
Integer keys and JSON string keys are equivalent. Names follow the ordered
columns of `X` and `U`, respectively. `None` and `{}` preserve unrestricted
behavior.

For a two-state model, identify the states outside the library using the base
fit on training data. For example, if that fit identifies internal state 0 as
Engaged and state 1 as Disengaged:

```python
from glmhmmt import fit_hmm

# X, U and names are constructed by the analysis project.
# U includes numeric columns named "Drug" and "drug_x_cumulative_reward".
disengaged_idx = state_indices_from_training_baseline["Disengaged"]  # 1 here
frozen_emissions = {disengaged_idx: {"drug_x_choice_lag_param": 0.0}}
frozen_transitions = {
    disengaged_idx: {"Drug": 0.0, "drug_x_cumulative_reward": 0.0}
}
fit = fit_hmm(
    y, X, session_ids, U=U,
    num_states=2, num_classes=2,
    emission_names=names["X_cols"],
    transition_names=names["U_cols"],
    frozen_emissions=frozen_emissions,
    frozen_transitions=frozen_transitions,
)
```

The transition tensor remains `(K, K-1, D)`: each source row uses its own
self-transition as the implicit zero reference, and explicit target entries
follow internal state order with the source omitted. In this example, row 1
controls Disengaged → Engaged. Freezing only `drug_x_cumulative_reward` in
row 1 leaves that interaction free for Engaged → Disengaged. With more than
two states, a `(state, feature)` restriction fixes **all non-self targets**
for that feature in the source row. It does not freeze transition biases,
other features, or other source rows. Emission freezes similarly fix all
explicit choice classes for the selected state and feature.

For direct `SoftmaxGLMHMM` construction, pass the same freeze dictionaries and
`emission_feature_names=names["X_cols"]`,
`transition_feature_names=names["U_cols"]`. The transition component
`InputDrivenSoftmaxTransitions` accepts `frozen_transitions` and
`transition_feature_names` too. Direct initialization supports
`transition_weights` and `transition_bias`; component initialization uses
`weights` and `bias`. Supplied weights, random initialization, and EM warm
starts are projected onto the restrictions. The loss and regularization use
the restricted weights, their frozen gradients are zero, and fixed values
remain exact in the tensor dtype after every optimizer step, including values
outside the usual clipping range. This is a constrained fit, not a post-fit
ablation. `U[t]` continues to determine the transition **into** trial `t`.

The array API applies both options to every cross-validation fit and the final
full-data fit; the normalized specification is in `fit.frozen_transitions`.
If semantic state identification or regressor scaling must differ by fold,
construct each training fold, its base fit, and its restrictions externally,
then call the fitting API separately for that fold. Interactions, cumulative
or filtered rewards, and two-step `drug_z` normalization are prepared by the
analysis project (normalization from training data per fold); the library
treats the supplied columns as ordinary numeric regressors.

The task-aware GLM-HMM-T CLI accepts JSON with the same syntax:

```bash
python -m glmhmmt.cli.fit_glmhmmt --K 2 \
  --frozen_emissions '{"1":{"drug_x_choice_lag_param":0.0}}' \
  --frozen_transitions '{"1":{"Drug":0.0,"drug_x_cumulative_reward":0.0}}'
```

The named columns must already be supplied by the calling analysis or its
design-matrix provider. No task adapter generates interactions for this option.
The CLI propagates freezes to ordinary and CV fits, saves them in `config.json`
and as `frozen_transitions_json` in result archives, and excludes frozen
weights from the BIC parameter count. The model manager preserves the option
when loading and saving configurations. Configurations use JSON string keys:

```json
{
  "frozen_emissions": {"1": {"drug_x_choice_lag_param": 0.0}},
  "frozen_transitions": {"1": {"Drug": 0.0, "drug_x_cumulative_reward": 0.0}}
}
```

## Direct Model Use

See [`examples/use_softmax_glmhmm.py`](examples/use_softmax_glmhmm.py) for a minimal example that builds the model directly from arrays, without any task adapter.

For a baseline GLM example, see [`examples/glm_lapses/example.py`](examples/glm_lapses/example.py).

The package root uses lazy imports, so importing `SoftmaxGLMHMM` does not require task adapters.

The CLI entrypoints under `glmhmmt.cli.*` are wrappers around task adapters, runtime paths, and result directories. They are useful for command-line workflows, but they are not the recommended import interface for another project.

## Runtime Config

`glmhmmt` now looks for `config.toml` by searching upward from the current working directory. That means each analysis project can keep its own config next to its notebooks and scripts.

The clean way to initialise one is:

```bash
uv run glmhmmt-init-config
```

That writes `config.toml` in the current working directory. You can also choose the destination explicitly:

```bash
uv run glmhmmt-init-config \
  --path ./config.toml \
  --data-dir /absolute/path/to/data \
  --results-dir /absolute/path/to/results
```

At runtime, config precedence is:

1. `configure_paths(...)`
2. `GLMHMMT_CONFIG_PATH`
3. nearest `config.toml` found by upward search from the current working directory
4. repo-local `config.toml` for editable installs
5. packaged defaults in `src/glmhmmt/resources/default_config.toml`

## Runtime Compatibility

The published package is tested against:

- Python 3.11–3.13
- `jax==0.4.35`
- `jaxlib==0.4.35`
- `tensorflow-probability==0.25.0`
- `optax==0.2.5`
- `numpy>=2.0,<2.1`

These bounds are intentionally conservative because newer JAX / TFP and NumPy
combinations have broken APIs used by `dynamax` and `glmhmmt.model`.

## Task Adapters Are Optional

This package does not need task adapters when someone only wants the reusable model classes and fitting utilities.

If a user wants task-aware CLIs or notebooks, they can provide adapters in either of these ways:

1. Put an `adapters/` package in their own working directory, or configure `[plugins].adapter_paths` / `GLMHMMT_TASK_PATHS`.
2. Install a separate package that exposes entry points in the `glmhmmt.tasks` group.

Minimal entry-point example:

```toml
[project.entry-points."glmhmmt.tasks"]
my_lab_task = "my_lab_glmhmmt.task:MyLabTaskAdapter"
```

## Recommended Sharing Workflow

For lab use, the simplest setup is:

1. Keep `glmhmmt` in its own Git repo.
2. Keep task adapters in a separate companion repo.
3. Install both from Git or from local editable paths during development.

That is usually better than publishing to PyPI immediately, because:

- it avoids exposing unrelated analysis code
- updates are simple
- private sharing inside the lab is easy

Publish to PyPI later only if you want a public, versioned release.

## Local Development

For work inside this repository:

```bash
uv sync
uv run python -c "from glmhmmt import SoftmaxGLMHMM; print(SoftmaxGLMHMM)"
```

If you want the notebook extras too:

```bash
uv sync --extra notebooks
```

The project-local runtime overrides live in `config.toml`. Packaged defaults live in `src/glmhmmt/resources/default_config.toml`.
