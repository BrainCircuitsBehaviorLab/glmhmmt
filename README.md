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
