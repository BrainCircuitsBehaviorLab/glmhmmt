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

## Task Adapters Are Optional

This package does not need task adapters when someone only wants the reusable model classes and fitting utilities.

If a user wants task-aware CLIs or notebooks, they can provide adapters in either of these ways:

1. Put a `tasks/` package in their own working directory and point `GLMHMMT_TASK_PATHS` at it.
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
