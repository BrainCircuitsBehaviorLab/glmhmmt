# glmhmmt

`glmhmmt` is the installable package for the Dynamax-based GLM-HMM / GLM-HMMT code.

If someone in the lab only wants to import the model class in their own code, this package is enough:

```bash
pip install "git+ssh://git@github.com/<org>/glmhmmt.git"
```

or with `uv`:

```bash
uv pip install "git+ssh://git@github.com/<org>/glmhmmt.git"
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

For the baseline GLM fitter, see [`examples/fit_glm_with_lapses.py`](examples/fit_glm_with_lapses.py). That is the right interface for someone who already has their own `X` and `y`.

The package root uses lazy imports, so importing `SoftmaxGLMHMM` does not require task adapters.

The CLI entrypoints under `glmhmmt.cli.*` are repo-oriented wrappers around task adapters, runtime paths, and result directories. They are useful for command-line workflows, but they are not the recommended import interface for another project.

## Task Adapters Are Optional

The thesis repo keeps editable task adapters in `code/tasks`, but they are not required when someone only wants the core model class.

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

1. Put `code/glmhmmt` in its own Git repo.
2. Keep that repo focused on the reusable package only.
3. Let people install it straight from Git.

That is usually better than publishing to PyPI immediately, because:

- it avoids exposing the rest of the thesis repo
- updates are simple
- private sharing inside the lab is easy

Publish to PyPI later only if you want a public, versioned release.

## Local Development In This Thesis Repo

For work inside this thesis repository:

```bash
cd code
uv sync
uv run python -c "from glmhmmt import SoftmaxGLMHMM; print(SoftmaxGLMHMM)"
```

If you want the marimo notebooks too:

```bash
cd code
uv sync --extra notebooks
uv run marimo edit notebooks/glmhmmt_analysis.py
```

If you specifically want to work on `glmhmmt` as a standalone package project:

```bash
cd code/glmhmmt
uv sync --extra notebooks
```

The project-local runtime overrides live in `config.toml`. Packaged defaults live in `src/glmhmmt/resources/default_config.toml`.
