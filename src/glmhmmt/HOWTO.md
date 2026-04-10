# `glmhmmt` HOWTO

This package lives inside the thesis repo at `/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/glmhmmt`, but it is structured as its own installable `uv` project.

## Install

```bash
cd /Users/javierrodriguezmartinez/Documents/MAMME/TFM/code
uv sync
```

For the marimo notebooks:

```bash
cd /Users/javierrodriguezmartinez/Documents/MAMME/TFM/code
uv sync --extra notebooks
```

## Runtime Configuration

The editable repo install reads runtime settings from:

1. CLI flags such as `--data-dir`, `--results-dir`, `--config-path`
2. Environment variables
3. `[paths]` in `/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/glmhmmt/config.toml`
4. Repo-aware fallbacks

`config.toml` is also allowed to override plotting and model settings from the packaged defaults in `src/glmhmmt/resources/default_config.toml`.
Task discovery uses `GLMHMMT_TASK_PATHS`, then `[plugins].adapter_paths` in `config.toml`, then `./adapters` or `./tasks` in the current working directory.

## Run Fits

Console scripts are exposed through the package:

```bash
uv run glmhmmt-fit-glm --help
uv run glmhmmt-fit-glmhmm --help
uv run glmhmmt-fit-glmhmmt --help
uv run glmhmmt-fit-tau-sweep --help
```

Examples:

```bash
uv run glmhmmt-fit-glm --task MCDR --subjects A83 A84 --tau 50
uv run glmhmmt-fit-glmhmm --task 2AFC --K 2 3 --tau 50
uv run glmhmmt-fit-glmhmmt --task MCDR --K 2 --tau 50
uv run glmhmmt-fit-tau-sweep --model glmhmmt --K 2 --tau_min 10 --tau_max 60 --tau_step 5
```

## Open Notebooks

```bash
cd /Users/javierrodriguezmartinez/Documents/MAMME/TFM/code
uv run marimo edit notebooks/glm_analysis.py
uv run marimo edit notebooks/glmhmm_analysis.py
uv run marimo edit notebooks/glmhmmt_analysis.py
```

The notebooks import `glmhmmt.runtime`, `glmhmmt.tasks`, and the packaged CLI modules directly. They no longer depend on repo-root `paths.py` or `scripts/` shims.

## Add Or Remove Repo Tasks

The thesis repo keeps its mutable task package at `/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/tasks`.
Any `*.py` file you place there is auto-imported by `glmhmmt.tasks`, so you can
drop in a new adapter module or delete one of the shipped examples without
editing `pyproject.toml`.

Each task adapter is responsible for:

- loading and filtering the task dataset
- defining emission and transition regressors
- exposing task-specific plotting helpers

Task-local plot modules live under `tasks/plots/` and are typically loaded from
the adapter with:

```python
import tasks.plots.my_task as plots
```

## Add an External Lab Task

External packages can provide additional tasks without editing this repo by exposing the same entry-point group:

```toml
[project.entry-points."glmhmmt.tasks"]
my_lab_task = "my_lab_glmhmmt.task:MyLabTaskAdapter"
```

Once that package is installed in the same environment, `glmhmmt.tasks.get_adapter(...)` will discover it automatically.
