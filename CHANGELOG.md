# Changelog

## 0.1.17
- Switched the model-manager anywidget assets from in-memory JS/CSS strings to file-backed assets so marimo no longer serves them through ephemeral virtual-file shared-memory blobs.
- Kept the simplified load-existing table and server-owned selection lifecycle, so stop/rerun and task/model switches stay backend-driven.
- Added regression tests covering the file-backed asset configuration and the load-selection lifecycle across task and model-type refreshes.
- Extended `2AFC_delay` to build the same grouped one-hot regressor families used by `2AFC`, including session-bias and explicit choice-lag columns.
- Added session-bias one-hot regressors and per-choice lag regressors to `MCDR`, and taught its adapter metadata to expose those dynamic families consistently.
- Updated the model-manager widget to treat `2AFC_delay` like the other binary tasks for grouped regressor selection, and added grouped handling for the new dynamic `MCDR` families.
- Commented out the widget `tau` selector and `max_lapse` control so they no longer appear in the current UI.
- Added regression tests covering the new `2AFC_delay` and `MCDR` regressor families plus the `2AFC_delay` widget grouping behavior.

## 0.1.16
- Switched model-manager actions to the same explicit JS-to-Python command channel used by the stable toml editor widget, avoiding direct JS click-counter mutations for save, delete, and run actions.
- Added command-handler regression tests for save, delete, and run-fit flows.

## 0.1.13
- Fixed bugs in the model manager widget.


## 0.1.13
- Added by subject parametrization.

## 0.1.12
- Optimized the glm function for lapses mode.

## 0.1.11
- Fixed progress bar callback in fit_glm.py.

## 0.1.10

- Standardized the 2AFC one-hot regressor family labels to `stim_hot`, `bias_hot`, and `choice_lag` in the model-manager widget.
- Collapsed fully selected one-hot regressor families back to their grouped names in the saved-model table, so the regressor summary matches the selector UI instead of listing raw `stim_*`, `bias_*`, or `choice_lag_*` members.

## 0.1.9

- Updated the 2AFC model-manager widget so one-hot regressor families can be selected as a single row-level toggle without exposing individual member cells in the table.
- Added grouped widget support for `stim_hot`, `bias_hot`, and `at_choice_lag` so large one-hot feature families no longer clutter the regressor selector UI.
- Extended fitted-regressor source selection to support prefix-based feature-family resolution, which lets mixed source fits feed multiple cumulative regressors cleanly.

## 0.1.8

- Fixed the shared `history` lapse-mode optimizer constraints so binary GLMs no longer index past the two shared repeat/alternate lapse parameters.
- Added a regression test covering the shared-history constraint path used by the SLSQP optimizer.

## 0.1.7

- Reload local task plot modules when their source file changes instead of caching them forever by path alone.
- Prevent stale `src/plots/*.py` imports from surviving notebook edits and causing attribute mismatches in marimo sessions.

## 0.1.6

- Dropped wrapper-level anywidget UIElement caching so marimo rebuilds widget views with a fresh virtual-file JavaScript module on each rerun.
- Preserved stable anywidget model ids by continuing to initialize the underlying widget comm before constructing the marimo wrapper.

## 0.1.5

- Split history lapses into two explicit modes: `history` now fits shared repeat/alternate lapse rates, while `history_conditioned` preserves the per-previous-choice formulation.
- Updated the CLI and notebook model manager to expose both history lapse modes explicitly.
- Extended the lapse-rate boxplot to support collapsing conditioned history lapses into shared repeat/alternate summaries, add pairwise significance annotations, and keep the custom white box styling.

## 0.1.4

- Fixed the marimo anywidget compatibility layer to serve widget JavaScript through marimo-managed virtual files instead of embedding untrusted `data:` URLs.
- Updated the notebook dependency pin to `anywidget==0.10.0` for compatibility with the latest marimo release.

## 0.1.3

- Restored `lapse_mode="history"` as a true probability lapse model with per-previous-choice `repeat` and `alternate` lapse parameters.
- Kept multinomial alternate lapses uniform over all non-previous classes, so a 3-choice alternation lapse uses targets like `[1/2, 0, 1/2]`.
- Updated lapse-rate plotting to use the project's custom boxplot style with subject-connecting lines and no scatter overlay.
- Refreshed history-lapse examples and labels to match the repeat/alternate lapse interpretation.

## 0.1.2

- Initial packaged release of `glmhmmt` with GLM, GLM-HMM, CLI entry points, and plotting utilities.
