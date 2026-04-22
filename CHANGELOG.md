# Changelog

## 0.1.22
- Restored the legacy visual style for the new payload-based common model plots, including the session deep dive, change-triggered posteriors, state occupancy, posterior histograms, state accuracy, and dwell-time summaries.
- Kept the strict postprocess/plot separation while enriching the plot payloads with the extra metadata needed for the original visual grammar: action traces, rolling accuracy, transition-derived dwell predictions, posterior histograms, and subject-level summaries.
- Added `build_change_triggered_posteriors_payload(...)` and corrected engaged-to-disengaged / disengaged-to-engaged event extraction to use confident posterior `argmax` changes, preserving partial edge windows and plotting non-engaged probability as `1 - P(engaged)`.
- Updated state accuracy payloads to include the pooled `All` condition and restored the project custom boxplot style with subject-connecting lines and percent accuracy.
- Extended dwell-time payloads to support transition matrices or views during postprocessing, so plots can draw predicted-vs-empirical dwell curves without doing model-data preparation internally.
- Added smoke and regression coverage for the common state/session plot payload contracts, including change-triggered posterior directionality.

## 0.1.21
- Split `glmhmmt.model_plots` into a thin public facade over smaller `glmhmmt.model_plotting` modules for emissions, transitions, sessions, states, and shared utilities.
- Moved common model plot preparation into `glmhmmt.postprocess` payload builders, so common plots consume explicit DataFrames or payloads instead of `views` as their public contract.
- Added payload builders for state accuracy, session trajectories, state occupancy, dwell times, posterior counts, and session deep dives, and exported them through the package lazy imports.
- Preserved non-migrated plotting helpers through a legacy bridge while introducing the new DataFrame/payload API for emission weights, transition matrices, posterior/session diagnostics, occupancy, dwell, and state accuracy.
- Made GLM fit configs store both requested and resolved emission columns, so downstream fitted-regressor resolution can use the actual expanded design matrix columns.
- Hardened fitted-regressor source resolution with fallback reads from saved arrays when older configs do not include resolved emission columns.
- Fixed multi-subject trial concatenation for subject-local one-hot feature columns by padding missing fitted feature columns with zeros before concatenation.
- Added regression coverage for trial-frame alignment, plot payload builders, and the new emission-weight plotting signature.

## 0.1.20
- Updated the model-manager widget to honor each task adapter's full `default_emission_cols(...)` list for GLM fits instead of truncating the preselected emission regressors to the first 10 columns.
- Added regression coverage for the full-default widget path so task-owned custom emission defaults remain stable across widget refreshes.

## 0.1.19
- Added choice-lag tau fitting and export support, including the new `glmhmmt-fit-choice-lag-tau` CLI, subject-level summary tables, and saved summary plots for GLM fits with `choice_lag_*` regressors.
- Validated categorical emission labels before entering the JIT-backed EM and smoothing paths, so invalid class indices now fail fast with clear `ValueError` messages instead of surfacing as harder-to-debug downstream errors.
- Added a reusable single-family custom boxplot helper, collapsed summary `bias_*` coefficients into per-subject `|bias|` absolute means for boxplot payloads, and kept one-hot family plotting aligned with the project boxplot style.
- Hardened notebook selector compatibility with a fallback `build_selector_groups` path and extended `2AFC_delay` regression coverage to ensure task-wide delay one-hot columns remain stable across subjects.
- Added regression tests covering choice-lag tau export, emission-label validation, custom feature boxplots, summary bias-family collapsing, and task-wide `2AFC_delay` delay columns.

## 0.1.18
- Replaced the raw `2AFC_delay` stimulus one-hot family with `delay_hot`, and added `bias_param`, `delay_param`, and `choice_lag_param` columns to the `2AFC_delay` and `MCDR` task adapters.
- Collapsed the `MCDR` `stim1..4{L,C,R}` family under a single hidden `stim_hot` widget row and exposed the pooled `stim_param` plus `choice_lag_param` scalar rows alongside the grouped raw families.
- Moved model-manager regressor grouping onto the task adapters, so the widget now asks each adapter for its emission and transition groups instead of hard-coding task-specific selector layouts.

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
