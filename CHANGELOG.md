# Changelog

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
