# Changelog

## 0.1.3

- Restored `lapse_mode="history"` as a true probability lapse model with per-previous-choice `repeat` and `alternate` lapse parameters.
- Kept multinomial alternate lapses uniform over all non-previous classes, so a 3-choice alternation lapse uses targets like `[1/2, 0, 1/2]`.
- Updated lapse-rate plotting to use the project's custom boxplot style with subject-connecting lines and no scatter overlay.
- Refreshed history-lapse examples and labels to match the repeat/alternate lapse interpretation.

## 0.1.2

- Initial packaged release of `glmhmmt` with GLM, GLM-HMM, CLI entry points, and plotting utilities.
