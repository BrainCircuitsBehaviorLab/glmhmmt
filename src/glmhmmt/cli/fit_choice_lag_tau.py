from __future__ import annotations

import argparse

from glmhmmt.choice_lag_tau import export_choice_lag_tau_table, resolve_fit_dir
from glmhmmt.runtime import add_runtime_path_args, configure_paths_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a single exponential decay to choice-lag GLM weights and export "
            "per-subject tau estimates."
        )
    )
    add_runtime_path_args(parser)
    parser.add_argument("--fit-dir", type=str, default=None, help="Direct path to an existing GLM fit directory.")
    parser.add_argument("--task", type=str, default=None, help="Task name under RESULTS/fits/<task>/...")
    parser.add_argument("--model-kind", type=str, default="glm", help="Model kind under the fit tree (default: glm).")
    parser.add_argument("--model-id", type=str, default=None, help="Model id / alias under the fit tree.")
    parser.add_argument("--out", type=str, default=None, help="Output path (.parquet default, or .csv).")
    parser.add_argument("--arrays-suffix", type=str, default="glm_arrays.npz")
    parser.add_argument("--state-idx", type=int, default=0)
    parser.add_argument("--class-idx", type=int, default=0)
    args = parser.parse_args()

    configure_paths_from_args(args)
    fit_dir = resolve_fit_dir(
        fit_dir=args.fit_dir,
        task=args.task,
        model_kind=args.model_kind,
        model_id=args.model_id,
    )
    table = export_choice_lag_tau_table(
        fit_dir=fit_dir,
        out_path=args.out,
        arrays_suffix=args.arrays_suffix,
        state_idx=args.state_idx,
        class_idx=args.class_idx,
    )
    print(table)


if __name__ == "__main__":
    main()
