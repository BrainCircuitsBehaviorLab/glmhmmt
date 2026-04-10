from __future__ import annotations

import argparse

from glmhmmt.runtime import PROJECT_CONFIG_NAME, init_project_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a project config.toml for glmhmmt."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Destination config path. Defaults to "
            f"{PROJECT_CONFIG_NAME} in the current working directory."
        ),
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory to write into [paths].")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory to write into [paths].")
    parser.add_argument(
        "--task-path",
        dest="task_paths",
        action="append",
        default=None,
        help="Optional local tasks package root. Repeat the flag to add more than one path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file if it already exists.",
    )
    args = parser.parse_args()

    target = init_project_config(
        path=args.path,
        force=args.force,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        task_paths=args.task_paths,
    )
    print(f"Wrote glmhmmt config to {target}")


if __name__ == "__main__":
    main()
