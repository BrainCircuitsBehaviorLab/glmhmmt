import json
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
import polars as pl

from glmhmmt.cli.fit_common import (
    apply_valid_trial_mask,
    build_session_kfold_split,
    fit_best_restart,
    normalize_cv_mode,
    score_split,
    stable_model_id,
)
from glmhmmt.model import SoftmaxGLMHMM, normalize_frozen_emissions, serialize_frozen_emissions
from glmhmmt.runtime import (
    add_runtime_path_args,
    configure_paths_from_args,
    get_results_dir,
)
from glmhmmt.tasks import get_adapter

ProgressCallback = Callable[[dict[str, Any]], None]


def generate_model_id(
    task: str,
    K: int,
    tau: float,
    emission_cols: list | None = None,
    transition_cols: list | None = None,
    frozen_emissions: dict | None = None,
    cv_mode: str = "none",
    cv_repeats: int = 0,
) -> str:
    """Stable 8-char hash over the GLMHMMT-defining model configuration."""
    return stable_model_id(
        task=task,
        K=K,
        tau=tau,
        emission_cols=emission_cols,
        transition_cols=transition_cols,
        frozen_emissions=frozen_emissions,
        cv_mode=cv_mode,
        cv_repeats=cv_repeats,
    )


def _load_subject_feature_df(subject: str, task: str, tau: float) -> tuple[Any, pl.DataFrame]:
    adapter = get_adapter(task)
    df = adapter.read_dataset()
    df = adapter.subject_filter(df)
    df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)
    feature_df = adapter.build_feature_df(df_sub, tau=tau)
    return adapter, feature_df


def _prepare_arrays(
    adapter,
    feature_df: pl.DataFrame,
    emission_cols,
    transition_cols,
    session_col: str,
    *,
    min_session_length: int = 2,
):
    y, X, U, names = adapter.build_design_matrices(
        feature_df,
        emission_cols=emission_cols,
        transition_cols=transition_cols,
    )
    session_ids = feature_df[session_col].to_numpy()
    y, X, U, session_ids = apply_valid_trial_mask(
        session_ids,
        y,
        X,
        U,
        min_length=min_session_length,
    )
    return jnp.asarray(y), jnp.asarray(X), jnp.asarray(U), np.asarray(session_ids), names


def _build_model(
    K: int,
    num_classes: int,
    x_dim: int,
    u_dim: int,
    m_step_num_iters: int,
    stickiness: float,
    frozen: dict | None,
    names: dict[str, Any],
) -> SoftmaxGLMHMM:
    return SoftmaxGLMHMM(
        num_states=K,
        num_classes=num_classes,
        emission_input_dim=x_dim,
        transition_input_dim=u_dim,
        m_step_num_iters=m_step_num_iters,
        transition_matrix_stickiness=stickiness,
        frozen_emissions=frozen or None,
        emission_feature_names=names.get("X_cols", []),
    )

def fit_subject(
    subject: str,
    K: int,
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    m_step_num_iters: int = 100,
    stickiness: float = 10.0,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    tau: float = 50.0,
    task: str = "MCDR",
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    adapter, feature_df = _load_subject_feature_df(subject, task, tau)
    y, X, U, session_ids, names = _prepare_arrays(
        adapter,
        feature_df,
        emission_cols,
        transition_cols,
        adapter.session_col,
    )
    frozen = normalize_frozen_emissions(frozen_emissions)
    inputs_all = jnp.concatenate([X, U], axis=1)
    model = _build_model(
        K,
        adapter.num_classes,
        X.shape[1],
        U.shape[1],
        m_step_num_iters,
        stickiness,
        frozen,
        names,
    )
    best_params, best_lps, best_restart = fit_best_restart(
        model,
        n_restarts=n_restarts,
        base_seed=base_seed,
        fit_once=lambda params, props: model.fit_em_multisession(
            params=params,
            props=props,
            emissions=y,
            inputs=inputs_all,
            session_ids=session_ids,
            num_iters=num_iters,
            verbose=verbose,
        ),
        failure_message=(
            "GLMHMM-T fitting did not produce valid parameters. "
            "All restarts ended with non-finite log-probabilities. "
            "Check the selected regressors for scaling issues."
        ),
        progress_callback=progress_callback,
        progress_payload={"subject": subject, "K": K},
    )
    smoothed_probs = model.smoother_multisession(
        params=best_params, emissions=y, inputs=inputs_all, session_ids=session_ids
    )
    p_pred = model.predict_choice_probs_multisession(best_params, y, inputs_all, session_ids=session_ids)
    predictive_state_probs = model.predict_state_probs_multisession(
        best_params, y, inputs_all, session_ids=session_ids
    )
    split_metrics = score_split(model, best_params, y, inputs_all, session_ids)

    return {
        "subject": subject,
        "K": K,
        "num_classes": adapter.num_classes,
        "model": model,
        "fitted_params": best_params,
        "lps": best_lps,
        "best_restart": int(best_restart),
        "smoothed_probs": smoothed_probs,
        "p_pred": p_pred,
        "predictive_state_probs": predictive_state_probs,
        "raw_ll": float(split_metrics["raw_ll"]),
        "T": int(y.shape[0]),
        "names": names,
        "y": np.asarray(y),
        "X": np.asarray(X),
        "U": np.asarray(U),
        "frozen_emissions": serialize_frozen_emissions(frozen),
        "cv_mode": "none",
        "cv_repeats": 0,
    }


def fit_subject_cv(
    subject: str,
    K: int,
    num_iters: int = 50,
    n_restarts: int = 1,
    cv_repeats: int = 5,
    base_seed: int = 0,
    m_step_num_iters: int = 100,
    stickiness: float = 10.0,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    tau: float = 50.0,
    task: str = "2AFC",
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    adapter, feature_df = _load_subject_feature_df(subject, task, tau)
    frozen = normalize_frozen_emissions(frozen_emissions)
    repeats: list[dict[str, Any]] = []
    best_repeat_idx = -1
    best_test_ll = -np.inf
    best_repeat_params = None
    best_repeat_lps = None
    best_repeat_model = None
    n_folds = 5
    for repeat_idx in range(n_folds):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "cv_repeat_start",
                    "subject": subject,
                    "K": K,
                    "cv_repeat_index": repeat_idx + 1,
                    "cv_repeat_total": n_folds,
                }
            )
        train_df, test_df, split_meta = build_session_kfold_split(
            feature_df,
            adapter.session_col,
            fold_index=repeat_idx,
            n_splits=n_folds,
            seed=base_seed,
        )
        if verbose:
            print(
                f"[CV fold {repeat_idx + 1}/{n_folds}] "
                f"sessions train/test={split_meta['train_session_count']}/{split_meta['test_session_count']}"
            )
        y_train, X_train, U_train, session_train, names = _prepare_arrays(
            adapter,
            train_df,
            emission_cols,
            transition_cols,
            adapter.session_col,
        )
        y_test, X_test, U_test, session_test, _ = _prepare_arrays(
            adapter,
            test_df,
            emission_cols,
            transition_cols,
            adapter.session_col,
        )
        inputs_train = jnp.concatenate([X_train, U_train], axis=1)
        inputs_test = jnp.concatenate([X_test, U_test], axis=1)
        model = _build_model(
            K,
            adapter.num_classes,
            X_train.shape[1],
            U_train.shape[1],
            m_step_num_iters,
            stickiness,
            frozen,
            names,
        )
        best_params, best_lps, best_restart = fit_best_restart(
            model,
            n_restarts=n_restarts,
            base_seed=base_seed + 1000 * (repeat_idx + 1),
            fit_once=lambda params, props: model.fit_em_multisession(
                params=params,
                props=props,
                emissions=y_train,
                inputs=inputs_train,
                session_ids=session_train,
                num_iters=num_iters,
                verbose=verbose,
            ),
            failure_message=(
                "GLMHMM-T fitting did not produce valid parameters. "
                "All restarts ended with non-finite log-probabilities. "
                "Check the selected regressors for scaling issues."
            ),
            progress_callback=progress_callback,
            progress_payload={
                "subject": subject,
                "K": K,
                "cv_repeat_index": repeat_idx + 1,
                "cv_repeat_total": n_folds,
            },
        )
        train_metrics = score_split(model, best_params, y_train, inputs_train, session_train)
        test_metrics = score_split(model, best_params, y_test, inputs_test, session_test)
        if float(test_metrics["ll_per_trial"]) > best_test_ll:
            best_test_ll = float(test_metrics["ll_per_trial"])
            best_repeat_idx = repeat_idx + 1
            best_repeat_params = best_params
            best_repeat_lps = np.asarray(best_lps)
            best_repeat_model = model
        repeats.append(
            {
                "subject": subject,
                "repeat_index": repeat_idx + 1,
                "best_restart": int(best_restart),
                "train_T": int(train_metrics["T"]),
                "test_T": int(test_metrics["T"]),
                "train_raw_ll": float(train_metrics["raw_ll"]),
                "train_ll_per_trial": float(train_metrics["ll_per_trial"]),
                "train_acc": float(train_metrics["acc"]),
                "test_raw_ll": float(test_metrics["raw_ll"]),
                "test_ll_per_trial": float(test_metrics["ll_per_trial"]),
                "test_acc": float(test_metrics["acc"]),
                "objective_lp": float(best_lps[-1]),
                "train_session_count": int(split_meta["train_session_count"]),
                "test_session_count": int(split_meta["test_session_count"]),
                "train_label_counts_json": json.dumps(split_meta["train_label_counts"], sort_keys=True),
                "test_label_counts_json": json.dumps(split_meta["test_label_counts"], sort_keys=True),
                "balance_score": None,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "cv_repeat_complete",
                    "subject": subject,
                    "K": K,
                    "cv_repeat_index": repeat_idx + 1,
                    "cv_repeat_total": n_folds,
                    "test_ll_per_trial": float(test_metrics["ll_per_trial"]),
                    "test_acc": float(test_metrics["acc"]),
                }
            )

    if best_repeat_model is None or best_repeat_params is None or best_repeat_lps is None:
        raise ValueError("CV fitting did not produce a valid best repeat.")

    y_full, X_full, U_full, session_full, full_names = _prepare_arrays(
        adapter,
        feature_df,
        emission_cols,
        transition_cols,
        adapter.session_col,
    )
    inputs_full = jnp.concatenate([X_full, U_full], axis=1)
    smoothed_probs = best_repeat_model.smoother_multisession(
        params=best_repeat_params,
        emissions=y_full,
        inputs=inputs_full,
        session_ids=session_full,
    )
    p_pred = best_repeat_model.predict_choice_probs_multisession(
        best_repeat_params,
        y_full,
        inputs_full,
        session_ids=session_full,
    )
    predictive_state_probs = best_repeat_model.predict_state_probs_multisession(
        best_repeat_params,
        y_full,
        inputs_full,
        session_ids=session_full,
    )
    full_metrics = score_split(best_repeat_model, best_repeat_params, y_full, inputs_full, session_full)

    return {
        "subject": subject,
        "K": K,
        "num_classes": adapter.num_classes,
        "repeats": repeats,
        "best_repeat_index": int(best_repeat_idx),
        "best_repeat_test_ll_per_trial": float(best_test_ll),
        "model": best_repeat_model,
        "fitted_params": best_repeat_params,
        "lps": best_repeat_lps,
        "smoothed_probs": smoothed_probs,
        "p_pred": p_pred,
        "predictive_state_probs": predictive_state_probs,
        "raw_ll": float(full_metrics["raw_ll"]),
        "T": int(y_full.shape[0]),
        "names": full_names,
        "y": np.asarray(y_full),
        "X": np.asarray(X_full),
        "U": np.asarray(U_full),
        "frozen_emissions": serialize_frozen_emissions(frozen),
        "cv_mode": "balanced_session_holdout",
        "cv_repeats": n_folds,
    }


def save_results(result: dict, out_dir: Path) -> None:
    subj = result["subject"]
    K = result["K"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{subj}_K{K}_glmhmmt"

    T = result["T"]
    p_pred = result["p_pred"]
    acc = float(np.mean(np.argmax(p_pred, axis=1) == result["y"]))
    raw_ll = float(result["raw_ll"])
    ll_per_trial = raw_ll / T
    num_classes = result["num_classes"]
    n_params = result["K"] * (result["K"] - 1) * (1 + result["U"].shape[1]) + result["K"] * (num_classes - 1) * result["X"].shape[1]
    bic = -2 * raw_ll + n_params * np.log(T)

    pl.DataFrame(
        {
            "subject": [subj],
            "K": [K],
            "model_kind": ["glmhmm-t"],
            "ll_per_trial": [ll_per_trial],
            "raw_ll": [raw_ll],
            "bic": [bic],
            "acc": [acc],
            "best_restart": [int(result.get("best_restart", 1))],
            "cv_mode": [result.get("cv_mode", "none")],
            "cv_repeats": [int(result.get("cv_repeats", 0))],
        }
    ).write_parquet(str(prefix) + "_metrics.parquet")

    np.savez(
        str(prefix) + "_arrays.npz",
        lps=result["lps"],
        p_pred=p_pred,
        smoothed_probs=result["smoothed_probs"],
        predictive_state_probs=result["predictive_state_probs"],
        initial_probs=np.asarray(result["fitted_params"].initial.probs),
        emission_weights=np.asarray(result["fitted_params"].emissions.weights),
        transition_bias=np.asarray(result["fitted_params"].transitions.bias),
        transition_weights=np.asarray(result["fitted_params"].transitions.weights),
        names=result["names"],
        y=result["y"],
        X=result["X"],
        U=result["U"],
        X_cols=np.array(result["names"].get("X_cols", []), dtype=object),
        U_cols=np.array(result["names"].get("U_cols", []), dtype=object),
        frozen_emissions_json=np.array(json.dumps(result["frozen_emissions"], sort_keys=True)),
    )


def save_cv_results(result: dict, out_dir: Path) -> None:
    subj = result["subject"]
    K = result["K"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{subj}_K{K}_glmhmmt"
    repeats_df = pl.DataFrame(result["repeats"])
    repeats_df.write_parquet(str(prefix) + "_cv_repeats.parquet")

    test_ll = repeats_df["test_ll_per_trial"].to_numpy()
    test_acc = repeats_df["test_acc"].to_numpy()
    train_ll = repeats_df["train_ll_per_trial"].to_numpy()
    train_acc = repeats_df["train_acc"].to_numpy()
    T = int(result["T"])
    p_pred = result["p_pred"]
    raw_ll = float(result["raw_ll"])
    ll_per_trial = raw_ll / T
    acc = float(np.mean(np.argmax(p_pred, axis=1) == result["y"]))
    num_classes = result["num_classes"]
    n_params = result["K"] * (result["K"] - 1) * (1 + result["U"].shape[1]) + result["K"] * (num_classes - 1) * result["X"].shape[1]
    bic = -2 * raw_ll + n_params * np.log(T)
    pl.DataFrame(
        {
            "subject": [subj],
            "K": [K],
            "model_kind": ["glmhmm-t"],
            "ll_per_trial": [ll_per_trial],
            "raw_ll": [raw_ll],
            "bic": [bic],
            "acc": [acc],
            "best_repeat_index": [int(result["best_repeat_index"])],
            "best_repeat_test_ll_per_trial": [float(result["best_repeat_test_ll_per_trial"])],
            "cv_mode": [result["cv_mode"]],
            "cv_repeats": [int(result["cv_repeats"])],
            "test_ll_per_trial_mean": [float(np.mean(test_ll))],
            "test_ll_per_trial_std": [float(np.std(test_ll, ddof=0))],
            "test_acc_mean": [float(np.mean(test_acc))],
            "test_acc_std": [float(np.std(test_acc, ddof=0))],
            "train_ll_per_trial_mean": [float(np.mean(train_ll))],
            "train_ll_per_trial_std": [float(np.std(train_ll, ddof=0))],
            "train_acc_mean": [float(np.mean(train_acc))],
            "train_acc_std": [float(np.std(train_acc, ddof=0))],
        }
    ).write_parquet(str(prefix) + "_metrics.parquet")

    np.savez(
        str(prefix) + "_arrays.npz",
        lps=result["lps"],
        p_pred=result["p_pred"],
        smoothed_probs=result["smoothed_probs"],
        predictive_state_probs=result["predictive_state_probs"],
        initial_probs=np.asarray(result["fitted_params"].initial.probs),
        emission_weights=np.asarray(result["fitted_params"].emissions.weights),
        transition_bias=np.asarray(result["fitted_params"].transitions.bias),
        transition_weights=np.asarray(result["fitted_params"].transitions.weights),
        names=result["names"],
        y=result["y"],
        X=result["X"],
        U=result["U"],
        X_cols=np.array(result["names"].get("X_cols", []), dtype=object),
        U_cols=np.array(result["names"].get("U_cols", []), dtype=object),
        frozen_emissions_json=np.array(json.dumps(result["frozen_emissions"], sort_keys=True)),
        cv_selected_repeat=np.array(int(result["best_repeat_index"])),
        cv_selected_metric=np.array(float(result["best_repeat_test_ll_per_trial"])),
    )


def main(
    subjects: list[str] | None = None,
    K_list: list[int] = [2, 3],
    num_iters: int = 50,
    n_restarts: int = 1,
    base_seed: int = 0,
    out_dir: Path | None = None,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    tau: float = 50.0,
    task: str = "MCDR",
    cv_mode: str = "none",
    cv_repeats: int = 0,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
):
    adapter = get_adapter(task)
    cv_mode = normalize_cv_mode(cv_mode)
    cv_repeats = int(cv_repeats) if cv_mode != "none" else 0
    frozen_spec = serialize_frozen_emissions(frozen_emissions)
    df = None
    if emission_cols is None or subjects is None:
        df = adapter.read_dataset()
        df = adapter.subject_filter(df)
    resolved_emission_cols = emission_cols or adapter.default_emission_cols(df)
    if out_dir is None:
        model_id = generate_model_id(
            task=task,
            K=K_list[0],
            tau=tau,
            emission_cols=resolved_emission_cols,
            transition_cols=transition_cols or adapter.default_transition_cols(),
            frozen_emissions=frozen_spec,
            cv_mode=cv_mode,
            cv_repeats=cv_repeats,
        )
        out_dir = get_results_dir() / "fits" / task / "glmhmmt" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as _f:
        json.dump(
            {
                "task": task,
                "tau": tau,
                "subjects": subjects,
                "emission_cols": resolved_emission_cols,
                "transition_cols": transition_cols or adapter.default_transition_cols(),
                "frozen_emissions": frozen_spec,
                "K_list": K_list,
                "model_id": out_dir.name,
                "cv_mode": cv_mode,
                "cv_repeats": int(cv_repeats),
            },
            _f,
            indent=4,
        )
    if subjects is None:
        if df is None:
            df = adapter.read_dataset()
            df = adapter.subject_filter(df)
        subjects = df["subject"].unique().sort().to_list()

    for subj_idx, subj in enumerate(subjects, start=1):
        for k_idx, K in enumerate(K_list, start=1):
            if verbose:
                print(
                    f"Fitting glmhmm-t | subject={subj} K={K} "
                    f"task={task} cv_mode={cv_mode} ..."
                )

            def _progress(info: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    {
                        **info,
                        "subject_index": subj_idx,
                        "subject_total": len(subjects),
                        "k_index": k_idx,
                        "k_total": len(K_list),
                    }
                )

            if cv_mode == "balanced_session_holdout":
                result = fit_subject_cv(
                    subj,
                    K,
                    num_iters=num_iters,
                    n_restarts=n_restarts,
                    cv_repeats=cv_repeats,
                    base_seed=base_seed,
                    emission_cols=emission_cols,
                    transition_cols=transition_cols,
                    frozen_emissions=frozen_spec,
                    tau=tau,
                    task=task,
                    verbose=verbose,
                    progress_callback=_progress if progress_callback is not None else None,
                )
                save_cv_results(result, out_dir)
            else:
                result = fit_subject(
                    subj,
                    K,
                    num_iters=num_iters,
                    n_restarts=n_restarts,
                    base_seed=base_seed,
                    emission_cols=emission_cols,
                    transition_cols=transition_cols,
                    frozen_emissions=frozen_spec,
                    tau=tau,
                    task=task,
                    verbose=verbose,
                    progress_callback=_progress if progress_callback is not None else None,
                )
                save_results(result, out_dir)
            if verbose:
                print(f"  ✓ saved to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_runtime_path_args(parser)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--K", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--tau", type=float, default=50.0, help="Half-life for exponential action traces.")
    parser.add_argument("--task", type=str, default="MCDR", help="Task to fit: 'MCDR', '2AFC', or 'nuo_auditory'.")
    parser.add_argument(
        "--cv_mode",
        type=str,
        default="none",
        choices=["none", "balanced_session_holdout", "balanced_holdout"],
    )
    parser.add_argument("--cv_repeats", type=int, default=0)
    parser.add_argument(
        "--frozen_emissions",
        type=str,
        default=None,
        help='JSON object mapping state indices to {feature: fixed_value}, e.g. \'{"0":{"SL":0.0}}\'.',
    )
    args = parser.parse_args()
    configure_paths_from_args(args)
    frozen_emissions = json.loads(args.frozen_emissions) if args.frozen_emissions else None
    main(
        subjects=args.subjects,
        K_list=args.K,
        num_iters=args.num_iters,
        n_restarts=args.n_restarts,
        base_seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        emission_cols=None,
        transition_cols=None,
        frozen_emissions=frozen_emissions,
        tau=args.tau,
        task=args.task,
        cv_mode=args.cv_mode,
        cv_repeats=args.cv_repeats,
    )
