from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from glmhmmt.postprocess import (
    build_change_triggered_posteriors_payload,
    build_session_deepdive_payload,
    build_session_trajectories_payload,
    build_state_accuracy_payload,
    build_state_dwell_times_payload,
    build_state_occupancy_payload,
    build_state_posterior_count_payload,
)
from glmhmmt import model_plots


def _trial_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["s1"] * 6 + ["s2"] * 6,
            "session": [1, 1, 1, 2, 2, 2] * 2,
            "trial_idx": list(range(6)) * 2,
            "state_idx": [0, 0, 1, 1, 1, 0] * 2,
            "state_rank": [0, 0, 1, 1, 1, 0] * 2,
            "state_label": ["Engaged", "Engaged", "Disengaged", "Disengaged", "Disengaged", "Engaged"] * 2,
            "p_state_0": [0.8, 0.7, 0.2, 0.3, 0.4, 0.9] * 2,
            "p_state_1": [0.2, 0.3, 0.8, 0.7, 0.6, 0.1] * 2,
            "correct_bool": [1, 1, 0, 1, 0, 1] * 2,
            "response": [0, 0, 1, 1, 1, 0] * 2,
            "A_plus": [0.1, 0.2, 0.4, 0.3, 0.2, 0.1] * 2,
            "A_minus": [0.3, 0.2, 0.1, 0.2, 0.3, 0.4] * 2,
        }
    )


def test_common_plot_payload_builders_return_explicit_tables():
    df = _trial_df()

    accuracy = build_state_accuracy_payload(df)
    trajectories = build_session_trajectories_payload(df)
    change = build_change_triggered_posteriors_payload(df, window=1)
    occupancy = build_state_occupancy_payload(df)
    dwell = build_state_dwell_times_payload(df)
    counts = build_state_posterior_count_payload(df)
    deepdive = build_session_deepdive_payload(df, subject="s1", session=1)

    assert set(accuracy["accuracy_df"].columns) >= {"subject", "state_label", "accuracy"}
    assert set(trajectories["trajectory_df"].columns) >= {"subject", "session", "trial_in_session", "probability"}
    assert set(change["change_df"].columns) >= {"subject", "direction", "relative_trial", "probability"}
    assert set(occupancy["occupancy_df"].columns) >= {"subject", "state_label", "occupancy"}
    assert set(dwell["dwell_df"].columns) >= {"subject", "session", "state_label", "dwell"}
    assert set(dwell["dwell_summary_df"].columns) >= {"state_label", "bin_center", "empirical"}
    assert set(counts["count_df"].columns) >= {"subject", "state_label", "n_trials"}
    assert set(counts["posterior_df"].columns) >= {"subject", "state_label", "probability"}
    assert set(deepdive["posterior_df"].columns) >= {"trial_in_session", "state_label", "probability"}


def test_common_state_plots_keep_payload_contracts():
    df = _trial_df()

    fig_acc, summary = model_plots.plot_state_accuracy(build_state_accuracy_payload(df, chance_level=0.5))
    assert isinstance(fig_acc, Figure)
    assert "mean_acc (%)" in summary.columns

    assert isinstance(model_plots.plot_state_posterior_count_kde(build_state_posterior_count_payload(df)), Figure)
    assert isinstance(model_plots.plot_state_occupancy(build_state_occupancy_payload(df)), Figure)
    assert isinstance(model_plots.plot_state_occupancy_overall(build_state_occupancy_payload(df)), Figure)
    assert isinstance(model_plots.plot_state_session_occupancy(build_state_occupancy_payload(df)), Figure)
    assert isinstance(model_plots.plot_state_switches(build_state_occupancy_payload(df)), Figure)
    assert isinstance(model_plots.plot_state_dwell_times_summary(build_state_dwell_times_payload(df)), Figure)
    assert isinstance(
            model_plots.plot_change_triggered_posteriors_summary(
                build_change_triggered_posteriors_payload(df, window=1)
            ),
        Figure,
    )
    assert isinstance(model_plots.plot_session_deepdive(build_session_deepdive_payload(df, subject="s1", session=1)), Figure)


def test_change_triggered_payload_uses_posterior_direction_and_non_engaged_trace():
    df = pd.DataFrame(
        {
            "subject": ["s1"] * 6,
            "session": [1] * 6,
            "trial_idx": list(range(6)),
            "state_idx": [0, 0, 1, 1, 0, 0],
            "state_rank": [0, 0, 1, 1, 0, 0],
            "state_label": ["Engaged", "Engaged", "Disengaged", "Disengaged", "Engaged", "Engaged"],
            "p_state_0": [0.9, 0.8, 0.2, 0.1, 0.8, 0.9],
            "p_state_1": [0.1, 0.2, 0.8, 0.9, 0.2, 0.1],
            "correct_bool": [1, 1, 0, 0, 1, 1],
        }
    )

    payload = build_change_triggered_posteriors_payload(df, window=1)
    change_df = payload["change_df"]

    assert payload["state_order"] == ["Engaged", "Disengaged"]
    assert set(change_df["direction"]) == {"out_of_engaged", "into_engaged"}

    out_before = change_df[
        (change_df["direction"] == "out_of_engaged")
        & (change_df["relative_trial"] == -0.5)
        & (change_df["state_label"] == "Engaged")
    ]["probability"].iloc[0]
    out_after = change_df[
        (change_df["direction"] == "out_of_engaged")
        & (change_df["relative_trial"] == 0.5)
        & (change_df["state_label"] == "Engaged")
    ]["probability"].iloc[0]
    in_after = change_df[
        (change_df["direction"] == "into_engaged")
        & (change_df["relative_trial"] == 0.5)
        & (change_df["state_label"] == "Engaged")
    ]["probability"].iloc[0]
    non_engaged_out = change_df[
        (change_df["direction"] == "out_of_engaged")
        & (change_df["relative_trial"] == 0.5)
        & (change_df["state_label"] == "Disengaged")
    ]["probability"].iloc[0]

    assert out_before == 0.8
    assert out_after == 0.2
    assert in_after == 0.8
    assert non_engaged_out == 0.8
