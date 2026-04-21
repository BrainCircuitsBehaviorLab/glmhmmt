from __future__ import annotations

import pandas as pd

from glmhmmt.postprocess import (
    build_session_deepdive_payload,
    build_session_trajectories_payload,
    build_state_accuracy_payload,
    build_state_dwell_times_payload,
    build_state_occupancy_payload,
    build_state_posterior_count_payload,
)


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
        }
    )


def test_common_plot_payload_builders_return_explicit_tables():
    df = _trial_df()

    accuracy = build_state_accuracy_payload(df)
    trajectories = build_session_trajectories_payload(df)
    occupancy = build_state_occupancy_payload(df)
    dwell = build_state_dwell_times_payload(df)
    counts = build_state_posterior_count_payload(df)
    deepdive = build_session_deepdive_payload(df, subject="s1", session=1)

    assert set(accuracy["accuracy_df"].columns) >= {"subject", "state_label", "accuracy"}
    assert set(trajectories["trajectory_df"].columns) >= {"subject", "session", "trial_in_session", "probability"}
    assert set(occupancy["occupancy_df"].columns) >= {"subject", "state_label", "occupancy"}
    assert set(dwell["dwell_df"].columns) >= {"subject", "session", "state_label", "dwell"}
    assert set(counts["count_df"].columns) >= {"subject", "state_label", "n_trials"}
    assert set(deepdive["posterior_df"].columns) >= {"trial_in_session", "state_label", "probability"}
