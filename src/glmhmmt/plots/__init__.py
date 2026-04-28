"""Task-agnostic model plotting API.

Plot functions in this package only render already prepared DataFrames or
payloads. Data preparation belongs in :mod:`glmhmmt.postprocess`.
"""

from glmhmmt.plots.emissions import (
     emission_weights_by_subject,
     emission_weights_summary_lineplot,
     emission_weights_summary_boxplot,
     emission_regressor_lr_boxplot,
     lapse_rates_boxplot,
)
from glmhmmt.plots.sessions import (
    posterior_probs,
    session_deepdive,
    session_trajectories,
)
from glmhmmt.plots.states import (
    change_triggered_posteriors_by_subject,
    change_triggered_posteriors_summary,
    state_accuracy,
    state_dwell_times_by_subject,
    state_dwell_times_summary,
    state_occupancy,
    state_occupancy_overall,
    state_occupancy_overall_by_subject,
    state_occupancy_overall_summary,
    state_posterior_count_kde,
    state_session_occupancy,
    state_session_occupancy_by_subject,
    state_session_occupancy_summary,
    state_switches,
    state_switches_by_subject,
    state_switches_summary,
)
from glmhmmt.plots.transitions import (
    plot_transition_matrix as transition_matrix,
    plot_transition_matrix_by_subject as transition_matrix_by_subject,
)

from glmhmmt.plots.metrics import ll_boxplot 

__all__ = [
    "emission_weights_by_subject",
    "emission_weights_summary_lineplot",
    "emission_weights_summary_boxplot",
    "emission_regressor_lr_boxplot",
    "change_triggered_posteriors_by_subject",
    "change_triggered_posteriors_summary",
    "emission_regressor_lr_boxplot",
    "lapse_rates_boxplot",
    "ll_boxplot",
    "emission_weights",
    "emission_weights_by_subject",
    "emission_weights_summary",
    "emission_weights_summary_boxplot",
    "emission_weights_summary_lineplot",
    "posterior_probs",
    "session_deepdive",
    "session_trajectories",
    "state_accuracy",
    "state_dwell_times",
    "state_dwell_times_by_subject",
    "state_dwell_times_summary",
    "state_occupancy",
    "state_occupancy_overall",
    "state_occupancy_overall_by_subject",
    "state_occupancy_overall_summary",
    "state_posterior_count_kde",
    "state_session_occupancy",
    "state_session_occupancy_by_subject",
    "state_session_occupancy_summary",
    "state_switches",
    "state_switches_by_subject",
    "state_switches_summary",
    "transition_matrix",
    "transition_matrix_by_subject",
]
