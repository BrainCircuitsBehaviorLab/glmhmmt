"""Task-agnostic model plotting API.

Plot functions in this package only render already prepared DataFrames or
payloads. Data preparation belongs in :mod:`glmhmmt.postprocess`.
"""

from glmhmmt.model_plotting.emissions import (
    plot_binary_emission_weights,
    plot_binary_emission_weights_by_subject,
    plot_binary_emission_weights_summary,
    plot_binary_emission_weights_summary_boxplot,
    plot_binary_emission_weights_summary_lineplot,
    plot_emission_weights,
    plot_emission_weights_by_subject,
    plot_emission_weights_summary,
    plot_emission_weights_summary_boxplot,
    plot_emission_weights_summary_lineplot,
)
from glmhmmt.model_plotting.sessions import (
    plot_posterior_probs,
    plot_session_deepdive,
    plot_session_trajectories,
)
from glmhmmt.model_plotting.states import (
    plot_change_triggered_posteriors_by_subject,
    plot_change_triggered_posteriors_summary,
    plot_state_accuracy,
    plot_state_dwell_times,
    plot_state_dwell_times_by_subject,
    plot_state_dwell_times_summary,
    plot_state_occupancy,
    plot_state_occupancy_overall,
    plot_state_occupancy_overall_boxplot,
    plot_state_posterior_count_kde,
    plot_state_session_occupancy,
    plot_state_switches,
)
from glmhmmt.model_plotting.transitions import (
    plot_transition_matrix,
    plot_transition_matrix_by_subject,
)

__all__ = [
    "plot_binary_emission_weights",
    "plot_binary_emission_weights_by_subject",
    "plot_binary_emission_weights_summary",
    "plot_binary_emission_weights_summary_boxplot",
    "plot_binary_emission_weights_summary_lineplot",
    "plot_change_triggered_posteriors_by_subject",
    "plot_change_triggered_posteriors_summary",
    "plot_emission_weights",
    "plot_emission_weights_by_subject",
    "plot_emission_weights_summary",
    "plot_emission_weights_summary_boxplot",
    "plot_emission_weights_summary_lineplot",
    "plot_posterior_probs",
    "plot_session_deepdive",
    "plot_session_trajectories",
    "plot_state_accuracy",
    "plot_state_dwell_times",
    "plot_state_dwell_times_by_subject",
    "plot_state_dwell_times_summary",
    "plot_state_occupancy",
    "plot_state_occupancy_overall",
    "plot_state_occupancy_overall_boxplot",
    "plot_state_posterior_count_kde",
    "plot_state_session_occupancy",
    "plot_state_switches",
    "plot_transition_matrix",
    "plot_transition_matrix_by_subject",
]
