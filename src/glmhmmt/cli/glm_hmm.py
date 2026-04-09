import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dynamax.utils.plotting import gradient_cmap
from glmhmmt.model import SoftmaxGLMHMM

from glmhmmt.runtime import get_data_dir
from glmhmmt.tasks import get_adapter

adapter = get_adapter("mcdr")
plots = adapter.get_plots()

sns.set_style("white")

color_names = [
    "red",
    "amber",
    "windows blue",
]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)    

df = pl.read_parquet(get_data_dir() / adapter.data_file)
y, X, U, names = adapter.load_subject(df.filter(pl.col("subject") == "A89"))

num_states= 2        # nº estados
emission_dim = 3          # 3 choices
input_dim = X.shape[1]          # intercept + delay + stim_L + stim_C + stim_R + previous_outcome
print(input_dim)

model = SoftmaxGLMHMM(num_states=num_states, num_classes=emission_dim, emission_input_dim=input_dim, transition_input_dim=0, m_step_num_iters=100, transition_matrix_stickiness=10.0)

# model2 = CategoricalRegressionHMM(num_states=num_states, num_classes=emission_dim, input_dim=input_dim-2,  transition_matrix_stickiness=10.0, m_step_optimizer=optax.adam(1e-3), m_step_num_iters=500,)

model3 = SoftmaxGLMHMM(num_states=num_states, num_classes=emission_dim, emission_input_dim=input_dim-2, transition_input_dim=0, m_step_num_iters=100, transition_matrix_stickiness=10.0)


key = jr.PRNGKey(12345)
params, props = model.initialize(key=key)
# params2, props2 = model2.initialize(key=key)
params3, props3 = model3.initialize(key=key)


model4 = SoftmaxGLMHMM( num_states=num_states, num_classes=emission_dim, emission_input_dim=X.shape[1], transition_input_dim=U.shape[1], transition_matrix_stickiness=10.0, m_step_num_iters=100,)
params4, props4 = model4.initialize(key=key)

inputs_all = jnp.concatenate([X, U], axis=1)
print("T:", y.shape[0], "inputs_all:", inputs_all.shape)
print("emission_input_dim:", model4.emission_input_dim, "transition_input_dim:", model4.transition_input_dim)

fitted_params, lps = model.fit_em(params=params, props=props, emissions=y, inputs=X, num_iters=50)
# fitted_params2, lps2 = model2.fit_em(params=params2, props=props2, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 4:]], axis=1), num_iters=50)
# fitted_params3, lps3 = model3.fit_em(params=params3, props=props3, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 4:]], axis=1), num_iters=50)

fitted_params4, lps4 = model4.fit_em(params=params4, props=props4, emissions=y, inputs=jnp.concatenate([X, U], axis=1), num_iters=50)


posterior = model.smoother(params=fitted_params, emissions=y, inputs=X)
# posterior2 = model2.smoother(params=fitted_params2, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 4:]], axis=1))
# posterior3 = model3.smoother(params=fitted_params3, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 4:]], axis=1))
posterior4 = model4.smoother(params=fitted_params4, emissions=y, inputs=jnp.concatenate([X, U], axis=1))

T = int(y.shape[0])

lps_np = np.asarray(lps)
# lps2_np = np.asarray(lps2)

plt.figure(figsize=(6,3))
plt.plot(lps_np / T, "-o", ms=3)
# plt.plot(lps2_np / T, "-o", ms=3)
# plt.plot(np.asarray(lps3) / T, "-o", ms=3)
plt.plot(np.asarray(lps4) / T, "-o", ms=3)
plt.legend(["SoftmaxGLMHMM", "SoftmaxGLMHMM (no bias)", "SoftmaxGLMHMM-t"])
plt.xlabel("EM Iteration")
plt.ylabel("Avg log prob per trial")
plt.title("EM log-likelihood")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

W = np.asarray(fitted_params.emissions.weights)  # (K, C, M)
K, C, M = W.shape


vmax = np.max(np.abs(W))  # escala simétrica alrededor de 0

fig, axs = plt.subplots(1, C, figsize=(3.5*C, 3), sharey=True)
if C == 1:
    axs = [axs]

for c in range(C):
    im = axs[c].imshow(W[:, c, :], aspect="auto", interpolation="none",
                       cmap="Greys", vmin=-vmax, vmax=vmax)
    axs[c].set_title(f"Emission weights (class {c})")
    axs[c].set_xlabel("input dim")
    axs[c].set_xticks(np.arange(M))
    axs[c].set_xticklabels(names["X_cols"])
    if c == 0:
        axs[c].set_ylabel("state")
        axs[c].set_yticks(np.arange(K))
        axs[c].set_yticklabels(np.arange(K) + 1)
    plt.colorbar(im, ax=axs[c], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

gamma = np.asarray(posterior.smoothed_probs)  # (T, K)

# Elige una ventana para visualizar
plot_slice = (0, min(1000, T))   # ajusta

fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

# 1) Heatmap de probabilidades posteriores
im = axs[0].imshow(gamma[plot_slice[0]:plot_slice[1]].T,
                   aspect="auto", interpolation="none",
                   cmap="Greys", vmin=0, vmax=1)
axs[0].set_ylabel("state k")
axs[0].set_yticks(np.arange(K))
axs[0].set_yticklabels(np.arange(K) + 1)
axs[0].set_title(r"Posterior probs $p(z_t=k \mid y_{1:T}, x_{1:T})$")
plt.colorbar(im, ax=axs[0], fraction=0.02, pad=0.02)

# 2) Estado MAP (argmax) como “barra”
z_map = np.argmax(gamma, axis=1)
axs[1].imshow(z_map[None, plot_slice[0]:plot_slice[1]],
              aspect="auto", interpolation="none",
              cmap=cmap, vmin=0, vmax=len(colors)-1)
axs[1].set_yticks([])
axs[1].set_xlabel("time (trial)")
axs[1].set_title("Most likely state (MAP)")

plt.tight_layout()
plt.show()

A = np.asarray(fitted_params.transitions.transition_matrix)
print("Transition matrix A:")
print(np.round(A, 3))


# p_pred = np.asarray(model3.predict_choice_probs(fitted_params3, y, jnp.concatenate([X[:, :1], X[:, 4:]], axis=1)))
p_pred = np.asarray(model.predict_choice_probs(fitted_params, y, X))
# p_pred = np.asarray(model4.predict_choice_probs(fitted_params4, y, jnp.concatenate([X[:, :], U], axis=1)))


pL, pC, pR = p_pred[:,0], p_pred[:,1], p_pred[:,2]

df_sub = df.filter(pl.col("subject") == "A89").sort("trial_idx")

df_sub = df_sub.with_columns([pl.Series("pL", pL), pl.Series("pC", pC), pl.Series("pR", pR), pl.Series("pred_choice", np.argmax(p_pred, axis=1))])

df_sub.write_parquet(get_data_dir() / "predictions.parquet")

plot_df = plots.prepare_predictions_df(df_sub)

plots.plot_categorical_performance_all(plot_df, "glmhmm")

print("Saved:", get_data_dir() / "predictions.parquet")

# --- Transition weights ---
import pandas as pd
bias_np = np.asarray(fitted_params4.transitions.bias)    # (K, K)
W_tr    = np.asarray(fitted_params4.transitions.weights)  # (K, K, D)

state_labels = [f"s{k+1}" for k in range(bias_np.shape[0])]

print("\n=== Transition bias (K×K) ===")
df_bias = pd.DataFrame(np.round(bias_np, 4), index=state_labels, columns=state_labels)
df_bias.index.name = "from \\ to"
print(df_bias.to_string())

print("\n=== Transition input weights (one table per U_col) ===")
for d, col_name in enumerate(names["U_cols"]):
    print(f"\n  [{col_name}]")
    df_w = pd.DataFrame(np.round(W_tr[:, :, d], 4), index=state_labels, columns=state_labels)
    df_w.index.name = "from \\ to"
    print(df_w.to_string())
