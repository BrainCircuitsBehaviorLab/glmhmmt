import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import jax.numpy as jnp
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from glmhmmt.tasks import get_adapter

sns.set_style("white")

# ── Data ──────────────────────────────────────────────────────────────────────

adapter = get_adapter("mcdr")
plots = adapter.get_plots()


def build_glm_cat_sequence(df_sub: pl.DataFrame, tau = 50):
    # ttype_n levels: 0, 1, 2, 3  → dummies ttype_1..ttype_3 (ref = 0)
    # stimd_n levels: 1, 2, 3, 4  → per-side dummies SL_2..SL_4, SC_2..SC_4, SR_2..SR_4 (ref = 1)
    TTYPE_LEVELS = [1, 2, 3]
    STIMD_LEVELS = [2, 3, 4]

    df_sub = df_sub.sort("trial_idx")
    df_sub = df_sub.with_columns([
        pl.col("response").cast(pl.Int32),

        (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
        (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),

        # ttype_n dummies (reference = 0)
        *[
            (pl.col("ttype_n") == k).cast(pl.Float32).alias(f"ttype_{k}")
            for k in TTYPE_LEVELS
        ],

        # stimd_n × side dummies (reference stimd_n = 1)
        *[
            ((pl.col("x_c") == "L") & (pl.col("stimd_n") == k)).cast(pl.Float32).alias(f"SL_{k}")
            for k in STIMD_LEVELS
        ],
        *[
            ((pl.col("x_c") == "C") & (pl.col("stimd_n") == k)).cast(pl.Float32).alias(f"SC_{k}")
            for k in STIMD_LEVELS
        ],
        *[
            ((pl.col("x_c") == "R") & (pl.col("stimd_n") == k)).cast(pl.Float32).alias(f"SR_{k}")
            for k in STIMD_LEVELS
        ],

        pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).alias("previous_outcome"),
        pl.col("response").shift(1).fill_null(0.0).eq(0).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_L"),
        pl.col("response").shift(1).fill_null(0.0).eq(2).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_R"),
    ])
    df_sub = df_sub.with_columns([
        pl.col("previous_outcome").shift(1).fill_null(0.0).ewm_mean(half_life=tau, adjust=False).alias("A_plus"),
        (1.0 - pl.col("previous_outcome")).shift(1).fill_null(0.0).ewm_mean(half_life=tau, adjust=False).alias("A_minus"),
    ])

    ttype_cols = [f"ttype_{k}" for k in TTYPE_LEVELS]
    stimd_cols = (
        [f"SL_{k}" for k in STIMD_LEVELS]
        + [f"SC_{k}" for k in STIMD_LEVELS]
        + [f"SR_{k}" for k in STIMD_LEVELS]
    )
    x_cols = ["biasL", "biasR"] + ttype_cols + stimd_cols + ["A_L", "A_R"]

    y = df_sub["response"].to_numpy()

    X_base = df_sub.select(x_cols).to_numpy().astype(jnp.float32)
    X = jnp.asarray(X_base)
    U_base = df_sub.select(["A_plus", "A_minus"]).to_numpy().astype(jnp.float32)
    U = jnp.asarray(U_base)

    A_plus = jnp.asarray(df_sub["A_plus"].to_numpy())[:, None]
    A_minus = jnp.asarray(df_sub["A_minus"].to_numpy())[:, None]

    names = {
        "X_cols": x_cols,
        "U_cols": ["A_plus", "A_minus"],
    }
    return jnp.asarray(y), jnp.asarray(X), jnp.asarray(U), names, jnp.concatenate([A_plus, A_minus], axis=1)

df = adapter.read_dataset()
y, X, U, names, _ = build_glm_cat_sequence(df.filter(pl.col("subject") == "A89"))

y_np = np.asarray(y)        # (T,)  int {0,1,2}
X_np = np.asarray(X)        # (T, M)
T, M = X_np.shape
print(f"T: {T}  input_dim: {M}")
print("Features:", names["X_cols"])

# ── Fit multinomial logistic regression (GLM, no hidden states) ───────────────
clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    fit_intercept=False,   # bias already in X
    C=1e6,                 # essentially no regularisation
)
clf.fit(X_np, y_np)

# clf.coef_: (C-1, M) for multinomial with sklearn — but with 3 classes it's (3, M)
W = clf.coef_         # (C, M)  — one row per class
C = W.shape[0]

# ── Log-likelihood per trial ──────────────────────────────────────────────────
p_pred = clf.predict_proba(X_np)          # (T, C)
ll_per_trial = -log_loss(y_np, p_pred)   # already averaged, negate for LL
print(f"LL per trial: {ll_per_trial:.4f}")

n_params = C * M
bic = -2 * ll_per_trial * T + n_params * np.log(T)
print(f"BIC: {bic:.1f}")

# ── Plot weights ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, 0.8 * M), 3))
vmax = np.max(np.abs(W))
im = ax.imshow(W, aspect="auto", interpolation="none",
               cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title("GLM weights (multinomial softmax)")
ax.set_xlabel("input feature")
ax.set_ylabel("class")
ax.set_xticks(np.arange(M))
ax.set_xticklabels(names["X_cols"], rotation=45, ha="right")
ax.set_yticks(np.arange(C))
ax.set_yticklabels(["L", "C", "R"][:C])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# ── Bar plot per feature ──────────────────────────────────────────────────────
fig, axs = plt.subplots(1, C, figsize=(4 * C, 4), sharey=False)
if C == 1:
    axs = [axs]
class_names = ["L", "C", "R"][:C]
for c in range(C):
    axs[c].barh(np.arange(M), W[c], color=["steelblue" if v >= 0 else "tomato" for v in W[c]])
    axs[c].set_title(f"Class {class_names[c]}")
    axs[c].set_yticks(np.arange(M))
    axs[c].set_yticklabels(names["X_cols"])
    axs[c].axvline(0, color="k", lw=0.8)
    axs[c].set_xlabel("weight")
plt.suptitle("GLM weights per class", y=1.01)
plt.tight_layout()
plt.show()

# ── Save predictions ──────────────────────────────────────────────────────────
pL, pC, pR = p_pred[:, 0], p_pred[:, 1], p_pred[:, 2]
df_sub = df.filter(pl.col("subject") == "A89").sort("trial_idx")
df_sub = df_sub.with_columns([
    pl.Series("pL", pL),
    pl.Series("pC", pC),
    pl.Series("pR", pR),
    pl.Series("pred_choice", np.argmax(p_pred, axis=1)),
])
df_sub.write_parquet(adapter.dataset_path().parent / "predictions_glm.parquet")
print("Saved:", adapter.dataset_path().parent / "predictions_glm.parquet")

plot_df = plots.prepare_predictions_df(df_sub)
plots.plot_categorical_performance_all(plot_df, "glm")
plots.plot_categorical_strat_by_side(plot_df, subject = "A89", model_name = "GLM")

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
plots.plot_delay_or_stim_1d_on_ax(ax1, plot_df, subject="A89", n_bins=7, which="delay")
plots.plot_delay_or_stim_1d_on_ax(ax2, plot_df, subject="A89", n_bins=7, which="stim")
plt.show()
