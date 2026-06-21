import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BASE = "C:/Users/Lenovo/Marginal-Likelihood-calculation/bayes logistic regression"
OUT = "C:/Users/Lenovo/Desktop"

datasets = ["pima", "creditcard", "tcga"]
labels  = ["Pima (d = 9)", "CreditCard (d = 29)", "TCGA (d = 61)"]
top_dims = {
    "pima":       ["BMI", "Glucose", "Intercept"],
    "creditcard": ["Intercept", "V8", "V21"],
    "tcga":       ["PC4", "PC3", "PC2"],
}
dim_labels = {
    "pima":       ["BMI", "Glucose", "Intercept"],
    "creditcard": [r"$\beta_0$", r"V8", r"V21"],
    "tcga":       ["PC4", "PC3", "PC2"],
}
n_lags = 50
max_chains = 10

def autocorr(x):
    x0 = x - np.mean(x)
    n = len(x0)
    r = np.correlate(x0, x0, mode="full")[-n:]
    r /= r[0]
    return r[:n_lags]

meta = pd.read_csv(f"{BASE}/hmc_runs/hmc_metadata.csv")

fig1 = plt.figure(figsize=(7.5, 6.5))
gs1  = gridspec.GridSpec(3, 3, hspace=0.40, wspace=0.35,
                         left=0.07, right=0.97, bottom=0.07, top=0.94)

colors = ["#457B9D", "#E63946", "#2A9D8F"]

for row, ds in enumerate(datasets):
    df = pd.read_csv(f"{BASE}/hmc_runs/{ds}/run_000.csv")
    n = len(df)
    x_axis = np.arange(n)
    dims = top_dims[ds]
    dlab = dim_labels[ds]

    for col, (d, dl) in enumerate(zip(dims, dlab)):
        ax = fig1.add_subplot(gs1[row, col])
        vals = df[d].values
        ax.plot(x_axis, vals, color=colors[col], linewidth=0.35, alpha=0.7)
        ax.set_xlim(0, n)
        ax.set_xlabel("Iteration", fontsize=7.5)
        ax.set_ylabel(dl, fontsize=7.5, rotation=0, ha="right", va="center",
                      labelpad=12)

        mean_val = np.mean(vals)
        ax.axhline(mean_val, color="grey", linewidth=0.4, linestyle="--", alpha=0.6)

        ax.tick_params(labelsize=7)

        if col == 1:
            ax.set_title(labels[row], fontsize=9, fontweight="bold", pad=6)

fig1.suptitle("HMC Trace Plots", fontsize=11, fontweight="bold", y=0.98)
fig1.savefig(f"{OUT}/Appendix_HMC_TracePlots.pdf", dpi=300)
fig1.savefig(f"{OUT}/Appendix_HMC_TracePlots.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("[OK] Trace plots")

fig2 = plt.figure(figsize=(7.5, 6.5))
gs2  = gridspec.GridSpec(3, 3, hspace=0.40, wspace=0.35,
                         left=0.07, right=0.97, bottom=0.07, top=0.94)

for row, ds in enumerate(datasets):
    df = pd.read_csv(f"{BASE}/hmc_runs/{ds}/run_000.csv")
    dims = top_dims[ds]
    dlab = dim_labels[ds]

    for col, (d, dl) in enumerate(zip(dims, dlab)):
        ax = fig2.add_subplot(gs2[row, col])
        vals = df[d].values
        acf  = autocorr(vals)
        lag_axis = np.arange(len(acf))

        ax.bar(lag_axis, acf, width=0.8, color=colors[col], alpha=0.75,
               edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.axhline(1.96 / np.sqrt(len(vals)), color="grey", linewidth=0.4,
                   linestyle="--")
        ax.axhline(-1.96 / np.sqrt(len(vals)), color="grey", linewidth=0.4,
                   linestyle="--")

        ax.set_xlim(-0.5, n_lags - 0.5)
        ax.set_ylim(-0.15, 1.05)
        ax.set_xlabel("Lag", fontsize=7.5)
        ax.set_ylabel(dl, fontsize=7.5, rotation=0, ha="right", va="center",
                      labelpad=12)
        ax.tick_params(labelsize=7)

        if col == 1:
            ax.set_title(labels[row], fontsize=9, fontweight="bold", pad=6)

fig2.suptitle("Autocorrelation Function of HMC Samples", fontsize=11,
              fontweight="bold", y=0.98)
fig2.savefig(f"{OUT}/Appendix_HMC_ACF.pdf", dpi=300)
fig2.savefig(f"{OUT}/Appendix_HMC_ACF.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("[OK] ACF plots")

fig3, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

box_colors = ["#A8DADC", "#F4A261", "#E76F51"]
ds_labels_short = ["Pima\n(d=9)", "CreditCard\n(d=29)", "TCGA\n(d=61)"]

ax = axes[0]
acc_data = []
for ds in datasets:
    sub = meta[meta.dataset == ds]
    acc_data.append(sub["acc_rate"].values * 100)

bp1 = ax.boxplot(acc_data, patch_artist=True, widths=0.45,
                 medianprops={"color": "black", "linewidth": 1.2},
                 whiskerprops={"linewidth": 0.8},
                 capprops={"linewidth": 0.8})

for patch, c in zip(bp1["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.6)

for i, data in enumerate(acc_data):
    jitter = np.random.normal(0, 0.04, size=len(data))
    ax.scatter(np.full_like(data, i + 1) + jitter, data, s=12,
               color=box_colors[i], edgecolors="black", linewidth=0.3,
               alpha=0.7, zorder=3)

ax.set_xticklabels(ds_labels_short, fontsize=8)
ax.set_ylabel("Acceptance Rate (%)", fontsize=9)
ax.tick_params(labelsize=7.5)
ax.set_title("(a) HMC Acceptance Rate", fontsize=10, fontweight="bold")

ax = axes[1]
rt_data = []
for ds in datasets:
    sub = meta[meta.dataset == ds].copy()
    if ds == "tcga":
        sub = sub[sub["runtime"] < 1000]
    rt_data.append(sub["runtime"].values)

bp2 = ax.boxplot(rt_data, patch_artist=True, widths=0.45,
                 medianprops={"color": "black", "linewidth": 1.2},
                 whiskerprops={"linewidth": 0.8},
                 capprops={"linewidth": 0.8})

for patch, c in zip(bp2["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.6)

for i, data in enumerate(rt_data):
    jitter = np.random.normal(0, 0.04, size=len(data))
    ax.scatter(np.full_like(data, i + 1) + jitter, data, s=12,
               color=box_colors[i], edgecolors="black", linewidth=0.3,
               alpha=0.7, zorder=3)

ax.set_xticklabels(ds_labels_short, fontsize=8)
ax.set_ylabel("Runtime (seconds)", fontsize=9)
ax.tick_params(labelsize=7.5)
ax.set_title("(b) HMC Runtime", fontsize=10, fontweight="bold")

fig3.tight_layout(pad=1.5)
fig3.savefig(f"{OUT}/Appendix_HMC_AccRate_Runtime.pdf", dpi=300)
fig3.savefig(f"{OUT}/Appendix_HMC_AccRate_Runtime.png", dpi=300,
             bbox_inches="tight")
plt.close(fig3)
print("[OK] Acceptance rate + Runtime")

def compute_rhat(chains):
    m = len(chains)
    n = len(chains[0])
    chain_means = np.array([np.mean(c) for c in chains])
    chain_vars  = np.array([np.var(c, ddof=1) for c in chains])
    overall_mean = np.mean(chain_means)

    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(chain_vars)
    var_plus = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_plus / W)
    return rhat

fig4, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))

for idx, (ds, label) in enumerate(zip(datasets, labels)):
    rhat_vals = []
    col_names = None

    for run_id in range(max_chains):
        path = f"{BASE}/hmc_runs/{ds}/run_{run_id:03d}.csv"
        df_run = pd.read_csv(path)
        if col_names is None:
            col_names = df_run.columns.tolist()

    for col in col_names:
        chains = []
        for run_id in range(max_chains):
            path = f"{BASE}/hmc_runs/{ds}/run_{run_id:03d}.csv"
            df_run = pd.read_csv(path)
            chains.append(df_run[col].values)
        rhat_vals.append(compute_rhat(chains))

    rhat_vals = np.array(rhat_vals)

    ax = axes[idx]
    ax.scatter(np.arange(len(rhat_vals)), rhat_vals, s=8, color="#2A9D8F",
               alpha=0.7, edgecolors="none")
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="-")
    ax.axhline(1.01, color="#E63946", linewidth=0.7, linestyle="--",
               label=r"$\hat{R}=1.01$")

    ax.set_xlabel("Parameter index", fontsize=7.5)
    if idx == 0:
        ax.set_ylabel(r"$\hat{R}$", fontsize=9)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_ylim(0.998, min(1.04, max(rhat_vals) * 1.05 + 0.01))
    ax.tick_params(labelsize=7)

    n_above = np.sum(rhat_vals > 1.01)
    ax.text(0.95, 0.95, f"max {np.max(rhat_vals):.4f}\nn > 1.01: {n_above}",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", linewidth=0.3))

    if idx == 2:
        ax.legend(fontsize=7, loc="lower right")

fig4.suptitle(r"Gelman-Rubin $\hat{R}$ Diagnostic (10 Chains)", fontsize=10,
              fontweight="bold", y=1.02)
fig4.tight_layout()
fig4.savefig(f"{OUT}/Appendix_HMC_Rhat.pdf", dpi=300)
fig4.savefig(f"{OUT}/Appendix_HMC_Rhat.png", dpi=300, bbox_inches="tight")
plt.close(fig4)
print("[OK] Rhat")

print("\nAll figures saved to Desktop.")
