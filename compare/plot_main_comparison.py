"""Figure 1: Dual half-violin — logZ (left) + runtime (right), per dataset."""
import matplotlib.pyplot as plt
import numpy as np
from utils import COLORS, DATASET_NAMES, load_all_runs, load_all_runtimes

logz_data = load_all_runs()
rt_data   = load_all_runtimes()
methods = ['NS', 'BS', 'AIS-HMC']
datasets = ['pima', 'creditcard', 'tcga']

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.subplots_adjust(wspace=0.42, left=0.06, right=0.93, top=0.87, bottom=0.18)

for panel_idx, (ds, ax) in enumerate(zip(datasets, axes)):
    positions = np.arange(len(methods))
    half_width = 0.38

    # Determine shared y-axis ranges
    all_logz = []
    all_rts  = []
    for m in methods:
        all_logz.extend(logz_data[(m, ds)])
        all_rts.extend(rt_data[(m, ds)])

    logz_lo = np.percentile(all_logz, 1)
    logz_hi = np.percentile(all_logz, 99)
    logz_pad = (logz_hi - logz_lo) * 0.08
    ax.set_ylim(logz_lo - logz_pad, logz_hi + logz_pad)

    # ── Right y-axis for runtime ──
    ax_rt = ax.twinx()
    rt_lo = 0
    rt_hi = np.percentile(all_rts, 99) * 1.18
    ax_rt.set_ylim(rt_lo, rt_hi)

    for i, m in enumerate(methods):
        lz = np.array(logz_data[(m, ds)])
        rt = np.array(rt_data[(m, ds)])
        x_center = positions[i]
        c = COLORS[m]

        # ── Left half-violin: logZ ──
        try:
            from scipy.stats import gaussian_kde
            kde_lz = gaussian_kde(lz)
            y_lz = np.linspace(np.percentile(lz, 0.5), np.percentile(lz, 99.5), 200)
            d_lz = kde_lz(y_lz)
            d_lz = d_lz / d_lz.max() * half_width
            ax.fill_betweenx(y_lz, x_center - d_lz, x_center, alpha=0.7, color=c,
                             edgecolor=None, zorder=4)
            ax.plot(x_center - d_lz, y_lz, color=c, linewidth=1, alpha=0.95, zorder=5)
        except:
            pass

        # ── Right half-violin: runtime ──
        try:
            from scipy.stats import gaussian_kde
            kde_rt = gaussian_kde(rt)
            y_rt = np.linspace(np.percentile(rt, 0.5), np.percentile(rt, 99.5), 200)
            d_rt = kde_rt(y_rt)
            d_rt = d_rt / d_rt.max() * half_width
            ax_rt.fill_betweenx(y_rt, x_center, x_center + d_rt, alpha=0.5, color=c,
                                edgecolor=None, zorder=4)
            ax_rt.plot(x_center + d_rt, y_rt, color=c, linewidth=1, alpha=0.8, zorder=5)
        except:
            pass

        # ── Summary annotation ──
        lz_mean = np.mean(lz)
        lz_std  = np.std(lz)
        rt_mean = np.mean(rt)
        rt_std  = np.std(rt)
        ax.annotate(f'log $\\mathcal{{Z}}$:\n{ lz_mean:.2f} $\\pm$ {lz_std:.2f}',
                    xy=(x_center - half_width * 0.2, lz_mean),
                    fontsize=6.5, ha='right', va='center', color='#333333', alpha=0.85)
        ax_rt.annotate(f'{rt_mean:.0f}s $\\pm$ {rt_std:.0f}s',
                       xy=(x_center + half_width * 0.2, rt_mean),
                       fontsize=6.5, ha='left', va='center', color='#666666', alpha=0.85)

    # ── Styling ──
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, fontsize=13, fontweight='medium')
    ax.set_title(DATASET_NAMES[ds], fontsize=15, fontweight='bold', pad=16)
    ax.set_xlim(-0.65, 2.65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(axis='y', alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=11)
    if panel_idx == 0:
        ax.set_ylabel('log $\\mathcal{Z}$', fontsize=14)

    ax_rt.spines['top'].set_visible(False)
    ax_rt.spines['left'].set_visible(False)
    ax_rt.spines['right'].set_color('#cccccc')
    ax_rt.tick_params(axis='y', labelsize=10, labelcolor='#888888')
    if panel_idx == 2:
        ax_rt.set_ylabel('Runtime (s)', fontsize=13, color='#888888', labelpad=10)

    # ── Center divider line per group ──
    for xi in positions:
        ax.axvline(x=xi, ymin=0.02, ymax=0.98, color='#cccccc', linewidth=0.6, linestyle=':', alpha=0.5)

# ── Legend ──
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[m], edgecolor='#888888', alpha=0.7, label=m) for m in methods]
# dummy entries for left/right explanation
from matplotlib.lines import Line2D
legend_elements += [
    Line2D([0], [0], marker='s', color='none', markerfacecolor='#333333', markersize=10,
           markeredgecolor='none', alpha=0.6, label='log $\\mathcal{Z}$ (left)'),
    Line2D([0], [0], marker='s', color='none', markerfacecolor='#888888', markersize=10,
           markeredgecolor='none', alpha=0.3, label='Runtime (right)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.005), fontsize=10.5, frameon=True,
           edgecolor='#dddddd', facecolor='white')

fig.suptitle('Log Marginal Likelihood and Runtime Distributions Across 24–30 Independent Runs',
             fontsize=16, fontweight='bold', y=0.99)
fig.savefig('fig1_main_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 1 saved: fig1_main_comparison.pdf')
