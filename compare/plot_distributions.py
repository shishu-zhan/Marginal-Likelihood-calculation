"""Figure 3: Run distributions — 3 panels (one per dataset)."""
import matplotlib.pyplot as plt
import numpy as np
from utils import COLORS, DATASET_NAMES, load_all_runs

data = load_all_runs()
methods = ['NS', 'BS', 'AIS-HMC']
datasets = ['pima', 'creditcard', 'tcga']

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

for panel_idx, (ds, ax) in enumerate(zip(datasets, axes)):
    positions = np.arange(len(methods))
    vwidth = 0.4

    for i, m in enumerate(methods):
        vals = np.array(data[(m, ds)])
        x = positions[i]
        c = COLORS[m]

        # Half-violin
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            y_lo = min(vals) - 0.8 * np.std(vals)
            y_hi = max(vals) + 0.8 * np.std(vals)
            y_range = np.linspace(y_lo, y_hi, 200)
            density = kde(y_range)
            density = density / density.max() * vwidth
            ax.fill_betweenx(y_range, x - density, x, alpha=0.35, color=c, edgecolor=None)
            ax.plot(x - density, y_range, color=c, linewidth=1, alpha=0.6)
        except:
            pass

        # Boxplot
        bp = ax.boxplot(vals, positions=[x], widths=vwidth * 0.7, patch_artist=True,
                        medianprops={'color': '#333333', 'linewidth': 1.8},
                        whiskerprops={'linewidth': 1.2, 'color': '#555555'},
                        capprops={'linewidth': 1.2, 'color': '#555555'},
                        flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.4, 'markerfacecolor': c})
        bp['boxes'][0].set_facecolor(c)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][0].set_edgecolor('#555555')
        bp['boxes'][0].set_linewidth(0.8)

        # Scatter jitter
        n_pts = len(vals)
        jitter = np.random.normal(0, vwidth * 0.2, n_pts)
        ax.scatter(np.full(n_pts, x) + vwidth * 0.5 + jitter, vals,
                   alpha=0.45, s=11, color=c, edgecolor='none', zorder=3)

    # Y-axis: tight around data
    all_vals = np.concatenate([np.array(data[(m, ds)]) for m in methods])
    y_c = np.median(all_vals)
    y_spread = max(np.ptp(all_vals) / 2, 3.0 * np.std(all_vals))
    ax.set_ylim(y_c - y_spread * 1.18, y_c + y_spread * 1.18)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_title(DATASET_NAMES[ds], fontsize=13, fontweight='bold', pad=8)
    if panel_idx == 0:
        ax.set_ylabel('log $Z$', fontsize=13)

    # σ annotation
    lines = []
    for m in methods:
        v = np.array(data[(m, ds)])
        lines.append(f'{m}: $\\sigma = {np.std(v):.3f}$')
    ax.text(0.02, 0.97, '\n'.join(lines), transform=ax.transAxes,
            fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85,
                      edgecolor='#cccccc', linewidth=0.5))

fig.suptitle('Distribution of 24–30 Independent Runs per Dataset', fontsize=16, fontweight='bold', y=1.03)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[m], edgecolor='#555555', alpha=0.7, label=m) for m in methods]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05),
           fontsize=11, frameon=True)

plt.tight_layout()
fig.savefig('fig3_distributions.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Figure 3 saved: fig3_distributions.pdf')
