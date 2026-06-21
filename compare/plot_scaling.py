"""Figure 2: Dimensional scaling — sigma vs d and N_call vs d, side by side."""
import matplotlib.pyplot as plt
import numpy as np
from utils import COLORS, DIMS, load_comparison_csv

comp = load_comparison_csv()
methods = ['NS', 'BS', 'AIS-HMC']
ds_ordered = ['pima', 'creditcard', 'tcga']
dims_plot = [DIMS[d] for d in ds_ordered]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.subplots_adjust(wspace=0.32, left=0.07, right=0.96)

# ── Panel A: sigma vs d (log-log) ──
for m in methods:
    sigmas = [comp[(m, ds)]['logZ_sigma'] for ds in ds_ordered]
    ax1.plot(dims_plot, sigmas, 'o-', color=COLORS[m], linewidth=2.5, markersize=10,
             label=m, markeredgecolor='#555555', markeredgewidth=0.8)

    log_d = np.log(dims_plot)
    log_s = np.log(sigmas)
    slope, intercept = np.polyfit(log_d, log_s, 1)
    d_fine = np.linspace(8, 70, 100)
    ax1.plot(d_fine, np.exp(intercept) * d_fine**slope, '--', color=COLORS[m],
             linewidth=1.2, alpha=0.6)
    ax1.annotate(f'$\\alpha \\approx {slope:.2f}$',
                 xy=(dims_plot[-1]*0.78, sigmas[-1]*1.2), color='#555555', fontsize=9,
                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Dimension $d$', fontsize=13)
ax1.set_ylabel('$\\sigma_{\\log Z}$', fontsize=13)
ax1.set_title('(a) Precision vs. Dimension', fontsize=14, fontweight='bold', pad=10)
ax1.legend(fontsize=10, frameon=True, loc='lower right')

# ── Panel B: N_call vs d ──
for m in methods:
    ncalls = [comp[(m, ds)]['ncall'] for ds in ds_ordered]
    ax2.plot(dims_plot, ncalls, 's-', color=COLORS[m], linewidth=2.5, markersize=10,
             label=m, markeredgecolor='#555555', markeredgewidth=0.8)

ax2.set_yscale('log')
ax2.set_xlabel('Dimension $d$', fontsize=13)
ax2.set_ylabel('Number of Likelihood / Gradient Evaluations', fontsize=13)
ax2.set_title('(b) Computational Cost vs. Dimension', fontsize=14, fontweight='bold', pad=10)
ax2.legend(fontsize=10, frameon=True, loc='lower right')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))

fig.suptitle('Dimensional Scaling of Precision and Cost', fontsize=16, fontweight='bold', y=1.005)
plt.tight_layout()
fig.savefig('fig2_scaling.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Figure 2 saved: fig2_scaling.pdf')
