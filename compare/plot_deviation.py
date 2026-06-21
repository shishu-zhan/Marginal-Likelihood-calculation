"""Figure 4: Deviation from three-method median."""
import matplotlib.pyplot as plt
import numpy as np
from utils import COLORS, DATASET_NAMES, DIMS, load_comparison_csv

comp = load_comparison_csv()
methods = ['NS', 'BS', 'AIS-HMC']
datasets = ['pima', 'creditcard', 'tcga']

medians = {}
for ds in datasets:
    vals = [comp[(m, ds)]['logZ_mean'] for m in methods]
    medians[ds] = np.median(vals)

fig, ax = plt.subplots(figsize=(10, 5.5))

group_gap = 1.6
x_positions = []
for j, ds in enumerate(datasets):
    offset = j * (len(methods) + group_gap)
    for i, m in enumerate(methods):
        x = i + offset
        x_positions.append(x)
        delta = comp[(m, ds)]['logZ_mean'] - medians[ds]
        bar = ax.bar(x, delta, 0.45, color=COLORS[m], edgecolor='#555555', linewidth=0.6, zorder=3)
        y_pos = delta + (0.25 if delta >= 0 else -0.5)
        va = 'bottom' if delta >= 0 else 'top'
        ax.text(x, y_pos, f'{delta:+.2f}', ha='center', va=va, fontsize=8.5, fontweight='bold',
                color='#333333')

# Zero reference line per dataset
for j in range(3):
    x0 = j * (len(methods) + group_gap) - 0.6
    x1 = j * (len(methods) + group_gap) + len(methods) - 0.4
    ax.axhline(y=0, color='#999999', linewidth=0.8, linestyle=':' if j > 0 else '-', alpha=0.6)

all_ticks = []
all_labels = []
for j in range(3):
    offset = j * (len(methods) + group_gap)
    all_ticks.extend(np.arange(len(methods)) + offset)
    all_labels.extend(methods)
ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels, fontsize=11)
ax.set_ylabel('Deviation from Method Median (nats)', fontsize=13)
ax.set_title('Cross-Method Agreement: Deviation from Three-Method Median', fontsize=14, fontweight='bold', pad=12)

# Dataset group labels below axis
for j, ds in enumerate(datasets):
    offset = j * (len(methods) + group_gap)
    mid = np.mean([offset, offset + len(methods) - 1])
    ax.annotate(f'{DATASET_NAMES[ds]}\n(median = {medians[ds]:.2f})',
                xy=(mid, 0), xycoords=('data', 'axes fraction'),
                ha='center', va='top', fontsize=9.5, fontweight='bold',
                xytext=(0, -22), textcoords='offset points')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[m], edgecolor='#555555', label=m) for m in methods]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)

plt.tight_layout()
fig.savefig('fig4_deviation.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Figure 4 saved: fig4_deviation.pdf')
