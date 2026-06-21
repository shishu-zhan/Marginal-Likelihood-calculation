"""Figure 5: NS rwalk efficiency degradation — epsilon vs nlive/d."""
import matplotlib.pyplot as plt
import numpy as np
from utils import COLORS, DATASET_NAMES, DIMS, load_comparison_csv

comp = load_comparison_csv()

fig, ax = plt.subplots(figsize=(8, 5.5))

# Data points from comparative analysis
# (nlive/d, epsilon, label)
points = [
    (300/9,    0.95,  'Pima\nrwalk, walks=30\nnlive=300'),
    (400/29,   2.01,  'CreditCard\nrwalk, walks=40\nnlive=400'),
    (500/29,   2.29,  'CreditCard V1 rwalk, walks=30 nlive=500'),
    (400/61,   7.66,  'TCGA V1 (FAILED)\nrwalk, walks=30\nnlive=400'),
]

# Plot rwalk data points (efficiency > 1.5 are failures)
colors_rwalk = []
markers_rwalk = []
for _, eps, _ in points:
    if eps < 3:
        colors_rwalk.append('#e6d592')
        markers_rwalk.append('o')
    else:
        colors_rwalk.append('#CC3333')
        markers_rwalk.append('X')

for i, (x, y, label) in enumerate(points):
    ax.scatter(x, y, s=200, c=colors_rwalk[i], marker=markers_rwalk[i],
               edgecolors='black', linewidth=1, zorder=5)

    # 针对第二个点（索引 1）特殊处理：正下方居中
    if i == 1:
        # 使用 offset points 实现更稳定的偏移，或者直接数据坐标
        ax.annotate(label, xy=(x, y),
                    xytext=(0, -10),          # 相对点向下偏移 10 点（约 0.15 数据单位，视坐标轴范围而定）
                    textcoords='offset points',
                    fontsize=8, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    else:
        # 原有逻辑（右上）
        offset_y = 0.4 if y < 3 else -0.6
        ax.annotate(label, xy=(x, y), xytext=(x + 0.5, y + offset_y),
                    fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

# Fit exponential trend to the 3 normal points
x_fit = np.array([p[0] for p in points[:3]])
y_fit = np.array([p[1] for p in points[:3]])
log_y = np.log(y_fit)
coeffs = np.polyfit(x_fit, log_y, 1)
x_smooth = np.linspace(5, 35, 100)
y_smooth = np.exp(coeffs[1]) * np.exp(coeffs[0] * x_smooth)
ax.plot(x_smooth, y_smooth, '--', color='gray', linewidth=1.5, alpha=0.7,
        label=f'$\\varepsilon \\propto e^{{{coeffs[0]:.3f} \\cdot N_{{live}}/d}}$')

# Shade danger zone
ax.axhspan(3, ax.get_ylim()[1], xmin=0, xmax=1, facecolor='red', alpha=0.06)
ax.text(7, 3.8, 'rwalk unstable', fontsize=10, color='#CC3333', fontstyle='italic', alpha=0.8)
ax.axhline(y=1, color='black', linestyle=':', alpha=0.5, linewidth=0.8)
ax.text(28, 1.05, 'ideal ($\\varepsilon$ = 1)', fontsize=9, color='gray')

ax.set_xlabel('$N_{\\rm live} / d$ (Particle Density Ratio)', fontsize=13)
ax.set_ylabel('Efficiency Factor  $\\varepsilon = \\sigma_{\\rm actual} / \\sigma_{\\rm theory}$',
              fontsize=13)
ax.set_title('NS rwalk Efficiency Degradation with Dimension', fontsize=14, fontweight='bold')

# rslice reference point
ax.scatter(250/61, 1.28, s=200, c='#DDAA33', marker='D', edgecolors='black',
           linewidth=1, zorder=5)
ax.annotate('TCGA rslice\nnlive=250\n(recovered)',
            xy=(250/61, 1.28), xytext=(5, 1.8),
            fontsize=8, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', alpha=0.9, edgecolor='#DDAA33'))

ax.legend(fontsize=9, frameon=True, loc='upper right')
plt.tight_layout()
fig.savefig('fig5_rwalk_efficiency.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Figure 5 saved: compare/fig5_rwalk_efficiency.pdf')
