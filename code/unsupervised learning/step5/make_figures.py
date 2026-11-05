# -*- coding: utf-8 -*-
"""Step 5 figures.

(a) detector comparison on the physics feature set: threshold-free PR-AUC and
    F1 at the reference operating point
(b) what labels buy: best F1 per label-budget layer

Grayscale only. Reads results_step5.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / 'results_step5.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2),
                               gridspec_kw={'width_ratios': [1, 1.05]})

# ---------------------------------------------------------------- panel (a)
sub = d[(d.feature_set == 'F2_physics')].copy()
pr = sub[sub.layer == 'L0 POT target 1%'].set_index('detector')['pr_auc']
f1 = sub[sub.layer == 'Lref ctm=3.39%'].set_index('detector')['f1']
order = pr.sort_values(ascending=True).index.tolist()
y = np.arange(len(order))
h = 0.38
ax1.barh(y + h / 2, [pr[k] for k in order], h, label='PR-AUC (threshold-free)',
         color='0.35', edgecolor='black', linewidth=0.6)
ax1.barh(y - h / 2, [f1[k] for k in order], h,
         label='F1 at the reference operating point', color='0.80',
         edgecolor='black', linewidth=0.6, hatch='//')
for yi, k in zip(y, order):
    ax1.text(pr[k] + 0.008, yi + h / 2, f'{pr[k]:.3f}', va='center', fontsize=7.5)
    ax1.text(f1[k] + 0.008, yi - h / 2, f'{f1[k]:.3f}', va='center', fontsize=7.5)
ax1.set_yticks(y)
ax1.set_yticklabels([k.replace(' (boundary violation)', '\n(boundary violation)')
                     for k in order], fontsize=8)
ax1.set_xlim(0, 0.52)
ax1.set_xlabel('score on the 3,000-sample test set', fontsize=9.5)
ax1.set_title('(a) Eight detectors, physics feature set (F2)', fontsize=11)
ax1.legend(fontsize=8, frameon=False, loc='lower right')
ax1.grid(axis='x', color='0.88', linewidth=0.5)
ax1.set_axisbelow(True)
for sp in ('top', 'right'):
    ax1.spines[sp].set_visible(False)

# ---------------------------------------------------------------- panel (b)
keep = ['L0 POT target 1%', 'L1 ctm=1%', 'Lref ctm=3.39%',
        'L2b rate from M=100', 'L2b rate from M=300', 'L2b rate from M=1000',
        'L2a F1max N=10', 'L2a F1max N=20', 'L2a F1max N=50',
        'L2a F1max N=100']
g = (d[d.layer.isin(keep)].groupby('layer')[['f1', 'precision', 'recall']]
     .max().reindex(keep))
label = {
    'L0 POT target 1%': 'L0\nPOT 1%\n(no prior)',
    'L1 ctm=1%': 'L1\n1% prior\n(3x wrong)',
    'Lref ctm=3.39%': 'Lref\ntrue rate\n3.39%',
    'L2b rate from M=100': 'L2b\nM=100',
    'L2b rate from M=300': 'L2b\nM=300',
    'L2b rate from M=1000': 'L2b\nM=1000',
    'L2a F1max N=10': 'L2a\nN=10',
    'L2a F1max N=20': 'L2a\nN=20',
    'L2a F1max N=50': 'L2a\nN=50',
    'L2a F1max N=100': 'L2a\nN=100',
}
x = np.arange(len(keep))
shade = ['0.25', '0.35', '0.45', '0.30', '0.20', '0.30',
         '0.88', '0.88', '0.88', '0.88']
hatch = ['', '', '', '..', '..', '..', 'xx', 'xx', 'xx', 'xx']
kinds = ['label-free', 'label-free', 'reference (label-derived)',
         'labels: rate only', 'labels: rate only', 'labels: rate only',
         'labels: naive F1-max', 'labels: naive F1-max',
         'labels: naive F1-max', 'labels: naive F1-max']
ax2.bar(x, g['f1'].values, 0.62, color=shade, edgecolor='black',
        linewidth=0.6, hatch=hatch)
for xi, v in zip(x, g['f1'].values):
    ax2.text(xi, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels([label[k] for k in keep], fontsize=7.6)
ax2.set_ylim(0, 0.55)
ax2.set_ylabel('best F1 over all detectors and feature sets', fontsize=9.5)
ax2.set_title('(b) What labels actually buy', fontsize=11)
ax2.grid(axis='y', color='0.88', linewidth=0.5)
ax2.set_axisbelow(True)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
ax2.axhline(g.loc['L0 POT target 1%', 'f1'], color='0.45', linewidth=0.8,
            linestyle='--')
ax2.text(9.55, g.loc['L0 POT target 1%', 'f1'] + 0.006,
         'zero-label baseline', fontsize=7.5, ha='right', color='0.35')

fig.suptitle('Step 5 — the matrix, and the label-budget layers', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = HERE / 'fig_step5_matrix_and_labels.png'
fig.savefig(out, dpi=180)
print('saved:', out)
