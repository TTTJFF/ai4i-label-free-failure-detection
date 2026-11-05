# -*- coding: utf-8 -*-
"""Step 3 figure: physics arm vs the four control arms.

Grayscale only. Reads results_step3.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / 'results_step3.csv')

ARMS = ['A_physics', 'B1b_poly2_all', 'B1a_naive3', 'B0_raw', 'B2_nystroem']
ARM_LABEL = {
    'A_physics': 'A physics\n(3 derived)',
    'B1b_poly2_all': 'B1b all\ndegree-2 (15)',
    'B1a_naive3': 'B1a 3 naive\nproducts',
    'B0_raw': 'B0 raw\n(no expansion)',
    'B2_nystroem': 'B2 Nystroem\nRBF features',
}
SHADE = {'A_physics': '0.20', 'B1b_poly2_all': '0.35', 'B1a_naive3': '0.50',
         'B0_raw': '0.72', 'B2_nystroem': '0.88'}
HATCH = {'A_physics': '', 'B1b_poly2_all': '\\\\', 'B1a_naive3': 'xx',
         'B0_raw': '', 'B2_nystroem': '..'}
DETS = sorted(df.detector.unique())
DET_SHORT = {'Isolation Forest': 'IF', 'One-Class SVM': 'OCSVM',
             'Local Outlier Factor': 'LOF'}

METRICS = [('f1', 'F1 at fixed contamination (3.39%)'),
           ('roc_auc', 'ROC-AUC (threshold-free)'),
           ('pr_auc', 'PR-AUC (threshold-free)')]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
x = np.arange(len(ARMS))
w = 0.26

for ax, (metric, title) in zip(axes, METRICS):
    for k, d in enumerate(DETS):
        vals = [df[(df.arm == a) & (df.detector == d)][metric].values[0]
                for a in ARMS]
        ax.bar(x + (k - 1) * w, vals, w, label=DET_SHORT[d],
               color=[SHADE[a] for a in ARMS], edgecolor='black',
               linewidth=0.6, hatch=[HATCH[a] for a in ARMS])
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.008, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=5.6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=8)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.30))
    ax.grid(axis='y', color='0.88', linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

axes[0].legend(fontsize=8, frameon=False, loc='upper right', ncol=1)
fig.suptitle('Step 3 — Q3 control: does choosing the physically right products '
             'matter, or would any expansion do?', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = HERE / 'fig_step3_control.png'
fig.savefig(out, dpi=180)
print('saved:', out)
