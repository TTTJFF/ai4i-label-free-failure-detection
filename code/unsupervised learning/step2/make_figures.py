# -*- coding: utf-8 -*-
"""Step 2 figure: feature set vs detector, three metrics.

Grayscale only (the project's reports are black-and-white).
Reads results_step2.csv produced by run_step2.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / 'results_step2.csv')

SETS = ['F1_raw', 'F2_physics', 'F3_physics_pt']
LABELS = {'F1_raw': 'F1 raw sensors', 'F2_physics': '+ physics',
          'F3_physics_pt': '+ physics + per-type z'}
SHADE = {'F1_raw': '0.85', 'F2_physics': '0.45', 'F3_physics_pt': '0.15'}
HATCH = {'F1_raw': '', 'F2_physics': '', 'F3_physics_pt': '//'}
DETS = sorted(df.detector.unique())

METRICS = [('f1', 'F1 at fixed contamination (3.39%)', None),
           ('roc_auc', 'ROC-AUC (threshold-free)', None),
           ('pr_auc', 'PR-AUC (threshold-free)', None)]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
x = np.arange(len(DETS))
w = 0.26

for ax, (metric, title, _) in zip(axes, METRICS):
    for k, s in enumerate(SETS):
        vals = [df[(df.feature_set == s) & (df.detector == d)][metric].values[0]
                for d in DETS]
        ax.bar(x + (k - 1) * w, vals, w, label=LABELS[s],
               color=SHADE[s], edgecolor='black', linewidth=0.6, hatch=HATCH[s])
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.012, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Isolation\nForest', 'One-Class\nSVM', 'LOF'],
                       fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.15))
    ax.grid(axis='y', color='0.85', linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

axes[0].legend(fontsize=8, frameon=False, loc='upper left')
fig.suptitle('Step 2 — controlled ablation: only the feature set changes',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / 'fig_step2_feature_sets.png'
fig.savefig(out, dpi=180)
print('saved:', out)
