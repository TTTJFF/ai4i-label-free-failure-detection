# -*- coding: utf-8 -*-
"""Step 4 figure: (a) the label-free performance ladder,
(b) per-mode recall — where the boundary-violation frame moves detection to.

Grayscale only. Reads results_step4_scanner.csv and ../step2/results_step2.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
s4 = pd.read_csv(HERE / 'results_step4_scanner.csv')
s2 = pd.read_csv(HERE.parent / 'step2' / 'results_step2.csv')

nat = s4[s4.decision == 'train-quantile threshold'].set_index('method')
msv_full = nat.loc['MSV_full']
msv_sing = nat.loc['MSV_singletons_only']
best2 = s2.sort_values('f1', ascending=False).iloc[0]

# ---------------------------------------------------------------- panel (a)
ladder = [
    ('Frozen baseline\n(OCSVM, raw)', 0.2029, 'label-free'),
    ('Step 2\n(iForest + physics)', 0.2439, 'label-free'),
    ('Step 3\n(best arm, mean)', 0.2304, 'label-free'),
    ('Step 4\nMSV all subspaces', float(msv_full.f1), 'label-free'),
    ('Step 4\nMSV singletons only', float(msv_sing.f1), 'label-free'),
    ('Supervised MLP\n(baseline, uses labels)', 0.6923, 'supervised'),
    ('Oracle rules\n(true documented constants)', 0.9263, 'oracle'),
]
SHADE = {'label-free': '0.35', 'supervised': '0.72', 'oracle': '0.93'}
HATCH = {'label-free': '', 'supervised': 'xx', 'oracle': '..'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={'width_ratios': [1.15, 1]})

names = [n for n, _, _ in ladder]
vals = [v for _, v, _ in ladder]
kinds = [k for _, _, k in ladder]
y = np.arange(len(ladder))[::-1]
for yi, v, k in zip(y, vals, kinds):
    ax1.barh(yi, v, 0.62, color=SHADE[k], edgecolor='black',
             linewidth=0.6, hatch=HATCH[k])
    ax1.text(v + 0.012, yi, f'{v:.3f}', va='center', fontsize=9)
ax1.set_yticks(y)
ax1.set_yticklabels(names, fontsize=8.5)
ax1.set_xlim(0, 1.05)
ax1.set_xlabel('F1 on the same 3,000-sample test set', fontsize=9.5)
ax1.set_title('(a) Label-free performance ladder', fontsize=11)
ax1.grid(axis='x', color='0.88', linewidth=0.5)
ax1.set_axisbelow(True)
for sp in ('top', 'right'):
    ax1.spines[sp].set_visible(False)

# ---------------------------------------------------------------- panel (b)
MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
series = [('Step 2 best (iForest + physics)',
           [0.15, 0.12, 0.52, 0.37, 0.00], '0.80', ''),
          ('Step 4 MSV, all subspaces',
           [float(msv_full['recall_' + m]) for m in MODES], '0.45', '//'),
          ('Step 4 MSV, singletons only',
           [float(msv_sing['recall_' + m]) for m in MODES], '0.15', '')]
x = np.arange(len(MODES))
w = 0.26
for k, (lab, v, shade, h) in enumerate(series):
    ax2.bar(x + (k - 1) * w, v, w, label=lab, color=shade, edgecolor='black',
            linewidth=0.6, hatch=h)
    for xi, vi in zip(x + (k - 1) * w, v):
        ax2.text(xi, vi + 0.02, f'{vi:.2f}', ha='center', va='bottom',
                 fontsize=7.5)
ax2.set_xticks(x)
ax2.set_xticklabels(['TWF\n(13)', 'HDF\n(42)', 'PWF\n(23)', 'OSF\n(27)',
                     'RNF\n(6)'], fontsize=9)
ax2.set_ylim(0, 1.18)
ax2.set_ylabel('recall within the mode', fontsize=9.5)
ax2.set_title('(b) Where the detection moves to (test-set counts in brackets)',
              fontsize=11)
ax2.legend(fontsize=8, frameon=False, loc='upper left')
ax2.grid(axis='y', color='0.88', linewidth=0.5)
ax2.set_axisbelow(True)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)

fig.suptitle('Step 4 — scoring boundary violation instead of rarity',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / 'fig_step4_ladder_and_modes.png'
fig.savefig(out, dpi=180)
print('saved:', out)
