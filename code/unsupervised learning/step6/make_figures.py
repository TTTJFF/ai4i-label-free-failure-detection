# -*- coding: utf-8 -*-
"""Step 6 figures.

(a) expected maintenance cost vs alarm threshold, with the operating points
    actually used and the supervised baseline's own cost-optimal point
(b) what each reason code contributes: true detections vs false alarms

Grayscale only.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step1'))
sys.path.insert(0, str(HERE))
import features as F        # noqa: E402
import pipeline as P        # noqa: E402

df = F.load_data()
y_all = df[F.LABEL].to_numpy(int)
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=42,
                                 stratify=y_all)
pipe = P.MaintenancePipeline().fit(df.iloc[idx_tr])
out = pipe.predict(df.iloc[idx_te])
score = out['score'].to_numpy()
y_te = y_all[idx_te]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8),
                               gridspec_kw={'width_ratios': [1.15, 1]})

# ---------------------------------------------------------------- panel (a)
ths = np.unique(np.quantile(score, np.linspace(0.0, 0.9995, 500)))
cost = []
for t in ths:
    pred = (score >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    cost.append(P.C_FN * fn + P.C_FP * fp)
cost = np.array(cost)

ax1.plot(ths, cost, color='0.25', linewidth=1.6)
def cost_at(t):
    pred = (score >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return P.C_FN * fn + P.C_FP * fp

c_opt = cost_at(pipe.threshold)
ax1.plot([pipe.threshold], [c_opt], marker='o', markersize=9, color='black',
         zorder=5)
ax1.annotate(f'cost-optimal threshold used\n{c_opt:,.0f}', xy=(pipe.threshold, c_opt),
             xytext=(pipe.threshold + 3.2, c_opt - 26000), fontsize=8.5,
             arrowprops=dict(arrowstyle='-', linewidth=0.6, color='0.4'))
ax1.axhline(102000, color='0.6', linewidth=0.9, linestyle='--')
ax1.text(0.4, 105000, 'never alarm: 102,000', fontsize=8.5, color='0.4')
ax1.axhline(24800, color='0.6', linewidth=0.9, linestyle=':')
ax1.text(0.4, 21000, "supervised MLP optimum: 24,800 (same cost model)",
         fontsize=8, color='0.4')
ax1.set_xlabel('alarm threshold on the violation score', fontsize=9.5)
ax1.set_ylabel('expected cost on the test set  (C_FN=1000, C_FP=100)',
               fontsize=9.5)
ax1.set_title('(a) The cost curve and the operating points used', fontsize=11)
ax1.set_ylim(0, 125000)
ax1.set_xlim(-0.5, 30)
ax1.grid(color='0.9', linewidth=0.5)
ax1.set_axisbelow(True)
for sp in ('top', 'right'):
    ax1.spines[sp].set_visible(False)

# ---------------------------------------------------------------- panel (b)
c = pd.read_csv(HERE / 'results_step6_contrib.csv')
c = c[c.reason.isin(['PWF', 'OSF', 'HDF'])]
x = np.arange(len(c))
w = 0.36
ax2.bar(x - w / 2, c.true_detections, w, label='true detections',
        color='0.30', edgecolor='black', linewidth=0.6)
ax2.bar(x + w / 2, c.false_alarms, w, label='false alarms', color='0.85',
        edgecolor='black', linewidth=0.6, hatch='//')
for xi, (a, b) in enumerate(zip(c.true_detections, c.false_alarms)):
    ax2.text(xi - w / 2, a + 0.5, str(a), ha='center', fontsize=8.5)
    ax2.text(xi + w / 2, b + 0.5, str(b), ha='center', fontsize=8.5)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{r}\n(subspace)' for r in c.reason], fontsize=9)
ax2.set_ylabel('alarms at the cost-optimal point', fontsize=9.5)
ax2.set_title('(b) Where the alarms come from', fontsize=11)
ax2.legend(fontsize=8.5, frameon=False)
ax2.grid(axis='y', color='0.9', linewidth=0.5)
ax2.set_axisbelow(True)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)

fig.suptitle('Step 6 — the end-to-end decision system', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out_png = HERE / 'fig_step6_cost_and_reason.png'
fig.savefig(out_png, dpi=180)
print('saved:', out_png)
