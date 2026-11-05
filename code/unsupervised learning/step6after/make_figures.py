# -*- coding: utf-8 -*-
"""Figures for step 6-after. English labels only (matplotlib has no CJK font
guaranteed on this machine, and the report figures are English throughout).

    fig_a1_multiseed.png  F1 across 10 seeds, per configuration
    fig_a2_triage.png     expected cost vs watch-list size, by monitor price
    fig_a3_a5.png         fusion variants  |  label-free rate estimation
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                     'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linestyle': '-', 'axes.axisbelow': True,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})

C = dict(base='#888780', blue='#378ADD', blue2='#85B7EB', blue3='#185FA5',
         teal='#1D9E75', coral='#D85A30', amber='#BA7517', gray='#D3D1C7')

ORDER = [('F2_physics', 'MSV_no_hdf', 'MSV\nno-HDF'),
         ('F2_physics', 'MSV_singletons', 'MSV\nsingletons'),
         ('F2_physics', 'MSV_full', 'MSV\nfull'),
         ('F2_physics', 'Local Outlier Factor', 'LOF\n@F2'),
         ('F2_physics', 'One-Class SVM', 'OCSVM\n@F2'),
         ('F2_physics', 'Isolation Forest', 'iForest\n@F2'),
         ('F1_raw', 'Isolation Forest', 'iForest\n@F1 raw')]
COLORS = [C['teal'], C['blue'], C['blue2'], C['base'], C['base'],
          C['base'], C['gray']]


def load(name):
    return pd.read_csv(os.path.join(HERE, name))


# ---------------------------------------------------------------- figure 1
def fig1():
    d = load('results_a1_multiseed.csv')
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    data, labels = [], []
    for fs, det, lab in ORDER:
        v = d[(d.feature_set == fs) & (d.detector == det)]
        v = v.sort_values('seed')['f1'].to_numpy()
        data.append(v)
        labels.append(f'{lab}\nmean {v.mean():.3f}')
    bp = ax.boxplot(data, patch_artist=True, widths=0.52,
                    medianprops=dict(color='#2C2C2A', linewidth=1.2),
                    whiskerprops=dict(color='#5F5E5A', linewidth=0.7),
                    capprops=dict(color='#5F5E5A', linewidth=0.7),
                    flierprops=dict(marker='', linestyle='none'))
    for patch, col in zip(bp['boxes'], COLORS):
        patch.set(facecolor=col, alpha=0.30, edgecolor=col, linewidth=0.9)
    rng = np.random.RandomState(0)
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i + 1) + rng.uniform(-0.13, 0.13, len(v)),
                   v, s=9, color=COLORS[i], zorder=5, linewidths=0)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('F1 on the test split')
    ax.set_title('Step 6-after / A1 — stability across 10 random splits '
                 '(70/30, stratified)', fontsize=10.5)
    ax.set_ylim(0, 0.76)
    ax.axhline(0.6923, color=C['coral'], linewidth=0.9, linestyle='--')
    ax.text(len(labels), 0.706, 'supervised MLP  0.692', color=C['coral'],
            fontsize=8.5, va='bottom', ha='center')
    ax.set_xlim(0.4, len(labels) + 0.6)
    fig.savefig(os.path.join(HERE, 'fig_a1_multiseed.png'))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2():
    wl = load('results_a2_watchlist.csv')
    be = load('results_a2_breakeven.csv')
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    style = {10.0: (C['teal'], 'o', '-'), 20.0: (C['blue'], 's', '-'),
             50.0: (C['amber'], '^', '-'), 100.0: (C['base'], 'D', '--')}
    for cm, (col, mk, ls) in style.items():
        s = wl[wl.c_monitor == cm].sort_values('quantile')
        ax.plot(s['quantile'] * 100, s['cost'] / 1000, marker=mk, color=col,
                linestyle=ls, markersize=5, linewidth=1.4,
                label=f'monitor price = {int(cm)}')
    for y, lab, col in [(67.5, 'published step 6 pipeline  67.5k', C['coral']),
                        (58.1, 'oracle 2-tier (test-tuned)  58.1k', '#993C1D'),
                        (102.0, 'never alarm  102.0k', '#5F5E5A')]:
        ax.axhline(y, color=col, linewidth=0.9, linestyle='--')
        ax.text(0.4, y + 1.3, lab, color=col, fontsize=8.5, va='bottom')
    s20 = wl[wl.c_monitor == 20.0]
    j = s20['cost'].idxmin()
    ax.annotate(f"best at a 20% watch-list:\n"
                f"{s20.loc[j,'cost']/1000:.1f}k, recall {s20.loc[j,'recall']:.2f}",
                xy=(20, s20.loc[j, 'cost'] / 1000), xytext=(9.6, 84),
                fontsize=8.5, color='#2C2C2A',
                arrowprops=dict(arrowstyle='->', color='#5F5E5A', linewidth=0.7))
    ax.text(0.4, 27.5,
            f"break-even triage quality: a 20% watch-list already wins once "
            f"alpha >= {be[be.c_monitor==20.0].alpha_breakeven_vs_67500.min():.2f}",
            fontsize=8.5, color='#2C2C2A')
    ax.set_xlabel('watch-list size (% of points flagged for monitoring)')
    ax.set_ylabel('expected maintenance cost  (thousands)')
    ax.set_title('Step 6-after / A2 — one threshold, priced as triage instead '
                 'of as an inspection', fontsize=10.5)
    ax.set_xlim(0, 21)
    ax.set_ylim(25, 108)
    ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=False)
    fig.savefig(os.path.join(HERE, 'fig_a2_triage.png'))
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig3():
    fus = load('results_a3_fusion.csv')
    rate = load('results_a5_rate_estimates.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    fus = fus.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(fus))
    h = 0.38
    ax.barh(y + h / 2, fus['f1'], height=h, color=C['blue'], label='F1')
    ax.barh(y - h / 2, fus['pr_auc'], height=h, color=C['teal'],
            label='PR-AUC')
    for i, r in fus.iterrows():
        ax.text(r['f1'] + 0.008, i + h / 2, f"{r['f1']:.3f}", va='center',
                fontsize=8, color='#2C2C2A')
        ax.text(r['pr_auc'] + 0.008, i - h / 2, f"{r['pr_auc']:.3f}",
                va='center', fontsize=8, color='#2C2C2A')
    ax.set_yticks(y)
    ax.set_yticklabels(fus['variant'], fontsize=8.5)
    ax.set_xlim(0, 0.60)
    ax.set_xlabel('metric at the fixed 3.39% operating point')
    ax.set_title('A3 — subspace fusion: dropping the HDF arm helps',
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False)
    ax.grid(axis='y', visible=False)

    ax = axes[1]
    r = rate.copy()
    r['lab'] = r['method'].str.replace('[reference] ', '', regex=False)
    r['mult'] = r['ratio_to_truth']
    cols = [C['gray'] if np.isfinite(m) and m > 2 else
            (C['amber'] if np.isfinite(m) and m > 1.3 else C['teal'])
            for m in r['mult']]
    cols[-1] = C['base']
    ax.bar(range(len(r)), r['mult'], color=cols, width=0.6)
    for i, row in r.iterrows():
        if np.isfinite(row['f1']):
            ax.text(i, row['mult'] + 0.08, f"F1 {row['f1']:.3f}", ha='center',
                    fontsize=8, color='#2C2C2A')
        else:
            ax.text(i, 0.06, 'no alarm', ha='center', fontsize=8,
                    color='#2C2C2A')
    ax.axhline(1.0, color=C['base'], linewidth=1.0, linestyle='--')
    ax.text(-0.42, 1.10, 'true rate = 3.39%', fontsize=8.5,
            color='#5F5E5A', ha='left')
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels([l.replace(' plateau', '\nplateau')
                        .replace('largest gap', 'largest\ngap')
                        for l in r['lab']], fontsize=8.5)
    ax.set_ylabel('estimated rate / true rate')
    ax.set_title('A5 — label-free rate estimation fails here', fontsize=10.5)
    ax.set_ylim(0, 4.7)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_a3_a5.png'))
    plt.close(fig)


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    print('figures written to', HERE)
