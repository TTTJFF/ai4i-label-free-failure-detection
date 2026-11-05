# -*- coding: utf-8 -*-
"""Two extra multi-label models for Task B.

The course requires at least five models per task and results_taskB_multilabel.csv
holds four. This adds two more under the same protocol (same split, same seven
features, same five mode flags) and writes a separate CSV so the frozen results
file is left untouched.

    python scripts/add_taskB_models.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
SEED = 42
MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
SHORT = {'Air temperature [K]': 'Air temp [K]',
         'Process temperature [K]': 'Process temp [K]',
         'Rotational speed [rpm]': 'Rot. speed [rpm]'}

df = pd.read_csv(ROOT / 'data' / 'ai4i2020.csv')
dfn = df.rename(columns=SHORT)
y_all = dfn['Machine failure'].values
Y_all = dfn[MODES].values
X_all = dfn.drop(columns=['UDI', 'Product ID', 'Machine failure'] + MODES)
X_all = pd.get_dummies(X_all, columns=['Type'], drop_first=True)

X_tr, X_te, Y_tr, Y_te = train_test_split(
    X_all, Y_all, test_size=0.3, random_state=SEED, stratify=y_all)

models = {
    'LR (OvR)': MultiOutputClassifier(
        LogisticRegression(max_iter=2000, random_state=SEED)),
    'KNN (k=5)': MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
}

rows = []
for name, mdl in models.items():
    t0 = time.time()
    mdl.fit(X_tr, Y_tr)
    tt = time.time() - t0
    pred = mdl.predict(X_te)
    p, r, f, _ = precision_recall_fscore_support(Y_te, pred, average='micro',
                                                 zero_division=0)
    rows.append({
        'Model': name,
        'Micro-Precision': round(p, 6),
        'Micro-Recall': round(r, 6),
        'Micro-F1': round(f, 6),
        'Macro-F1': round(f1_score(Y_te, pred, average='macro', zero_division=0), 6),
        'ExactMatch': round(float((pred == Y_te).all(axis=1).mean()), 6),
        'HammingLoss': round(float((pred != Y_te).mean()), 6),
        'TrainTime(s)': round(tt, 3),
    })
    print('%-12s Micro-F1=%.4f Macro-F1=%.4f Exact=%.4f Hamming=%.4f'
          % (name, rows[-1]['Micro-F1'], rows[-1]['Macro-F1'],
             rows[-1]['ExactMatch'], rows[-1]['HammingLoss']))

out = pd.DataFrame(rows)
out.to_csv(RES / 'results_taskB_extra_models.csv', index=False)
print('written:', RES / 'results_taskB_extra_models.csv')
print('features used:', list(X_all.columns), '| train', X_tr.shape, '| test', X_te.shape)
