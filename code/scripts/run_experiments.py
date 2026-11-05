"""Full experimental pipeline for the course project.

Machine Failure Prediction and Failure-Mode Diagnosis in Manufacturing
Dataset: UCI AI4I 2020 Predictive Maintenance

Produces ~20 figures (figures/) and a metrics summary (results/summary.json).
Run:  D:\\Python\\python.exe scripts\\run_experiments.py
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, roc_curve,
                             average_precision_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'figures'
RES = ROOT / 'results'
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['figure.dpi'] = 130
plt.rcParams['savefig.bbox'] = 'tight'
PAL = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6a994e']

NUM = ['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
SHORT = {'Air temperature [K]': 'Air temp [K]',
         'Process temperature [K]': 'Process temp [K]',
         'Rotational speed [rpm]': 'Rot. speed [rpm]',
         'Torque [Nm]': 'Torque [Nm]',
         'Tool wear [min]': 'Tool wear [min]'}
MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# ============================ data ============================
df = pd.read_csv(ROOT / 'data' / 'ai4i2020.csv')
dfn = df.rename(columns=SHORT)
NUM_S = [SHORT[c] for c in NUM]

X_all = dfn.drop(columns=['UDI', 'Product ID', 'Machine failure'] + MODES)
y_all = dfn['Machine failure'].values
Y_all = dfn[MODES].values

idx_tr, idx_te = train_test_split(
    np.arange(len(dfn)), test_size=0.3, random_state=SEED, stratify=y_all)
df_tr, df_te = dfn.iloc[idx_tr], dfn.iloc[idx_te]
y_tr, y_te = y_all[idx_tr], y_all[idx_te]
Y_tr, Y_te = Y_all[idx_tr], Y_all[idx_te]

CAT_COLS = sorted(dfn['Type'].unique())[1:]  # drop_first


def build(df_in, scaler=None):
    num = df_in[NUM_S].values.astype(float)
    cat = pd.get_dummies(df_in['Type']).reindex(columns=CAT_COLS, fill_value=0).values
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(num)
    X = np.hstack([scaler.transform(num), cat])
    return X, scaler


X_tr_raw = df_tr[NUM_S].values.astype(float)
from sklearn.preprocessing import StandardScaler
SCALER = StandardScaler().fit(X_tr_raw)
X_tr, _ = build(df_tr, SCALER)
X_te, _ = build(df_te, SCALER)
FEAT_NAMES = NUM_S + ['Type=' + c for c in CAT_COLS]

summary = {
    'n_samples': int(len(dfn)),
    'n_features': len(FEAT_NAMES),
    'n_train': int(len(df_tr)),
    'n_test': int(len(df_te)),
    'fail_rate': float(y_all.mean()),
    'fail_train': int(y_tr.sum()),
    'fail_test': int(y_te.sum()),
    'mode_counts': {m: int(df[m].sum()) for m in MODES},
    'type_counts': df['Type'].value_counts().to_dict(),
}


def save(fig, name):
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print('  saved', name)


# ============================ EDA figures ============================
print('[EDA] generating figures ...')

# 01 class distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
vc = dfn['Machine failure'].value_counts().sort_index()
axes[0].bar(['Normal (0)', 'Failure (1)'], vc.values, color=[PAL[0], PAL[2]])
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 60, f'{v}\n({v / len(dfn) * 100:.2f}%)', ha='center', fontsize=10)
axes[0].set_ylabel('Count')
axes[0].set_title('(a) Machine failure label distribution')
axes[0].set_ylim(0, max(vc.values) * 1.25)

mc = df[MODES].sum().sort_values(ascending=False)
axes[1].bar(mc.index, mc.values, color=PAL[1])
for i, v in enumerate(mc.values):
    axes[1].text(i, v + 2, str(int(v)), ha='center', fontsize=10)
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Failure mode occurrences (can co-occur)')
axes[1].set_ylim(0, max(mc.values) * 1.25)
save(fig, 'fig01_class_distribution.png')

# 02 feature distributions, failure vs normal
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for i, c in enumerate(NUM_S):
    ax = axes.flat[i]
    ax.hist(dfn.loc[dfn['Machine failure'] == 0, c], bins=40, alpha=0.6,
            label='Normal', color=PAL[0], density=True)
    ax.hist(dfn.loc[dfn['Machine failure'] == 1, c], bins=40, alpha=0.7,
            label='Failure', color=PAL[2], density=True)
    ax.set_xlabel(c)
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
axes.flat[5].axis('off')
fig.suptitle('Feature distributions: normal vs failure samples', y=1.02)
save(fig, 'fig02_feature_distributions.png')

# 03 correlation heatmap
fig, ax = plt.subplots(figsize=(7.2, 5.8))
corr = dfn[NUM_S + ['Machine failure']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Pearson correlation of numeric features')
save(fig, 'fig03_correlation_heatmap.png')

# 04 boxplots by failure label
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for i, c in enumerate(NUM_S):
    sns.boxplot(data=dfn, x='Machine failure', y=c, ax=axes[i],
                palette=[PAL[0], PAL[2]], showfliers=False)
    axes[i].set_xlabel('Machine failure')
    axes[i].set_xticklabels(['Normal', 'Failure'])
fig.suptitle('Feature ranges grouped by failure label (outliers hidden)', y=1.04)
save(fig, 'fig04_boxplots_by_failure.png')

# 05 failure rate by machine type
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fr = dfn.groupby('Type')['Machine failure'].mean().sort_values(ascending=False)
axes[0].bar(fr.index, fr.values, color=PAL[3])
for i, v in enumerate(fr.values):
    axes[0].text(i, v + 0.001, f'{v * 100:.2f}%', ha='center')
axes[0].set_ylabel('Failure rate')
axes[0].set_title('(a) Failure rate by machine type')
tc = df['Type'].value_counts()
axes[1].bar(tc.index, tc.values, color=PAL[4])
for i, v in enumerate(tc.values):
    axes[1].text(i, v + 60, str(int(v)), ha='center')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Sample count by machine type')
save(fig, 'fig05_failure_by_type.png')

# 06 scatter: torque vs rotational speed
fig, ax = plt.subplots(figsize=(7, 5.6))
norm = dfn[dfn['Machine failure'] == 0]
fail = dfn[dfn['Machine failure'] == 1]
ax.scatter(norm['Rot. speed [rpm]'], norm['Torque [Nm]'], s=9, alpha=0.35,
           label='Normal', color=PAL[0])
ax.scatter(fail['Rot. speed [rpm]'], fail['Torque [Nm]'], s=34, alpha=0.9,
           label='Failure', color=PAL[2], marker='x')
ax.set_xlabel('Rotational speed [rpm]')
ax.set_ylabel('Torque [Nm]')
ax.set_title('Failure region in the speed-torque plane')
ax.legend()
save(fig, 'fig06_scatter_speed_torque.png')

print('[EDA] done.')


# ============================ helpers ============================
def eval_binary(y_true, y_pred, y_score=None):
    out = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        out['ROC-AUC'] = float(roc_auc_score(y_true, y_score))
        out['PR-AUC'] = float(average_precision_score(y_true, y_score))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        out['TP'], out['FP'], out['TN'], out['FN'] = int(tp), int(fp), int(tn), int(fn)
    return out


def smote(X, y, k=5, seed=SEED):
    """Simple SMOTE: synthetic minority oversampling."""
    rng = np.random.default_rng(seed)
    minority = 1 if (y == 1).sum() < (y == 0).sum() else 0
    Xm = X[y == minority]
    n_need = (y == 1 - minority).sum() - Xm.shape[0]
    if n_need <= 0:
        return X, y
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    _, idxs = nn.kneighbors(Xm)
    idxs = idxs[:, 1:]
    synth = []
    for i in range(n_need):
        base = rng.integers(0, len(Xm))
        neigh = Xm[rng.choice(idxs[base])]
        lam = rng.random()
        synth.append(Xm[base] + lam * (neigh - Xm[base]))
    X_new = np.vstack([X, np.array(synth)])
    y_new = np.hstack([y, np.full(n_need, minority)])
    return X_new, y_new


def random_oversample(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    minority = 1 if (y == 1).sum() < (y == 0).sum() else 0
    Xm, ym = X[y == minority], y[y == minority]
    Xmaj, ymaj = X[y == 1 - minority], y[y == 1 - minority]
    Xm_up, ym_up = resample(Xm, ym, replace=True,
                            n_samples=len(Xmaj), random_state=seed)
    idx = rng.permutation(len(Xmaj) * 2)
    X_new = np.vstack([Xmaj, Xm_up])[idx]
    y_new = np.hstack([ymaj, ym_up])[idx]
    return X_new, y_new


# ============================ Task A ============================
print('[Task A] binary classification ...')
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=SEED, class_weight='balanced',
                                            max_depth=8),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=SEED,
                                            class_weight='balanced_subsample', n_jobs=-1),
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                     class_weight='balanced', random_state=SEED),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
    'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced',
                                              random_state=SEED),
}

rows, scores, probs = [], {}, {}
for name, mdl in models.items():
    t0 = time.time()
    mdl.fit(X_tr, y_tr)
    tt = time.time() - t0
    pred = mdl.predict(X_te)
    prob = mdl.predict_proba(X_te)[:, 1]
    scores[name], probs[name] = prob, prob
    m = eval_binary(y_te, pred, prob)
    m.update({'Model': name, 'TrainTime(s)': round(tt, 3)})
    rows.append(m)
    print(f'  {name:22s} F1={m["F1"]:.4f} AUC={m["ROC-AUC"]:.4f} ({tt:.2f}s)')

resA = pd.DataFrame(rows)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1',
                           'ROC-AUC', 'PR-AUC', 'TrainTime(s)', 'TP', 'FP', 'FN', 'TN']]
resA.to_csv(RES / 'results_taskA_binary.csv', index=False)
summary['taskA'] = resA.to_dict(orient='records')

# fig07 model comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
melt = resA.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1', 'ROC-AUC'],
                 var_name='Metric', value_name='Score')
sns.barplot(data=melt, x='Model', y='Score', hue='Metric', ax=axes[0], palette=PAL)
axes[0].set_ylim(0, 1.05)
axes[0].tick_params(axis='x', rotation=18)
axes[0].set_title('(a) Task A: metric comparison')
axes[0].legend(loc='lower right', fontsize=9)

axes[1].bar(resA['Model'], resA['Accuracy'], color=PAL[0])
axes[1].axhline(y=1 - y_te.mean(), color=PAL[3], ls='--',
                label=f'Majority baseline = {1 - y_te.mean():.4f}')
axes[1].set_ylim(0.9, 1.0)
axes[1].tick_params(axis='x', rotation=18)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('(b) Accuracy is misleading (all models > 93%)')
axes[1].legend(fontsize=9)
save(fig, 'fig07_taskA_model_comparison.png')

# fig08 ROC curves
fig, ax = plt.subplots(figsize=(6.6, 5.6))
for i, (name, prob) in enumerate(probs.items()):
    fpr, tpr, _ = roc_curve(y_te, prob)
    ax.plot(fpr, tpr, color=PAL[i % len(PAL)], lw=2,
            label=f'{name} (AUC={roc_auc_score(y_te, prob):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Task A: ROC curves')
ax.legend(fontsize=9, loc='lower right')
save(fig, 'fig08_taskA_roc.png')

# fig09 PR curves
fig, ax = plt.subplots(figsize=(6.6, 5.6))
for i, (name, prob) in enumerate(probs.items()):
    p, r, _ = precision_recall_curve(y_te, prob)
    ax.plot(r, p, color=PAL[i % len(PAL)], lw=2,
            label=f'{name} (AP={average_precision_score(y_te, prob):.3f})')
ax.axhline(y=y_te.mean(), color='k', ls='--', lw=1,
           label=f'Random (={y_te.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Task A: Precision-Recall curves (informative under imbalance)')
ax.legend(fontsize=9, loc='upper right')
save(fig, 'fig09_taskA_pr.png')

# fig10 confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for i, name in enumerate(models):
    cm = confusion_matrix(y_te, models[name].predict(X_te))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes.flat[i],
                xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
    axes.flat[i].set_title(f'{name}\nF1={resA.loc[resA.Model == name, "F1"].values[0]:.3f}')
    axes.flat[i].set_xlabel('Predicted')
    axes.flat[i].set_ylabel('True')
axes.flat[5].axis('off')
fig.suptitle('Task A: confusion matrices on the test set (n=3000)', y=1.02)
save(fig, 'fig10_taskA_confusion_matrices.png')

# ---- threshold & cost analysis (best model by F1) ----
best_name = resA.sort_values('F1', ascending=False).iloc[0]['Model']
best_prob = probs[best_name]
thr = np.linspace(0.02, 0.98, 97)
prec, rec, f1s = [], [], []
for t in thr:
    pred = (best_prob >= t).astype(int)
    prec.append(precision_score(y_te, pred, zero_division=0))
    rec.append(recall_score(y_te, pred, zero_division=0))
    f1s.append(f1_score(y_te, pred, zero_division=0))
C_FN, C_FP = 1000.0, 100.0   # cost of a missed failure vs a false inspection
costs = []
for t in thr:
    pred = (best_prob >= t).astype(int)
    fn = int(((pred == 0) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    costs.append(fn * C_FN + fp * C_FP)
best_f1_t = float(thr[int(np.argmax(f1s))])
best_cost_t = float(thr[int(np.argmin(costs))])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(thr, prec, label='Precision', color=PAL[0])
axes[0].plot(thr, rec, label='Recall', color=PAL[1])
axes[0].plot(thr, f1s, label='F1', color=PAL[2], lw=2.5)
axes[0].axvline(0.5, color='k', ls=':', label='default = 0.50')
axes[0].axvline(best_f1_t, color=PAL[3], ls='--', label=f'best F1 @ {best_f1_t:.2f}')
axes[0].set_xlabel('Decision threshold')
axes[0].set_ylabel('Score')
axes[0].set_title(f'(a) Threshold sweep - {best_name}')
axes[0].legend(fontsize=9)

axes[1].plot(thr, np.array(costs) / 1000, color=PAL[3], lw=2)
axes[1].axvline(best_cost_t, color=PAL[0], ls='--',
                label=f'min cost @ {best_cost_t:.2f}')
axes[1].axvline(0.5, color='k', ls=':', label='default = 0.50')
axes[1].set_xlabel('Decision threshold')
axes[1].set_ylabel('Total cost (x1000 units)')
axes[1].set_title(f'(b) Cost curve (C_FN={C_FN:.0f}, C_FP={C_FP:.0f})')
axes[1].legend(fontsize=9)
save(fig, 'fig11_threshold_and_cost.png')

summary['threshold'] = {
    'model': best_name,
    'best_f1_threshold': best_f1_t,
    'best_f1': float(np.max(f1s)),
    'f1_at_0.5': float(f1_score(y_te, (best_prob >= 0.5).astype(int))),
    'precision_at_best': float(prec[int(np.argmax(f1s))]),
    'recall_at_best': float(rec[int(np.argmax(f1s))]),
    'min_cost_threshold': best_cost_t,
    'cost_at_default': float(costs[int(np.argmin(np.abs(thr - 0.5)))]),
    'cost_at_optimal': float(np.min(costs)),
    'C_FN': C_FN, 'C_FP': C_FP,
}

# ---- resampling strategies ----
print('[Task A] resampling comparison ...')
X_sm, y_sm = smote(X_tr, y_tr)
X_ro, y_ro = random_oversample(X_tr, y_tr)
strategies = {
    'None (raw)': (X_tr, y_tr),
    'class_weight': (X_tr, y_tr),
    'Random oversampling': (X_ro, y_ro),
    'SMOTE': (X_sm, y_sm),
}
rows_r = []
for sname, (Xs, ys) in strategies.items():
    for mname in ['Decision Tree', 'Random Forest', 'MLP']:
        if mname == 'Decision Tree':
            m = DecisionTreeClassifier(random_state=SEED, max_depth=8,
                                       class_weight=None if sname != 'class_weight' else 'balanced')
        elif mname == 'Random Forest':
            m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1,
                                       class_weight=None if sname != 'class_weight' else 'balanced_subsample')
        else:
            m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED)
        m.fit(Xs, ys)
        pred = m.predict(X_te)
        prob = m.predict_proba(X_te)[:, 1]
        mm = eval_binary(y_te, pred, prob)
        rows_r.append({'Strategy': sname, 'Model': mname, 'Precision': mm['Precision'],
                       'Recall': mm['Recall'], 'F1': mm['F1'], 'ROC-AUC': mm['ROC-AUC'],
                       'PR-AUC': mm['PR-AUC']})
        print(f'  {sname:22s} {mname:14s} F1={mm["F1"]:.4f}')

resR = pd.DataFrame(rows_r)
resR.to_csv(RES / 'results_taskA_resampling.csv', index=False)
summary['resampling'] = resR.to_dict(orient='records')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for i, metric in enumerate(['F1', 'ROC-AUC', 'PR-AUC']):
    sns.barplot(data=resR, x='Model', y=metric, hue='Strategy', ax=axes[i], palette=PAL)
    axes[i].tick_params(axis='x', rotation=18)
    axes[i].set_ylim(0, 1.05)
    axes[i].set_title(f'({chr(97 + i)}) {metric}')
    axes[i].legend(fontsize=8)
fig.suptitle('Effect of imbalance-handling strategies (Task A)', y=1.04)
save(fig, 'fig12_resampling_comparison.png')

# ---- permutation importance ----
print('[Task A] permutation importance ...')
rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                            class_weight='balanced_subsample', n_jobs=-1).fit(X_tr, y_tr)
pi = permutation_importance(rf, X_te, y_te, n_repeats=15, random_state=SEED,
                            scoring='f1', n_jobs=-1)
imp = pd.DataFrame({'Feature': FEAT_NAMES,
                    'Permutation': pi.importances_mean,
                    'Std': pi.importances_std,
                    'Gini': rf.feature_importances_}).sort_values('Permutation', ascending=False)
imp.to_csv(RES / 'results_feature_importance.csv', index=False)
summary['importance'] = imp.to_dict(orient='records')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
sns.barplot(data=imp, x='Permutation', y='Feature', ax=axes[0], color=PAL[0],
            xerr=imp['Std'], error_kw={'lw': 1})
axes[0].set_title('(a) Permutation importance (F1 drop)')
sns.barplot(data=imp, x='Gini', y='Feature', ax=axes[1], color=PAL[1])
axes[1].set_title('(b) Random Forest Gini importance')
save(fig, 'fig13_feature_importance.png')

# ---- learning curve ----
print('[Task A] learning curve ...')
sizes, tr_sc, va_sc = learning_curve(
    RandomForestClassifier(n_estimators=200, random_state=SEED,
                           class_weight='balanced_subsample', n_jobs=-1),
    X_tr, y_tr, train_sizes=np.linspace(0.1, 1.0, 6), cv=3,
    scoring='f1', n_jobs=-1)
fig, ax = plt.subplots(figsize=(6.8, 5))
ax.plot(sizes, tr_sc.mean(1), 'o-', color=PAL[0], label='Training F1')
ax.fill_between(sizes, tr_sc.mean(1) - tr_sc.std(1), tr_sc.mean(1) + tr_sc.std(1),
                alpha=0.15, color=PAL[0])
ax.plot(sizes, va_sc.mean(1), 'o-', color=PAL[2], label='CV F1')
ax.fill_between(sizes, va_sc.mean(1) - va_sc.std(1), va_sc.mean(1) + va_sc.std(1),
                alpha=0.15, color=PAL[2])
ax.set_xlabel('Training set size')
ax.set_ylabel('F1 (failure class)')
ax.set_title('Learning curve - Random Forest (Task A)')
ax.legend()
save(fig, 'fig14_learning_curve.png')
summary['learning_curve'] = {
    'sizes': sizes.tolist(),
    'train_f1': tr_sc.mean(1).tolist(),
    'cv_f1': va_sc.mean(1).tolist(),
}

# ---- PCA / t-SNE ----
print('[Task A] PCA / t-SNE ...')
sub = np.random.default_rng(SEED).choice(len(X_te), 3000, replace=False)
Xs, ys = X_te[sub], y_te[sub]
pca = PCA(n_components=2, random_state=SEED).fit(X_tr)
P = pca.transform(Xs)
ts = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca').fit_transform(Xs)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, Z, ttl in [(axes[0], P, f'PCA (explained var = {pca.explained_variance_ratio_.sum() * 100:.1f}%)'),
                   (axes[1], ts, 't-SNE (perplexity = 30)')]:
    ax.scatter(Z[ys == 0, 0], Z[ys == 0, 1], s=8, alpha=0.35, color=PAL[0], label='Normal')
    ax.scatter(Z[ys == 1, 0], Z[ys == 1, 1], s=28, color=PAL[2], marker='x', label='Failure')
    ax.set_title(ttl)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend()
fig.suptitle('2-D projections of the test set: are failures separable?', y=1.03)
save(fig, 'fig15_pca_tsne.png')

print('[Task A] done.')

# ============================ Task B ============================
print('[Task B] multi-label failure modes ...')
multi = {
    'Decision Tree': MultiOutputClassifier(DecisionTreeClassifier(random_state=SEED,
                                                                  class_weight='balanced')),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    'SVM (RBF)': MultiOutputClassifier(SVC(kernel='rbf', C=10, probability=True,
                                           class_weight='balanced', random_state=SEED)),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=SEED),
}
rows_b, per_mode = [], {}
for name, mdl in multi.items():
    t0 = time.time()
    mdl.fit(X_tr, Y_tr)
    tt = time.time() - t0
    pred = mdl.predict(X_te)
    p, r, f, _ = precision_recall_fscore_support(Y_te, pred, average='micro', zero_division=0)
    macro_f = f1_score(Y_te, pred, average='macro', zero_division=0)
    exact = float((pred == Y_te).all(axis=1).mean())
    ham = float((pred != Y_te).mean())
    per_mode[name] = f1_score(Y_te, pred, average=None, zero_division=0).tolist()
    rows_b.append({'Model': name, 'Micro-Precision': p, 'Micro-Recall': r,
                   'Micro-F1': f, 'Macro-F1': macro_f, 'ExactMatch': exact,
                   'HammingLoss': ham, 'TrainTime(s)': round(tt, 3)})
    print(f'  {name:22s} Micro-F1={f:.4f} Macro-F1={macro_f:.4f} ({tt:.2f}s)')

resB = pd.DataFrame(rows_b)
resB.to_csv(RES / 'results_taskB_multilabel.csv', index=False)
summary['taskB'] = resB.to_dict(orient='records')
summary['taskB_per_mode'] = per_mode
summary['taskB_mode_support'] = Y_te.sum(axis=0).tolist()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
melt_b = resB.melt(id_vars='Model', value_vars=['Micro-Precision', 'Micro-Recall',
                                                'Micro-F1', 'Macro-F1'],
                   var_name='Metric', value_name='Score')
sns.barplot(data=melt_b, x='Model', y='Score', hue='Metric', ax=axes[0], palette=PAL)
axes[0].set_ylim(0, 1.05)
axes[0].tick_params(axis='x', rotation=18)
axes[0].set_title('(a) Task B: overall metrics')
axes[0].legend(fontsize=9, loc='lower right')

pm = pd.DataFrame(per_mode, index=MODES).T
pm.plot(kind='bar', ax=axes[1], color=PAL[:5], width=0.8)
axes[1].set_ylabel('F1 per mode')
axes[1].set_ylim(0, 1.05)
axes[1].tick_params(axis='x', rotation=18)
axes[1].set_title('(b) Per-mode F1 (support: ' +
                  ', '.join(f'{m}={int(s)}' for m, s in zip(MODES, Y_te.sum(axis=0))) + ')')
axes[1].legend(fontsize=9)
save(fig, 'fig16_taskB_multilabel.png')
print('[Task B] done.')

# ============================ Task C ============================
print('[Task C] unsupervised anomaly detection ...')
cont = float(y_tr.mean())
anom = {
    'Isolation Forest': IsolationForest(n_estimators=300, contamination=cont,
                                        random_state=SEED),
    'One-Class SVM': OneClassSVM(kernel='rbf', nu=cont, gamma='auto'),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=cont,
                                               novelty=True),
}
rows_c, scores_c = [], {}
for name, mdl in anom.items():
    t0 = time.time()
    mdl.fit(X_tr)
    tt = time.time() - t0
    if name == 'Local Outlier Factor':
        pred = (mdl.predict(X_te) == -1).astype(int)
        sc = -mdl.score_samples(X_te)
    else:
        pred = (mdl.predict(X_te) == -1).astype(int)
        sc = (-mdl.score_samples(X_te) if hasattr(mdl, 'score_samples')
              else -mdl.decision_function(X_te))
    scores_c[name] = sc
    m = eval_binary(y_te, pred, sc)
    m.update({'Model': name, 'TrainTime(s)': round(tt, 3)})
    rows_c.append(m)
    print(f'  {name:22s} F1={m["F1"]:.4f} AUC={m["ROC-AUC"]:.4f} ({tt:.2f}s)')

resC = pd.DataFrame(rows_c)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1',
                             'ROC-AUC', 'PR-AUC', 'TrainTime(s)', 'TP', 'FP', 'FN', 'TN']]
resC.to_csv(RES / 'results_taskC_anomaly.csv', index=False)
summary['taskC'] = resC.to_dict(orient='records')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
melt_c = resC.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1', 'ROC-AUC'],
                   var_name='Metric', value_name='Score')
sns.barplot(data=melt_c, x='Model', y='Score', hue='Metric', ax=axes[0], palette=PAL)
axes[0].set_ylim(0, 1.05)
axes[0].tick_params(axis='x', rotation=15)
axes[0].set_title('(a) Task C: metric comparison')
axes[0].legend(fontsize=9, loc='upper right')

for i, (name, sc) in enumerate(scores_c.items()):
    axes[1].hist(sc[y_te == 0], bins=60, alpha=0.5, density=True,
                 color=PAL[i], label=f'{name} - normal')
    axes[1].hist(sc[y_te == 1], bins=60, alpha=0.85, density=True, histtype='step',
                 color=PAL[i], lw=1.8, label=f'{name} - failure')
axes[1].set_yscale('log')
axes[1].set_xlabel('Anomaly score')
axes[1].set_ylabel('Density (log)')
axes[1].set_title('(b) Score distributions (filled=normal, line=failure)')
axes[1].legend(fontsize=7, ncol=2)
save(fig, 'fig17_taskC_anomaly.png')

# supervised vs unsupervised headline comparison
best_sup = resA.sort_values('F1', ascending=False).iloc[0]
best_unsup = resC.sort_values('F1', ascending=False).iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
comp = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'],
    'Supervised (best)': [best_sup['Precision'], best_sup['Recall'], best_sup['F1'],
                          best_sup['ROC-AUC'], best_sup['PR-AUC']],
    'Unsupervised (best)': [best_unsup['Precision'], best_unsup['Recall'], best_unsup['F1'],
                            best_unsup['ROC-AUC'], best_unsup['PR-AUC']],
})
comp_m = comp.melt(id_vars='Metric', var_name='Setting', value_name='Score')
sns.barplot(data=comp_m, x='Metric', y='Score', hue='Setting', ax=axes[0],
            palette=[PAL[0], PAL[2]])
axes[0].set_ylim(0, 1.05)
axes[0].set_title(f'(a) {best_sup["Model"]} vs {best_unsup["Model"]}')
axes[0].legend(fontsize=9, loc='upper right')

for i, (name, prob) in enumerate(probs.items()):
    fpr, tpr, _ = roc_curve(y_te, prob)
    axes[1].plot(fpr, tpr, color=PAL[i % len(PAL)], lw=1.8, ls='-',
                 label=f'{name} (sup, {roc_auc_score(y_te, prob):.3f})')
for i, (name, sc) in enumerate(scores_c.items()):
    fpr, tpr, _ = roc_curve(y_te, sc)
    axes[1].plot(fpr, tpr, color=PAL[i % len(PAL)], lw=1.8, ls='--',
                 label=f'{name} (unsup, {roc_auc_score(y_te, sc):.3f})')
axes[1].plot([0, 1], [0, 1], 'k:', lw=1)
axes[1].set_xlabel('False positive rate')
axes[1].set_ylabel('True positive rate')
axes[1].set_title('(b) ROC: solid=supervised, dashed=unsupervised')
axes[1].legend(fontsize=8, loc='lower right')
save(fig, 'fig18_supervised_vs_unsupervised.png')

summary['headline'] = {
    'best_supervised_model': str(best_sup['Model']),
    'best_supervised_f1': float(best_sup['F1']),
    'best_supervised_auc': float(best_sup['ROC-AUC']),
    'best_unsupervised_model': str(best_unsup['Model']),
    'best_unsupervised_f1': float(best_unsup['F1']),
    'best_unsupervised_auc': float(best_unsup['ROC-AUC']),
}
print('[Task C] done.')

# ============================ efficiency ============================
fig, ax = plt.subplots(figsize=(7, 5.2))
allm = pd.concat([
    resA[['Model', 'F1', 'TrainTime(s)']].assign(Kind='Supervised (Task A)'),
    resC[['Model', 'F1', 'TrainTime(s)']].assign(Kind='Unsupervised (Task C)'),
])
for kind, marker in [('Supervised (Task A)', 'o'), ('Unsupervised (Task C)', 's')]:
    d = allm[allm.Kind == kind]
    ax.scatter(d['TrainTime(s)'], d['F1'], s=90, marker=marker, label=kind)
    for _, r in d.iterrows():
        ax.annotate(r['Model'], (r['TrainTime(s)'], r['F1']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
ax.set_xscale('log')
ax.set_xlabel('Training time (s, log scale)')
ax.set_ylabel('F1 (failure class)')
ax.set_title('Accuracy-efficiency trade-off')
ax.legend()
save(fig, 'fig19_efficiency_tradeoff.png')

# ============================ save summary ============================
with open(RES / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print('\nALL DONE. Figures ->', FIG, '| Results ->', RES)
