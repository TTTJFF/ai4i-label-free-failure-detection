# Support-Boundary Violation Scoring for Label-Free Machine-Failure Detection

*A controlled ablation on the AI4I 2020 benchmark*

On the AI4I 2020 predictive-maintenance dataset the strongest supervised model reaches F1 **0.692** while the best of three standard unsupervised detectors reaches only **0.203**. This gap is usually read as a limitation of unsupervised methods. This repository shows that, on this dataset, it is a property of **two choices made before any model is trained** — which coordinates the detector sees, and what it is asked to score. Changing those two things, and nothing else, raises zero-label F1 from **0.203 to 0.463** (± 0.032 over ten stratified splits), with the alarm budget held fixed and no label used for fitting, for model selection, or for thresholding.

---

## Result at a glance

| Zero-label configuration | F1 | PR-AUC |
|---|---|---|
| Rarity scoring on raw sensor features — one-class SVM | 0.203 | 0.168 |
| + three derived coordinates (power, temperature difference, wear × torque) — isolation forest | 0.244 | 0.185 |
| + score *support-boundary violation* instead of rarity — fixed 3.39 % alarm budget | 0.383 | 0.424 |
| + final subspace pool — ten-split mean and standard deviation | **0.463 ± 0.032** | **0.516** |
| Supervised ceiling (MLP, trained on 339 labelled failures) | 0.692 | 0.747 |

Rows 1–2 are the best of the three standard detectors on that representation; rows 3–4 use the support-violation detector with the alarm budget fixed. Every row changes **one** thing — features, score and threshold are never changed together, so the gain can be attributed. The response to the coordinate change differs across families: **+0.105** for isolation forest, **+0.047** for LOF, **+0.005** for one-class SVM, whose RBF kernel already bends around the product constraint.

The operating-point consequence, at an assumed cost ratio of 1,000 for a missed failure to 100 for an inspection:

| Policy | Expected cost |
|---|---|
| Never alarm | 102,000 |
| Released pipeline, single threshold | 67,500 |
| One threshold + watch-list top 20 % (recall 0.735) | 43,580 |

The last figure holds only if a monitored failure is eventually caught with probability α ≥ 0.68. That condition is stated with the number, not after it.

---

## Quick start

```bash
git clone <this-repo>.git && cd <this-repo>
pip install -r code/requirements.txt        # 6 packages, no GPU

cd code
python scripts/eda.py                       # data checks: 10,000 × 14, no missing values
python scripts/run_experiments.py           # tasks A / B / C: 5 + 4 + 3 models, costs, curves
python "unsupervised learning/step2/run_step2.py"   # the controlled ablation above
```

The full pipeline is eight scripts, listed with their order, outputs and expected values in [`code/README.md`](code/README.md). It runs in **3–8 minutes** on a laptop CPU. Every random process uses a fixed seed; results reproduce to the last digit apart from the training-time columns, which track machine load.

---

## What is inside

```
.
├── README.md                  this file
├── README-CN.md               简体中文版
├── LICENSE                    MIT (code) · CC BY 4.0 (dataset)
└── code/
    ├── README.md              environment, per-script guide, expected numbers
    ├── requirements.txt       pinned versions
    ├── data/                  ai4i2020.csv (unmodified) + field dictionary
    ├── scripts/               eda · run_experiments · add_taskB_models
    ├── unsupervised learning/ step1–step6after: the label-free ladder
    ├── baseline/              frozen reference results (inputs to step 4 and 6)
    ├── results/               final task A / B / C tables
    └── figures/               output directory for the per-step diagnostic plots
```

`code/` is kept in its original layout on purpose: the scripts locate the project root through `Path(__file__)`, and `step4` / `step6` read `baseline/results/*`. Flattening the tree breaks those paths.

---

## The two changes

**1. Coordinates.** Three of the five failure modes are deterministic functions of derived quantities: mechanical power outside an interval, a process-to-air temperature difference below a threshold at low rotational speed, and the product of torque and accumulated tool wear above a machine-type threshold. Verified against the released data rather than the documentation, all three recover every affected record with zero false alarms. In raw coordinates the power constraint is a hyperbola; handed the product as a column, a single axis-aligned cut suffices.

**2. Score.** The failures sit *inside* the normal cloud, so rarity is the wrong question. The detector is asked instead how far a record departs from the support of normal data, which is monotone in the violation.

Which detector is best depends on the coordinates: one-class SVM leads on raw features, isolation forest leads once the products are explicit. The response to the same coordinate change differs by a factor of twenty across detector families, which is why the report compares detectors *within* a fixed representation instead of ranking them across one.

---

## Methods inventory

Everything below was actually run, and every row is stored under `code/**/results*.csv` — nothing here is a method that was only considered.

| Layer | Count | What was run |
|---|---|---|
| Supervised classifiers | **11** | Task A: decision tree, random forest, SVM (RBF), MLP, logistic regression. Task B: those four plus LR (one-vs-rest) and KNN (k = 5) |
| Class-imbalance handling | **4 × 3** | none / class_weight / random oversampling / SMOTE, each with decision tree, random forest and MLP |
| Unsupervised detectors | **8** | isolation forest, one-class SVM, LOF, Gaussian mixture, kernel density, Mahalanobis, HBOS-style, and the support-violation detector (MSV) |
| Feature sets | **3** | F1_raw, F2_physics, F3_physics_pt |
| Statistical control arms | **5** | raw features, naive three-feature, full polynomial expansion, Nystroem RBF approximation, physics rules |
| Threshold strategies | **10** | POT (extreme-value), contamination 1 %, contamination 3.39 %, F1-maximising on N = 10/20/50/100 labels, rate estimated from M = 100/300/1000 labels |
| Fusion variants | **9** | max, second-largest, mean of top-2, singletons only, no-HDF pool, Power+Overstrain, vote ≥ 1 / ≥ 2 / ≥ 3 of 3 |
| Contamination-rate estimators | **4** | Kneedle, two-component GMM, largest gap, Hill plateau — all four fail, and are reported as a negative result |
| Cost protocols | **5** | cost-optimal, POT 1 %, watch-list at 3× the rate, never alarm, always alarm |

Step 5 alone — 8 detectors × 3 feature sets × 10 threshold settings — is **240 rows**; all committed result files together hold **590 rows** of evaluated configurations.

---

## Reproducibility

- Labels are used for **evaluation only** — never for fitting, model selection, or threshold choice. The one exception is a deliberately reported arm (L2) that spends 300 labels to calibrate a threshold; it buys the operating point (F1 0.383 → 0.435) and leaves ranking untouched (PR-AUC 0.424 either way).
- All comparisons are paired over ten stratified splits, tested with a Wilcoxon signed-rank test. A claim of improvement requires p < 0.05, a mean difference above 0.02, **and** PR-AUC moving in the same direction; otherwise the report says "not established".
- 27 result CSVs are committed so numbers can be checked without running anything.

---

## Data and citation

The dataset is included unmodified for convenience.

> S. Matzka. *Explainable Artificial Intelligence for Predictive Maintenance Applications.* AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, 2020. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). Licensed CC BY 4.0.

References for the cited methods: isolation forest (Liu, Ting & Zhou, 2008), local outlier factor (Breunig et al., 2000), one-class SVM (Schölkopf et al., 2001), cost-sensitive thresholding (Elkan, 2001), precision–recall analysis (Saito & Rehmsmeier, 2015).

This covers only the methods that carry a primary reference. The support-violation detector is implemented in this repository, and the generic estimators (Gaussian mixture, kernel density, Mahalanobis, HBOS) use their standard formulations — see the inventory above for everything that was run.

---

## License

Code is released under the MIT License — see [LICENSE](LICENSE). The dataset remains under CC BY 4.0 and is redistributed here unmodified, with attribution.
