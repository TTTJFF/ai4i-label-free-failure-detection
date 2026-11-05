# -*- coding: utf-8 -*-
"""Step 1 — Feature layer (label-free).

Builds the three feature sets compared in step 2:

    F1_raw         : the 5 raw sensors + machine type (this is exactly the
                     representation used by the frozen baseline)
    F2_physics     : F1 + three derived physical quantities
    F3_physics_pt  : F2 + within-machine-type standardised sensors

Every transformation here is deterministic and label-free. The derived
quantities encode *domain knowledge*, not labels: they are the physical
quantities on which the failure mechanisms are actually defined. Per the
project decision Q1 (2026-10-11) this counts as introducing an inductive
bias, which is allowed; nothing in this file looks at `Machine failure`,
the five mode flags, or any split-dependent label.

Design notes
------------
* The three derived quantities are not arbitrary. On 2026-10-11 a read-only
  rule check showed that three of the five failure modes are deterministic
  functions of exactly these quantities (zero false positives over all
  10,000 rows):

      PWF (power failure)   : Power = Torque * 2*pi*Speed/60 outside [3500, 9000] W
      HDF (heat dissipation): Temp diff = Process - Air < 8.6 K  AND  Speed < 1380 rpm
      OSF (overstrain)      : Overstrain = Torque * Tool wear > threshold
                              (L: 11000, M: 12000, H: 13000 Nm*min)

  In the raw coordinate system each of those regions is a *product* half
  space, which axis-aligned detectors cannot cut out. Adding the products
  as explicit coordinates turns them into interval constraints.

* Within-type standardisation exists because the OSF threshold differs by
  machine type (L/M/H), so the same raw value means different things on
  different machines.

* Per-type statistics are estimated on the TRAINING split only and then
  applied to the test split. Using full-dataset statistics would leak test
  information into the features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'ai4i2020.csv'

LABEL = 'Machine failure'
FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TYPES = ['L', 'M', 'H']

RAW_SENSORS = ['Air temperature [K]', 'Process temperature [K]',
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# --- the three derived physical quantities (the core of the hypothesis) ---
DERIVED_PHYSICS = ['Power [W]', 'Temp diff [K]', 'Overstrain [Nm*min]']

# --- per-type standardised copies of the raw sensors ---
PER_TYPE_Z = [c + '_z' for c in RAW_SENSORS]

FEATURE_SETS = {
    'F1_raw': RAW_SENSORS,
    'F2_physics': RAW_SENSORS + DERIVED_PHYSICS,
    'F3_physics_pt': RAW_SENSORS + DERIVED_PHYSICS + PER_TYPE_Z,
}

DERIVED_DOC = {
    'Power [W]': ('Torque * 2*pi*Speed/60', 'mechanical power', 'PWF'),
    'Temp diff [K]': ('Process temp - Air temp', 'heat-dissipation margin', 'HDF'),
    'Overstrain [Nm*min]': ('Torque * Tool wear', 'accumulated mechanical strain', 'OSF'),
}


def load_data():
    """Read the AI4I 2020 dataset from the project data folder."""
    return pd.read_csv(DATA_PATH)


def add_physics(df):
    """Add the three derived physical quantities. No statistics involved."""
    out = df.copy()
    torque = out['Torque [Nm]'].astype(float)
    speed = out['Rotational speed [rpm]'].astype(float)
    wear = out['Tool wear [min]'].astype(float)
    air = out['Air temperature [K]'].astype(float)
    proc = out['Process temperature [K]'].astype(float)

    out['Power [W]'] = torque * 2.0 * np.pi * speed / 60.0
    out['Temp diff [K]'] = proc - air
    out['Overstrain [Nm*min]'] = torque * wear
    return out


def fit_per_type_stats(df_train):
    """Per-type mean/std of the raw sensors, estimated on the training split only."""
    stats = {}
    for t in TYPES:
        sub = df_train[df_train['Type'] == t]
        stats[t] = {c: (float(sub[c].mean()), float(sub[c].std(ddof=0)))
                    for c in RAW_SENSORS}
    return stats


def add_per_type_z(df, stats):
    """Standardise each raw sensor within its machine type, using train statistics."""
    out = df.copy()
    for c in RAW_SENSORS:
        z = np.full(len(out), np.nan)
        for t in TYPES:
            m = (out['Type'] == t).to_numpy()
            mean, std = stats[t][c]
            std = std if std > 0 else 1.0
            z[m] = (out.loc[m, c].to_numpy(dtype=float) - mean) / std
        out[c + '_z'] = z
    return out


def build_frames(df, train_mask):
    """Return the three feature matrices (as DataFrames) plus the label vector.

    Parameters
    ----------
    df : full dataframe (all 10,000 rows, in original order)
    train_mask : boolean array, True for rows belonging to the training split;
        used only to estimate the per-type standardisation statistics.

    Returns
    -------
    (dict of feature_name -> DataFrame, y : np.ndarray)
    """
    df_phys = add_physics(df)
    stats = fit_per_type_stats(df_phys[train_mask])
    df_all = add_per_type_z(df_phys, stats)

    frames = {}
    for name, cols in FEATURE_SETS.items():
        X = df_all[cols].copy()
        # Machine type enters as a drop-first one-hot block, identical in every
        # feature set. Dropping the first level (L) matches the encoder used by
        # the frozen baseline, so F1_raw is directly comparable to it.
        for t in TYPES[1:]:
            X['Type=' + t] = (df_all['Type'] == t).astype(int)
        frames[name] = X

    return frames, df['Machine failure'].to_numpy(dtype=int)


if __name__ == '__main__':
    df = load_data()
    frames, y = build_frames(df, np.ones(len(df), dtype=bool))
    print(f'rows {len(df)}, failures {int(y.sum())} ({y.mean() * 100:.2f}%)')
    for name, X in frames.items():
        print(f'{name:16s} shape={X.shape}  cols={list(X.columns)}')
    d = add_physics(df)
    print('\nderived quantity summaries (full data):')
    print(d[DERIVED_PHYSICS].describe().T[['min', '25%', '50%', '75%', 'max']].round(2))
    for k, (formula, meaning, mode) in DERIVED_DOC.items():
        print(f'  {k:22s} = {formula:26s} -> {meaning} (targets {mode})')
