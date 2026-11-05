# -*- coding: utf-8 -*-
"""Step 6 — single-reading demonstration entry point.

Run it with no arguments:

    python "unsupervised learning/step6/demo.py"

It fits the pipeline once on the project's training split, then pushes four
sensor readings through it:

  * one row taken verbatim from the dataset, unmodified;
  * three readings constructed by hand to sit just outside one constraint each
    (power band, overstrain limit, heat-dissipation conjunction).

The three constructed cases are labelled as constructed. They exist to show
the interface (what a technician would see) and to show honestly which of them
the system actually catches.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step1'))
sys.path.insert(0, str(HERE))
import features as F        # noqa: E402
import pipeline as P        # noqa: E402

COLS = F.RAW_SENSORS + ['Type']


def show(title, readings, pipe, labels=None):
    df = pd.DataFrame(readings, columns=COLS)
    out = pipe.predict(df)
    d = F.add_physics(df)
    print(f'\n{title}')
    print('  ' + '-' * 100)
    for i in range(len(df)):
        r = df.iloc[i]
        o = out.iloc[i]
        print(f'  reading {i + 1}: air={r["Air temperature [K]"]:.0f}K  '
              f'proc={r["Process temperature [K]"]:.0f}K  '
              f'speed={r["Rotational speed [rpm]"]:.0f}rpm  '
              f'torque={r["Torque [Nm]"]:.0f}Nm  '
              f'wear={r["Tool wear [min]"]:.0f}min  type={r["Type"]}')
        print(f'      derived: power={d.iloc[i]["Power [W]"]:.0f} W, '
              f'tempdiff={d.iloc[i]["Temp diff [K]"]:.1f} K, '
              f'overstrain={d.iloc[i]["Overstrain [Nm*min]"]:.0f} Nm*min')
        verdict = 'ALARM' if o['alarm'] else ('watch' if o['watch'] else 'clear')
        truth = ''
        if labels is not None:
            truth = f'   [true label: {"FAILURE" if labels[i] else "normal"}]'
        print(f'      -> score {o["score"]:.2f}  [{verdict}]  '
              f'reason: {o["reason"]} ({o["reason_text"]}){truth}')
        if o['alarm'] or o['watch']:
            print(f'      -> action: {o["action"]}')


def main():
    pipe, df, idx_tr = P.fit_from_project()
    info = pipe.explain()
    print('=' * 104)
    print('End-to-end maintenance decision pipeline (unsupervised + weak label)')
    print('=' * 104)
    print(f'  feature layer : 5 sensors -> power, temp diff, overstrain (label-free)')
    print(f'  scorer        : union-of-subspaces support violation, '
          f'{info["subspaces"]} subspaces (label-free)')
    print(f'  decision      : alarm threshold {info["threshold"]:.2f} from the '
          f'{info["cost_model"]} cost model')
    print(f'  calibration   : failure rate {info["calibration_rate"] * 100:.2f}% '
          f'estimated from {P.CAL_SAMPLE} labelled random rows (labels used '
          f'for this one number only)')
    print(f'  reason code   : whichever constraint subspace produced the score; '
          f'mapped through a small action table')

    # 1) two real rows, unmodified, with their true labels printed so you can
    #    judge the alarms yourself instead of taking our word for it
    fail_rows = df.index[df[F.LABEL] == 1]
    real_idx = [1234, int(fail_rows[0])]
    real = df.loc[real_idx, COLS]
    show('Case 1 - two rows taken verbatim from the dataset '
         '(indices %s), with true labels shown' % real_idx,
         real.values.tolist(), pipe,
         labels=[int(df.loc[i, F.LABEL]) for i in real_idx])

    # 2) constructed readings: just outside one constraint each
    constructed = [
        [298.0, 308.0, 1300.0, 70.0, 100.0, 'L'],
        [298.0, 308.0, 1200.0, 60.0, 200.0, 'L'],
        [300.0, 308.0, 1200.0, 40.0, 100.0, 'L'],
    ]
    show('Case 2 — constructed readings, one per constraint\n'
         '         (L1 power band | L2 overstrain limit | L3 heat-dissipation corner)',
         constructed, pipe)

    print('\n' + '=' * 104)
    print('What to notice')
    print('=' * 104)
    print('  1. Readings 1 and 2 of case 2 (power band, overstrain) are caught and')
    print('     attributed correctly. These are the two failure modes whose region')
    print('     is a genuine support edge, so a label-free detector can see them.')
    print('  2. Reading 3 (the heat-dissipation corner) is alarmed but attributed to')
    print('     PWF, not HDF. That is the central finding of this project: the')
    print('     (temp diff, low speed) corner is populated by the failures')
    print('     themselves, so it is not a support edge and cannot be isolated')
    print('     without labels. Step 4 measured this: 1 true detection for 24')
    print('     false alarms from that subspace.')
    print('  3. Case 1 shows an alarm on a real row; compare it with the printed')
    print('     true label to see whether it was justified. The system trades')
    print('     precision for recall at a 1000:100 cost ratio, so false alarms are')
    print('     expected and are the price of not missing failures.')
    print('  The three constructed readings are hand-built to sit just outside one')
    print('  constraint each; they are not sampled from the data and are not')
    print('  evidence of performance -- they only demonstrate the interface.')
    print('=' * 104)


if __name__ == '__main__':
    main()
