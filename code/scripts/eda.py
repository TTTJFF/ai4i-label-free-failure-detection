from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import pandas as pd

pd.set_option('display.width', 200)
df = pd.read_csv(r'C:\Users\TTTJJF\WorkBuddy\2026-09-28-16-16-01\data\ai4i2020.csv')

print('shape:', df.shape)
print('\ncolumns:', list(df.columns))
print('\ndtypes:\n', df.dtypes)
print('\nmissing:\n', df.isna().sum()[df.isna().sum() > 0])
print('\ndescribe:\n', df.describe().T)

print('\nMachine failure counts:')
print(df['Machine failure'].value_counts())
print(df['Machine failure'].value_counts(normalize=True))

for c in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
    print(c, int(df[c].sum()))

print('\nType counts:')
print(df['Type'].value_counts())

print('\nUDI range:', df['UDI'].min(), df['UDI'].max())
print('Product ID nunique:', df['Product ID'].nunique())
print('\nhead:\n', df.head())
