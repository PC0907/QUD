# scripts/check_annotators.py
import itertools
import numpy as np
import pandas as pd

tr = pd.read_parquet('data/processed/train.parquet')
dv = pd.read_parquet('data/processed/dev.parquet')
te = pd.read_parquet('data/processed/official_test.parquet')
full_train = pd.concat([tr, dv], ignore_index=True)
A = ['annotator1', 'annotator2', 'annotator3']

print("=" * 70, "\n1. OVERLAP BETWEEN TRAIN AND TEST\n", "=" * 70)
for keys in [['url', 'question'], ['url', 'question', 'interview_answer'], ['interview_answer']]:
    k1 = set(map(tuple, full_train[keys].astype(str).values))
    k2 = set(map(tuple, te[keys].astype(str).values))
    print(f"  key={keys}: {len(k1 & k2)} of {len(te)} test rows also in train")

print("\n", "=" * 70, "\n2. WHAT IS IN THE ANNOTATOR COLUMNS\n", "=" * 70)
for c in A:
    vals = te[c].dropna().unique()
    print(f"  {c}: {len(vals)} distinct -> {sorted(map(str, vals))[:12]}")

print("\n", "=" * 70, "\n3. MAJORITY VOTE VS GOLD\n", "=" * 70)
def majority(row):
    v = [row[c] for c in A if pd.notna(row[c])]
    if not v:
        return None, 0
    s = pd.Series(v).value_counts()
    return (s.index[0], int(s.iloc[0])) if s.iloc[0] > 1 else (None, 1)

maj = te.apply(lambda r: majority(r), axis=1)
te['maj'] = [m[0] for m in maj]
te['maj_n'] = [m[1] for m in maj]
has = te.maj.notna()
print(f"  rows with a majority: {has.sum()} of {len(te)}")
print(f"  NO majority (all three differ, expert-resolved): {(~has).sum()}")
for col in ['evasion_label', 'clarity_label']:
    if te[col].notna().any():
        agree = (te.loc[has, 'maj'].astype(str) == te.loc[has, col].astype(str)).mean()
        print(f"  majority reproduces {col}: {agree:.3f}")
print(f"  unanimous (3/3): {(te.maj_n == 3).sum()}")

print("\n", "=" * 70, "\n4. FLEISS KAPPA ON THE 308\n", "=" * 70)
def fleiss(df, cols):
    cats = sorted({str(v) for c in cols for v in df[c].dropna()})
    M = np.array([[sum(str(r[c]) == k for c in cols if pd.notna(r[c])) for k in cats]
                  for _, r in df.iterrows()], dtype=float)
    n = M.sum(1)
    M, n = M[n > 1], n[n > 1]
    P = ((M ** 2).sum(1) - n) / (n * (n - 1))
    pj = M.sum(0) / M.sum()
    Pe = (pj ** 2).sum()
    return (P.mean() - Pe) / (1 - Pe), len(M)

k, n = fleiss(te, A)
print(f"  raw labels: kappa={k:.3f} over n={n}   (paper: 0.48 evasion, 0.644 clarity)")

print("\n", "=" * 70, "\n5. PER-ANNOTATOR DIAGNOSTICS ON TRAIN\n", "=" * 70)
if 'annotator_id' in full_train.columns and full_train.annotator_id.notna().any():
    g = full_train.groupby('annotator_id')
    print(f"  annotators: {sorted(full_train.annotator_id.dropna().unique())}")
    print("\n  Explicit rate per annotator:")
    print((g.evasion_label.apply(lambda s: (s == 'Explicit').mean())).round(3).to_string())
    print("\n  rows per annotator:")
    print(g.size().to_string())
    print("\n  clarity distribution per annotator:")
    print(pd.crosstab(full_train.annotator_id, full_train.clarity_label,
                      normalize='index').round(3).to_string())
else:
    print("  annotator_id not populated")