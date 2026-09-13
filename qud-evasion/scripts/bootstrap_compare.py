"""Bootstrap confidence intervals for test-set macro-F1, and a paired test
between two systems evaluated on the same rows.

Resampling is at TURN level, not row level. Sibling sub-questions share a
single answer, so rows within a turn are not independent and a row-level
bootstrap would report an interval narrower than the data supports.

With two systems the paired difference is bootstrapped directly, which is
the right test for "is 0.596 better than 0.588": both systems are scored on
the same resampled rows every iteration, so the shared sampling noise
cancels.

    # interval for one system
    python scripts/bootstrap_compare.py \
        --a outputs/cbm/v1/head_lr/cbm_test_predictions.csv

    # paired comparison
    python scripts/bootstrap_compare.py \
        --a outputs/cbm/v1/head_lr/cbm_test_predictions.csv --a-name cbm \
        --b outputs/encoder/clarity/test_predictions.csv    --b-name encoder
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from qud_evasion.data.taxonomy import CLARITY_LABELS


def bootstrap(df: pd.DataFrame, names: list[str], n: int, seed: int) -> dict:
    turns = df.turn_id.unique()
    rows_for = {t: np.flatnonzero(df.turn_id.values == t) for t in turns}
    rng = np.random.default_rng(seed)
    gold = df.gold.values
    preds = {c: df[c].values for c in names}

    out: dict[str, list] = {c: [] for c in names}
    out["_diff"] = []
    for _ in range(n):
        picked = rng.choice(turns, size=len(turns), replace=True)
        idx = np.concatenate([rows_for[t] for t in picked])
        g = gold[idx]
        scores = {}
        for c in names:
            scores[c] = f1_score(g, preds[c][idx], labels=CLARITY_LABELS,
                                 average="macro", zero_division=0)
            out[c].append(scores[c])
        if len(names) == 2:
            out["_diff"].append(scores[names[0]] - scores[names[1]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV with example_id + a prediction column")
    ap.add_argument("--a-col", default="clarity_pred")
    ap.add_argument("--a-name", default="system_a")
    ap.add_argument("--b", default=None)
    ap.add_argument("--b-col", default="clarity_pred")
    ap.add_argument("--b-name", default="system_b")
    ap.add_argument("--data", default="data/processed/official_test.parquet")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    te = pd.read_parquet(args.data)[["example_id", "turn_id", "clarity_label"]]
    te = te.rename(columns={"clarity_label": "gold"})

    a = pd.read_csv(args.a)[["example_id", args.a_col]].rename(
        columns={args.a_col: args.a_name})
    df = te.merge(a, on="example_id", validate="1:1")
    names = [args.a_name]

    if args.b:
        b = pd.read_csv(args.b)[["example_id", args.b_col]].rename(
            columns={args.b_col: args.b_name})
        df = df.merge(b, on="example_id", validate="1:1")
        names.append(args.b_name)

    print(f"{len(df)} rows over {df.turn_id.nunique()} turns, "
          f"{args.n} bootstrap samples (resampled by turn)\n")

    res = bootstrap(df, names, args.n, args.seed)
    for c in names:
        point = f1_score(df.gold, df[c], labels=CLARITY_LABELS,
                         average="macro", zero_division=0)
        lo, hi = np.percentile(res[c], [2.5, 97.5])
        print(f"  {c:<12} macro-F1 {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
              f"   width {hi - lo:.4f}")

    if len(names) == 2:
        d = np.array(res["_diff"])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        print(f"\n  paired difference ({names[0]} - {names[1]}):")
        print(f"    mean {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p ~ {p:.3f}")
        print("    -> interval spans zero: no detectable difference"
              if lo < 0 < hi else
              "    -> interval excludes zero: difference is detectable")


if __name__ == "__main__":
    main()