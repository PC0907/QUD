"""Paired significance and a length-confound check for the faithfulness trials.

McNemar's exact test is the right test for this design: each row is tested
twice (cited span deleted vs control sentence deleted), so the two flip rates
are paired, and only the discordant pairs carry information.

The length check asks whether control flips are an artifact of deleting a
large share of a short answer, which matters most for the refusal concept,
where short answers are common.

    python scripts/faithfulness_stats.py \
        --trials outputs/cbm/v1/faithfulness_train/trials.jsonl \
        --data data/processed/train.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-pairs", type=int, default=20)
    args = ap.parse_args()

    t = pd.read_json(args.trials, lines=True)
    t["flip"] = (t.new_value != "yes").astype(int)
    words = pd.read_parquet(args.data).set_index("example_id")["interview_answer"] \
              .str.split().str.len()

    wide = t.pivot_table(index=["concept", "example_id"], columns="condition",
                         values="flip", aggfunc="first").dropna().reset_index()
    spans = t[t.condition == "control"].set_index(["concept", "example_id"])["span"]
    wide = wide.join(spans, on=["concept", "example_id"])
    wide["answer_words"] = wide.example_id.map(words)
    wide["deleted_frac"] = wide.span.str.split().str.len() / wide.answer_words

    print("McNemar exact test, cited vs control deletion (paired)\n")
    print(f"{'concept':<8}{'pairs':>6}{'cited':>8}{'control':>9}"
          f"{'only cited':>12}{'only ctrl':>11}{'p':>10}")
    for cid, g in wide.groupby("concept", sort=False):
        b = int(((g.cited == 1) & (g.control == 0)).sum())
        c = int(((g.cited == 0) & (g.control == 1)).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
        print(f"{cid:<8}{len(g):>6}{g.cited.mean():>8.2f}{g.control.mean():>9.2f}"
              f"{b:>12}{c:>11}{p:>10.4f}")

    print("\nControl flip rate by answer length and by share of answer deleted")
    print("(if control flips concentrate in short answers / large deletions,")
    print(" the control flip rate is a length artifact, not concept fragility)\n")
    for cid, g in wide.groupby("concept", sort=False):
        if len(g) < args.min_pairs:
            continue
        print(f"--- {cid}  (n={len(g)}, median answer {g.answer_words.median():.0f} words)")
        for col, label in [("answer_words", "answer length"),
                           ("deleted_frac", "share deleted")]:
            try:
                bins = pd.qcut(g[col], 3, labels=["low", "mid", "high"],
                               duplicates="drop")
            except ValueError:
                continue
            s = g.groupby(bins, observed=True).control.agg(["mean", "size"])
            cells = "  ".join(f"{i}: {r['mean']:.2f} (n={int(r['size'])})"
                              for i, r in s.iterrows())
            print(f"  {label:<14} {cells}")
        print()


if __name__ == "__main__":
    main()