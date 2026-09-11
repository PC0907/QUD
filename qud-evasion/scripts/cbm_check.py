"""Manual concept check: is the LLM's concept answer right?

Shows one row at a time (target sub-question, answer, the model's value and
its justification) and asks for your judgment. Saves after every row, so you
can stop with Ctrl-C and resume later.

Sampling is disagreement-weighted by default: rows where the model's concept
answer conflicts with what the gold label implies are the informative ones,
but a random slice is included so you can also estimate the base rate.

    # review c1 and c6, 30 rows each
    python scripts/cbm_check.py --concepts-to-check c1 c6 --n 30

    # resume where you left off
    python scripts/cbm_check.py --concepts-to-check c1 c6 --n 30

    # once done, print accuracy
    python scripts/cbm_check.py --report
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from qud_evasion.cbm.annotate import build_turn_context, load_annotations
from qud_evasion.cbm.concepts import get_concepts

# Gold labels that imply the concept SHOULD be yes. Used only to weight the
# sample toward informative rows, never to score the answer.
IMPLIED_YES = {
    "c1": {"Explicit"},
    "c2": {"Implicit"},
    "c3": {"Partial/half-answer"},
    "c4": {"General"},
    "c5": {"Deflection"},
    "c6": {"Declining to answer", "Claims ignorance", "Clarification"},
    "c7": {"Deflection"},
    "c8": set(),
}

VALID = {"y": "yes", "p": "partly", "n": "no", "?": "unsure"}


def wrap(text: str, width: int = 96, limit: int | None = None) -> str:
    if limit and len(text) > limit:
        text = text[:limit] + " [...]"
    return "\n".join(textwrap.wrap(text, width=width)) or "(empty)"


def build_queue(df, ann, cid, n, seed):
    m = ann[["example_id", f"{cid}_value", f"{cid}_why"]].merge(
        df[["example_id", "question", "interview_answer", "turn_id",
            "question_order", "evasion_label"]],
        on="example_id", how="inner")
    implied = IMPLIED_YES.get(cid, set())
    if implied:
        conflict = (m.evasion_label.isin(implied) & (m[f"{cid}_value"] != "yes")) | \
                   (~m.evasion_label.isin(implied) & (m[f"{cid}_value"] == "yes"))
    else:
        conflict = pd.Series(False, index=m.index)
    rng = np.random.default_rng(seed)
    take_conf = min(int(n * 0.6), int(conflict.sum()))
    idx = list(rng.choice(m.index[conflict], take_conf, replace=False)) if take_conf else []
    rest = m.index.difference(idx)
    idx += list(rng.choice(rest, min(n - len(idx), len(rest)), replace=False))
    out = m.loc[idx].copy()
    out["concept"] = cid
    return out


def report(path: Path, concepts) -> None:
    if not path.exists():
        print(f"no judgments yet at {path}")
        return
    d = pd.read_csv(path)
    d = d[d.human.isin(["yes", "partly", "no"])]
    if d.empty:
        print("no scored rows yet")
        return
    print(f"{len(d)} scored rows\n")
    name = {c.id: c.name for c in concepts}
    for cid, g in d.groupby("concept"):
        exact = (g.human == g.model_value).mean()
        # binary: does the concept fire at all (yes/partly) vs not
        fires_h = g.human.isin(["yes", "partly"])
        fires_m = g.model_value.isin(["yes", "partly"])
        binary = (fires_h == fires_m).mean()
        miss = ((fires_h) & (~fires_m)).sum()
        false_fire = ((~fires_h) & (fires_m)).sum()
        print(f"{cid} {name.get(cid,''):<24} n={len(g):<4} "
              f"exact={exact:.2f} binary={binary:.2f} "
              f"missed={miss} over-fired={false_fire}")
        print("   confusion (rows=human, cols=model):")
        print(textwrap.indent(
            pd.crosstab(g.human, g.model_value).to_string(), "     "))
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="v0")
    ap.add_argument("--concepts-to-check", nargs="+", default=["c1", "c6"])
    ap.add_argument("--n", type=int, default=30, help="rows per concept")
    ap.add_argument("--data", default="data/processed/dev.parquet")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--out", default="outputs/cbm/v0/concept_judgments.csv")
    ap.add_argument("--answer-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    concepts = get_concepts(args.concepts)
    out_path = Path(args.out)
    if args.report:
        report(out_path, concepts)
        return

    df = pd.read_parquet(args.data)
    ann = load_annotations(Path(args.ann or f"outputs/cbm/{args.concepts}/dev.jsonl"))
    ctx = build_turn_context(df)
    qtext = {c.id: c.question for c in concepts}

    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev.example_id, prev.concept))
        print(f"resuming: {len(done)} rows already judged")

    queue = pd.concat([build_queue(df, ann, cid, args.n, args.seed)
                       for cid in args.concepts_to_check], ignore_index=True)
    queue = queue[~queue.apply(lambda r: (r.example_id, r.concept) in done, axis=1)]
    if queue.empty:
        print("nothing left to judge; run with --report")
        return

    print(f"{len(queue)} rows to review. y=yes p=partly n=no ?=unsure s=skip q=quit\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = not out_path.exists()

    for i, row in enumerate(queue.itertuples(), start=1):
        sibs, target_idx = ctx[row.example_id]
        print("=" * 98)
        print(f"[{i}/{len(queue)}]  {row.concept}  ({row.example_id})")
        print("-" * 98)
        print("CONCEPT:", wrap(qtext[row.concept]))
        print("-" * 98)
        if len(sibs) > 1:
            print(f"TURN ({len(sibs)} sub-questions, target is [{target_idx}]):")
            for j, q in enumerate(sibs, start=1):
                mark = " <== TARGET" if j == target_idx else ""
                print(f"  [{j}] {wrap(q, 90, 200)}{mark}")
        else:
            print("TARGET:", wrap(row.question, 90, 400))
        print("-" * 98)
        print("ANSWER:", wrap(row.interview_answer, 96, args.answer_chars))
        print("-" * 98)
        print(f"MODEL SAID : {getattr(row, row.concept + '_value')}")
        print("WHY        :", wrap(str(getattr(row, row.concept + '_why')), 96))
        print(f"GOLD LABEL : {row.evasion_label}")
        print("-" * 98)

        while True:
            r = input("your judgment [y/p/n/?/s/q]: ").strip().lower()
            if r in VALID or r in {"s", "q"}:
                break
            print("  use y, p, n, ?, s or q")
        if r == "q":
            print("stopped. rerun to resume.")
            break
        if r == "s":
            continue

        rec = pd.DataFrame([{
            "example_id": row.example_id,
            "concept": row.concept,
            "model_value": getattr(row, row.concept + "_value"),
            "human": VALID[r],
            "gold_evasion": row.evasion_label,
            "turn_size": len(sibs),
        }])
        rec.to_csv(out_path, mode="a", header=header, index=False)
        header = False

    print(f"\nsaved to {out_path}. run with --report for the summary.")


if __name__ == "__main__":
    main()