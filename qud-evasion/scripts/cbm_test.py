"""Evaluate the concept bottleneck on the official test split.

The head is trained on train (evasion labels) exactly as in cbm_train.py,
then applied to official_test. That split ships without fine-grained
evasion gold, so only clarity is scorable: the predicted evasion label is
projected up the taxonomy hierarchy and compared against clarity_label.

Results are reported overall and stratified by annotator agreement
(unanimous / majority / expert-resolved), because agreement is far from
uniform on this split and a single macro-F1 hides where the errors are.

    python scripts/cbm_test.py --concepts v1 --head lr
    python scripts/cbm_test.py --concepts v1 --head lr --no-structural
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from qud_evasion.cbm.annotate import load_annotations
from qud_evasion.cbm.classify import ConceptClassifier
from qud_evasion.cbm.concepts import get_concepts
from qud_evasion.cbm.featurize import align, featurize
from qud_evasion.data.taxonomy import CLARITY_LABELS, EVASION_LABELS
from qud_evasion.eval.stratified import disagreement_profile, stratified_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("cbm_test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="v1")
    ap.add_argument("--head", choices=["lr", "hgb"], default="lr")
    ap.add_argument("--ann-dir", default=None)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--no-structural", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir or f"outputs/cbm/{args.concepts}")
    out_dir = Path(args.out or ann_dir / f"head_{args.head}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)

    concepts = get_concepts(args.concepts)
    tr = pd.read_parquet(data / "train.parquet")
    te = pd.read_parquet(data / "official_test.parquet")

    test_ann_path = ann_dir / "official_test.jsonl"
    if not test_ann_path.exists():
        raise SystemExit(
            f"{test_ann_path} not found. Annotate the test split first:\n"
            f"  sbatch scripts/11_cbm_annotate_a100.sh {args.concepts}\n"
            f"(after adding official_test to the split loop in that script)"
        )

    ann_tr = load_annotations(ann_dir / "train.jsonl")
    ann_te = load_annotations(test_ann_path)
    log.info("annotations: train %d, test %d | parse failures %d / %d",
             len(ann_tr), len(ann_te), int((~ann_te.parse_ok).sum()), len(ann_te))

    use_struct = not args.no_structural
    X_tr, feats = featurize(ann_tr, tr, concepts, include_structural=use_struct)
    X_tr, y_tr = align(X_tr, tr, "evasion_label")

    # The test split has no evasion gold, so it cannot go through align();
    # order the feature frame against the test parquet directly instead.
    X_te, _ = featurize(ann_te, te, concepts, include_structural=use_struct)
    X_te = te[["example_id"]].merge(X_te, on="example_id", how="left", validate="1:1")
    missing = X_te[feats].isna().any(axis=1).sum()
    if missing:
        log.warning("%d test rows have no annotation; filling with 0", missing)
        X_te[feats] = X_te[feats].fillna(0.0)

    clar_f1, acc, first = [], [], None
    for seed in args.seeds:
        clf = ConceptClassifier(args.head, seed=seed).fit(X_tr, y_tr, feats)
        pred = clf.predict(X_te)
        clar_f1.append(f1_score(te["clarity_label"], pred["clarity_pred"],
                                labels=CLARITY_LABELS, average="macro", zero_division=0))
        acc.append(accuracy_score(te["clarity_label"], pred["clarity_pred"]))
        if first is None:
            first = pred

    log.info("=" * 64)
    log.info("OFFICIAL TEST | concepts=%s head=%s structural=%s seeds=%d",
             args.concepts, args.head, use_struct, len(args.seeds))
    log.info("clarity macro-F1 : %.4f +/- %.4f", np.mean(clar_f1), np.std(clar_f1))
    log.info("clarity accuracy : %.4f", np.mean(acc))
    log.info("=" * 64)
    print(classification_report(te["clarity_label"], first["clarity_pred"],
                                labels=CLARITY_LABELS, zero_division=0))

    strat = stratified_report(te, first["clarity_pred"], "clarity_label", CLARITY_LABELS)
    strat.to_csv(out_dir / "cbm_test_stratified.csv", index=False)
    log.info("stratified by annotator agreement:\n%s", strat.to_string(index=False))
    disagreement_profile(te, "clarity_label").to_csv(
        out_dir / "cbm_test_disagreement_profile.csv")

    # Predicted evasion distribution: no gold to score against, but a
    # collapsed distribution here explains a weak clarity number upstream.
    log.info("predicted evasion distribution on test:\n%s",
             first["evasion_pred"].value_counts().to_string())

    out = te[["example_id", "clarity_label"]].merge(first, on="example_id")
    out.to_csv(out_dir / "cbm_test_predictions.csv", index=False)
    (out_dir / "cbm_test_summary.json").write_text(json.dumps({
        "concepts": args.concepts, "head": args.head, "structural": use_struct,
        "seeds": args.seeds,
        "test_clarity_macro_f1_mean": float(np.mean(clar_f1)),
        "test_clarity_macro_f1_std": float(np.std(clar_f1)),
        "test_clarity_accuracy": float(np.mean(acc)),
    }, indent=2))
    log.info("wrote results to %s", out_dir)


if __name__ == "__main__":
    main()