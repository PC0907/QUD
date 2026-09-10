"""Train and evaluate the concept bottleneck. CPU only; runs on the login node.

Reads the JSONL written by qud_evasion.cbm.annotate for train and dev,
builds concept features, fits the head, reports macro-F1 for evasion and
clarity, prints logistic-regression weights, runs interventions, and
optionally dumps rows for manual concept checking.

Example:
    python scripts/cbm_train.py --concepts v0 --head lr
    python scripts/cbm_train.py --concepts v0 --head lr --no-structural
    python scripts/cbm_train.py --concepts v0 --head hgb --seeds 13 17 23 29 31
    python scripts/cbm_train.py --concepts v0 --dump-check 100
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
from qud_evasion.cbm.classify import ConceptClassifier, counterfactual_consistency, intervene
from qud_evasion.cbm.concepts import get_concepts
from qud_evasion.cbm.featurize import align, featurize
from qud_evasion.data.taxonomy import CLARITY_LABELS, EVASION_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("cbm_train")


def macro(gold, pred, labels):
    return f1_score(gold, pred, labels=labels, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="v0")
    ap.add_argument("--ann-dir", default=None, help="defaults to outputs/cbm/<concepts>")
    ap.add_argument("--train-parquet", default="data/processed/train.parquet")
    ap.add_argument("--dev-parquet", default="data/processed/dev.parquet")
    ap.add_argument("--head", choices=["lr", "hgb"], default="lr")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--no-structural", action="store_true")
    ap.add_argument("--with-confidence", action="store_true")
    ap.add_argument("--dump-check", type=int, default=0,
                    help="write N stratified dev rows to CSV for manual concept checking")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir or f"outputs/cbm/{args.concepts}")
    out_dir = Path(args.out or ann_dir / f"head_{args.head}")
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = get_concepts(args.concepts)
    tr = pd.read_parquet(args.train_parquet)
    dv = pd.read_parquet(args.dev_parquet)
    ann_tr = load_annotations(ann_dir / "train.jsonl")
    ann_dv = load_annotations(ann_dir / "dev.jsonl")

    log.info("parse failures: train %d / %d, dev %d / %d",
             (~ann_tr.parse_ok).sum(), len(ann_tr), (~ann_dv.parse_ok).sum(), len(ann_dv))

    # --- Stage B ---------------------------------------------------------
    X_tr, feats = featurize(ann_tr, tr, concepts,
                            include_structural=not args.no_structural,
                            include_confidence=args.with_confidence)
    X_dv, _ = featurize(ann_dv, dv, concepts,
                        include_structural=not args.no_structural,
                        include_confidence=args.with_confidence)
    X_tr, y_tr = align(X_tr, tr, "evasion_label")
    X_dv, y_dv = align(X_dv, dv, "evasion_label")
    gold_clar = dv.set_index("example_id").loc[X_dv.example_id, "clarity_label"].to_numpy()

    # --- concept marginals: are the concepts even firing? ----------------
    log.info("concept value distribution on train:")
    for c in concepts:
        vc = ann_tr[f"{c.id}_value"].value_counts(normalize=True)
        log.info("  %-4s %-24s yes=%.2f partly=%.2f no=%.2f", c.id, c.name,
                 vc.get("yes", 0), vc.get("partly", 0), vc.get("no", 0))

    # --- Stage C ---------------------------------------------------------
    ev, cl, preds = [], [], None
    for seed in args.seeds:
        clf = ConceptClassifier(args.head, seed=seed).fit(X_tr, y_tr, feats)
        p = clf.predict(X_dv)
        ev.append(macro(y_dv, p.evasion_pred, EVASION_LABELS))
        cl.append(macro(gold_clar, p.clarity_pred, CLARITY_LABELS))
        if preds is None:
            preds, first_clf = p, clf
    ev, cl = np.array(ev), np.array(cl)
    log.info("=" * 64)
    log.info("head=%s structural=%s seeds=%d", args.head, not args.no_structural, len(args.seeds))
    log.info("dev evasion macro-F1 : %.4f +/- %.4f", ev.mean(), ev.std())
    log.info("dev clarity macro-F1 : %.4f +/- %.4f", cl.mean(), cl.std())
    log.info("dev clarity accuracy : %.4f", accuracy_score(gold_clar, preds.clarity_pred))
    log.info("=" * 64)
    print(classification_report(y_dv, preds.evasion_pred, labels=EVASION_LABELS, zero_division=0))

    preds.merge(dv[["example_id", "evasion_label", "clarity_label"]], on="example_id") \
         .to_csv(out_dir / "dev_predictions.csv", index=False)
    first_clf.save(out_dir / "model.joblib")

    # --- weights ---------------------------------------------------------
    coef = first_clf.coefficients()
    if coef is not None:
        coef.to_csv(out_dir / "lr_coefficients.csv")
        log.info("top positive weights per label:")
        for lab, row in coef.iterrows():
            top = row.sort_values(ascending=False).head(3)
            log.info("  %-22s %s", lab, ", ".join(f"{k}={v:+.2f}" for k, v in top.items()))

    # --- Stage D ---------------------------------------------------------
    iv = intervene(first_clf, X_dv, concepts)
    iv.to_csv(out_dir / "interventions.csv", index=False)
    log.info("interventions (force concept, share of dev predictions that change):")
    for _, r in iv.iterrows():
        log.info("  %-4s -> %.1f : %5.1f%% change, mostly to %s (%.0f%%)",
                 r.concept, r.forced_to, 100 * r.frac_changed,
                 r.most_common_new_label or "-", 100 * r.most_common_share)
    cc = counterfactual_consistency(first_clf, X_dv, concepts)
    cc.to_csv(out_dir / "counterfactual_consistency.csv", index=False)
    log.info("counterfactual consistency (share of changes landing in the taxonomy-expected label):")
    for _, r in cc.iterrows():
        log.info("  %-4s n=%-4d expected-share=%s", r.concept, r.n_changed,
                 "nan" if np.isnan(r.share_in_expected) else f"{r.share_in_expected:.2f}")

    # --- manual concept check sample ------------------------------------
    if args.dump_check:
        rng = np.random.default_rng(13)
        per = max(1, args.dump_check // len(EVASION_LABELS))
        picks = []
        for lab in EVASION_LABELS:
            ids = dv.loc[dv.evasion_label == lab, "example_id"].to_numpy()
            picks += list(rng.choice(ids, size=min(per, len(ids)), replace=False))
        chk = dv[dv.example_id.isin(picks)][
            ["example_id", "question", "interview_answer", "evasion_label", "clarity_label"]]
        chk = chk.merge(ann_dv[["example_id"] + [f"{c.id}_value" for c in concepts]
                               + [f"{c.id}_why" for c in concepts]], on="example_id")
        for c in concepts:
            chk[f"{c.id}_human"] = ""
        chk.to_csv(out_dir / "concept_check.csv", index=False)
        log.info("wrote %d rows for manual checking to %s", len(chk), out_dir / "concept_check.csv")

    summary = {"concepts": args.concepts, "head": args.head, "structural": not args.no_structural,
               "seeds": args.seeds, "evasion_f1_mean": float(ev.mean()), "evasion_f1_std": float(ev.std()),
               "clarity_f1_mean": float(cl.mean()), "clarity_f1_std": float(cl.std())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()