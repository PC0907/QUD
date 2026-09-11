"""Stratified evaluation by annotator agreement.

The official test split carries three independent annotator labels, and
gold is the majority vote projected up the clarity hierarchy (verified:
275/275 agreement where a majority exists). Agreement is far from uniform:

    3/3 unanimous      125 rows
    2/3 majority       150 rows
    1/3 no majority     33 rows   (expert-resolved)

and disagreement concentrates almost entirely in Ambivalent (31 of the 33
no-majority rows). A single macro-F1 cannot distinguish a system that
matches humans where humans agree and degrades where they do not, from a
system that is uniformly mediocre. This module reports them separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNOTATOR_COLS = ["annotator1", "annotator2", "annotator3"]
STRATUM_NAME = {3: "unanimous", 2: "majority", 1: "no majority (expert-resolved)"}


def agreement_level(df: pd.DataFrame, cols: list[str] = ANNOTATOR_COLS) -> pd.Series:
    """Size of the modal annotator label per row (3, 2, or 1)."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)

    def n_modal(row):
        vals = [row[c] for c in present if pd.notna(row[c])]
        if not vals:
            return np.nan
        return int(pd.Series(vals).value_counts().iloc[0])

    return df.apply(n_modal, axis=1)


def stratified_report(
    df: pd.DataFrame,
    predictions: list[str] | pd.Series,
    label_col: str = "clarity_label",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Macro-F1 and accuracy overall and per agreement stratum."""
    from sklearn.metrics import accuracy_score, f1_score

    d = df.copy()
    d["_pred"] = list(predictions)
    d["_agree"] = agreement_level(d)

    def score(sub: pd.DataFrame, name: str) -> dict:
        if sub.empty:
            return {"stratum": name, "n": 0, "macro_f1": np.nan, "accuracy": np.nan}
        return {
            "stratum": name,
            "n": len(sub),
            "macro_f1": f1_score(sub[label_col], sub["_pred"],
                                 labels=labels, average="macro", zero_division=0),
            "accuracy": accuracy_score(sub[label_col], sub["_pred"]),
        }

    rows = [score(d, "all")]
    for lvl in (3, 2, 1):
        rows.append(score(d[d._agree == lvl], STRATUM_NAME[lvl]))
    return pd.DataFrame(rows)


def disagreement_profile(df: pd.DataFrame, label_col: str = "clarity_label") -> pd.DataFrame:
    """Gold label distribution per agreement stratum, for the analysis section."""
    d = df.copy()
    d["_agree"] = agreement_level(d)
    return pd.crosstab(d["_agree"].map(STRATUM_NAME), d[label_col])