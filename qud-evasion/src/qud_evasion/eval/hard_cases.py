"""Hard-case analysis: the Ambivalent <-> Clear Non-Reply boundary.

Course requirement: analyze performance specifically on cases where
"Ambivalent" overlaps with "Clear Non-Reply". In QUD terms, this boundary
is the difference between a *partially overlapping* addressed question
(Ambivalent) and a *disjoint or absent* one (Clear Non-Reply), so the
QUD overlap score produced by the pipeline doubles as a continuous
ambiguity measure for this analysis.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score

HARD_PAIR = ("Ambivalent", "Clear Non-Reply")


def hard_case_slice(df: pd.DataFrame, true_col="clarity_true", pred_col="clarity_pred") -> pd.DataFrame:
    """Rows where truth or prediction falls in the hard pair."""
    mask = df[true_col].isin(HARD_PAIR) | df[pred_col].isin(HARD_PAIR)
    return df[mask].copy()


def boundary_report(df: pd.DataFrame, true_col="clarity_true", pred_col="clarity_pred") -> dict:
    sl = df[df[true_col].isin(HARD_PAIR)]
    cross_confusions = sl[
        (sl[true_col] != sl[pred_col]) & sl[pred_col].isin(HARD_PAIR)
    ]
    return {
        "n_hard_true": len(sl),
        "binary_f1_on_pair": f1_score(
            sl[true_col], sl[pred_col].where(sl[pred_col].isin(HARD_PAIR), HARD_PAIR[0]),
            labels=list(HARD_PAIR), average="macro", zero_division=0,
        ),
        "n_cross_confusions": len(cross_confusions),
        "cross_confusion_rate": len(cross_confusions) / max(len(sl), 1),
        "examples": cross_confusions.head(25)[
            [c for c in ("example_id", "question", true_col, pred_col, "qud_overlap")
             if c in cross_confusions.columns]
        ].to_dict(orient="records"),
    }


def overlap_vs_error(df: pd.DataFrame, overlap_col="qud_overlap",
                     true_col="clarity_true", pred_col="clarity_pred",
                     n_bins: int = 5) -> pd.DataFrame:
    """Bin examples by QUD overlap score and report error rate per bin.
    Expectation: errors concentrate in the middle bins, where the
    addressed question only partially covers the asked one.
    """
    d = df.dropna(subset=[overlap_col]).copy()
    d["correct"] = (d[true_col] == d[pred_col]).astype(int)
    d["bin"] = pd.qcut(d[overlap_col], q=n_bins, duplicates="drop")
    return (
        d.groupby("bin", observed=True)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             mean_overlap=(overlap_col, "mean"))
        .reset_index()
    )
