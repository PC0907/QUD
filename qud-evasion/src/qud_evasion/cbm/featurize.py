"""Stage B: concept vector.

yes -> 1.0, partly -> 0.5, no -> 0.0 for every concept. `has_which`
concepts additionally contribute a binary "points at a different sibling".
Structural features come straight from the parquet (turn_id,
question_order) and do not depend on any LLM output.

Encoder probabilities are deliberately NOT built here. If you want the
stacking ablation, add them out-of-fold in cbm_train.py, never from a
model that saw the rows it is scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .concepts import Concept

VALUE_MAP = {"no": 0.0, "partly": 0.5, "yes": 1.0}


def structural_features(df: pd.DataFrame) -> pd.DataFrame:
    s = df[["example_id", "turn_id", "question_order"]].copy()
    s["rank_in_turn"] = s.groupby("turn_id")["question_order"].rank(method="first")
    size = s.groupby("turn_id")["example_id"].transform("size")
    s["turn_size"] = size.astype(float)
    s["is_multi_question"] = (size > 1).astype(float)
    return s[["example_id", "rank_in_turn", "turn_size", "is_multi_question"]]


def concept_columns(concepts: list[Concept]) -> list[str]:
    cols = [c.id for c in concepts]
    cols += [f"{c.id}_other" for c in concepts if c.has_which]
    return cols


def featurize(
    ann: pd.DataFrame,
    df: pd.DataFrame,
    concepts: list[Concept],
    include_structural: bool = True,
    include_confidence: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Returns (frame with example_id + features, feature column names)."""
    X = ann[["example_id"]].copy()
    for c in concepts:
        X[c.id] = ann[f"{c.id}_value"].map(VALUE_MAP).fillna(0.0).astype(float)
        if c.has_which:
            X[f"{c.id}_other"] = ann[f"{c.id}_is_other"].fillna(False).astype(float)
    feats = concept_columns(concepts)

    if include_confidence and "llm_confidence" in ann.columns:
        X["llm_confidence"] = ann["llm_confidence"].fillna(0.5).astype(float)
        feats.append("llm_confidence")

    if include_structural:
        X = X.merge(structural_features(df), on="example_id", how="left")
        feats += ["rank_in_turn", "turn_size", "is_multi_question"]

    X[feats] = X[feats].fillna(0.0)
    return X, feats


def align(X: pd.DataFrame, df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Order X to df and return the gold labels alongside."""
    merged = df[["example_id", label_col]].merge(X, on="example_id", how="left", validate="1:1")
    assert merged[X.columns.drop("example_id")].notna().all().all(), "missing annotations"
    return merged, merged[label_col]