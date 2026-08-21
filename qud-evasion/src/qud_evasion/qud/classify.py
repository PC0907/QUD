"""Stage 3: from QUD relations to taxonomy labels.

Two interchangeable heads:

  RuleClassifier    : direct application of the taxonomy's relation
                      signatures (zero learned parameters; the cleanest
                      test of the QUD hypothesis).
  LearnedClassifier : gradient-boosted trees over the relation features
                      (dominant relation one-hot, overlap stats, speech
                      act, NLI asymmetry, embedding similarity, turn-level
                      allocation, and answer-commitment). Trained on the
                      internal train split only.

Both predict the 9-way evasion label; clarity is derived through the
taxonomy hierarchy (joint learning for free).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.taxonomy import (
    EVASION_LABELS, EVASION2ID, ID2EVASION,
    QUDRelation, RELATION_RULES, SpeechAct, clarity_from_evasion,
)
from .llm_client import LLMClient, parse_json
from .prompts import DIRECTNESS_SYSTEM, DIRECTNESS_USER

logger = logging.getLogger(__name__)

# Allowed commitment labels (must match relations.commitment_features output).
COMMITMENT_LABELS = ["full", "partial", "evasive", "none"]


# ---------------------------------------------------------------------------
# Directness check (Explicit vs Implicit) for equivalent-relation cases
# ---------------------------------------------------------------------------

def directness_check(
    df: pd.DataFrame,
    client: LLMClient,
    question_col: str = "question",
    answer_col: str = "interview_answer",
    batch_size: int = 256,
) -> pd.Series:
    users = [
        DIRECTNESS_USER.format(question=q, answer=a)
        for q, a in zip(df[question_col], df[answer_col])
    ]
    out = []
    for s in range(0, len(users), batch_size):
        out.extend(client.chat_batch(DIRECTNESS_SYSTEM, users[s:s + batch_size]))
    return pd.Series(
        ["Explicit" if parse_json(t, {}).get("directness") == "explicit" else "Implicit"
         for t in out],
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Rule-based head
# ---------------------------------------------------------------------------

class RuleClassifier:
    """Taxonomy rules over (dominant_relation, speech_act), with the
    Explicit/Implicit split resolved by the directness check."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client

    def predict(self, df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(agg, on="example_id", how="left", validate="1:1")
        preds = []
        for _, row in merged.iterrows():
            rel = QUDRelation(row["dominant_relation"])
            sa = SpeechAct(row["speech_act"]) if rel == QUDRelation.NONE else SpeechAct.ANSWER
            label = RELATION_RULES.get((rel, sa))
            if label is None:
                # NONE relation but speech_act == answer: content-free
                # answer attempt -> treat as Dodging.
                label = "Dodging"
            preds.append(label)
        merged["evasion_pred"] = preds

        # Resolve Explicit vs Implicit on equivalent-relation cases.
        eq_mask = merged["evasion_pred"] == "Explicit"
        if eq_mask.any() and self.client is not None:
            merged.loc[eq_mask, "evasion_pred"] = directness_check(
                merged.loc[eq_mask], self.client
            ).values

        merged["clarity_pred"] = merged["evasion_pred"].map(clarity_from_evasion)
        return merged


# ---------------------------------------------------------------------------
# Learned head
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "qud_overlap", "mean_overlap", "n_addressed",
    "emb_cosine_max", "nli_asym_best",
    "rank_in_turn", "turn_size", "is_multi_question",
    # turn-level allocation (sibling competition)
    "alloc_share", "overlap_rank_in_turn", "is_best_in_turn",
    "sibling_max_overlap", "self_minus_sibling", "won_allocation",
    # answer-commitment (orthogonal to topical overlap)
    "commitment",
]


def _featurize(agg: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    rel_dummies = pd.get_dummies(
        agg["dominant_relation"].astype(
            pd.CategoricalDtype([r.value for r in QUDRelation])
        ), prefix="rel",
    )
    sa_dummies = pd.get_dummies(
        agg["speech_act"].astype(
            pd.CategoricalDtype([s.value for s in SpeechAct])
        ), prefix="sa",
    )
    # commitment label one-hot (full / partial / evasive / none)
    commit_series = (
        agg["commitment_label"] if "commitment_label" in agg.columns
        else pd.Series(["evasive"] * len(agg), index=agg.index)
    )
    commit_dummies = pd.get_dummies(
        commit_series.astype(pd.CategoricalDtype(COMMITMENT_LABELS)),
        prefix="commit",
    )
    num = agg[FEATURE_COLUMNS].fillna(0.0)
    X = pd.concat([num, rel_dummies, sa_dummies, commit_dummies], axis=1)
    return X.to_numpy(dtype=float), list(X.columns)


class LearnedClassifier:
    def __init__(self, **hgb_kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            max_iter=hgb_kwargs.pop("max_iter", 400),
            class_weight="balanced",
            random_state=13,
            **hgb_kwargs,
        )
        self.feature_names: list[str] = []

    def fit(self, agg: pd.DataFrame, y_evasion: pd.Series) -> "LearnedClassifier":
        X, self.feature_names = _featurize(agg)
        self.model.fit(X, y_evasion.map(EVASION2ID))
        return self

    def predict(self, df: pd.DataFrame, agg: pd.DataFrame,
                use_commitment_override: bool = False,
                overlap_hi: float = 0.6,
                commitment_lo: float = 0.30) -> pd.DataFrame:
        merged = df.merge(agg, on="example_id", how="left", validate="1:1")
        X, _ = _featurize(merged)
        merged["evasion_pred"] = [ID2EVASION[i] for i in self.model.predict(X)]
        proba = self.model.predict_proba(X)
        for j, cls in enumerate(self.model.classes_):
            merged[f"p_{ID2EVASION[int(cls)]}"] = proba[:, j]

        # --- Confidence-conditional commitment override ---------------------
        # Surgical correction for the diagnosed high-overlap failure mode: when
        # an answer is highly on-topic (qud_overlap high) but commits to little
        # (commitment low) yet was predicted as a genuine reply, it is the
        # signature of vague evasion -> relabel toward General. This applies the
        # signal ONLY where it is diagnostic (high-overlap region), rather than
        # as a flat feature, mirroring the confidence-conditional routing that
        # was the strongest strategy in the shared task.
        if use_commitment_override and {"qud_overlap", "commitment"}.issubset(merged.columns):
            ov = pd.to_numeric(merged["qud_overlap"], errors="coerce").fillna(0.0)
            cm = pd.to_numeric(merged["commitment"], errors="coerce").fillna(0.25)
            mask = (
                (ov >= overlap_hi)
                & (cm <= commitment_lo)
                & (merged["evasion_pred"].isin(["Explicit", "Implicit"]))
            )
            merged.loc[mask, "evasion_pred"] = "General"
            merged["commitment_override"] = mask.astype(int)

        merged["clarity_pred"] = merged["evasion_pred"].map(clarity_from_evasion)
        return merged

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedClassifier":
        obj = cls()
        blob = joblib.load(path)
        obj.model, obj.feature_names = blob["model"], blob["features"]
        return obj
    
    