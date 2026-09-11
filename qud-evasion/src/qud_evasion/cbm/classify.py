"""Stage C: classifier over the concept vector. Stage D: interventions.

Two heads, selected by name:
  "lr"  : multinomial logistic regression, class-balanced. Weights are the
          explanation; use this for every interpretability claim.
  "hgb" : HistGradientBoosting, same settings as the existing learned
          head in qud/classify.py. Accuracy ablation only.

Interventions flip one concept at a time on held-out rows and record how
often, and to what, the prediction changes. A concept whose flip never
moves the output is not load-bearing.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.taxonomy import EVASION2ID, ID2EVASION, EVASION_LABELS, clarity_from_evasion
from .concepts import Concept


class ConceptClassifier:
    def __init__(self, kind: str = "lr", seed: int = 13, **kw):
        self.kind = kind
        self.seed = seed
        if kind == "lr":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                C=kw.pop("C", 1.0), class_weight="balanced",
                max_iter=kw.pop("max_iter", 5000), random_state=seed,
            )
        elif kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=kw.pop("max_iter", 400), class_weight="balanced",
                random_state=seed,
            )
        else:
            raise ValueError(f"unknown head {kind!r}")
        self.features: list[str] = []

    # -- training / inference ---------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, features: list[str]) -> "ConceptClassifier":
        self.features = list(features)
        self.model.fit(X[self.features].to_numpy(float), y.map(EVASION2ID).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        A = X[self.features].to_numpy(float)
        out = X[["example_id"]].copy()
        out["evasion_pred"] = [ID2EVASION[int(i)] for i in self.model.predict(A)]
        out["clarity_pred"] = out["evasion_pred"].map(clarity_from_evasion)
        proba = self.model.predict_proba(A)
        for j, cls in enumerate(self.model.classes_):
            out[f"p_{ID2EVASION[int(cls)]}"] = proba[:, j]
        out["p_max"] = proba.max(axis=1)
        return out

    # -- explanations -----------------------------------------------------

    def coefficients(self) -> pd.DataFrame | None:
        """Per-label weights (LR only). Rows = labels, cols = features."""
        if self.kind != "lr":
            return None
        coef = self.model.coef_
        labels = [ID2EVASION[int(c)] for c in self.model.classes_]
        return pd.DataFrame(coef, index=labels, columns=self.features)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": self.kind, "model": self.model, "features": self.features}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ConceptClassifier":
        blob = joblib.load(path)
        obj = cls(kind=blob["kind"])
        obj.model, obj.features = blob["model"], blob["features"]
        return obj


# ---------------------------------------------------------------------------
# Stage D: interventions
# ---------------------------------------------------------------------------

def intervene(
    clf: ConceptClassifier,
    X: pd.DataFrame,
    concepts: list[Concept],
    values: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """For each concept and each forced value, fraction of rows whose
    predicted evasion label changes, and the most common new label."""
    base = clf.predict(X)["evasion_pred"].to_numpy()
    rows = []
    for c in concepts:
        col = c.id
        if col not in clf.features:
            continue
        for v in values:
            Xi = X.copy()
            Xi[col] = v
            pred = clf.predict(Xi)["evasion_pred"].to_numpy()
            changed = pred != base
            top = pd.Series(pred[changed]).value_counts()
            rows.append({
                "concept": c.id,
                "name": c.name,
                "forced_to": v,
                "frac_changed": float(changed.mean()),
                "n_changed": int(changed.sum()),
                "most_common_new_label": (top.index[0] if len(top) else ""),
                "most_common_share": (float(top.iloc[0] / changed.sum()) if changed.any() else 0.0),
            })
    return pd.DataFrame(rows)


def counterfactual_consistency(clf: ConceptClassifier, X: pd.DataFrame,
                               concepts: list[Concept]) -> pd.DataFrame:
    """Does forcing a concept move predictions in the direction the taxonomy
    implies? Reports, per concept, the share of changed predictions that
    land in the label(s) the concept is supposed to indicate."""
    expected = {
        "p1": {"Explicit", "Implicit"},
        "p2": {"Explicit"},
        "p3": {"General"},
        "p4": {"Partial/half-answer"},
        "p5": {"Explicit", "Implicit", "General", "Partial/half-answer", "Deflection"},
        "p6": {"Deflection"},
        "p7": {"Declining to answer"},
        "p8": {"Claims ignorance"},
        "p9": {"Clarification"},
        "p10": {"Dodging", "Deflection", "Partial/half-answer"},
    }
    base = clf.predict(X)["evasion_pred"].to_numpy()
    rows = []
    for c in concepts:
        if c.id not in clf.features or c.id not in expected:
            continue
        Xi = X.copy(); Xi[c.id] = 1.0
        pred = clf.predict(Xi)["evasion_pred"].to_numpy()
        changed = pred != base
        if not changed.any():
            rows.append({"concept": c.id, "n_changed": 0, "share_in_expected": float("nan")})
            continue
        ok = np.isin(pred[changed], list(expected[c.id]))
        rows.append({"concept": c.id, "n_changed": int(changed.sum()),
                     "share_in_expected": float(ok.mean())})
    return pd.DataFrame(rows)