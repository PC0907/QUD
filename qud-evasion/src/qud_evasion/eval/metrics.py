"""Evaluation: Macro-F1 (course primary metric), per-class breakdown,
confusion matrices, and the official any-annotator matching rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ..data.taxonomy import CLARITY_LABELS, EVASION_LABELS


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {l: report[l] for l in labels},
        "accuracy": report["accuracy"],
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n": len(y_true),
    }


def evaluate_clarity(y_true, y_pred) -> dict:
    return evaluate_predictions(list(y_true), list(y_pred), CLARITY_LABELS)


def evaluate_evasion(y_true, y_pred) -> dict:
    return evaluate_predictions(list(y_true), list(y_pred), EVASION_LABELS)


def any_annotator_macro_f1(
    pred: list[str],
    annotator_labels: list[list[str]],
    labels: list[str],
) -> float:
    """SemEval-2026 Task 6 scoring: a prediction counts as correct if it
    matches ANY annotator's label. Operationalized by snapping y_true to
    the prediction whenever the prediction is in the annotator set,
    otherwise to the first annotator label.

    Only usable on data with per-annotator labels (check the official
    test split; train rows carry a single adjudicated label).
    """
    snapped = [
        p if p in anns else anns[0]
        for p, anns in zip(pred, annotator_labels)
    ]
    return f1_score(snapped, pred, labels=labels, average="macro", zero_division=0)


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str))


def confusion_df(metrics: dict) -> pd.DataFrame:
    labels = metrics["labels"]
    return pd.DataFrame(
        np.array(metrics["confusion_matrix"]),
        index=[f"true:{l}" for l in labels],
        columns=[f"pred:{l}" for l in labels],
    )
