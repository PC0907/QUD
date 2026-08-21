"""Baseline (a): direct LLM prompting (zero-shot / few-shot CoT).

The strategy class that dominated SemEval-2026 Task 6. Reimplemented as a
comparison point: same backbone model as the QUD pipeline so that any
gain is attributable to the QUD decomposition, not model choice.

Supports two prompt styles via `prompt_style`:
  - "flat"          : original 9-way CoT (DIRECT_BASELINE_*)
  - "hierarchical"  : taxonomy-tree CoT (DIRECT_HIER_*), reasons clarity
                      branch first, then fine-grained strategy.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..data.taxonomy import EVASION_LABELS, clarity_from_evasion, normalize_evasion
from ..qud.llm_client import LLMClient, parse_json
from ..qud.prompts import (
    DIRECT_BASELINE_SYSTEM, DIRECT_BASELINE_USER,
    DIRECT_HIER_SYSTEM, DIRECT_HIER_USER,
)

logger = logging.getLogger(__name__)


def _few_shot_block(train_df: pd.DataFrame, k_per_class: int, seed: int) -> str:
    parts = []
    for label in EVASION_LABELS:
        pool = train_df[train_df["evasion_label"] == label]
        for _, row in pool.sample(min(k_per_class, len(pool)), random_state=seed).iterrows():
            answer = str(row["interview_answer"])
            answer = answer[:800] + (" [...]" if len(answer) > 800 else "")
            parts.append(
                f"Question: {row['question']}\n"
                f"Answer: {answer}\n"
                f"Label: {label}\n"
            )
    return "Here are labeled examples:\n\n" + "\n".join(parts) + "\n---\n"


def run_direct_baseline(
    df: pd.DataFrame,
    client: LLMClient,
    out_path: str | Path,
    few_shot_train: pd.DataFrame | None = None,
    k_per_class: int = 1,
    seed: int = 13,
    batch_size: int = 256,
    prompt_style: str = "flat",
) -> pd.DataFrame:
    if prompt_style == "hierarchical":
        system_base, user_tmpl = DIRECT_HIER_SYSTEM, DIRECT_HIER_USER
    elif prompt_style == "flat":
        system_base, user_tmpl = DIRECT_BASELINE_SYSTEM, DIRECT_BASELINE_USER
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style!r}")

    system = system_base
    if few_shot_train is not None:
        system = system + "\n\n" + _few_shot_block(few_shot_train, k_per_class, seed)

    users = [
        user_tmpl.format(question=q, answer=a)
        for q, a in zip(df["question"], df["interview_answer"])
    ]

    raw: list[str] = []
    for s in range(0, len(users), batch_size):
        raw.extend(client.chat_batch(system, users[s:s + batch_size]))

    preds, fallbacks = [], 0
    for text in raw:
        rec = parse_json(text, default={})
        try:
            preds.append(normalize_evasion(rec.get("evasion_label", "")))
        except ValueError:
            preds.append("Dodging")  # neutral fallback on parse failure
            fallbacks += 1
    if fallbacks:
        logger.warning("Direct baseline (%s): %d/%d label parse fallbacks",
                       prompt_style, fallbacks, len(preds))

    out = df[["example_id"]].copy()
    out["evasion_pred"] = preds
    out["clarity_pred"] = out["evasion_pred"].map(clarity_from_evasion)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(out_path, orient="records", lines=True)
    return out