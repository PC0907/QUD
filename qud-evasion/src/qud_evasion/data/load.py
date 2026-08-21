"""Loading and cleaning the ailsntua/QEvasion dataset.

The HF dataset has two splits:
  train : 3,448 rows  -> we re-split this into train / internal dev
  test  : 308 rows    -> the OFFICIAL evaluation set; touched exactly once,
                         at the very end (course protocol).

Each row is one (sub-question, full answer) pair. Sub-questions extracted
from the same interviewer turn share the same `interview_question`,
`interview_answer`, and `question_order`, which we expose through a
`turn_id` column (needed for sibling-aware modeling and leak-free splits).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from .taxonomy import normalize_clarity, normalize_evasion

logger = logging.getLogger(__name__)

HF_DATASET = "ailsntua/QEvasion"

KEEP_COLUMNS = [
    "title", "date", "president", "url", "question_order",
    "interview_question", "interview_answer", "question",
    "clarity_label", "evasion_label",
    "annotator_id", "annotator1", "annotator2", "annotator3",
    "inaudible", "multiple_questions", "affirmative_questions",
]


def _turn_id(row: pd.Series) -> str:
    """Stable identifier for an interviewer turn (shared by sibling
    sub-questions)."""
    key = f"{row['url']}||{row['question_order']}||{row['interview_question']}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def load_qevasion(cache_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Load both splits as DataFrames with normalized labels and IDs.

    Returns {"train": df, "official_test": df}.
    """
    ds = load_dataset(HF_DATASET, cache_dir=cache_dir)
    out: dict[str, pd.DataFrame] = {}
    for hf_split, name in [("train", "train"), ("test", "official_test")]:
        df = ds[hf_split].to_pandas()

        # Some dataset versions store both labels in a single `label`
        # column; the current version has clarity_label / evasion_label.
        if "clarity_label" not in df.columns and "label" in df.columns:
            raise RuntimeError(
                "Dataset schema mismatch: expected clarity_label/evasion_label "
                "columns. Inspect the raw columns and update load.py."
            )

        df = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()
        df["clarity_label"] = df["clarity_label"].map(normalize_clarity)
        df["evasion_label"] = df["evasion_label"].map(normalize_evasion)
        n_missing_evasion = df["evasion_label"].isna().sum()
        if n_missing_evasion:
            logger.warning(
                "%s: %d/%d rows have no fine-grained evasion label; "
                "evasion-level metrics will be skipped for this split.",
                name, n_missing_evasion, len(df),
            )
        df["turn_id"] = df.apply(_turn_id, axis=1)
        df["interview_id"] = df["url"].map(
            lambda u: hashlib.md5(str(u).encode()).hexdigest()[:12]
        )
        df["example_id"] = [f"{name}-{i:05d}" for i in range(len(df))]

        n_dropped = df["question"].isna().sum() + df["interview_answer"].isna().sum()
        if n_dropped:
            logger.warning("Dropping %d rows with missing question/answer", n_dropped)
            df = df.dropna(subset=["question", "interview_answer"])

        out[name] = df.reset_index(drop=True)
        logger.info(
            "%s: %d rows, %d interviews, %d turns",
            name, len(df), df["interview_id"].nunique(), df["turn_id"].nunique(),
        )
    return out


def save_processed(dfs: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(df))


def load_processed(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        p.stem: pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))
    }
