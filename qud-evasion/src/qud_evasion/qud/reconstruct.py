"""Stage 1: inverse QUD reconstruction.

Given ONLY the answer text, generate the set of questions the answer
actually addresses, plus the speech act performed. The asked question is
withheld on purpose: showing it would let the model rationalize the
answer as responsive (anchoring), which is exactly the failure mode the
pipeline is designed to avoid.

Output: one JSON record per example, written to a JSONL file so that
Stage 2 can run independently (and so reconstructions can be inspected
as rationales for the paper's qualitative analysis).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # headless fallback
    def tqdm(it, **kwargs):
        return it

from .llm_client import LLMClient, parse_json
from .prompts import RECONSTRUCT_SYSTEM, RECONSTRUCT_USER

logger = logging.getLogger(__name__)

VALID_SPEECH_ACTS = {"answer", "decline", "ignorance", "clarify"}


def reconstruct_quds(
    df: pd.DataFrame,
    client: LLMClient,
    out_path: str | Path,
    max_quds: int = 3,
    answer_col: str = "interview_answer",
    batch_size: int = 256,
) -> pd.DataFrame:
    """Run Stage 1 over a dataframe; returns and persists records:
    {example_id, addressed_questions: [...], speech_act, evidence}

    Note: sibling sub-questions share the same answer, so we deduplicate
    by answer text and broadcast results back — a ~2-3x cost saving and
    a guarantee of consistent reconstructions within a turn.
    """
    system = RECONSTRUCT_SYSTEM.format(max_quds=max_quds)

    unique_answers = df[answer_col].drop_duplicates().tolist()
    logger.info("Stage 1: %d rows -> %d unique answers", len(df), len(unique_answers))

    answer2result: dict[str, dict] = {}
    for start in tqdm(range(0, len(unique_answers), batch_size), desc="reconstruct"):
        chunk = unique_answers[start:start + batch_size]
        users = [RECONSTRUCT_USER.format(answer=a) for a in chunk]
        raw = client.chat_batch(system, users)
        for ans, text in zip(chunk, raw):
            rec = parse_json(text, default={})
            qs = rec.get("addressed_questions") or []
            sa = rec.get("speech_act", "answer")
            answer2result[ans] = {
                "addressed_questions": [q for q in qs if isinstance(q, str)][:max_quds],
                "speech_act": sa if sa in VALID_SPEECH_ACTS else "answer",
                "evidence": rec.get("evidence", ""),
                "parse_ok": bool(rec),
            }

    records = []
    for _, row in df.iterrows():
        rec = dict(answer2result[row[answer_col]])
        rec["example_id"] = row["example_id"]
        records.append(rec)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Stage 1 wrote %d records to %s", len(records), out_path)

    n_fail = sum(1 for r in records if not r["parse_ok"])
    if n_fail:
        logger.warning("Stage 1: %d/%d JSON parse failures", n_fail, len(records))
    return pd.DataFrame(records)


def load_reconstructions(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)
