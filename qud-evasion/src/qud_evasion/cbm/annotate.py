"""Stage A: concept annotation.

One LLM call per row. The prompt shows the full interviewer turn broken
into numbered sub-questions with the TARGET marked, plus the respondent's
full answer, and asks the K concept questions about the TARGET only.

Every call goes through qud_evasion.qud.llm_client.LLMClient, so results
are cached on disk keyed by (model, system, user) and re-runs are free.

Run as a module:
    python -m qud_evasion.cbm.annotate \
        --data data/processed/train.parquet \
        --out  outputs/cbm/v0/train.jsonl \
        --model Qwen/Qwen3-4B-Instruct-2507 --concepts v0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ..qud.llm_client import LLMClient, parse_json
from .concepts import Concept, get_concepts

logger = logging.getLogger(__name__)

VALID_VALUES = {"yes", "partly", "no"}

SYSTEM_TEMPLATE = """\
You analyze exchanges from political interviews and press conferences.

You will see the interviewer's full turn, broken into numbered sub-questions. \
One sub-question is marked as the TARGET. You will also see the respondent's \
full answer, which responds to the whole turn, not only to the TARGET.

Answer each concept question below ABOUT THE TARGET SUB-QUESTION ONLY. Judge \
what the answer does with respect to the TARGET, not with respect to the turn \
as a whole. Other sub-questions in the turn may have been answered; that does \
not make the TARGET answered.

Use exactly one of "yes", "partly", "no" for each concept. Give a one-line \
justification quoting or paraphrasing the relevant part of the answer.

Concept questions:
{concept_list}

Respond with JSON only, in exactly this schema (all keys required):
{schema}"""

USER_TEMPLATE = """\
Interviewer's full turn:
\"\"\"{interview_question}\"\"\"

Sub-questions in this turn:
{sibling_list}

TARGET sub-question: [{target_idx}]

Respondent's answer:
\"\"\"{answer}\"\"\"

JSON:"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def render_system(concepts: list[Concept]) -> str:
    lines = [f"{c.id}: {c.question}" for c in concepts]
    schema_parts = []
    for c in concepts:
        if c.has_which:
            schema_parts.append(
                f'"{c.id}": {{"value": "yes|partly|no", "which": <number or null>, '
                f'"why": "..."}}'
            )
        else:
            schema_parts.append(f'"{c.id}": {{"value": "yes|partly|no", "why": "..."}}')
    schema_parts.append('"confidence": <number from 0.0 to 1.0>')
    schema = "{" + ",\n ".join(schema_parts) + "}"
    return SYSTEM_TEMPLATE.format(concept_list="\n".join(lines), schema=schema)


def build_turn_context(df: pd.DataFrame) -> dict[str, tuple[list[str], int]]:
    """example_id -> (ordered sibling questions, 1-based index of this row)."""
    ctx: dict[str, tuple[list[str], int]] = {}
    for _, grp in df.groupby("turn_id", sort=False):
        grp = grp.sort_values("question_order", kind="stable")
        sibs = grp["question"].tolist()
        for i, ex_id in enumerate(grp["example_id"].tolist(), start=1):
            ctx[ex_id] = (sibs, i)
    return ctx


def render_user(row: pd.Series, ctx: dict, max_answer_chars: int | None) -> str:
    sibs, target_idx = ctx[row["example_id"]]
    sibling_list = "\n".join(
        f"[{i}]{' (TARGET)' if i == target_idx else ''} {q}"
        for i, q in enumerate(sibs, start=1)
    )
    answer = row["interview_answer"]
    if max_answer_chars and len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars] + " [...]"
    return USER_TEMPLATE.format(
        interview_question=row["interview_question"],
        sibling_list=sibling_list,
        target_idx=target_idx,
        answer=answer,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_record(text: str, concepts: list[Concept], target_idx: int) -> dict:
    rec = parse_json(text, default={})
    out: dict = {"parse_ok": bool(rec)}
    for c in concepts:
        node = rec.get(c.id) or {}
        if not isinstance(node, dict):
            node = {}
        val = str(node.get("value", "")).strip().lower()
        out[f"{c.id}_value"] = val if val in VALID_VALUES else "no"
        out[f"{c.id}_valid"] = val in VALID_VALUES
        out[f"{c.id}_why"] = str(node.get("why", ""))[:300]
        if c.has_which:
            which = node.get("which")
            try:
                which = int(which) if which is not None else None
            except (TypeError, ValueError):
                which = None
            out[f"{c.id}_which"] = which
            # A "which" equal to the target is not a sibling redirect.
            out[f"{c.id}_is_other"] = bool(which is not None and which != target_idx)
    try:
        conf = float(rec.get("confidence", float("nan")))
        out["llm_confidence"] = conf if 0.0 <= conf <= 1.0 else float("nan")
    except (TypeError, ValueError):
        out["llm_confidence"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def annotate(
    df: pd.DataFrame,
    client: LLMClient,
    concepts: list[Concept],
    out_path: str | Path,
    batch_size: int = 256,
    max_answer_chars: int | None = None,
) -> pd.DataFrame:
    system = render_system(concepts)
    ctx = build_turn_context(df)
    users = [render_user(row, ctx, max_answer_chars) for _, row in df.iterrows()]
    logger.info("CBM annotate: %d rows, %d concepts", len(df), len(concepts))

    raw: list[str] = []
    for s in range(0, len(users), batch_size):
        raw.extend(client.chat_batch(system, users[s:s + batch_size]))
        logger.info("  %d / %d", min(s + batch_size, len(users)), len(users))

    records = []
    for (_, row), text in zip(df.iterrows(), raw):
        _, target_idx = ctx[row["example_id"]]
        rec = parse_record(text, concepts, target_idx)
        rec["example_id"] = row["example_id"]
        rec["raw"] = text
        records.append(rec)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_fail = sum(1 for r in records if not r["parse_ok"])
    logger.info("Wrote %d records to %s (%d parse failures)", len(records), out_path, n_fail)
    return pd.DataFrame(records)


def load_annotations(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--concepts", default="v0")
    ap.add_argument("--cache", default="outputs/llm_cache.sqlite")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-answer-chars", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    df = pd.read_parquet(args.data)
    concepts = get_concepts(args.concepts)
    client = LLMClient(
        args.model, cache_path=args.cache, max_tokens=args.max_tokens,
        temperature=0.0, seed=args.seed, batch_size=args.batch_size,
    )
    annotate(df, client, concepts, args.out, max_answer_chars=args.max_answer_chars)


if __name__ == "__main__":
    main()