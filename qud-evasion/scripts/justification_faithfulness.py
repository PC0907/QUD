"""Justification faithfulness for the concept layer.

Every concept answer comes with a one-line justification that usually quotes
or paraphrases the part of the answer it relied on. If that justification is
faithful, deleting the cited span should often flip the concept away from
"yes"; deleting an unrelated sentence of similar length should rarely do so.

For each concept, on rows where the annotator answered "yes":

  1. locate the cited span in the answer (a verbatim quote from the
     justification if there is one, otherwise the answer sentence sharing the
     most content words with the justification);
  2. CITED condition: re-annotate with that span deleted;
  3. CONTROL condition: re-annotate with a different sentence of similar
     length deleted instead.

A faithful justification shows flip_rate(CITED) well above flip_rate(CONTROL).
The control matters: deleting any text shifts the input distribution, so the
raw flip rate alone would overstate faithfulness.

This applies the chain-of-thought faithfulness question (Turpin et al. 2023)
to the concept layer, using only the existing pipeline. Output is a per-trial
JSONL plus a per-concept summary; `--report` recomputes the summary from the
JSONL on the login node without a GPU.

    python scripts/justification_faithfulness.py --concepts v1
    python scripts/justification_faithfulness.py --concepts v1 --report
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("faithfulness")

STOP = set("""
a an the and or but if of to in on at for with by from as is are was were be
been being it its this that these those i we you he she they them our your
their my me us not no do does did have has had will would can could should
shall may might must there here what which who whom whose when where why how so
than then too very just also about into over under again further answer
speaker target question sub respondent says said states stated
""".split())

_QUOTE_RE = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{12,})["\u201c\u201d]')
# Transcripts often lack a space after the full stop ("soon.But no"), so the
# split does not require whitespace, only a capital or quote after the mark.
_SENT_RE = re.compile(r'(?<=[.!?])\s*(?=[A-Z"\u201c])')


# ---------------------------------------------------------------------------
# span location
# ---------------------------------------------------------------------------

def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts or [text.strip()]


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", s.lower())
            if w not in STOP and len(w) > 2}


def locate_span(answer: str, why: str, min_overlap: float = 0.25):
    """Return (span, method). span is None when nothing can be located."""
    low = answer.lower()
    for q in _QUOTE_RE.findall(why or ""):
        q = q.strip().strip(".,;:")
        if len(q.split()) >= 3 and q.lower() in low:
            i = low.index(q.lower())
            return answer[i:i + len(q)], "quote"

    wj = content_words(why or "")
    if not wj:
        return None, "empty_justification"
    best, best_score = None, 0.0
    for s in sentences(answer):
        ws = content_words(s)
        if not ws:
            continue
        score = len(wj & ws) / len(wj)
        if score > best_score:
            best, best_score = s, score
    if best is not None and best_score >= min_overlap:
        return best, "overlap"
    return None, "unlocatable"


def _interval(answer: str, span: str):
    i = answer.find(span)
    return (i, i + len(span)) if i >= 0 else None


def control_span(answer: str, cited: str, rng: np.random.Generator):
    """A sentence sharing no characters with the cited span, close to it in
    length. Overlap is tested by position in the answer, not by substring,
    because a cited quote and a sentence can differ only in punctuation."""
    ci = _interval(answer, cited)
    cands = []
    for s in sentences(answer):
        si = _interval(answer, s)
        if si is None or len(s.split()) < 3:
            continue
        if ci is not None and si[0] < ci[1] and ci[0] < si[1]:
            continue
        cands.append(s)
    if not cands:
        return None
    target = len(cited.split())
    cands.sort(key=lambda s: abs(len(s.split()) - target))
    top = cands[:3]
    return top[int(rng.integers(len(top)))]


def delete(answer: str, span: str) -> str:
    out = answer.replace(span, " ", 1)
    return re.sub(r"\s{2,}", " ", out).strip()


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarize(trials: pd.DataFrame, n_candidates: dict) -> pd.DataFrame:
    rows = []
    for cid in sorted(n_candidates, key=lambda c: (len(c), c)):
        t = trials[trials.concept == cid]
        cited = t[t.condition == "cited"]
        ctrl = t[t.condition == "control"]
        paired_ids = set(cited.example_id) & set(ctrl.example_id)
        c_p = cited[cited.example_id.isin(paired_ids)]
        k_p = ctrl[ctrl.example_id.isin(paired_ids)]

        def flip(df, strict=False):
            if df.empty:
                return np.nan
            return float((df.new_value == "no").mean() if strict
                         else (df.new_value != "yes").mean())

        rows.append({
            "concept": cid,
            "n_yes": n_candidates[cid],
            "n_located": len(cited),
            "located_by_quote": int((cited.method == "quote").sum()),
            "n_paired": len(paired_ids),
            "flip_cited": flip(c_p),
            "flip_control": flip(k_p),
            "delta": flip(c_p) - flip(k_p) if len(paired_ids) else np.nan,
            "flip_cited_strict": flip(c_p, strict=True),
        })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    fmt = summary.copy()
    for c in ["flip_cited", "flip_control", "delta", "flip_cited_strict"]:
        fmt[c] = fmt[c].map(lambda v: "  -  " if pd.isna(v) else f"{v:+.2f}"
                            if c == "delta" else f"{v:.2f}")
    log.info("\n%s", fmt.to_string(index=False))
    log.info(
        "flip = concept no longer 'yes' after deletion; strict = became 'no'. "
        "A faithful justification has flip_cited well above flip_control. "
        "Treat n_paired below ~20 as anecdotal.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="v1")
    ap.add_argument("--data", default="data/processed/dev.parquet")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--cache", default="outputs/llm_cache.sqlite")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-per-concept", type=int, default=60)
    ap.add_argument("--concepts-to-test", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--report", action="store_true",
                    help="recompute the summary from saved trials; no GPU")
    args = ap.parse_args()

    out_dir = Path(args.out or f"outputs/cbm/{args.concepts}/faithfulness")
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    meta_path = out_dir / "candidates.json"

    if args.report:
        trials = pd.read_json(trials_path, lines=True)
        n_candidates = json.loads(meta_path.read_text())
        summary = summarize(trials, n_candidates)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print_summary(summary)
        return

    # Imported here so --report works without the model stack.
    from qud_evasion.cbm.annotate import (build_turn_context, load_annotations,
                                          parse_record, render_system, render_user)
    from qud_evasion.cbm.concepts import get_concepts
    from qud_evasion.qud.llm_client import LLMClient

    concepts = get_concepts(args.concepts)
    test_ids = set(args.concepts_to_test or [c.id for c in concepts])
    df = pd.read_parquet(args.data)
    ann = load_annotations(Path(args.ann or f"outputs/cbm/{args.concepts}/dev.jsonl"))
    ctx = build_turn_context(df)
    rows = df.set_index("example_id")
    rng = np.random.default_rng(args.seed)

    # ---- build trials -----------------------------------------------------
    trials, n_candidates, unlocated = [], {}, {}
    for c in concepts:
        if c.id not in test_ids:
            continue
        yes = ann[ann[f"{c.id}_value"] == "yes"]
        n_candidates[c.id] = int(len(yes))
        if len(yes) > args.max_per_concept:
            yes = yes.sample(n=args.max_per_concept, random_state=args.seed)
        for rec in yes.itertuples():
            ex = rec.example_id
            if ex not in rows.index:
                continue
            answer = rows.at[ex, "interview_answer"]
            span, method = locate_span(answer, getattr(rec, f"{c.id}_why"))
            if span is None:
                unlocated[method] = unlocated.get(method, 0) + 1
                continue
            trials.append({"example_id": ex, "concept": c.id, "condition": "cited",
                           "method": method, "span": span,
                           "answer": delete(answer, span)})
            ctrl = control_span(answer, span, rng)
            if ctrl is not None:
                trials.append({"example_id": ex, "concept": c.id,
                               "condition": "control", "method": method,
                               "span": ctrl, "answer": delete(answer, ctrl)})

    log.info("candidates (concept=yes): %s", n_candidates)
    log.info("could not locate a cited span: %s", unlocated or "none")
    log.info("%d trials to annotate", len(trials))
    meta_path.write_text(json.dumps(n_candidates))

    # ---- annotate ---------------------------------------------------------
    system = render_system(concepts)
    users = []
    for t in trials:
        row = rows.loc[t["example_id"]].copy()
        row["example_id"] = t["example_id"]
        row["interview_answer"] = t["answer"]
        users.append(render_user(row, ctx, None))

    # Identical modified answers (same span cited by several concepts) share
    # one prompt; annotate each unique prompt once.
    uniq = list(dict.fromkeys(users))
    log.info("%d unique prompts after deduplication", len(uniq))
    client = LLMClient(args.model, cache_path=args.cache, max_tokens=args.max_tokens,
                       temperature=0.0, seed=args.seed, batch_size=args.batch_size)
    raw: dict[str, str] = {}
    step = 256
    for s in range(0, len(uniq), step):
        chunk = uniq[s:s + step]
        raw.update(zip(chunk, client.chat_batch(system, chunk)))
        log.info("  %d / %d", min(s + step, len(uniq)), len(uniq))

    # ---- parse and save ---------------------------------------------------
    with trials_path.open("w") as f:
        for t, u in zip(trials, users):
            _, target_idx = ctx[t["example_id"]]
            rec = parse_record(raw[u], concepts, target_idx)
            t_out = {k: v for k, v in t.items() if k != "answer"}
            t_out["new_value"] = rec[f"{t['concept']}_value"]
            t_out["new_why"] = rec[f"{t['concept']}_why"]
            t_out["parse_ok"] = rec["parse_ok"]
            f.write(json.dumps(t_out) + "\n")

    trials_df = pd.read_json(trials_path, lines=True)
    summary = summarize(trials_df, n_candidates)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print_summary(summary)
    log.info("wrote %s and %s", trials_path, out_dir / "summary.csv")


if __name__ == "__main__":
    main()