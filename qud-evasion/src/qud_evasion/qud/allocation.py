"""Turn-level answer allocation (sibling sub-question modeling).

The core novel contribution. In the CLARITY data, a single ``interview_answer``
is shared across all sibling sub-questions of a turn (grouped by ``turn_id``);
~70% of rows live in such multi-sibling turns. The shared answer is a finite
resource: when its content is "spent" addressing one sub-question, the siblings
are starved and surface as evasions (empirically, Dodging roughly doubles in
multi-sibling turns, and evasion probability rises monotonically with
sub-question position, ~0.35 -> ~0.92 by the 4th sub-question).

Prior shared-task systems classified every row independently. Here we instead
reason about the whole turn jointly: we treat the per-sub-question QUD overlap
as a bid for the answer's content and run a competitive allocation across
siblings. The sub-question that wins the answer is "answered"; the starved
siblings are the evaded ones, and *which* sibling won is an interpretable
rationale ("the answer addressed sub-question 1, leaving 2 and 3 unallocated").

This module derives, per example (row), a set of allocation features that
(a) feed the learned head as extra columns, and (b) can be read directly as a
zero-parameter allocation rule. It operates on the aggregated per-example frame
``agg`` (which must already carry ``turn_id`` and ``qud_overlap``; both are
attached by ``aggregate_per_example``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sub-questions whose own overlap is below this AND whose best sibling clears
# SIBLING_HI are treated by the allocation *rule* as "content went to a sibling".
SELF_LO = 0.30
SIBLING_HI = 0.50

ALLOCATION_FEATURE_COLUMNS = [
    "alloc_share",          # this sub-q's overlap / sum of turn overlaps
    "overlap_rank_in_turn", # 1 = most-addressed sub-q in the turn
    "is_best_in_turn",      # won the highest overlap in its turn
    "sibling_max_overlap",  # best overlap achieved by any *other* sibling
    "self_minus_sibling",   # qud_overlap - sibling_max_overlap (content tug-of-war)
    "won_allocation",       # competitive winner of the answer's content
]


def allocate_turns(agg: pd.DataFrame) -> pd.DataFrame:
    """Add turn-level allocation features to ``agg`` (one row per example).

    Requires columns: ``example_id``, ``turn_id``, ``qud_overlap``.
    Safe on single-sibling turns (a lone sub-question trivially wins its turn).
    """
    agg = agg.copy()

    if "turn_id" not in agg.columns or "qud_overlap" not in agg.columns:
        # No turn structure available; fill neutral defaults so downstream
        # featurization still has the columns.
        agg["alloc_share"] = 1.0
        agg["overlap_rank_in_turn"] = 1.0
        agg["is_best_in_turn"] = 1.0
        agg["sibling_max_overlap"] = 0.0
        agg["self_minus_sibling"] = agg.get("qud_overlap", 0.0)
        agg["won_allocation"] = 1.0
        return agg

    ov = agg["qud_overlap"].fillna(0.0)
    agg["_ov"] = ov

    # --- per-turn aggregates ---
    grp = agg.groupby("turn_id")["_ov"]
    turn_sum = grp.transform("sum")
    turn_max = grp.transform("max")
    turn_size = agg.groupby("turn_id")["example_id"].transform("size")

    # share of the turn's total addressed-content captured by this sub-question
    agg["alloc_share"] = np.where(turn_sum > 0, agg["_ov"] / turn_sum, 0.0)

    # rank by overlap within the turn (1 = most-addressed); ties -> min rank
    agg["overlap_rank_in_turn"] = (
        agg.groupby("turn_id")["_ov"].rank(method="min", ascending=False)
    )

    # did this sub-question achieve the turn's max overlap?
    agg["is_best_in_turn"] = (agg["_ov"] >= turn_max).astype(float)

    # best overlap any *sibling* reached: turn_max if I'm not the unique max,
    # else the second-highest in the turn.
    def _sibling_max(s: pd.Series) -> pd.Series:
        if len(s) == 1:
            return pd.Series([0.0], index=s.index)
        out = []
        arr = s.to_numpy()
        for i in range(len(arr)):
            others = np.delete(arr, i)
            out.append(float(others.max()) if len(others) else 0.0)
        return pd.Series(out, index=s.index)

    agg["sibling_max_overlap"] = (
        agg.groupby("turn_id")["_ov"].transform(lambda s: _sibling_max(s).values)
    )

    # tug-of-war: positive => I out-competed my siblings for the answer.
    agg["self_minus_sibling"] = agg["_ov"] - agg["sibling_max_overlap"]

    # competitive winner: top-ranked in the turn AND captured a non-trivial share.
    agg["won_allocation"] = (
        (agg["overlap_rank_in_turn"] <= 1.0) & (agg["_ov"] > 0.0)
    ).astype(float)

    agg.drop(columns=["_ov"], inplace=True)
    return agg


def allocation_rule_label(row: pd.Series) -> str | None:
    """Zero-parameter allocation rule (optional, interpretable).

    Returns an evasion label *only* when the allocation structure is decisive,
    else None (defer to the relation/learned head). The decisive case: this
    sub-question is starved (low self overlap) while a sibling captured the
    answer (high sibling overlap) -> the content went elsewhere.
    """
    self_ov = float(row.get("qud_overlap", 0.0) or 0.0)
    sib = float(row.get("sibling_max_overlap", 0.0) or 0.0)
    won = float(row.get("won_allocation", 0.0) or 0.0)

    if won >= 1.0 and self_ov >= SIBLING_HI:
        # captured the answer's content -> a real reply (Explicit/Implicit
        # resolved later by the directness check; default Explicit here).
        return "Explicit"
    if self_ov < SELF_LO and sib >= SIBLING_HI:
        # starved while a sibling won the content -> the answer was spent
        # elsewhere. This is the canonical allocation-driven evasion.
        return "Dodging"
    return None