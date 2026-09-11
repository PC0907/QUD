"""Leak-free train / val / internal-dev splitting.

Sibling sub-questions share the same full answer text, and questions
within one interview share topical and speaker context. A random row
split therefore leaks. We split at the INTERVIEW level (group split),
with greedy stratification so each held-out set approximately matches
the global evasion-label distribution.

`three_way_split` exists because epoch selection and reporting must not
use the same data: selecting the best epoch by dev macro-F1 and then
reporting that same dev macro-F1 is a maximum over epochs, not an
estimate. Train on `train`, select on `val`, report on `dev`.

The official 308-row test split is never passed through this module.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .taxonomy import EVASION_LABELS

logger = logging.getLogger(__name__)


def interview_level_split(
    df: pd.DataFrame,
    dev_fraction: float = 0.10,
    seed: int = 13,
    label_col: str = "evasion_label",
    group_col: str = "interview_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy stratified group split.

    Interviews are shuffled, then assigned to dev one at a time while the
    dev set is below the target size, preferring the interview whose
    addition keeps the dev label distribution closest (L1) to the global
    distribution.
    """
    rng = np.random.default_rng(seed)
    global_dist = (
        df[label_col].value_counts(normalize=True)
        .reindex(EVASION_LABELS, fill_value=0.0).to_numpy()
    )

    groups = df.groupby(group_col)
    group_ids = list(groups.groups.keys())
    rng.shuffle(group_ids)

    group_counts = {
        g: groups.get_group(g)[label_col]
        .value_counts().reindex(EVASION_LABELS, fill_value=0).to_numpy()
        for g in group_ids
    }

    target = int(round(dev_fraction * len(df)))
    dev_groups: list = []
    dev_vec = np.zeros(len(EVASION_LABELS), dtype=float)
    dev_size = 0

    remaining = set(group_ids)
    while dev_size < target and remaining:
        best_g, best_score = None, None
        # Evaluate a random candidate pool for tractability.
        # NOTE: sorted(), not list(). `remaining` is a set of interview-id
        # strings and Python randomizes string hashing per process, so
        # list(remaining) gives a different order in every run and the split
        # is not reproducible despite the seed.
        pool = rng.choice(sorted(remaining), size=min(64, len(remaining)), replace=False)
        for g in pool:
            cand = dev_vec + group_counts[g]
            cand_dist = cand / cand.sum()
            score = float(np.abs(cand_dist - global_dist).sum())
            if best_score is None or score < best_score:
                best_g, best_score = g, score
        dev_groups.append(best_g)
        dev_vec += group_counts[best_g]
        dev_size += int(group_counts[best_g].sum())
        remaining.discard(best_g)

    dev_mask = df[group_col].isin(dev_groups)
    train_df = df[~dev_mask].reset_index(drop=True)
    dev_df = df[dev_mask].reset_index(drop=True)

    logger.info(
        "Split: train=%d rows (%d interviews) | held-out=%d rows (%d interviews)",
        len(train_df), train_df[group_col].nunique(),
        len(dev_df), dev_df[group_col].nunique(),
    )
    assert set(train_df[group_col]) & set(dev_df[group_col]) == set(), "Group leak!"
    return train_df, dev_df


def three_way_split(
    df: pd.DataFrame,
    dev_fraction: float = 0.10,
    val_fraction: float = 0.10,
    seed: int = 13,
    label_col: str = "evasion_label",
    group_col: str = "interview_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, dev), all disjoint at the interview level.

    Both fractions are of the ORIGINAL dataframe. dev is carved first so
    that it is unaffected by val_fraction: changing the validation size
    must not change the reporting set, or numbers stop being comparable
    across experiments.
    """
    rest, dev = interview_level_split(
        df, dev_fraction=dev_fraction, seed=seed,
        label_col=label_col, group_col=group_col,
    )
    # val_fraction is relative to the original df, so rescale against `rest`.
    rel = val_fraction * len(df) / max(len(rest), 1)
    train, val = interview_level_split(
        rest, dev_fraction=min(rel, 0.9), seed=seed + 1,
        label_col=label_col, group_col=group_col,
    )

    for a, b, name in [(train, val, "train/val"), (train, dev, "train/dev"),
                       (val, dev, "val/dev")]:
        assert set(a[group_col]) & set(b[group_col]) == set(), f"Group leak in {name}!"

    logger.info(
        "Three-way: train=%d (%d iv) | val=%d (%d iv) | dev=%d (%d iv)",
        len(train), train[group_col].nunique(),
        len(val), val[group_col].nunique(),
        len(dev), dev[group_col].nunique(),
    )
    return train, val, dev