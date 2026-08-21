import numpy as np
import pandas as pd

from qud_evasion.data.splits import interview_level_split
from qud_evasion.data.taxonomy import EVASION_LABELS


def _toy_df(n_interviews=40, rows_per=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_interviews):
        for j in range(rows_per):
            rows.append({
                "interview_id": f"iv{i}",
                "turn_id": f"iv{i}-t{j // 2}",
                "evasion_label": rng.choice(EVASION_LABELS),
            })
    return pd.DataFrame(rows)


def test_no_interview_leak_and_reasonable_size():
    df = _toy_df()
    train, dev = interview_level_split(df, dev_fraction=0.10, seed=13)
    assert set(train["interview_id"]).isdisjoint(set(dev["interview_id"]))
    assert 0.05 * len(df) <= len(dev) <= 0.20 * len(df)


def test_split_is_deterministic():
    df = _toy_df()
    _, dev_a = interview_level_split(df, seed=13)
    _, dev_b = interview_level_split(df, seed=13)
    assert sorted(dev_a["interview_id"].unique()) == sorted(dev_b["interview_id"].unique())
