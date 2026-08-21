# Notebooks

Exploratory analysis only — anything needed to reproduce paper numbers
must live in `src/` and be runnable from `scripts/`.

Suggested notebooks (create as the project progresses):
- `01_data_exploration.ipynb` — label distributions, answer lengths,
  sibling sub-question structure, annotator columns on the official test split.
- `02_qud_qualitative.ipynb` — inspect Stage-1 reconstructions as rationales;
  pick examples for the paper.
- `03_hard_cases.ipynb` — Ambivalent vs Clear Non-Reply boundary,
  QUD-overlap-vs-error curves (uses `qud_evasion.eval.hard_cases`).
