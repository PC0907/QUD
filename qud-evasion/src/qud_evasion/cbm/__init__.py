"""Concept bottleneck model (CBM) for evasion detection.

Pipeline:
  annotate.py  -> Stage A: one LLM call per row answers K concept questions
                  about the TARGET sub-question, with the full interviewer
                  turn visible (sibling sub-questions enumerated).
  featurize.py -> Stage B: ordinal concept vector (+ structural features).
  classify.py  -> Stage C: logistic regression (readable weights) or HGB;
                  Stage D: concept interventions (faithfulness).

Concept sets are versioned data in concepts.py so that discovery rounds
(v1, v2, ...) are config changes, not code changes.
"""