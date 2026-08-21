# QUD-Evasion: Methods and Experiments Log

SemEval-2026 Task 6 (CLARITY / QEvasion) — political evasion detection.
This document records every method attempted in the `qud-evasion` project, the
result of each, and what was learned. All scores are **macro-F1 on the internal
dev split** (347 rows carved from the official train set; official test held
out). The two reported metrics throughout:

- **Clarity** = Task 1, the 3-way label (Clear Reply / Ambivalent / Clear Non-Reply). Primary graded metric.
- **Evasion** = Task 2 / bonus, the 9-way fine-grained label.

---

## 1. Task and data

Dataset: `ailsntua/QEvasion`. Official train split was divided interview-level
(leak-free) into train (3101 rows / 215 interviews) and dev (347 rows / 72
interviews). Official test = 308 rows (Task 1 labels; fine-grained Task 2 gold
absent for that split, so evasion is scored on dev only).

Task 1 labels: Clear Reply, Ambivalent, Clear Non-Reply.
Task 2 labels (9): Explicit, Implicit, Dodging, General, Deflection,
Partial/half-answer, Declining to answer, Claims ignorance, Clarification.
Clarity is derivable from evasion through a fixed taxonomy hierarchy
(`EVASION_TO_CLARITY`).

Reference points from the shared-task summary paper: best clarity system 0.89
macro-F1, fine-tuned Llama-70B baseline 0.82; evasion ceiling 0.68 (median
~0.52). Encoder-only models saturate ~0.76–0.81 clarity / ~0.50 evasion — a
known dead end. Annotator agreement on the hard boundary (Ambivalent vs Clear
Non-Reply) is only κ=0.65, vs κ=0.97 for Clear-Reply vs Clear-Non-Reply.

---

## 2. The core method: QUD pipeline

The central idea (novel for this task): instead of classifying each
(question, answer) row in isolation, **reconstruct the Question Under Discussion
(QUD)** that the answer actually addresses, then judge how that reconstructed
question relates to the asked question. Clarity/evasion is then derived from
that relation through the taxonomy.

Three stages, each a separate LLM call (Qwen3-4B-Instruct-2507) with strict JSON
output, all disk-cached in `outputs/llm_cache.sqlite`:

- **Stage 1 — RECONSTRUCT** (`reconstruct.py` / `prompts.py`): given ONLY the
  answer (the asked question is withheld to prevent anchoring — "inverse QUD
  reconstruction"), infer the addressed question(s) + a speech act
  (answer / decline / ignorance / clarify).
- **Stage 2 — RELATE** (`relations.py`): compare the asked question vs each
  reconstructed question → a structured relation (equivalent / specification /
  generalization / topic_shift / unrelated) plus a 0–1 `overlap` score.
  Augmented with embedding cosine (all-mpnet-base-v2) and bidirectional NLI
  features (DeBERTa-v3-base-mnli-fever-anli).
- **Stage 3 — CLASSIFY** (`classify.py`): two interchangeable heads.
  - **RuleClassifier**: zero-parameter taxonomy rules over
    (dominant_relation, speech_act). Explicit/Implicit resolved by a
    DIRECTNESS check (a 4th targeted LLM call).
  - **LearnedClassifier**: HistGradientBoosting over the relation/overlap/NLI/
    embedding features (+ later additions), class_weight="balanced",
    trained on the internal train split only.

---

## 3. Data findings (the empirical contribution)

Validated directly on `train.parquet` via groupby on `turn_id`. These motivate
the whole structural approach and were not reported by any prior shared-task
system:

1. **Sibling structure is dominant.** A single `interview_answer` is shared
   across multiple sibling sub-questions of a turn (grouped by `turn_id`):
   **~70% of rows live in multi-sibling turns** (1835 turns over 3101 rows).
2. **Multi-sibling turns are ~2× more evasive.** Dodging rate rises 0.131 →
   0.239 from single-question to multi-sibling turns; Implicit also rises.
3. **Evasion rises monotonically with sub-question position** within a turn:
   rank 1 = 34.6% evaded, rank 2 = 66%, rank 3 = 78%, rank 4 = 92%. Politicians
   answer the first thing asked and dodge the rest.
4. **42.1% of multi-sibling turns are "mixed"** (some sub-questions answered,
   some evaded) — the core answer-allocation case.
5. Annotator columns (annotator1/2/3) are NULL in train, populated only in the
   308-row test set → disagreement modelling is an analysis-section contribution
   on the 308-set, not a training contribution.

---

## 4. Experiment ladder (dev macro-F1)

Each row changes ONE thing from the row above (controlled ablations). Backbone
LLM is Qwen3-4B-Instruct-2507 unless stated.

| # | Configuration | Evasion | Clarity | Verdict |
|---|---|---|---|---|
| 0 | Direct LLM baseline (few-shot prompting) | 0.204 | 0.444 | Baseline to beat |
| 1 | QUD, RuleClassifier head | 0.142 | ~0.398 | Rule head collapses |
| 2 | QUD, LearnedClassifier (base features) | 0.194 | 0.380 | Learned recovers dead classes |
| 3 | + structural features (rank_in_turn, turn_size, is_multi_question) | 0.242 | 0.451 | **First to beat baseline clarity** |
| 4 | + allocation features (`allocation.py`) | **0.259** | **0.470** | **BEST CONFIG** |
| 5 | + commitment feature (flat fusion) | 0.244 | 0.457 | Fixed high-overlap bin but hurt aggregate |
| 6 | + commitment hard override | 0.249 | 0.429 | Overcorrected; reverted |
| 7 | reconstruction prompt v2 (strict "prefer empty") | ~0.219 | ~0.420 | Non-Reply recall ↑ but overshot |
| 8 | reconstruction prompt v3 (high bar for non-answer acts) | ~0.218 | ~0.399 | Overcorrected other way |
| 9 | swap backbone 4B → Qwen3.5-9B (best config) | 0.208 | 0.428 | Bigger model did NOT help |

**Best result: row 4 — evasion 0.259 / clarity 0.470** (QUD + structural +
allocation features, learned head, commitment off, original v1 reconstruction
prompt). This is the locked baseline.

---

## 5. Each method in detail

### 5.1 Direct LLM baseline (row 0)
Few-shot chain-of-thought prompting of Qwen3-4B straight to the 9-way label,
clarity derived via taxonomy. Clarity 0.444 / evasion 0.204. Both above chance.
Characteristic failure: **collapse toward the majority "Ambivalent" class** —
Clear-Reply recall 0.23, Clear-Non-Reply recall 0.14; the model dumps both
extremes into Ambivalent. This is exactly the boundary the QUD method targets,
making it an ideal contrast baseline. (Notable: speech-act classes "Declining"
and "Claims ignorance" hit precision 1.0 — directly detectable without QUD
reasoning.)

### 5.2 QUD rule head (row 1)
Zero-parameter taxonomy mapping. Collapsed to evasion 0.142 because the Stage-2
LLM over-emits topic_shift/specification relations → everything maps to
Deflection/Partial. Pure-thesis test; informative but not competitive.

### 5.3 QUD learned head (row 2)
HistGradientBoosting with balanced class weights over the relation features
recovers the dead classes the rule head collapsed (evasion 0.194). Still below
the direct baseline on clarity (0.380) — the base feature set wasn't enough.

### 5.4 Structural features (row 3) — FIRST WIN
Added three turn-structure features computed from `turn_id`/`question_order`
(no LLM): `rank_in_turn`, `turn_size`, `is_multi_question`. Clarity 0.380 →
0.451, evasion 0.194 → 0.242. **Dodging F1 0.257 → 0.388** — the class that
doubles in multi-sibling turns, exactly as the position finding predicted. This
is the cheap proxy that proved the sibling structure is usable signal.

### 5.5 Allocation features (row 4) — BEST
`allocation.py`: per turn, treat the per-sub-question overlap as a bid for the
shared answer's content and compute competitive features — `alloc_share`,
`overlap_rank_in_turn`, `is_best_in_turn`, `sibling_max_overlap`,
`self_minus_sibling`, `won_allocation`. Encodes "a sibling captured the answer,
so this sub-question is starved → evaded." Clarity → 0.470, evasion → 0.259.
Modest lift over structural-only, limited because the features derive from
`qud_overlap`, which is itself imperfectly calibrated.

### 5.6 Commitment signal (rows 5–6) — NEGATIVE RESULT
Diagnosis: `qud_overlap` measures *topicality*, not *answer-resolution*. A
vague/General evasion is fully on-topic yet commits to nothing, so high overlap
was being misread as "answered" (the 0.8–1.0 overlap bin was the worst, ~0.43
accuracy). Added a `commitment` signal — a separate LLM judgment on
(asked question, answer): full / partial / evasive / none (scored 1.0 / 0.6 /
0.25 / 0.0), orthogonal to overlap.
- **As a flat feature (row 5):** fixed the high-overlap bin (0.43 → 0.54) but
  *lowered* aggregate (clarity 0.470 → 0.457) — the balanced learned head let
  the new signal trade off against Deflection/General, which collapsed.
- **As a hard override (row 6):** "high overlap + low commitment + predicted
  reply → relabel General." Overcorrected; clarity dropped to 0.429.
Conclusion: commitment is a real signal (it fixes its target sub-problem) but
neither flat-fusion nor hard-override integrates it well — the 4B commitment
judgments are too noisy to drive a hard rule. Kept behind a config flag
(`use_commitment_features: false`) as a documented negative-result ablation.
This independently rediscovers the shared-task winners' finding that
confidence-conditional routing beats flat fusion.

### 5.7 Reconstruction-prompt calibration (rows 7–8) — NEGATIVE RESULT
Root-cause diagnostic showed Stage-1 was **hallucinating addressed-questions for
true non-replies** (refusals, ignorance claims, rambling), inflating overlap →
Clear Non-Reply misread as Ambivalent. Iterated the prompt:
- **v1 (original):** lenient → Clear-Non-Reply recall 0.28 (misses them).
- **v2 (strict, "prefer empty"):** overshot → over-fired ignorance/decline/
  clarify on vague-but-engaging answers; 92% of the Ambivalent→Non-Reply errors
  were vague answers mislabeled as a non-answer speech act. Non-Reply recall
  jumped to 0.64 but precision cratered to 0.21; clarity 0.420.
- **v3 (high bar for non-answer acts, "hedging = vague answer not ignorance"):**
  overcorrected the other way — Non-Reply recall fell to 0.19; clarity 0.399.
The precision/recall on Clear-Non-Reply is controllable via reconstruction
strictness, but **no variant beat the v1/best aggregate (0.470)**. The class is
bounded by the κ=0.65 annotator-disagreement ceiling. Reverted to v1.
(Operational note: this arc also surfaced several workflow hazards — cache must
bust on prompt change via the system-prompt hash; `git checkout <commit> --
file` reverts the *whole* file and dropped the commitment prompt once; a forgot-
to-`git pull` made a "v3" run silently re-use v2.)

### 5.8 Backbone scaling 4B → 9B (row 9) — NEGATIVE RESULT
Swapped only the LLM (Qwen3-4B → Qwen3.5-9B, newer generation, ~2× params),
everything else at the best config. **Clarity 0.470 → 0.428, evasion 0.259 →
0.208 — bigger model made it worse.** Decisive evidence that the bottleneck is
the *pipeline architecture / task ambiguity*, not model capacity. (Independently
corroborated by teammates' encoder sweep: DeBERTa-xlarge was their best Task 1
at 0.632 but their *worst* Task 2 at 0.267 — scale does not monotonically help.)
Setup notes: Qwen3.5 is a thinking model (`<think>` blocks must be disabled via
`enable_thinking=False`); A100 nodes (AMD EPYC) need a separate env built on the
AMD module stack or the run dies with `Illegal instruction`; the experiment
otherwise runs fine single-GPU on A40 in `qud-env`.

---

## 6. What was NOT the bottleneck (component choices held fixed)

To keep ablations clean, these were held constant; each is a candidate late-
stage ablation but none is the limiting factor:
- Embedding model: `all-mpnet-base-v2` (2021-era; newer embedders exist).
- NLI model: `DeBERTa-v3-base-mnli-fever-anli` (base, not large).
- Encoder baseline (`microsoft/deberta-v3-base`) — confirmed the shared task's
  "encoders saturate ~0.76–0.81 clarity / ~0.50 evasion" ceiling. (The encoder
  run itself required: torch ≥2.6 for the CVE-guarded `.bin` load, fp32 instead
  of bf16 to avoid an immediate NaN explosion in DeBERTa-v3, and dodging a node
  with an uncorrectable ECC GPU fault.)

---

## 7. Headline conclusions

1. **Best system: QUD + structural + allocation features, learned head —
   clarity 0.470 / evasion 0.259** on dev. Beats the direct-LLM baseline
   (0.444 / 0.204).
2. **The novel, validated contribution is the data finding**: political answers
   are allocated front-to-back across sibling sub-questions; evasion rises
   monotonically with position (35% → 92%); multi-sibling turns are ~2× more
   Dodging-heavy. No prior system used this.
3. **Three rigorous negative results** that characterize the ceiling rather than
   just fail: commitment fusion, reconstruction-strictness tuning, and backbone
   scaling all fail to beat 0.470 — establishing that the limit is architectural
   and tied to the κ=0.65 Ambivalent/Non-Reply human-disagreement boundary, not
   model capacity or prompt wording.
4. **Interpretability** is the standing advantage: the method exposes *which*
   sub-question received the answer's content, which the encoder/leaderboard
   systems cannot.

---

## 8. Open / untried directions (as of this log)

- Inject the structural features into a strong supervised encoder
  (DeBERTa-v3-base, teammates' 0.605) — does the sibling signal lift the best
  system? (Cheapest remaining experiment.)
- Hierarchical Task-2→Task-1 derivation (the shared-task *winning* strategy;
  unrun by the team).
- Disagreement / hard-case analysis on the 308-row triple-annotated set
  (course requirement; the natural home for the κ=0.65 ceiling story).
- Full bipartite (Hungarian) answer-allocation model — the proper form of the
  feature-based allocation lower bound used in row 4.
- Late component ablations: stronger embedder, larger NLI, frontier/CoT-distilled
  LLM.
