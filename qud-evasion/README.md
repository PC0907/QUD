# QUD-Based Political Question Evasion Detection

Code for the project **Political Clarity and Evasion Detection** on the
CLARITY task ([QEvasion dataset](https://huggingface.co/datasets/ailsntua/QEvasion),
Thomas et al., 2024; SemEval-2026 Task 6).

## Idea

Every evasion category in the CLARITY taxonomy is, formally, a **relation
between two questions**: the question that was *asked* and the question the
response *actually addresses* — its Question Under Discussion (QUD).
Instead of classifying question–answer pairs directly, we:

1. **Reconstruct** (Stage 1): given *only the answer*, an LLM infers which
   question(s) the answer would directly answer, and which speech act it
   performs (answer / decline / claims ignorance / asks for clarification).
   Withholding the asked question prevents anchoring.
2. **Relate** (Stage 2): compute the structured relation between the asked
   question and each reconstructed QUD — *equivalent*, *specification*,
   *generalization*, *topic shift*, *unrelated* — plus a continuous overlap
   score, supplemented by NLI and embedding features.
3. **Classify** (Stage 3): read the evasion label off the relation pattern,
   either with zero-parameter taxonomy rules or a light learned head.
   The clarity label follows deterministically from the taxonomy hierarchy
   (joint learning of both levels for free).

| QUD relation of addressed vs. asked | Evasion label |
|---|---|
| equivalent (+ stated in requested form) | Explicit |
| equivalent (inferable only) | Implicit |
| specification (one facet answered) | Partial/half-answer |
| generalization (only generic info) | General |
| topic shift (different subject/agent) | Deflection |
| unrelated | Dodging |
| none + decline / ignorance / clarify | Declining / Claims ignorance / Clarification |

Reconstructed QUDs double as human-readable rationales, and the overlap
score gives a continuous handle on the hardest boundary in the data
(Ambivalent vs. Clear Non-Reply, κ = 0.65 between human annotators).

## Repository structure

```
qud-evasion/
├── README.md
├── pyproject.toml             # installable package + console script
├── requirements.txt
├── configs/                   # YAML experiment configs (inherit base.yaml)
│   ├── base.yaml
│   ├── baseline_encoder.yaml
│   ├── baseline_llm.yaml
│   └── qud_pipeline.yaml
├── scripts/                   # numbered, GPU-server-ready (SLURM headers included)
│   ├── 00_prepare_data.sh
│   ├── 01_train_encoder_baseline.sh
│   ├── 02_llm_baseline.sh
│   ├── 03_qud_pipeline.sh
│   └── 04_final_evaluation.sh # one-time official test run (guarded)
├── src/qud_evasion/
│   ├── cli.py                 # single entry point: python -m qud_evasion.cli
│   ├── data/
│   │   ├── load.py            # HF download, label normalization, turn/interview IDs
│   │   ├── splits.py          # leak-free interview-level stratified split
│   │   └── taxonomy.py        # labels, evasion→clarity map, QUD relation rules
│   ├── baselines/
│   │   ├── encoder.py         # DeBERTa-v3 fine-tuning (weighted CE)
│   │   └── llm_prompting.py   # direct few-shot CoT prompting
│   ├── qud/
│   │   ├── prompts.py         # all prompt templates (the scientific core)
│   │   ├── llm_client.py      # vLLM batch inference + sqlite prompt cache
│   │   ├── reconstruct.py     # Stage 1
│   │   ├── relations.py       # Stage 2 (LLM judgment, NLI, embeddings)
│   │   └── classify.py        # Stage 3 (rule head + learned head)
│   ├── eval/
│   │   ├── metrics.py         # Macro-F1 (primary), per-class, any-annotator rule
│   │   └── hard_cases.py      # Ambivalent ↔ Clear Non-Reply boundary analysis
│   └── utils/                 # config inheritance, logging, seeding
├── tests/                     # fast unit tests (no GPU / no network)
├── notebooks/                 # exploratory analysis only
└── outputs/                   # gitignored artifacts (models, predictions, metrics)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q          # sanity check (CPU-only, < 5 s)
```

Requires Python ≥ 3.10 and, for the LLM stages, CUDA GPUs (vLLM).
Set `tensor_parallel` in `configs/*.yaml` to your GPU count and swap
`llm_model` for whatever is available locally.

## Reproduction pipeline (Uni Bonn HPC / Marvin)

The job scripts target the [Marvin cluster](https://wiki.hpc.uni-bonn.de/running_jobs):
A100 partitions (`sgpu_*`), `--gpus` for GPU requests, and the mandatory
`--account` for your group (find yours with `sshare -U`). Configure once in
`scripts/_env.sh` (account, modules, storage paths) and in the `#SBATCH
--account=` lines, then — **submitting from the repo root**:

```bash
# login node (needs internet; downloads data + pre-fetches all model weights):
bash scripts/00_prepare_data.sh

# compute jobs (run fully offline via HF_HUB_OFFLINE=1):
sbatch scripts/01_train_encoder_baseline.sh          # baseline (b): DeBERTa-v3, 1x A100
sbatch scripts/02_llm_baseline.sh dev                # baseline (a): direct CoT, 4x A100
sbatch scripts/03_qud_pipeline.sh dev                # ours, rule head
sbatch scripts/03_qud_pipeline.sh dev learned        # ours, learned head

# ONE-TIME official test run (env-var guard replaces a TTY prompt):
sbatch --export=ALL,CONFIRM_FINAL=yes scripts/04_final_evaluation.sh
```

Useful while running: `squeue --me`, `scontrol show job <id>`,
`seff <id>` (efficiency after completion), logs land in `logs/`.
For quick prompt iterations use the 1-hour devel queue:
`sbatch --partition=sgpu_devel --time=00:59:00 scripts/03_qud_pipeline.sh`.

Each run writes `*_predictions.jsonl` and `*_metrics.json` under
`outputs/<system>/`. All LLM calls are cached in `outputs/llm_cache.sqlite`
keyed by (model, prompt) — this doubles as **checkpointing**: if a job hits
the partition wall time, resubmitting resumes where it stopped, and head
ablations reuse Stage 1/2 outputs at zero GPU cost.

## Data protocol (course requirements)

- The official HF **train** split (3,448 rows) is divided into an internal
  **train** and **dev** split at the **interview level** (sibling
  sub-questions share answer text; a random row split leaks).
- The official HF **test** split (308 rows) is loaded once during data
  preparation and then **evaluated exactly once at the end**
  (`scripts/04_final_evaluation.sh` has an interactive guard).
- Primary metric: **Macro-F1**, for both clarity (3-way) and evasion (9-way).
- Hard-case analysis (Ambivalent vs. Clear Non-Reply) is produced
  automatically in every `*_metrics.json` (`hard_cases` key), including
  the QUD-overlap-vs-error breakdown.

## Systems compared

| System | Module | Role |
|---|---|---|
| Direct few-shot CoT prompting | `baselines/llm_prompting.py` | dominant SemEval-2026 strategy class, same backbone as ours |
| Fine-tuned DeBERTa-v3-large | `baselines/encoder.py` | encoder paradigm (saturates ≈ 0.81 / 0.50) |
| QUD pipeline, rule head | `qud/` | ours — zero-parameter test of the QUD hypothesis |
| QUD pipeline, learned head | `qud/` | ours — relation features → HGB classifier |

Planned ablations: Stage 1 with vs. without question withholding;
rule vs. learned head; relation features added to the encoder;
max_quds ∈ {1, 3}.

## References

- Thomas et al. (2024). *"I Never Said That": A dataset, taxonomy and
  baselines on response clarity classification.* Findings of EMNLP 2024.
- Thomas et al. (2026). *SemEval-2026 Task 6: CLARITY — Unmasking
  Political Question Evasions.* arXiv:2603.14027.
- Roberts (1996/2012). *Information structure in discourse: Towards an
  integrated formal theory of pragmatics* (QUD framework).
- Louis et al. (2020). *"I'd rather just go to bed": Understanding
  Indirect Answers.* EMNLP 2020 (Circa dataset — candidate transfer source).

## License / data note

The QEvasion dataset is distributed under **CC BY-NC-ND 4.0**; this
repository contains code only and downloads the data from Hugging Face at
runtime. Model predictions analyze response clarity as a discourse
phenomenon and are not judgments of any speaker's honesty.
