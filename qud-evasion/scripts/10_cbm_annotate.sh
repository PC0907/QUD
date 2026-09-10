#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=cbm-annotate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
#
# Stage A of the concept bottleneck: one LLM call per row, train + dev.
# Cached in outputs/llm_cache.sqlite, so re-runs only pay for new rows.
# Submit from the repo root:  sbatch scripts/10_cbm_annotate.sh
# Override the concept set or model:
#   CONCEPTS=v1 MODEL=Qwen/Qwen3-8B sbatch scripts/10_cbm_annotate.sh
set -euo pipefail
unset SLURM_EXPORT_ENV

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"

CONCEPTS="${CONCEPTS:-v0}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUT="outputs/cbm/${CONCEPTS}"
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('cuda:', torch.cuda.is_available())"

for split in dev train; do
    echo "=== annotating ${split} with ${CONCEPTS} / ${MODEL} ==="
    python -m qud_evasion.cbm.annotate \
        --data "data/processed/${split}.parquet" \
        --out  "${OUT}/${split}.jsonl" \
        --model "$MODEL" \
        --concepts "$CONCEPTS" \
        --max-tokens 768 \
        --batch-size 8
done

echo "done. next: python scripts/cbm_train.py --concepts ${CONCEPTS} --head lr"