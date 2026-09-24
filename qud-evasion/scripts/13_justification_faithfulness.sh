#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=cbm-faithfulness
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
#
# Justification faithfulness on the dev split: for each concept answered
# "yes", delete the span its justification cites (vs a control sentence) and
# re-annotate. Submit from the repo root:
#   sbatch scripts/13_justification_faithfulness.sh v1
# The concept set must already be annotated (outputs/cbm/<set>/dev.jsonl).

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
set -euo pipefail

CONCEPTS="${1:-v1}"

grep "model name" /proc/cpuinfo | head -1 || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python scripts/justification_faithfulness.py \
    --concepts "$CONCEPTS" \
    --max-per-concept 60 \
    --batch-size 16