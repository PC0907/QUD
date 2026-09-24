#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cbm-faithfulness-a100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
#
# Justification faithfulness: for each concept answered "yes", delete the span
# its justification cites (vs a control sentence) and re-annotate.
#
# First argument is the concept set; everything after it is passed straight
# through to the Python script, and later flags override the defaults below.
#
#   sbatch scripts/13_justification_faithfulness.sh v1
#   sbatch scripts/13_justification_faithfulness.sh v1 \
#       --data data/processed/train.parquet \
#       --ann outputs/cbm/v1/train.jsonl \
#       --out outputs/cbm/v1/faithfulness_train \
#       --concepts-to-test p2 p7 p8 p9 p10 \
#       --max-per-concept 150

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env_a100.sh"
set -euo pipefail

CONCEPTS="${1:-v1}"

echo "=== environment ==="
grep "model name" /proc/cpuinfo | head -1 || true
echo "host : $(hostname)"
echo "args : ${*}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== environment OK ==="

python scripts/justification_faithfulness.py \
    --concepts "$CONCEPTS" \
    --max-per-concept 60 \
    --batch-size 16 \
    "${@:2}"