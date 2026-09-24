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

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env_a100.sh"
set -euo pipefail

CONCEPTS="${1:-v1}"

grep "model name" /proc/cpuinfo | head -1 || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python scripts/justification_faithfulness.py \
    --concepts "$CONCEPTS" \
    --max-per-concept 60 \
    --batch-size 16