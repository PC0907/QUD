#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cbm-dim-a100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
#
# Difference-in-means concept directions inside the annotator + causal tests.
# First argument is the concept set; everything after it passes through.
#
#   sbatch scripts/14_dim_directions.sh v1 --concept p7
#   sbatch scripts/14_dim_directions.sh v1 --concept p8

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

python scripts/dim_directions.py --concepts "$CONCEPTS" "${@:2}"