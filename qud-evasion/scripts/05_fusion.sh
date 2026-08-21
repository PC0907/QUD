#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=clarity-fusion
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Structural-feature fusion A/B for CLARITY Task 1.
# Runs BOTH arms sequentially so the comparison is one job:
#   sbatch scripts/05_fusion.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
nvidia-smi
echo "===== ARM A: baseline DeBERTa (no structural) ====="
python scripts/fusion_experiment.py --use_structural 0
echo "===== ARM B: DeBERTa + structural features ====="
python scripts/fusion_experiment.py --use_structural 1
echo "===== DONE \u2014 compare the two macro-F1 lines above ====="