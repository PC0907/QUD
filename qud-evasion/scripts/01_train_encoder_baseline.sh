#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=clarity-encoder
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Fine-tuned DeBERTa-v3-base baseline (evasion + clarity heads).
# Model name comes from configs/baseline_encoder.yaml (microsoft/deberta-v3-base).
# Fits on a single A40. Submit from the repo root: sbatch scripts/01_train_encoder_baseline.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
nvidia-smi
python -m qud_evasion.cli train-encoder --config configs/baseline_encoder.yaml

