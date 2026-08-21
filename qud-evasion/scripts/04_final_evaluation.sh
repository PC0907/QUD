#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=4:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=clarity-FINAL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# FINAL evaluation on the official 308-row test split. Per the course
# protocol this runs EXACTLY ONCE, after all decisions are frozen on
# the internal dev split.
#
# Guard: batch jobs have no TTY, so confirmation is via an environment
# variable. Submit deliberately with:
#   sbatch --export=ALL,CONFIRM_FINAL=yes scripts/04_final_evaluation.sh
set -euo pipefail
if [[ "${CONFIRM_FINAL:-}" != "yes" ]]; then
  echo "Refusing to run: this is the one-time official test evaluation."
  echo "Submit with: sbatch --export=ALL,CONFIRM_FINAL=yes \$0"
  exit 1
fi
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
nvidia-smi

python -m qud_evasion.cli llm-baseline --config configs/baseline_llm.yaml --split official_test
python -m qud_evasion.cli qud-pipeline --config configs/qud_pipeline.yaml --split official_test