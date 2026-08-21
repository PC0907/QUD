#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=clarity-qud
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Full QUD pipeline (Stage 1 reconstruction -> Stage 2 relations ->
# Stage 3 classification) via HF transformers (single GPU).
#   sbatch scripts/03_qud_pipeline.sh                  # dev split, rule head
#   sbatch scripts/03_qud_pipeline.sh dev learned      # learned head
#
# Loads three models (Qwen3-4B ~8GB + DeBERTa-base NLI ~0.75GB +
# mpnet embeddings ~0.42GB) — all fit on one A40 (48GB). device_map="auto"
# handles placement; the old vLLM tensor-parallel requirement is gone.
#
# This header targets A40devel (1h cap) for a smoke test. If the full dev
# run needs more time, switch to:  --partition=A40short --time=8:00:00
# (or A40medium for up to 24h). The sqlite prompt cache (outputs/llm_cache.sqlite)
# resumes the run for free after a wall-time kill — just resubmit.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
nvidia-smi
SPLIT=${1:-dev}
HEAD=${2:-rule}
CONFIG=configs/qud_pipeline.yaml
if [[ "$HEAD" == "learned" ]]; then
  TMP="configs/.qud_pipeline_learned.yaml"
  sed 's/^head: rule/head: learned/' "$CONFIG" > "$TMP"
  CONFIG="$TMP"
fi
python -m qud_evasion.cli qud-pipeline --config "$CONFIG" --split "$SPLIT"