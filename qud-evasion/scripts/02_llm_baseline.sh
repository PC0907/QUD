#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=clarity-llm-baseline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Direct few-shot CoT prompting baseline via HF transformers (single GPU).
#   sbatch scripts/02_llm_baseline.sh            # dev split (default)
#
# Qwen3-4B in bf16 (~8GB weights) fits comfortably on one A40 (48GB).
# device_map="auto" in LLMClient handles placement; the old vLLM
# tensor-parallel multi-GPU requirement no longer applies.
#
# This header targets A40devel (1h cap) for a smoke test. For the full
# dev-split run, switch to:  --partition=A40short --time=8:00:00
#
# Checkpointing: all LLM calls are cached in outputs/llm_cache.sqlite —
# if the job hits the wall, just resubmit; it resumes where it stopped.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
nvidia-smi
SPLIT=${1:-dev}
python -m qud_evasion.cli llm-baseline --config configs/baseline_llm.yaml --split "$SPLIT"