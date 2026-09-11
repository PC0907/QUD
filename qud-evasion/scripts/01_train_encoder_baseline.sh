#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=clarity-encoder
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
#
# Fine-tuned DeBERTa-v3-base baseline (evasion + clarity heads), multi-seed.
# Model, seeds, and epochs come from configs/baseline_encoder.yaml.
# Submit from the repo root:  sbatch scripts/01_train_encoder_baseline.sh
#
# Ten epochs x 5 seeds x 2 targets is the long pole here; save_total_limit=1
# keeps the disk cost bounded regardless.

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
set -euo pipefail

echo "=== environment ==="
grep "model name" /proc/cpuinfo | head -1 || true
echo "host   : $(hostname)"
echo "python : $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== environment OK ==="

python -m qud_evasion.cli train-encoder --config configs/baseline_encoder.yaml