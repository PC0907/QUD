#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cbm-annotate-a100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
#
# Stage A of the concept bottleneck, on the AMD A100 nodes.
# Identical to 10_cbm_annotate.sh except for the partition and the env file.
# Submit from the repo root:  sbatch scripts/11_cbm_annotate_a100.sh v1
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env_a100.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONCEPTS="${1:-v0}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUT="outputs/cbm/${CONCEPTS}"
mkdir -p "$OUT"

echo "=== environment ==="
grep "model name" /proc/cpuinfo | head -1 || true
echo "host   : $(hostname)"
echo "python : $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import qud_evasion, os; print('pkg', os.path.dirname(qud_evasion.__file__))"
echo "=== environment OK ==="

for split in dev train; do
    echo "=== annotating ${split} with ${CONCEPTS} / ${MODEL} ==="
    python -m qud_evasion.cbm.annotate \
        --data "data/processed/${split}.parquet" \
        --out  "${OUT}/${split}.jsonl" \
        --model "$MODEL" \
        --concepts "$CONCEPTS" \
        --max-tokens 1024 \
        --batch-size 24
done

echo "done. next: python scripts/cbm_train.py --concepts ${CONCEPTS} --head lr"