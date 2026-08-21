#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=prompt-sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Prompting-strategy sweep on the direct LLM baseline (Qwen-4B).
# Runs all 6 variants on dev. Each ~few min (cached after first gen).
#   sbatch scripts/07_prompt_sweep.sh
set -eu
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source scripts/_env.sh
export HF_HOME=/home/s44srizv/MMD2_Project/qud-evasion/.hf_cache

echo "=== GPU ==="
nvidia-smi || true

# the three hier configs — these will now actually use the hierarchical prompt
for name in hier_k1 hier_k3 hier_k0; do
  python -m qud_evasion.cli llm-baseline --config configs/prompt_${name}.yaml --split dev
done

echo "############## SWEEP DONE ##############"
echo "=== clarity / evasion macro-F1 per variant ==="
for name in flat_k1 hier_k1 flat_k3 hier_k3 flat_k0 hier_k0; do
  m="outputs/prompt_${name}/dev_metrics.json"
  if [ -f "$m" ]; then
    python - "$m" "$name" << 'PY' || true
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get("clarity", {}).get("macro_f1", float("nan"))
e = d.get("evasion", {}).get("macro_f1", float("nan"))
print(f"  {sys.argv[2]:10s}  clarity={c:.4f}  evasion={e:.4f}")
PY
  fi
done