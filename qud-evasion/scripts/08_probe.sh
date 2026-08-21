#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=probe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Linear-probe study on a GPU node. Submit:
#   sbatch scripts/08_probe.sh
set -eu
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source scripts/_env.sh
export HF_HOME=/home/s44srizv/MMD2_Project/qud-evasion/.hf_cache
echo "=== GPU ==="
nvidia-smi || true
echo "=== running probe ==="
python scripts/probe_experiment.py