#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# mmd2 project environment — QUD-based evasion detection (CLARITY/QEvasion)
#
# This script must be SOURCED, not executed, so that the module loads, venv
# activation, and cd persist in your current shell:
#
#     source ~/MMD2_Project/qud-evasion/activate_mmd2.sh
#
# Or use the `mmd2` alias (see install note at the bottom) and just type:
#
#     mmd2
# ---------------------------------------------------------------------------

# --- project location ---
export MMD2_DIR="$HOME/MMD2_Project/qud-evasion"

# --- modules (versions matched to the venv + installed torch build) ---
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# --- virtual environment ---
source "$HOME/qud-env/bin/activate"

# --- HuggingFace cache (local, under the repo; keeps home quota happy) ---
export HF_HOME="$MMD2_DIR/.hf_cache"

# --- land in the project directory ---
cd "$MMD2_DIR" || return 1

# --- friendly status so you know it worked ---
echo "── mmd2 environment ready ──────────────────────────────"
echo "  python : $(which python)"
echo "  cwd    : $(pwd)"
echo "  HF_HOME: $HF_HOME"
echo "  GPU jobs: sbatch scripts/02_llm_baseline.sh | scripts/03_qud_pipeline.sh"
echo "────────────────────────────────────────────────────────"