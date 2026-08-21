#!/usr/bin/env bash
# Shared environment setup, sourced by every job script.
# Bender cluster: no SLURM account needed (unlike Marvin/Bonna).

# --- modules ----------------------------------------------------------
# Module names verified on Bender (module avail Python / module avail CUDA).
# Python must match the interpreter used to build ~/qud-env (3.12.3).
# CUDA 12.1.1 matches the installed torch build (2.5.1+cu121).
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# --- storage ----------------------------------------------------------
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"

# Compute jobs run strictly offline; 00_prepare_data.sh pre-downloads
# data and model weights on the login node (and must override these to 0).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$PROJECT_DIR"

# Activate the project virtual environment (Python 3.12 venv, built against
# the module above). Previous path (~/mmd2) was removed during the restart.
source ~/qud-env/bin/activate

mkdir -p logs outputs
