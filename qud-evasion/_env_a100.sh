#!/usr/bin/env bash
# Shared environment for A100 (AMD EPYC) nodes on Bender.
# Use ONLY in jobs submitted with --export=NONE.
#
# The A40 nodes and the login node are Intel; the A100 nodes are AMD, and
# ~/qud-env is built against the Intel module stack, so importing torch there
# dies with "Illegal instruction". This script uses the AMD module tree and
# the AMD-side venv built for NLP_Lab instead.
#
# We deliberately do NOT `pip install -e .` into that venv: the package is
# made importable via PYTHONPATH, so nothing in the shared A100 environment
# is modified and NLP_Lab cannot be broken by this project.

source /etc/profile

# Force the AMD stack explicitly rather than relying on auto-switching.
module unuse /software/easybuild-INTEL_A40/modules/all 2>/dev/null || true
module use   /software/easybuild-AMD_A100/modules/all

module load Python/3.12.3
module load CUDA/12.4.0

source ~/nlp_lab_a100/bin/activate

# --- storage ----------------------------------------------------------
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"

# Compute jobs run strictly offline; 00_prepare_data.sh pre-downloads
# weights on the login node.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$PROJECT_DIR"
mkdir -p logs outputs