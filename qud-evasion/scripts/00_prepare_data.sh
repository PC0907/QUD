#!/usr/bin/env bash
# Run this ON THE LOGIN NODE (it needs internet, uses no GPU, and is
# light enough not to bother anyone).
#
# It (1) downloads QEvasion and builds the leak-free interview-level
# train/dev split, and (2) pre-fetches all model weights into HF_HOME
# so that compute-node jobs can run with HF_HUB_OFFLINE=1.
set -euo pipefail
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
source "$(dirname "$0")/_env.sh"

python -m qud_evasion.cli prepare-data --config configs/base.yaml

echo "Pre-fetching model weights into $HF_HOME ..."
python - << 'PYEOF'
import yaml
from huggingface_hub import snapshot_download

models = set()
for cfg_file in ["configs/baseline_encoder.yaml", "configs/baseline_llm.yaml",
                 "configs/qud_pipeline.yaml"]:
    cfg = yaml.safe_load(open(cfg_file))
    for key in ["encoder_model", "llm_model", "embedding_model", "nli_model"]:
        if cfg.get(key):
            models.add(cfg[key])

for m in sorted(models):
    print(f"  -> {m}")
    snapshot_download(m)
print("Done. Jobs can now run offline.")
PYEOF
