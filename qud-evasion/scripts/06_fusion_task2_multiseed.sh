#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=fusion-t2-seeds
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Multi-seed Task 2 (evasion) fusion A/B. Seeds 13 and 7, both arms each.
# Deletes each run's checkpoint dir after its result JSON is saved, so the
# sweep stays under the tight home quota. Submit (exclude the ECC-bad node):
#   sbatch --exclude=<badnode> scripts/06_fusion_task2_multiseed.sh
set -eu
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source scripts/_env.sh
export HF_HOME=/home/s44srizv/MMD2_Project/qud-evasion/.hf_cache

echo "=== GPU ==="
nvidia-smi || true

for s in 13 7; do
  for arm in 0 1; do
    echo "############## SEED $s : struct=$arm ##############"
    python scripts/fusion_experiment_task2.py --use_structural "$arm" --seed "$s"
    # result JSON is saved; drop the heavy checkpoint dir to free quota
    rm -rf "outputs/fusion_task2/struct${arm}_seed${s}" || true
    quota -s || true
  done
done

echo "############## ALL SEEDS DONE ##############"
ls -1 outputs/fusion_task2/result_struct*_seed*.json || true
echo
echo "=== Dodging F1 + macro per run ==="
python - << 'PY' || true
import json, glob
rows = []
for f in sorted(glob.glob("outputs/fusion_task2/result_struct*_seed*.json")):
    d = json.load(open(f))
    dod = d["report"].get("Dodging", {}).get("f1-score", float("nan"))
    rows.append((d["use_structural"], d["seed"], d["macro_f1"], dod))
    print(f"  struct={d['use_structural']} seed={d['seed']}  "
          f"macroF1={d['macro_f1']:.4f}  Dodging_F1={dod:.4f}")
# quick mean delta on Dodging if both arms present per seed
import statistics as st
by = {}
for u, s, m, dod in rows:
    by.setdefault(s, {})[u] = dod
deltas = [by[s][1] - by[s][0] for s in by if 0 in by[s] and 1 in by[s]]
if deltas:
    print(f"\n  Dodging F1 delta (struct1 - struct0) per seed: "
          f"{[round(x,4) for x in deltas]}")
    print(f"  mean delta = {st.mean(deltas):.4f}"
          + (f"  std = {st.pstdev(deltas):.4f}" if len(deltas) > 1 else ""))
PY