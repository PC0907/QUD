#!/bin/bash
# Generates the 6 prompting-sweep configs from the base baseline_llm.yaml.
# Run from repo root:  bash scripts/make_prompt_configs.sh
set -eu
cd "$(dirname "$0")/.."
BASE=configs/baseline_llm.yaml

mk () {  # name, prompt_style, few_shot, k
  local name="$1" style="$2" fs="$3" k="$4"
  local out="configs/prompt_${name}.yaml"
  # copy base, then override/append the three knobs
  grep -vE '^(prompt_style|few_shot|k_per_class):' "$BASE" > "$out"
  {
    echo "prompt_style: ${style}"
    echo "few_shot: ${fs}"
    echo "k_per_class: ${k}"
    echo "output_dir: outputs/prompt_${name}"
  } >> "$out"
  echo "wrote $out"
}

#    name              style          few_shot  k
mk   flat_k1           flat           true      1     # = your existing baseline (control)
mk   hier_k1           hierarchical   true      1
mk   flat_k3           flat           true      3
mk   hier_k3           hierarchical   true      3     # best-bet (ttda704: hier + more shots)
mk   flat_k0           flat           false     0     # zero-shot flat
mk   hier_k0           hierarchical   false     0     # zero-shot hierarchical