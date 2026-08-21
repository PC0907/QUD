#!/usr/bin/env python
"""Linear-probe study: where in Qwen3-4B is the Ambiguous vs Clear Non-Reply
distinction linearly decodable?

For each layer, extract the last-token hidden state for every Q-A pair, train a
logistic-regression probe on the binary (Ambiguous vs Clear Non-Reply) task, and
report macro-F1 by layer. The curve shows at which depth (if any) the hardest
clarity boundary becomes linearly available.

  python scripts/probe_experiment.py
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CLARITY_ALIASES = {"Ambiguous": "Ambivalent", "Ambivalent Reply": "Ambivalent"}
# the hard boundary: Ambiguous (=Ambivalent) vs Clear Non-Reply
POS, NEG = "Ambivalent", "Clear Non-Reply"
OUT = "outputs/probe"
SEED = 42
MAX_LEN = 512
BATCH = 16


def normalize(v):
    v = str(v).strip()
    return CLARITY_ALIASES.get(v, v)


@torch.no_grad()
def extract_layerwise_lasttoken(texts, tok, model, device):
    """Return array [n_examples, n_layers, hidden] of last-token states."""
    all_feats = []
    for s in range(0, len(texts), BATCH):
        batch = texts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=MAX_LEN, padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple(n_layers+1) each [B, T, H]
        hs = torch.stack(out.hidden_states, dim=1)  # [B, L, T, H]
        # last non-pad token index per example
        lengths = enc["attention_mask"].sum(dim=1) - 1  # [B]
        idx = lengths.view(-1, 1, 1, 1).expand(-1, hs.size(1), 1, hs.size(3))
        last = hs.gather(2, idx).squeeze(2)  # [B, L, H]
        all_feats.append(last.float().cpu().numpy())
        print(f"  extracted {min(s+BATCH, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(all_feats, axis=0)  # [N, L, H]


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data: keep only the two hard-boundary classes
    df = load_dataset("ailsntua/QEvasion")["train"].to_pandas()
    df["clarity"] = df["clarity_label"].map(normalize)
    df = df[df["clarity"].isin([POS, NEG])].reset_index(drop=True)
    df["y"] = (df["clarity"] == POS).astype(int)
    print(f"examples: {len(df)}  ({POS}={df['y'].sum()}, {NEG}={(1-df['y']).sum()})")

    texts = [f"Question: {q} Answer: {a}"
             for q, a in zip(df["question"].fillna(""),
                             df["interview_answer"].fillna(""))]

    # ---- model
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True).to(device)
    model.eval()

    # ---- extract activations for all layers
    print("extracting activations...")
    feats = extract_layerwise_lasttoken(texts, tok, model, device)  # [N, L, H]
    n, n_layers, hidden = feats.shape
    print(f"feature tensor: {feats.shape}")

    # ---- interview-level split (no leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["url"]))
    y = df["y"].to_numpy()

    # ---- probe per layer
    results = []
    for L in range(n_layers):
        Xtr, Xte = feats[tr, L, :], feats[te, L, :]
        # standardize on train
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        macro = f1_score(y[te], pred, average="macro")
        results.append({"layer": L, "macro_f1": float(macro)})
        print(f"  layer {L:2d}  macro-F1 = {macro:.4f}")

    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nBEST: layer {best['layer']} macro-F1 = {best['macro_f1']:.4f}")
    with open(f"{OUT}/probe_ambiguous_vs_nonreply.json", "w") as f:
        json.dump({"model": MODEL, "pos": POS, "neg": NEG,
                   "n": int(n), "n_layers": int(n_layers),
                   "results": results, "best": best}, f, indent=2)
    print(f"saved {OUT}/probe_ambiguous_vs_nonreply.json")


if __name__ == "__main__":
    main()