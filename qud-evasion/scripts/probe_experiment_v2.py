#!/usr/bin/env python
"""Linear-probe study (v2): multi-seed, two probing targets, cached activations.

Targets:
  task1  : binary  Ambivalent vs Clear Non-Reply        (the kappa=0.65 boundary)
  task2  : 4-class  Dodging / Deflection / General / Implicit
           (the proximal evasion cluster the organizers flag as hardest)

For each layer of Qwen3-4B we take the last-token hidden state, train a
logistic-regression probe, and report macro-F1, averaged over 3 interview-level
splits (seeds 42, 13, 7). Activations are extracted once and cached to disk, so
both targets and all seeds reuse the same forward pass.

  python scripts/probe_experiment_v2.py --target task1
  python scripts/probe_experiment_v2.py --target task2
"""
from __future__ import annotations

import argparse
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
EVASION_CLUSTER = ["Dodging", "Deflection", "General", "Implicit"]
SEEDS = [42, 13, 7]
OUT = "outputs/probe"
CACHE = f"{OUT}/acts_cache.npz"
MAX_LEN = 512
BATCH = 16


def normalize_clarity(v):
    v = str(v).strip()
    return CLARITY_ALIASES.get(v, v)


@torch.no_grad()
def extract(texts, tok, model, device):
    feats = []
    for s in range(0, len(texts), BATCH):
        enc = tok(texts[s:s + BATCH], return_tensors="pt", truncation=True,
                  max_length=MAX_LEN, padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)          # [B, L, T, H]
        lengths = enc["attention_mask"].sum(dim=1) - 1
        idx = lengths.view(-1, 1, 1, 1).expand(-1, hs.size(1), 1, hs.size(3))
        last = hs.gather(2, idx).squeeze(2)                 # [B, L, H]
        feats.append(last.float().cpu().numpy())
        if s % (BATCH * 20) == 0:
            print(f"  extracted {min(s+BATCH, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(feats, axis=0)                    # [N, L, H]


def load_or_build_cache():
    """Build activations for ALL rows once (both targets are subsets)."""
    df = load_dataset("ailsntua/QEvasion")["train"].to_pandas()
    df["clarity"] = df["clarity_label"].map(normalize_clarity)
    df["evasion"] = df["evasion_label"].astype(str).str.strip()
    texts = [f"Question: {q} Answer: {a}"
             for q, a in zip(df["question"].fillna(""), df["interview_answer"].fillna(""))]

    if os.path.exists(CACHE):
        print(f"loading cached activations from {CACHE}")
        z = np.load(CACHE, allow_pickle=True)
        return df, z["feats"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, output_hidden_states=True).to(device).eval()
    print("extracting activations for all rows (one-time)...")
    feats = extract(texts, tok, model, device)              # [N, L, H]
    print(f"feature tensor: {feats.shape}; caching to {CACHE}")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(CACHE, feats=feats)
    return df, feats


def subset(df, feats, target):
    if target == "task1":
        m = df["clarity"].isin(["Ambivalent", "Clear Non-Reply"])
        sub = df[m].reset_index(drop=True)
        y = (sub["clarity"] == "Ambivalent").astype(int).to_numpy()
        labels = ["Clear Non-Reply", "Ambivalent"]
    elif target == "task2":
        m = df["evasion"].isin(EVASION_CLUSTER)
        sub = df[m].reset_index(drop=True)
        y = sub["evasion"].map({c: i for i, c in enumerate(EVASION_CLUSTER)}).to_numpy()
        labels = EVASION_CLUSTER
    else:
        raise ValueError(target)
    X = feats[m.to_numpy()]                                  # [n, L, H]
    return sub, X, y, labels


def probe_all_layers(sub, X, y, seed):
    n_layers = X.shape[1]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(sub, groups=sub["url"]))
    out = []
    for L in range(n_layers):
        Xtr, Xte = X[tr, L, :], X[te, L, :]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr])
        out.append(f1_score(y[te], clf.predict(Xte), average="macro"))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["task1", "task2"], required=True)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df, feats = load_or_build_cache()
    sub, X, y, labels = subset(df, feats, args.target)
    counts = {l: int((y == i).sum()) for i, l in enumerate(labels)}
    print(f"[{args.target}] examples: {len(sub)}  classes: {counts}")

    curves = np.stack([probe_all_layers(sub, X, y, s) for s in SEEDS], axis=0)  # [seeds, L]
    mean, std = curves.mean(0), curves.std(0)
    best = int(mean.argmax())

    print(f"\nlayer  mean_F1  std")
    for L in range(len(mean)):
        print(f"  {L:2d}   {mean[L]:.4f}  {std[L]:.4f}")
    print(f"\nBEST: layer {best}  macro-F1 = {mean[best]:.4f} +/- {std[best]:.4f}")

    res = {"target": args.target, "labels": labels, "counts": counts,
           "seeds": SEEDS, "best_layer": best,
           "mean": mean.tolist(), "std": std.tolist()}
    with open(f"{OUT}/probe_{args.target}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {OUT}/probe_{args.target}.json")


if __name__ == "__main__":
    main()