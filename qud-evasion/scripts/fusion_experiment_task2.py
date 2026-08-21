#!/usr/bin/env python
"""Structural-feature fusion experiment for EVASION (Task 2, 9-way).

Same design as the Task 1 clarity fusion, retargeted to the fine-grained
evasion label where the sibling-position signal should matter most (the coarse
clarity metric washed it out; evasion shouldn't). Clean A/B via
--use_structural {0,1}; --seed for multi-seed runs.

  python scripts/fusion_experiment_task2.py --use_structural 0 --seed 42
  python scripts/fusion_experiment_task2.py --use_structural 1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModel, AutoTokenizer, Trainer, TrainingArguments,
    EarlyStoppingCallback,
)

# 9-way evasion taxonomy (canonical order)
EVASION_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Partial/half-answer", "Declining to answer", "Claims ignorance",
    "Clarification",
]
LABEL2ID = {l: i for i, l in enumerate(EVASION_LABELS)}

# evasion label surface-variant normalization (mirror taxonomy.py)
EVASION_ALIASES = {
    "Partial": "Partial/half-answer",
    "Half-answer": "Partial/half-answer",
    "Declining": "Declining to answer",
    "Ignorance": "Claims ignorance",
}

STRUCT_COLS = ["rank_in_turn", "turn_size", "is_multi_question"]


def normalize_evasion(v):
    v = str(v).strip()
    return EVASION_ALIASES.get(v, v)


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["turn_key"] = df["url"].astype(str) + "||" + df["interview_answer"].astype(str)
    df["rank_in_turn"] = df.groupby("turn_key")["question_order"].rank(method="first")
    size = df.groupby("turn_key")["question_order"].transform("size")
    df["turn_size"] = size.astype(float)
    df["is_multi_question"] = (size > 1).astype(float)
    return df


def standardize(train_feats, *others):
    mu = train_feats.mean(axis=0, keepdims=True)
    sd = train_feats.std(axis=0, keepdims=True) + 1e-6
    out = [(train_feats - mu) / sd]
    for o in others:
        out.append((o - mu) / sd)
    return out


class FusionClassifier(nn.Module):
    def __init__(self, model_name, num_labels, n_struct, use_structural, class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        h = self.encoder.config.hidden_size
        dp = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dp)
        self.use_structural = use_structural
        if use_structural:
            self.struct_proj = nn.Sequential(nn.Linear(n_struct, 32), nn.ReLU(), nn.Dropout(dp))
            self.classifier = nn.Linear(h + 32, num_labels)
        else:
            self.classifier = nn.Linear(h, num_labels)
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None,
            persistent=False,
        )

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, struct_features=None, labels=None, **kw):
        enc_in = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            enc_in["token_type_ids"] = token_type_ids
        out = self.encoder(**enc_in)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        if self.use_structural:
            pooled = torch.cat([pooled, self.struct_proj(struct_features)], dim=-1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits, labels,
                weight=self.class_weights.to(logits.dtype) if self.class_weights is not None else None,
            )
        return {"loss": loss, "logits": logits}


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, struct, labels):
        self.enc, self.struct, self.labels = enc, struct, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["struct_features"] = self.struct[i]
        item["labels"] = self.labels[i]
        return item


def collate(tokenizer):
    def _fn(batch):
        keys = [k for k in batch[0] if k not in ("struct_features", "labels")]
        maxlen = max(len(b["input_ids"]) for b in batch)
        out = {}
        pad_id = tokenizer.pad_token_id
        for k in keys:
            seqs = []
            for b in batch:
                seq = b[k]
                fill = pad_id if k == "input_ids" else 0
                seqs.append(seq + [fill] * (maxlen - len(seq)))
            out[k] = torch.tensor(seqs, dtype=torch.long)
        out["struct_features"] = torch.tensor(
            np.stack([b["struct_features"] for b in batch]), dtype=torch.float)
        out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return out
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_structural", type=int, default=0, choices=[0, 1])
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weighted", type=int, default=1, choices=[0, 1],
                    help="evasion is severely imbalanced; weighted CE on by default")
    ap.add_argument("--out", default="outputs/fusion_task2")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw = load_dataset("ailsntua/QEvasion")["train"].to_pandas()
    raw = add_structural_features(raw)
    raw["evasion"] = raw["evasion_label"].map(normalize_evasion)
    raw = raw[raw["evasion"].isin(EVASION_LABELS)].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    tr_idx, dv_idx = next(gss.split(raw, groups=raw["url"]))
    train_df, dev_df = raw.iloc[tr_idx].copy(), raw.iloc[dv_idx].copy()
    print(f"[seed {args.seed}] train={len(train_df)} dev={len(dev_df)} "
          f"interviews {train_df['url'].nunique()}/{dev_df['url'].nunique()}")

    tr_s = train_df[STRUCT_COLS].to_numpy(dtype=float)
    dv_s = dev_df[STRUCT_COLS].to_numpy(dtype=float)
    tr_s, dv_s = standardize(tr_s, dv_s)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    def texts(df):
        return [f"Question: {q} Answer: {a}"
                for q, a in zip(df["question"].fillna(""), df["interview_answer"].fillna(""))]

    def enc(df):
        e = tok(texts(df), truncation=True, max_length=args.max_len, padding=False)
        return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}

    tr_enc, dv_enc = enc(train_df), enc(dev_df)
    tr_y = train_df["evasion"].map(LABEL2ID).to_numpy()
    dv_y = dev_df["evasion"].map(LABEL2ID).to_numpy()

    # class weights (inverse frequency) — evasion is severely imbalanced
    cw = None
    if args.class_weighted:
        counts = np.bincount(tr_y, minlength=len(EVASION_LABELS)).astype(float)
        cw = (counts.sum() / (len(EVASION_LABELS) * np.maximum(counts, 1))).tolist()

    train_ds, dev_ds = DS(tr_enc, tr_s, tr_y), DS(dv_enc, dv_s, dv_y)
    model = FusionClassifier(args.model, len(EVASION_LABELS), len(STRUCT_COLS),
                             bool(args.use_structural), class_weights=cw).float()

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    targs = TrainingArguments(
        output_dir=f"{args.out}/struct{args.use_structural}_seed{args.seed}",
        learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.accum,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1,
        fp16=False, bf16=False, max_grad_norm=1.0,
        warmup_ratio=0.06, weight_decay=0.01, logging_steps=50,
        report_to="none", seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collate(tok), compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    pred = trainer.predict(dev_ds)
    preds = np.argmax(pred.predictions, axis=1)
    macro = f1_score(dv_y, preds, average="macro")
    print(f"\n===== TASK2 use_structural={args.use_structural} seed={args.seed}  "
          f"dev evasion macro-F1 = {macro:.4f} =====")
    print(classification_report(dv_y, preds, labels=list(range(len(EVASION_LABELS))),
                                target_names=EVASION_LABELS, digits=3, zero_division=0))

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/result_struct{args.use_structural}_seed{args.seed}.json", "w") as f:
        json.dump({"task": "evasion", "use_structural": args.use_structural,
                   "seed": args.seed, "macro_f1": float(macro),
                   "report": classification_report(
                       dv_y, preds, labels=list(range(len(EVASION_LABELS))),
                       target_names=EVASION_LABELS, output_dict=True, zero_division=0)},
                  f, indent=2)


if __name__ == "__main__":
    main()