"""Baseline (b): fine-tuned encoder classifier.

DeBERTa-v3 over "question [SEP] answer" with class-weighted cross-entropy.
This reproduces the encoder paradigm from SemEval-2026 Task 6; it is a
comparison system, not a contribution.

Three things changed from the original version, all of them correctness
rather than tuning:

1. Epoch selection now happens on a separate VALIDATION split. Previously
   `load_best_model_at_end` selected the best epoch by dev macro-F1 and the
   reported number was dev macro-F1 from the same 346 rows, i.e. a maximum
   over 10 epochs on the evaluation data rather than an estimate of it.
2. `save_total_limit=1`. Ten epochs of unbounded checkpointing writes
   roughly 15 GB per run; multi-seed runs then fill the disk quota.
3. The official test split is evaluated too. It ships without fine-grained
   evasion gold, so evasion is scored on dev only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, set_seed,
)

from ..data.taxonomy import (
    CLARITY2ID, CLARITY_LABELS, EVASION2ID, EVASION_LABELS,
    ID2CLARITY, ID2EVASION,
)
from ..eval.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


def _target_spec(target: str):
    if target == "evasion":
        return EVASION_LABELS, EVASION2ID, ID2EVASION, "evasion_label"
    return CLARITY_LABELS, CLARITY2ID, ID2CLARITY, "clarity_label"


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def _encode(df: pd.DataFrame, tokenizer, label2id: dict, max_length: int,
            target_col: str, with_labels: bool = True):
    import datasets as hfds

    enc = tokenizer(
        df["question"].tolist(), df["interview_answer"].tolist(),
        truncation="only_second", max_length=max_length,
    )
    if with_labels:
        enc["labels"] = df[target_col].map(label2id).tolist()
    return hfds.Dataset.from_dict(enc)


def train_encoder(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    target: str = "evasion",                     # "evasion" | "clarity"
    model_name: str = "microsoft/deberta-v3-base",
    output_dir: str | Path = "outputs/encoder",
    max_length: int = 512,
    lr: float = 1e-5,
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 4,
    seed: int = 13,
    save_best: bool = False,
) -> dict:
    """Train once and evaluate on dev (and test, if given).

    val_df is used ONLY for epoch selection. If it is None, selection is
    disabled entirely and the final epoch is reported: that is noisier but
    still honest. Selecting on dev_df is never an option here.
    """
    set_seed(seed)
    labels, label2id, id2label, col = _target_spec(target)

    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label={v: k for k, v in label2id.items()}, label2id=label2id,
    )

    raw_weights = compute_class_weight(
        "balanced", classes=np.arange(len(labels)),
        y=train_df[col].map(label2id).to_numpy(),
    )
    # Soften (sqrt) and clip: full inverse-frequency weighting destabilizes
    # DeBERTa fine-tuning when rare classes get enormous weights, collapsing
    # the model to majority-class prediction.
    raw_weights = np.clip(np.sqrt(raw_weights), a_min=None, a_max=5.0)
    weights = torch.tensor(raw_weights, dtype=torch.float)
    logger.info("Class weights (%s): %s", target,
                {labels[i]: round(float(w), 2) for i, w in enumerate(weights)})

    train_ds = _encode(train_df, tokenizer, label2id, max_length, col)
    select_on_val = val_df is not None and len(val_df) > 0
    val_ds = _encode(val_df, tokenizer, label2id, max_length, col) if select_on_val else None

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        pred = logits.argmax(-1)
        m = evaluate_predictions(
            [id2label[i] for i in gold], [id2label[i] for i in pred], labels
        )
        return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="epoch" if select_on_val else "no",
        save_strategy="epoch" if select_on_val else "no",
        save_total_limit=1,
        load_best_model_at_end=select_on_val,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=False,          # DeBERTa-v3 NaNs immediately in bf16 here
        fp16=False,
        max_grad_norm=1.0,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        seed=seed,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, compute_metrics=compute_metrics,
        class_weights=weights,
    )
    trainer.train()
    if save_best:
        trainer.save_model(str(Path(output_dir) / "best"))

    def _predict(df: pd.DataFrame) -> list[str]:
        ds = _encode(df, tokenizer, label2id, max_length, col, with_labels=False)
        logits = trainer.predict(ds).predictions
        return [id2label[int(i)] for i in logits.argmax(-1)]

    result: dict = {"seed": seed, "target": target, "selected_on_val": select_on_val}

    dev_pred = _predict(dev_df)
    result["dev"] = {
        "metrics": evaluate_predictions(dev_df[col].tolist(), dev_pred, labels),
        "predictions": dev_pred,
    }
    logger.info("[seed %d] encoder %s DEV macro-F1: %.4f",
                seed, target, result["dev"]["metrics"]["macro_f1"])

    if select_on_val:
        val_pred = _predict(val_df)
        result["val"] = {
            "metrics": evaluate_predictions(val_df[col].tolist(), val_pred, labels),
        }

    if test_df is not None and len(test_df):
        if test_df[col].notna().all():
            test_pred = _predict(test_df)
            result["test"] = {
                "metrics": evaluate_predictions(test_df[col].tolist(), test_pred, labels),
                "predictions": test_pred,
            }
            logger.info("[seed %d] encoder %s TEST macro-F1: %.4f",
                        seed, target, result["test"]["metrics"]["macro_f1"])
        else:
            # official_test ships without fine-grained evasion gold
            logger.info("[seed %d] %s: no gold in test split, skipping", seed, target)
            result["test"] = {"skipped": "no gold labels for this target in test"}

    del model, trainer
    torch.cuda.empty_cache()
    return result


def predict_encoder(
    df: pd.DataFrame,
    model_dir: str | Path,
    target: str = "evasion",
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> list[str]:
    labels, label2id, id2label, col = _target_spec(target)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, use_safetensors=True,
    ).to(device).eval()

    preds = []
    for s in range(0, len(df), batch_size):
        chunk = df.iloc[s:s + batch_size]
        enc = tokenizer(
            chunk["question"].tolist(), chunk["interview_answer"].tolist(),
            truncation="only_second", max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(id2label[int(i)] for i in logits.argmax(-1).cpu())
    return preds