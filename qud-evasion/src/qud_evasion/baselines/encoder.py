"""Baseline (b): fine-tuned encoder classifier.

DeBERTa-v3 over "question [SEP] answer" with class-weighted
cross-entropy. This reproduces the encoder paradigm that saturated at
~0.81 (clarity) / ~0.50 (evasion) in SemEval-2026 Task 6; it is a
comparison system, not a contribution.
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
            target_col: str):
    import datasets as hfds

    enc = tokenizer(
        df["question"].tolist(), df["interview_answer"].tolist(),
        truncation="only_second", max_length=max_length,
    )
    enc["labels"] = df[target_col].map(label2id).tolist()
    return hfds.Dataset.from_dict(enc)


def train_encoder(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    target: str = "evasion",                     # "evasion" | "clarity"
    model_name: str = "microsoft/deberta-v3-base",
    output_dir: str | Path = "outputs/encoder",
    max_length: int = 512,
    lr: float = 1e-5,
    epochs: int = 5,
    batch_size: int = 8,
    grad_accum: int = 4,
    seed: int = 13,
) -> dict:
    set_seed(seed)
    if target == "evasion":
        labels, label2id, id2label, col = EVASION_LABELS, EVASION2ID, ID2EVASION, "evasion_label"
    else:
        labels, label2id, id2label, col = CLARITY_LABELS, CLARITY2ID, ID2CLARITY, "clarity_label"

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
    # DeBERTa fine-tuning when rare classes (e.g. Partial, 8 examples) get
    # enormous weights, collapsing the model to majority-class prediction.
    raw_weights = np.sqrt(raw_weights)
    raw_weights = np.clip(raw_weights, a_min=None, a_max=5.0)
    weights = torch.tensor(raw_weights, dtype=torch.float)
    logger.info("Class weights (%s): %s", target,
                {labels[i]: round(float(w), 2) for i, w in enumerate(weights)})

    train_ds = _encode(train_df, tokenizer, label2id, max_length, col)
    dev_ds = _encode(dev_df, tokenizer, label2id, max_length, col)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        bf16=False,
        fp16=False,
        max_grad_norm=1.0,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        seed=seed,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        processing_class=tokenizer, compute_metrics=compute_metrics,
        class_weights=weights,
    )
    trainer.train()
    trainer.save_model(str(Path(output_dir) / "best"))

    logits = trainer.predict(dev_ds).predictions
    dev_pred = [id2label[i] for i in logits.argmax(-1)]
    metrics = evaluate_predictions(dev_df[col].tolist(), dev_pred, labels)
    logger.info("Encoder %s dev Macro-F1: %.4f", target, metrics["macro_f1"])
    return {"metrics": metrics, "predictions": dev_pred}


def predict_encoder(
    df: pd.DataFrame,
    model_dir: str | Path,
    target: str = "evasion",
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> list[str]:
    if target == "evasion":
        labels, label2id, id2label, col = EVASION_LABELS, EVASION2ID, ID2EVASION, "evasion_label"
    else:
        labels, label2id, id2label, col = CLARITY_LABELS, CLARITY2ID, ID2CLARITY, "clarity_label"

    # Load the fine-tuned model from its saved directory. num_labels/id2label
    # are already baked into the saved config, so they need not be re-passed.
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