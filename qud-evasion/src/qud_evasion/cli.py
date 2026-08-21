"""Command-line interface. One subcommand per pipeline stage so that
shell scripts stay one-liners and every run is reproducible from a YAML
config.

    python -m qud_evasion.cli prepare-data  --config configs/base.yaml
    python -m qud_evasion.cli train-encoder --config configs/baseline_encoder.yaml
    python -m qud_evasion.cli llm-baseline  --config configs/baseline_llm.yaml --split dev
    python -m qud_evasion.cli qud-pipeline  --config configs/qud_pipeline.yaml --split dev
    python -m qud_evasion.cli evaluate --pred outputs/qud/dev_predictions.jsonl --split dev
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .utils import load_config, seed_everything, setup_logging

logger = logging.getLogger("qud_evasion.cli")

# Default model names used when a config key is absent. These must match
# models that are actually pre-fetched by 00_prepare_data.sh, otherwise an
# offline compute-node run will fail with LocalEntryNotFoundError.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_split(cfg: dict, split: str) -> pd.DataFrame:
    from .data.load import load_processed

    dfs = load_processed(cfg["data_dir"])
    if split not in dfs:
        raise SystemExit(
            f"Split {split!r} not found in {cfg['data_dir']} "
            f"(available: {sorted(dfs)}). Run prepare-data first."
        )
    if split == "official_test":
        logger.warning(
            "You are loading the OFFICIAL test set. Per protocol this is "
            "evaluated exactly once, at the very end."
        )
    return dfs[split]


def _make_client(cfg: dict):
    from .qud.llm_client import LLMClient

    return LLMClient(
        model_name=cfg["llm_model"],
        cache_path=cfg.get("llm_cache", "outputs/llm_cache.sqlite"),
        tensor_parallel=cfg.get("tensor_parallel", 1),
        max_model_len=cfg.get("max_model_len", 8192),
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 512),
        seed=cfg.get("seed", 13),
    )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_prepare_data(cfg: dict, args) -> None:
    from .data.load import load_qevasion, save_processed
    from .data.splits import interview_level_split

    dfs = load_qevasion(cache_dir=cfg.get("hf_cache_dir"))
    train, dev = interview_level_split(
        dfs["train"],
        dev_fraction=cfg.get("dev_fraction", 0.10),
        seed=cfg.get("seed", 13),
    )
    save_processed(
        {"train": train, "dev": dev, "official_test": dfs["official_test"]},
        cfg["data_dir"],
    )


def cmd_train_encoder(cfg: dict, args) -> None:
    from .baselines.encoder import train_encoder
    from .eval.metrics import save_metrics

    train = _load_split(cfg, "train")
    dev = _load_split(cfg, "dev")
    for target in cfg.get("targets", ["evasion", "clarity"]):
        out_dir = Path(cfg["output_dir"]) / target
        result = train_encoder(
            train, dev,
            target=target,
            model_name=cfg.get("encoder_model", "microsoft/deberta-v3-base"),
            output_dir=out_dir,
            max_length=cfg.get("max_length", 512),
            lr=cfg.get("lr", 1e-5),
            epochs=cfg.get("epochs", 5),
            batch_size=cfg.get("batch_size", 8),
            grad_accum=cfg.get("grad_accum", 4),
            seed=cfg.get("seed", 13),
        )
        save_metrics(result["metrics"], out_dir / "dev_metrics.json")


def cmd_llm_baseline(cfg: dict, args) -> None:
    from .baselines.llm_prompting import run_direct_baseline

    df = _load_split(cfg, args.split)
    few_shot = _load_split(cfg, "train") if cfg.get("few_shot", True) else None
    client = _make_client(cfg)
    out = Path(cfg["output_dir"]) / f"{args.split}_predictions.jsonl"
    run_direct_baseline(
        df, client, out,
        few_shot_train=few_shot,
        k_per_class=cfg.get("k_per_class", 1),
        seed=cfg.get("seed", 13),
    )
    _evaluate_file(cfg, out, df)


def cmd_qud_pipeline(cfg: dict, args) -> None:
    from .qud.classify import LearnedClassifier, RuleClassifier
    from .qud.reconstruct import reconstruct_quds
    from .qud.allocation import allocate_turns
    from .qud.relations import (aggregate_per_example, commitment_features,
                                embedding_features, llm_relations, nli_features)

    df = _load_split(cfg, args.split)
    client = _make_client(cfg)
    out_dir = Path(cfg["output_dir"])

    def build_agg(d, split_tag):
        """Run the full feature pipeline for one split and return the
        aggregated per-example frame with all features attached."""
        recon = reconstruct_quds(
            d, client, out_dir / f"{split_tag}_stage1_quds.jsonl",
            max_quds=cfg.get("max_quds", 3),
        )
        pairs = llm_relations(
            d, recon, client, out_dir / f"{split_tag}_stage2_pairs.jsonl"
        )
        if cfg.get("use_embedding_features", True) and len(pairs):
            pairs = embedding_features(
                pairs, cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            )
        if cfg.get("use_nli_features", True) and len(pairs):
            pairs = nli_features(
                pairs, cfg.get("nli_model", DEFAULT_NLI_MODEL)
            )
        agg = aggregate_per_example(pairs, recon, d)
        agg = allocate_turns(agg)

        # Stage 2b: answer-commitment (orthogonal to topical overlap).
        if cfg.get("use_commitment_features", True):
            commit = commitment_features(
                d, client, out_dir / f"{split_tag}_commitment.jsonl"
            )
            agg = agg.merge(commit, on="example_id", how="left")
            agg["commitment"] = agg["commitment"].fillna(0.25)
            agg["commitment_label"] = agg["commitment_label"].fillna("evasive")
        else:
            agg["commitment"] = 0.25
            agg["commitment_label"] = "evasive"
        return agg

    agg = build_agg(df, args.split)
    agg.to_json(out_dir / f"{args.split}_stage2_agg.jsonl", orient="records", lines=True)

    head = cfg.get("head", "rule")
    if head == "rule":
        clf = RuleClassifier(client=client)
    elif head == "learned":
        model_path = out_dir / "learned_head.joblib"
        if args.split == "train":
            raise SystemExit("Run the learned head on dev/test; it trains on train internally.")
        if not model_path.exists():
            logger.info("Training learned head on train split features...")
            train_df = _load_split(cfg, "train")
            train_agg = build_agg(train_df, "train")
            # train_agg already has one row per example_id with all features;
            # align labels to it by example_id rather than re-merging (which
            # would collide on turn_id/question_order now present in both).
            train_labeled = train_agg.merge(
                train_df[["example_id", "evasion_label"]], on="example_id"
            )
            clf = LearnedClassifier().fit(
                train_labeled,
                train_labeled["evasion_label"],
            )
            clf.save(model_path)
        else:
            clf = LearnedClassifier.load(model_path)
    else:
        raise SystemExit(f"Unknown head: {head}")

    merged = clf.predict(df, agg)
    pred_path = out_dir / f"{args.split}_predictions.jsonl"
    keep = ["example_id", "evasion_pred", "clarity_pred",
            "dominant_relation", "speech_act", "qud_overlap",
            "commitment", "commitment_label"]
    merged[[c for c in keep if c in merged.columns]].to_json(
        pred_path, orient="records", lines=True
    )
    _evaluate_file(cfg, pred_path, df)


def _evaluate_file(cfg: dict, pred_path: Path, df: pd.DataFrame) -> None:
    from .eval.hard_cases import boundary_report, overlap_vs_error
    from .eval.metrics import evaluate_clarity, evaluate_evasion, save_metrics

    preds = pd.read_json(pred_path, lines=True)
    merged = df.merge(preds, on="example_id", validate="1:1")
    merged = merged.rename(columns={
        "clarity_label": "clarity_true", "evasion_label": "evasion_true",
    })

    metrics = {
        "clarity": evaluate_clarity(merged["clarity_true"], merged["clarity_pred"]),
    }
    # The official test split ships without fine-grained (Level 2) gold,
    # so evasion can only be scored where gold labels exist (train/dev).
    if "evasion_true" in merged.columns and merged["evasion_true"].notna().all():
        metrics["evasion"] = evaluate_evasion(
            merged["evasion_true"], merged["evasion_pred"]
        )
    else:
        metrics["evasion"] = {"skipped": "no gold evasion labels in this split"}

    metrics["hard_cases"] = boundary_report(merged)
    if "qud_overlap" in merged.columns and merged["qud_overlap"].notna().any():
        metrics["overlap_vs_error"] = overlap_vs_error(merged).to_dict(orient="records")

    out = pred_path.with_name(pred_path.stem.replace("_predictions", "_metrics") + ".json")
    save_metrics(metrics, out)
    ev = metrics["evasion"].get("macro_f1")
    logger.info(
        "Clarity Macro-F1: %.4f | Evasion Macro-F1: %s  ->  %s",
        metrics["clarity"]["macro_f1"],
        f"{ev:.4f}" if ev is not None else "n/a (no gold)",
        out,
    )


def cmd_evaluate(cfg: dict, args) -> None:
    df = _load_split(cfg, args.split)
    _evaluate_file(cfg, Path(args.pred), df)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="qud_evasion")
    parser.add_argument("command", choices=[
        "prepare-data", "train-encoder", "llm-baseline", "qud-pipeline", "evaluate",
    ])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="dev",
                        choices=["train", "dev", "official_test"])
    parser.add_argument("--pred", help="Predictions JSONL (evaluate command)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 13))
    logger.info("Config: %s", json.dumps(cfg, indent=2, default=str))

    {
        "prepare-data": cmd_prepare_data,
        "train-encoder": cmd_train_encoder,
        "llm-baseline": cmd_llm_baseline,
        "qud-pipeline": cmd_qud_pipeline,
        "evaluate": cmd_evaluate,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()