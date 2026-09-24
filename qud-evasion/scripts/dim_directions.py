"""Difference-in-means concept directions inside the annotator, with causal tests.

Question: when the annotator answers a concept (for example "does the speaker
refuse?"), where in the network is that decision represented, and does the
network represent the concept even on rows where its output gets it wrong?

Method (difference-in-means, as used by Arditi et al. 2024 and Marks and
Tegmark 2024 to find linear directions):

 1. Reconstruct the exact annotation prompt for each row and teacher-force the
    model's own recorded output up to the point where it is about to write the
    value of the target concept, i.e. just after  "p7": {"value": "
 2. Take the residual stream at that decision position at every layer.
 3. Direction per layer = mean(activation | model said yes)
                        - mean(activation | model said no), fit on half the rows.
 4. Readout: on held-out rows, AUROC of the projection onto the direction.

Latent-knowledge test (the main analysis): "missed" rows are rows where the
model said "no" but the GOLD label is the one this concept indicates (e.g.
the model says the speaker did not refuse, but the gold label is Declining to
answer). If held-out missed rows project higher than ordinary "no" rows at
some layer, the network represented the concept there and lost it before the
output. The layer profile shows where.

Causal tests at the chosen layer, measured on the next-token logit difference
logit("yes") - logit("no") at the decision position:
  - ablate the direction on held-out "yes" rows: does the answer flip to no?
  - ablate a random direction of the same norm: control, should do little
  - add the direction to held-out ordinary "no" rows and to missed rows:
    does the answer flip to yes?

    python scripts/dim_directions.py --concepts v1 --concept p7
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("dim")

# Gold labels a concept is meant to indicate, for the missed-row analysis.
GOLD_FOR = {
    "v1": {"p7": {"Declining to answer"}, "p8": {"Claims ignorance"},
           "p9": {"Clarification"}, "p2": {"Explicit"}},
    "v2": {"q7": {"Declining to answer"}, "q8": {"Claims ignorance"},
           "q9": {"Clarification"}, "q2": {"Explicit"}, "q5": {"Dodging"},
           "q6": {"Deflection"}},
}


def value_prefix_end(raw: str, cid: str):
    """Character offset just after  "<cid>": {"value": "  in the raw output."""
    m = re.search(r'"%s"\s*:\s*\{\s*"value"\s*:\s*"' % re.escape(cid), raw or "")
    return m.end() if m else None


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="v1")
    ap.add_argument("--concept", default="p7")
    ap.add_argument("--data", default="data/processed/train.parquet")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max-per-class", type=int, default=150)
    ap.add_argument("--layer", type=int, default=None,
                    help="layer for causal tests; default = best missed-row layer")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="addition strength, in units of the yes/no mean gap")
    ap.add_argument("--ablate-all-layers", action="store_true",
                    help="project the direction out of EVERY layer (Arditi et al. "
                         "2024), so later layers cannot rebuild it from context")
    ap.add_argument("--alphas", type=float, nargs="*", default=None,
                    help="dose-response sweep for addition, e.g. 1 2 4 8")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qud_evasion.cbm.annotate import (build_turn_context, load_annotations,
                                          render_system, render_user)
    from qud_evasion.cbm.concepts import get_concepts

    cid = args.concept
    out_dir = Path(args.out or f"outputs/cbm/{args.concepts}/dim/{cid}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- select rows ------------------------------------------------------
    df = pd.read_parquet(args.data)
    ann = load_annotations(Path(args.ann or f"outputs/cbm/{args.concepts}/train.jsonl"))
    ann = ann[ann.parse_ok].merge(df[["example_id", "evasion_label"]], on="example_id")
    ann["end"] = [value_prefix_end(r, cid) for r in ann.raw]
    ann = ann[ann.end.notna() & ann[f"{cid}_value"].isin(["yes", "no"])]
    gold_set = GOLD_FOR.get(args.concepts, {}).get(cid, set())

    yes = ann[ann[f"{cid}_value"] == "yes"]
    no = ann[ann[f"{cid}_value"] == "no"]
    missed = no[no.evasion_label.isin(gold_set)] if gold_set else no.iloc[:0]
    other_no = no[~no.evasion_label.isin(gold_set)]

    def take(frame, k):
        return frame.sample(n=min(k, len(frame)), random_state=args.seed)

    k = args.max_per_class
    yes, other_no, missed = take(yes, k), take(other_no, k), take(missed, k)
    log.info("rows: yes=%d  ordinary no=%d  missed (no, gold=%s)=%d",
             len(yes), len(other_no), sorted(gold_set) or "-", len(missed))
    if len(yes) < 10 or len(other_no) < 10:
        raise SystemExit("too few rows for a stable direction")

    def split(frame):
        idx = rng.permutation(len(frame))
        h = len(frame) // 2
        return frame.iloc[idx[:h]], frame.iloc[idx[h:]]

    yes_fit, yes_eval = split(yes)
    no_fit, no_eval = split(other_no)

    # ---- model ------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = model.model.layers
    n_layers = len(layers)

    def first_id(s):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            log.warning("%r tokenizes to %d tokens; using the first", s, len(ids))
        return ids[0]

    yes_id, no_id = first_id("yes"), first_id("no")

    concepts = get_concepts(args.concepts)
    system = render_system(concepts)
    ctx = build_turn_context(df)
    rows = df.set_index("example_id")

    def build_text(rec) -> str:
        row = rows.loc[rec.example_id].copy()
        row["example_id"] = rec.example_id
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": render_user(row, ctx, None)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        return prompt + rec.raw[: int(rec.end)]

    @torch.no_grad()
    def run(text, hook=None):
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        handle = hook() if hook else None
        try:
            out = model(**enc, output_hidden_states=hook is None)
        finally:
            if handle is not None:
                handle.remove()
        logits = out.logits[0, -1].float()
        diff = float(logits[yes_id] - logits[no_id])
        hs = None
        if hook is None:
            hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
        return diff, hs

    def collect(frame, name):
        feats, diffs = [], []
        for rec in frame.itertuples():
            d, hs = run(build_text(rec))
            feats.append(hs.numpy())
            diffs.append(d)
        log.info("collected %s: %d rows", name, len(frame))
        return np.stack(feats), np.array(diffs)

    F = {}
    for name, frame in [("yes_fit", yes_fit), ("yes_eval", yes_eval),
                        ("no_fit", no_fit), ("no_eval", no_eval), ("missed", missed)]:
        F[name] = collect(frame, name) if len(frame) else (np.zeros((0, n_layers + 1, 1)), np.zeros(0))

    # Sanity: teacher forcing should reproduce the recorded answers.
    agree_yes = float((np.r_[F["yes_fit"][1], F["yes_eval"][1]] > 0).mean())
    agree_no = float((np.r_[F["no_fit"][1], F["no_eval"][1]] < 0).mean())
    log.info("teacher-forcing sanity: logit sign matches recorded answer "
             "on %.2f of yes rows, %.2f of no rows", agree_yes, agree_no)
    if min(agree_yes, agree_no) < 0.8:
        log.warning("low agreement: the decision position may be off, "
                    "treat results with caution")

    # ---- directions and readout per layer -----------------------------------
    rows_out, dirs = [], {}
    for L in range(1, n_layers + 1):
        mu_y = F["yes_fit"][0][:, L].mean(0)
        mu_n = F["no_fit"][0][:, L].mean(0)
        d = mu_y - mu_n
        u = d / (np.linalg.norm(d) + 1e-8)
        mid = (mu_y + mu_n) / 2
        proj = lambda X: (X[:, L] - mid) @ u
        dirs[L] = (d, mid)
        rows_out.append({
            "layer": L,
            "auroc_yes_vs_no": auroc(proj(F["yes_eval"][0]), proj(F["no_eval"][0])),
            "auroc_missed_vs_no": auroc(proj(F["missed"][0]), proj(F["no_eval"][0]))
            if len(F["missed"][0]) else float("nan"),
            "gap_norm": float(np.linalg.norm(d)),
        })
    table = pd.DataFrame(rows_out)
    table.to_csv(out_dir / "layer_auroc.csv", index=False)
    log.info("per-layer readout (held-out):\n%s",
             table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- causal tests -------------------------------------------------------
    if args.layer is not None:
        L = args.layer
    elif table.auroc_missed_vs_no.notna().any():
        L = int(table.loc[table.auroc_missed_vs_no.idxmax(), "layer"])
    else:
        L = int(table.loc[table.auroc_yes_vs_no.idxmax(), "layer"])
    L = min(L, n_layers - 1)       # the last entry is post-final-norm
    d_np, _ = dirs[L]
    d = torch.tensor(d_np, dtype=torch.bfloat16, device=model.device)
    u = d / d.norm()
    r = torch.randn_like(u.float())
    r = (r / r.norm()).to(torch.bfloat16)
    log.info("causal tests at layer %d (hook on decoder block %d)%s", L, L - 1,
             "; ablation applied at ALL layers" if args.ablate_all_layers else "")

    def hook_factory(vec, mode, alpha=1.0):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if mode == "ablate":
                h = h - (h @ vec).unsqueeze(-1) * vec
            else:
                h = h + alpha * vec
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        targets = (list(layers) if (mode == "ablate" and args.ablate_all_layers)
                   else [layers[L - 1]])

        def register():
            handles = [t.register_forward_hook(hook) for t in targets]

            class _Handles:
                def remove(self):
                    for hd in handles:
                        hd.remove()
            return _Handles()
        return register

    def causal(frame, base_diffs, factory, flip_to):
        new = np.array([run(build_text(rec), hook=factory)[0]
                        for rec in frame.itertuples()])
        if flip_to == "no":
            flipped = (base_diffs > 0) & (new < 0)
        else:
            flipped = (base_diffs < 0) & (new > 0)
        return {"n": int(len(new)), "flip_rate": float(flipped.mean()) if len(new) else float("nan"),
                "mean_diff_before": float(base_diffs.mean()) if len(new) else float("nan"),
                "mean_diff_after": float(new.mean()) if len(new) else float("nan")}

    results = {"concept": cid, "layer": L, "alpha": args.alpha,
               "sanity_agree_yes": agree_yes, "sanity_agree_no": agree_no}
    results["ablate_direction_on_yes"] = causal(
        yes_eval, F["yes_eval"][1], hook_factory(u, "ablate"), "no")
    results["ablate_random_on_yes"] = causal(
        yes_eval, F["yes_eval"][1], hook_factory(r, "ablate"), "no")
    results["add_direction_on_no"] = causal(
        no_eval, F["no_eval"][1], hook_factory(d, "add", args.alpha), "yes")
    results["add_random_on_no"] = causal(
        no_eval, F["no_eval"][1], hook_factory(r * d.norm(), "add", args.alpha), "yes")
    if len(missed):
        results["add_direction_on_missed"] = causal(
            missed, F["missed"][1], hook_factory(d, "add", args.alpha), "yes")

    # Dose-response: at margins of ~20 logits a flip is a harsh criterion; the
    # informative quantity is whether the logit shift grows with strength and
    # stays above the random-direction control at every dose.
    if args.alphas:
        sweep = []
        for a in args.alphas:
            on_dir = causal(no_eval, F["no_eval"][1], hook_factory(d, "add", a), "yes")
            on_rnd = causal(no_eval, F["no_eval"][1],
                            hook_factory(r * d.norm(), "add", a), "yes")
            row = {"alpha": a,
                   "shift_direction": on_dir["mean_diff_after"] - on_dir["mean_diff_before"],
                   "shift_random": on_rnd["mean_diff_after"] - on_rnd["mean_diff_before"],
                   "flip_direction": on_dir["flip_rate"],
                   "flip_random": on_rnd["flip_rate"]}
            if len(missed):
                on_mis = causal(missed, F["missed"][1], hook_factory(d, "add", a), "yes")
                row["shift_missed"] = on_mis["mean_diff_after"] - on_mis["mean_diff_before"]
                row["flip_missed"] = on_mis["flip_rate"]
            sweep.append(row)
        results["dose_response_on_no"] = sweep
        log.info("dose-response (logit shift toward yes):\n%s",
                 pd.DataFrame(sweep).to_string(index=False,
                                               float_format=lambda v: f"{v:.2f}"))

    (out_dir / "causal.json").write_text(json.dumps(results, indent=2))
    torch.save({L: torch.tensor(dirs[L][0]) for L in dirs}, out_dir / "directions.pt")
    log.info("causal results:\n%s", json.dumps(results, indent=2))
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()