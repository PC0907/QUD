#!/usr/bin/env python
"""Plot layer-wise probe curves (mean +/- std) for task1 and/or task2.
Reads outputs/probe/probe_task1.json and probe_task2.json, writes a PDF figure.

  python scripts/plot_probe.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/probe"
KAPPA_T1 = 0.65  # human inter-annotator agreement, Ambiguous vs Non-Reply

fig, ax = plt.subplots(figsize=(6, 4))

for target, color, label in [("task1", "C0", "Ambiguous vs Non-Reply"),
                              ("task2", "C1", "Evasion cluster (4-way)")]:
    path = f"{OUT}/probe_{target}.json"
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    mean, std = np.array(d["mean"]), np.array(d["std"])
    layers = np.arange(len(mean))
    ax.plot(layers, mean, color=color, label=label, linewidth=1.8)
    ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.18)
    b = d["best_layer"]
    ax.scatter([b], [mean[b]], color=color, zorder=5, s=28)

ax.axhline(KAPPA_T1, ls="--", color="gray", linewidth=1,
           label=f"human agreement ($\\kappa$={KAPPA_T1})")
ax.set_xlabel("layer")
ax.set_ylabel("probe macro-F1")
ax.set_ylim(0.3, 0.8)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/probe_curves.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/probe_curves.png", dpi=150, bbox_inches="tight")
print(f"saved {OUT}/probe_curves.pdf and .png")