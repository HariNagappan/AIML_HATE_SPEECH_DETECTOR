# -*- coding: utf-8 -*-
"""Figure 8: Counter Context ablation ladder with seed error bars, from experiment_results.json."""
import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
OUT = ROOT / "paper_assets"
res = json.loads((OUT / "experiment_results.json").read_text(encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300,
})
C = {"blue": "#3B6EA5", "orange": "#E8853D", "green": "#4C9A6A", "red": "#C0504D",
     "purple": "#7E6BA8", "gray": "#8A8A8A"}

def agg(rows, key="test_macro_f1"):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)

cc = res.get("cc_ladder", [])
xe = res.get("cross_encoder", [])
order = [
    ("No context\n(baseline)", "cc_baseline", C["gray"]),
    ("Context concat\n[u; v]", "cc_concat", C["lblue"] if "lblue" in C else "#9DC3E6"),
    ("Context interact.\n[u; v; |u-v|; u*v]", "cc_inter", C["blue"]),
    ("Interaction\n+ InfoNCE", "cc_nce", C["orange"]),
    ("Cross-encoder\n[CLS] P [SEP] C", "cross_encoder", C["purple"]),
]
labels, means, stds, colors, ns = [], [], [], [], []
for label, key, color in order:
    rows = xe if key == "cross_encoder" else [r for r in cc if r.get("variant") == key]
    a = agg(rows)
    if a is None:
        continue
    m, s, n = a
    labels.append(label); means.append(m); stds.append(s); colors.append(color); ns.append(n)

fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.6,
              error_kw=dict(lw=1.1, ecolor="#333333"))
for i, (m, s, n) in enumerate(zip(means, stds, ns)):
    ax.text(i, m + s + 0.012, f"{m:.3f}±{s:.3f}\n(n={n})", ha="center", fontsize=7.6)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("test macro-F1")
ax.set_ylim(0.45, max(means) + max(stds) + 0.06)
ax.set_title("Counter Context test macro-F1 by variant (mean ± s.d. over seeds)")
ax.grid(axis="y", alpha=0.25, lw=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig8_cc_ladder.png", bbox_inches="tight", facecolor="white")
print("fig8 saved", list(zip(labels, [round(m,4) for m in means], [round(s,4) for s in stds], ns)))
