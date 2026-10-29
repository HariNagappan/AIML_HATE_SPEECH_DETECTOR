# Generate all paper figures from REAL repo data (checkpoints/*.json, processed datasets).
import json, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
OUT = ROOT / "paper_assets"
CKPT = ROOT / "backend" / "checkpoints"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})
C = {"blue": "#3B6EA5", "orange": "#E8853D", "green": "#4C9A6A", "red": "#C0504D",
     "purple": "#7E6BA8", "gray": "#8A8A8A", "lblue": "#9DC3E6", "lorange": "#F4B183"}

# ---------------------------------------------------------------- fig 1: architecture
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=8.4, bold=False, ls="-"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec=ec,
                           lw=0.9, linestyle=ls)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.0, color="#444444", ls=ls))

    # ---- row 1: inputs
    box(6, 64, 38, 8, "Previous comment $P$\n(conversational context, optional)", C["lorange"], fs=8.6)
    box(56, 64, 38, 8, "Current comment $C$", C["lblue"], fs=8.6)

    # ---- row 2: shared encoder
    box(33, 50, 34, 9, "Shared BERT encoder\n(identical weights for both inputs)",
        C["blue"], ec="#1c3a5e", fs=9)
    arrow(25, 63.5, 40, 59.6)
    arrow(75, 63.5, 60, 59.6)

    # ---- row 3: embeddings + token states
    box(6, 37, 24, 7.5, "$E_c$  =  [CLS] of $C$", C["lblue"], fs=8.8)
    box(36, 37, 28, 7.5, "$E_p$  =  [CLS] of $P$\n(zero vector if absent)", C["lorange"], fs=8.8)
    box(70, 37, 24, 7.5, "Token states $H_1..H_n$", "#DCE6F1", fs=8.8)
    arrow(43, 49.5, 18, 45.1)
    arrow(52, 49.5, 50, 45.1)
    arrow(60, 49.5, 82, 45.1)

    # ---- row 4: interaction
    box(14, 24, 60, 8.5, "Context interaction\n"
        "$[\,E_c;\,E_p;\,|E_c-E_p|;\,E_c\\odot E_p\,]$  \u2192  Linear(3072\u21921024) \u2192 GELU \u2192 Dropout",
        "#DCE6F1", fs=8.4)
    arrow(18, 36.5, 30, 33.1)
    arrow(50, 36.5, 48, 33.1)

    # ---- row 5: heads + contrastive
    box(2, 11, 22, 8, "Hate head\n3-class softmax", C["green"], fs=8.4)
    box(27, 11, 22, 8, "Target head\n(if annotated)", C["green"], fs=8.4)
    box(52, 11, 22, 8, "Reason head\n(if annotated)", C["green"], fs=8.4)
    box(77, 11, 21, 8, "Contrastive proj.\nInfoNCE (training only)", C["orange"], fs=7.8, ls="--")
    for hx in (13, 38, 63):
        arrow(hx, 23.5, hx, 19.6)
    arrow(74.5, 26, 87, 19.6, ls="--")

    # ---- row 6: evidence + explanation
    box(14, 0.5, 60, 7, "Evidence extraction: Integrated Gradients \u2192 token scores \u2192 spans + offsets\n"
        "Deterministic explanation builder (templates filled with real outputs)",
        "#EFEFEF", fs=8)
    arrow(13, 10.5, 25, 8.0)
    arrow(63, 10.5, 55, 8.0)

    ax.text(93, 49.5, "Training objective:\n"
            "$L = \\lambda_h L_{hate} + \\lambda_t L_{target} + \\lambda_r L_{reason}$\n"
            "$+ \\, \\lambda_e L_{evid} + \\lambda_{cl} L_{InfoNCE}$",
            ha="center", fontsize=7.2, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8EE", ec="#C9A227", lw=0.8))

    fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- fig 2: datasets
def fig2_datasets():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    # HateXplain train distribution
    hx = collections.Counter()
    for split in ("train", "val", "test"):
        with open(ROOT / f"backend/data/processed/hatexplain_{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                hx[json.loads(line)["hate_label"]] += 1
    labels = ["hate", "offensive", "normal"]
    vals = [hx["hate"], hx["offensive"], hx["normal"]]
    axes[0].bar(labels, vals, color=[C["red"], C["orange"], C["green"]])
    axes[0].set_title("HateXplain (all splits, n=19,229)")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 120, f"{v:,}", ha="center", fontsize=8)
    axes[0].set_ylabel("posts")

    # target distribution (train)
    tgt = collections.Counter()
    with open(ROOT / "backend/data/processed/hatexplain_train.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["target_label"]
            if t and t != "None":
                tgt[t] += 1
    items = sorted(tgt.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]; tv = [v for _, v in items]
    axes[1].barh(names[::-1], tv[::-1], color=C["blue"])
    axes[1].set_title("HateXplain target groups (train)")

    # counter context
    cc = collections.Counter()
    for split in ("train", "val", "test"):
        with open(ROOT / f"backend/data/processed/counter_context_{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                cc[json.loads(line)["hate_label"]] += 1
    labels = ["hate_speech", "counter_speech", "neither"]
    vals = [cc["hate_speech"], cc["counter_speech"], cc["neither"]]
    axes[2].bar(labels, vals, color=[C["red"], C["purple"], C["green"]])
    axes[2].set_title("Counter Context (gold splits, n=4,751)")
    for i, v in enumerate(vals):
        axes[2].text(i, v + 40, f"{v:,}", ha="center", fontsize=8)
    axes[2].tick_params(axis="x", labelsize=7.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_datasets.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- fig 3: training curves
def _history(name):
    with open(CKPT / name / "training_summary.json", encoding="utf-8") as f:
        h = json.load(f)["history"]
    return [e["epoch"] + 1 for e in h], [e["val_macro_f1"] for e in h]

def fig3_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for name, label, color in [
        ("baseline", "Baseline (no context)", C["gray"]),
        ("full", "Full multi-task (HateXplain)", C["blue"]),
        ("hx_full_long", "Full multi-task, 6-epoch budget", C["lblue"]),
    ]:
        x, y = _history(name)
        axes[0].plot(x, y, marker="o", ms=3.5, lw=1.4, color=color, label=label)
    for name, label, color, mk in [
        ("context", "Context, no contrastive", C["gray"], "^"),
        ("cc_baseline", "Comment-only (no context)", C["purple"], "D"),
        ("cc_context", "Context + interaction + InfoNCE", C["orange"], "s"),
    ]:
        x, y = _history(name)
        axes[1].plot(x, y, marker=mk, ms=3.5, lw=1.4, color=color, label=label)
    axes[0].set_title("HateXplain runs")
    axes[1].set_title("Counter Context runs")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25, lw=0.4); ax.set_xticks(range(1, 7))
        ax.set_ylim(0.43, 0.72)
    axes[0].set_ylabel("validation macro-F1")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- fig 4: confusion matrices
def _metrics(name):
    with open(CKPT / name / "best.metrics.json", encoding="utf-8") as f:
        return json.load(f)

def fig4_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, name, labels in [
        (axes[0], "hx_full", ["hate", "offensive", "normal"]),
        (axes[1], "cc_context", ["hate\nspeech", "counter\nspeech", "neither"]),
    ]:
        m = _metrics(name)["hate"]
        cm = np.array(m["confusion_matrix"])
        norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]*100:.0f}%)", ha="center", va="center",
                        fontsize=7.8, color="white" if norm[i, j] > 0.55 else "#222222")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        acc = m["accuracy"] * 100
        ax.set_title(f"{name} test (n={m['support'] and sum(sum(r) for r in m['confusion_matrix'])}, acc {acc:.1f}%)", fontsize=9)
        ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_confusion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- fig 5: per-class F1
def fig5_perclass():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    hx = _metrics("hx_full")
    cc = _metrics("cc_context")
    # hate head per-class F1 comparison across datasets
    ax = axes[0]
    classes = ["hate", "offensive", "normal"]
    f1 = [hx["hate"]["per_class_f1"][c] for c in classes]
    bars = ax.bar(classes, f1, color=[C["red"], C["orange"], C["green"]])
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 0.9); ax.set_ylabel("test F1")
    ax.set_title("hx_full hate head, per class")
    ax.tick_params(axis="x", labelsize=8.5)
    # target head per-class F1
    ax = axes[1]
    tg = hx["target"]["per_class_f1"]
    order = ["race", "religion", "nationality", "sexual_orientation", "ethnicity", "none", "gender", "other", "political_group"]
    vals = [tg[k] for k in order]
    cols = [C["blue"] if v > 0 else C["gray"] for v in vals]
    bars = ax.barh(order[::-1], vals[::-1], color=cols[::-1])
    for b, v in zip(bars, vals[::-1]):
        ax.text(max(v + 0.015, 0.03), b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=7.6)
    ax.set_xlim(0, 0.95)
    ax.set_title("hx_full target head, per class (test F1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_perclass.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- fig 6: reasoning pipeline
def fig6_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.set_xlim(0, 104); ax.set_ylim(0, 42); ax.axis("off")

    steps = [
        (1,  "1  Prediction",    "trained hate head\nlabel + confidence\n(WHAT)", C["lblue"]),
        (22, "2  Evidence",      "Integrated Gradients\nper-token scores \u2192\nspans + offsets\n(WHICH WORDS)", "#DCE6F1"),
        (43, "3  Context link",  "pronoun\u2192antecedent\n(WHO is\ntargeted)", C["lorange"]),
        (64, "4  Explanation",   "fixed templates,\nreal values only\n(WHY)", "#E2EFDA"),
        (85, "5  JSON + UI",     "structured JSON,\nhighlighted\nevidence", "#EFEFEF"),
    ]
    for x, title, body, fc in steps:
        ax.add_patch(FancyBboxPatch((x, 20), 18, 17, boxstyle="round,pad=0.4",
                                    fc=fc, ec="#333333", lw=0.9))
        ax.text(x + 9, 34.2, title, ha="center", fontsize=8.0, fontweight="bold")
        ax.text(x + 9, 26.4, body, ha="center", va="center", fontsize=6.8)
    for x in (19.6, 40.6, 61.6, 82.6):
        ax.add_patch(FancyArrowPatch((x, 28.5), (x + 1.8, 28.5), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.0, color="#444444"))

    ax.text(52, 8, 'Worked example (real output):  "They" \u2192 "group of immigrants";  evidence spans '
                    '"kicked" (1.00), "they should" (0.66);  hate_speech @ 0.757',
            ha="center", fontsize=7.8, style="italic",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFF8EE", ec="#C9A227", lw=0.8))
    fig.savefig(OUT / "fig6_reasoning_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- fig 7: real attribution
def fig7_attribution():
    with open(OUT / "explain_canonical.json", encoding="utf-8") as f:
        raw = f.read()
    part = raw.split("== attribution details ==")[1]
    data = json.loads(part)
    toks = [(t["token"], t["score"]) for t in data["tokens"] if t["token"] not in ("[CLS]", "[SEP]")]
    names = [t for t, _ in toks]; scores = [s for _, s in toks]
    colors = [C["red"] if s >= 0 else C["blue"] for s in scores]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.bar(names, scores, color=colors, width=0.62)
    for i, s in enumerate(scores):
        ax.text(i, s + (0.06 if s >= 0 else -0.13), f"{s:.2f}", ha="center", fontsize=7.6)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel("IG attribution")
    ax.set_title(f'Integrated-Gradients token attribution for the predicted class '
                 f'({data["predicted_label"]}, p={data["predicted_confidence"]:.3f})\n'
                 f'comment: "They should all be kicked out."  |  context: "I saw a group of immigrants protesting downtown."',
                 fontsize=8.6)
    ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_attribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

fig1_architecture()
fig2_datasets()
fig3_curves()
fig4_confusion()
fig5_perclass()
fig6_pipeline()
fig7_attribution()
for p in sorted(OUT.glob("fig*.png")):
    print(p.name, f"{p.stat().st_size/1024:.0f} KiB")
