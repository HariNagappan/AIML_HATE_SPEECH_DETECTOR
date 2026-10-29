# Dump per-example test predictions for the Counter Context ladder checkpoints
# and run paired tests (McNemar + bootstrap) between variants.
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os_like = None

from app.core.config import get_settings, resolve_device  # noqa: E402
from app.datasets.collator import MultitaskCollator, build_label_maps  # noqa: E402
from app.datasets.unified import load_unified_jsonl  # noqa: E402
from app.services.inference import build_model_from_checkpoint  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

settings = get_settings()
device = resolve_device(settings.device)
test = load_unified_jsonl(BACKEND / "data/processed/counter_context_test.jsonl")
print("n_test", len(test), "device", device)

LABELS = ["hate_speech", "neither", "counter_speech"]
lab2id = {n: i for i, n in enumerate(LABELS)}


@torch.no_grad()
def predict(ckpt: str):
    bundle = build_model_from_checkpoint(str(ckpt), settings)
    model = bundle.model
    hate_labels_names = bundle.model_kwargs.get("hate_labels") or LABELS
    label_maps = build_label_maps(hate_labels_names, [], [])
    collator = MultitaskCollator(bundle.tokenizer, label_maps, max_length=128)
    loader = DataLoader(test, batch_size=32, shuffle=False, collate_fn=collator)
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(bundle.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            kwargs = dict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            if getattr(model, "supports_context", False):
                kwargs.update(
                    context_input_ids=batch.get("context_input_ids"),
                    context_attention_mask=batch.get("context_attention_mask"),
                    context_present=batch.get("context_present"),
                )
            out = model(**kwargs)
            preds.extend(out["hate_logits"].argmax(dim=-1).cpu().tolist())
            golds.extend(batch["hate_labels"].cpu().tolist())
    del bundle
    torch.cuda.empty_cache()
    return golds, preds


def mcnemar(y, a, b):
    n01 = sum(1 for t, x, y_ in zip(y, a, b) if x != t and y_ == t)
    n10 = sum(1 for t, x, y_ in zip(y, a, b) if x == t and y_ != t)
    if n01 + n10 == 0:
        return 1.0, n01, n10
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    from math import erfc, sqrt
    p = erfc(sqrt(chi / 2.0))
    return p, n01, n10


def bootstrap_acc_ci(y, a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    a = np.asarray(a)
    b = np.asarray(b)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    da = (a[idx] == y[idx]).mean(1) - (b[idx] == y[idx]).mean(1)
    return float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))


variants = {
    "cc_baseline": [
        ROOT / "backend/checkpoints/cc_baseline_s42/best.pt",
        ROOT / "backend/checkpoints/cc_baseline_s1/best.pt",
        ROOT / "backend/checkpoints/cc_baseline_s2/best.pt",
    ],
    "cc_concat": [
        ROOT / "backend/checkpoints/cc_concat_s42/best.pt",
        ROOT / "backend/checkpoints/cc_concat_s1/best.pt",
        ROOT / "backend/checkpoints/cc_concat_s2/best.pt",
    ],
    "cc_inter": [
        ROOT / "backend/checkpoints/cc_inter_s42/best.pt",
        ROOT / "backend/checkpoints/cc_inter_s1/best.pt",
        ROOT / "backend/checkpoints/cc_inter_s2/best.pt",
    ],
    "cc_nce": [
        ROOT / "backend/checkpoints/cc_nce_s42/best.pt",
        ROOT / "backend/checkpoints/cc_nce_s1/best.pt",
        ROOT / "backend/checkpoints/cc_nce_s2/best.pt",
    ],
    "cc_context_orig": [ROOT / "backend/checkpoints/cc_context/best.pt"],
}

out = {"n": len(test), "labels": LABELS, "runs": {}, "paired": []}
golds_ref = None
for name, paths in variants.items():
    for i, p in enumerate(paths):
        if not p.exists():
            print("missing", p)
            continue
        tag = f"{name}_s{i}"
        golds, preds = predict(p)
        if golds_ref is None:
            golds_ref = golds
        acc = float(np.mean(np.asarray(preds) == np.asarray(golds)))
        out["runs"][tag] = {"path": str(p), "acc": acc, "pred": preds, "gold": golds}
        print(tag, "acc", round(acc, 4))

# paired: concat vs baseline, seed-wise if both exist, using seed 42 as primary
if "cc_concat_s0" in out["runs"] and "cc_baseline_s0" in out["runs"]:
    y = out["runs"]["cc_concat_s0"]["gold"]
    a = out["runs"]["cc_concat_s0"]["pred"]
    b = out["runs"]["cc_baseline_s0"]["pred"]
    p, n01, n10 = mcnemar(y, a, b)
    lo, hi = bootstrap_acc_ci(y, a, b)
    out["paired"].append(
        {
            "a": "cc_concat_s42",
            "b": "cc_baseline_s42",
            "acc_a": out["runs"]["cc_concat_s0"]["acc"],
            "acc_b": out["runs"]["cc_baseline_s0"]["acc"],
            "mcnemar_p": p,
            "n01": n01,
            "n10": n10,
            "boot_ci": [lo, hi],
        }
    )
    print("paired concat vs base", p, n01, n10, lo, hi)

if "cc_inter_s0" in out["runs"] and "cc_concat_s0" in out["runs"]:
    y = out["runs"]["cc_inter_s0"]["gold"]
    a = out["runs"]["cc_inter_s0"]["pred"]
    b = out["runs"]["cc_concat_s0"]["pred"]
    p, n01, n10 = mcnemar(y, a, b)
    lo, hi = bootstrap_acc_ci(y, a, b)
    out["paired"].append(
        {
            "a": "cc_inter_s42",
            "b": "cc_concat_s42",
            "acc_a": out["runs"]["cc_inter_s0"]["acc"],
            "acc_b": out["runs"]["cc_concat_s0"]["acc"],
            "mcnemar_p": p,
            "n01": n01,
            "n10": n10,
            "boot_ci": [lo, hi],
        }
    )

dest = ROOT / "paper_assets" / "cc_predictions.json"
dest.write_text(json.dumps(out), encoding="utf-8")
print("saved", dest, "size", dest.stat().st_size)
