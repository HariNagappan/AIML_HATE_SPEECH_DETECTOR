import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import get_settings  # noqa: E402
from app.services.inference import build_model_from_checkpoint  # noqa: E402

settings = get_settings()
OUT = ROOT / "paper_assets"
results = {}
if (OUT / "addenda_results.json").exists():
    results = json.loads((OUT / "addenda_results.json").read_text(encoding="utf-8"))

labels_hx = ["hate", "offensive", "normal"]


@torch.no_grad()
def predict_hate(model, tokenizer, text, device):
    enc = tokenizer.encode(text, max_length=128, padding=False, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(out["hate_logits"][0], dim=-1).cpu().numpy()
    return int(probs.argmax()), probs


print("=== HateCheck (fixed) ===", flush=True)
rows = list(csv.DictReader(open(OUT / "hc_cases.csv", encoding="utf-8")))
print("rows", len(rows), "fields", rows[0].keys(), flush=True)
print("gold values:", Counter((r.get("label_gold") or "").strip() for r in rows), flush=True)
bundle = build_model_from_checkpoint(str(ROOT / "backend/checkpoints/hx_full/best.pt"), settings)
func = {}
n_h = n_nh = 0
pred_hate_on_hateful = pred_off_on_hateful = pred_norm_on_hateful = 0
nh_normal = nh_off = nh_hate = 0
skipped = 0
with torch.no_grad():
    for r in rows:
        text = (r.get("test_case") or "").strip()
        g = (r.get("label_gold") or "").strip()
        if not text or g not in ("hateful", "non-hateful"):
            skipped += 1
            continue
        lab, _ = predict_hate(bundle.model, bundle.tokenizer, text, bundle.device)
        name = labels_hx[lab]
        if g == "hateful":
            n_h += 1
            if name == "hate":
                pred_hate_on_hateful += 1
            elif name == "offensive":
                pred_off_on_hateful += 1
            else:
                pred_norm_on_hateful += 1
            st = func.setdefault(r["functionality"], [0, 0])
            st[1] += 1
            st[0] += int(name == "hate")
        else:
            n_nh += 1
            if name == "normal":
                nh_normal += 1
            elif name == "offensive":
                nh_off += 1
            else:
                nh_hate += 1
print("skipped", skipped, "n_h", n_h, "n_nh", n_nh, flush=True)
results["hatecheck"] = {
    "n": n_h + n_nh,
    "skipped_rows": skipped,
    "checkpoint": "hx_full",
    "hateful_n": n_h,
    "hateful_recall_hate": round(pred_hate_on_hateful / n_h, 4) if n_h else None,
    "hateful_pred_offensive": pred_off_on_hateful,
    "hateful_pred_normal": pred_norm_on_hateful,
    "nonhateful_n": n_nh,
    "nonhateful_pred_normal": nh_normal,
    "nonhateful_pred_offensive": nh_off,
    "nonhateful_pred_hate": nh_hate,
    "nonhateful_overflag_hate": round(nh_hate / n_nh, 4) if n_nh else None,
    "functionality_recall_hateful": {
        k: {"recall": round(v[0] / v[1], 3), "n": v[1]}
        for k, v in sorted(func.items(), key=lambda kv: -(kv[1][0] / kv[1][1]))
    },
}
print(json.dumps({k: v for k, v in results["hatecheck"].items() if k != "functionality_recall_hateful"}, indent=2), flush=True)
print("best functionalities:", json.dumps(dict(list(results["hatecheck"]["functionality_recall_hateful"].items())[:6]), indent=2), flush=True)
print("worst functionalities:", json.dumps(dict(list(results["hatecheck"]["functionality_recall_hateful"].items())[-6:]), indent=2), flush=True)
del bundle
torch.cuda.empty_cache()

print("=== IG comprehensiveness/sufficiency ===", flush=True)
from app.reasoning.attribution import compute_attributions  # noqa: E402
from app.reasoning.evidence_extractor import extract_evidence  # noqa: E402
from app.datasets.tokenizer import is_special_token  # noqa: E402

bundle = build_model_from_checkpoint(str(ROOT / "backend/checkpoints/hx_full/best.pt"), settings)
hx_rows = []
with open(ROOT / "backend/data/processed/hatexplain_test.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("rationale_labels") and r["rationale_labels"] != "None":
            hx_rows.append(r)
        if len(hx_rows) >= 100:
            break
comp_list, suff_list = [], []
for r in hx_rows:
    enc = bundle.tokenizer.encode(r["current_text"], max_length=128, padding=False, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(bundle.device)
    attention_mask = enc["attention_mask"].to(bundle.device)
    with torch.no_grad():
        out = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
        p_orig = float(torch.softmax(out["hate_logits"][0], -1).max())
    attr = compute_attributions(
        bundle.model, bundle.tokenizer, r["current_text"], None, target="hate",
        method="integrated_gradients", steps=32, max_length=128, device=bundle.device,
    )
    spans = extract_evidence(
        attr, bundle.tokenizer, top_k=5, min_score_frac=0.15, original_text=r["current_text"]
    )
    ev_idx = set()
    for s in spans:
        for i in s.token_indices:
            ev_idx.add(i)
    ids = input_ids[0].tolist()
    toks = bundle.tokenizer.token_strings(ids)
    del_ids, keep_ids = ids[:], ids[:]
    for i, t in enumerate(toks):
        if i in ev_idx and not is_special_token(t):
            del_ids[i] = bundle.tokenizer.pad_token_id
        elif i not in ev_idx and not is_special_token(t):
            keep_ids[i] = bundle.tokenizer.pad_token_id
    del_ids_t = torch.tensor([del_ids], device=bundle.device)
    keep_ids_t = torch.tensor([keep_ids], device=bundle.device)
    with torch.no_grad():
        p_del = float(torch.softmax(bundle.model(input_ids=del_ids_t, attention_mask=attention_mask)["hate_logits"][0], -1).max())
        p_keep = float(torch.softmax(bundle.model(input_ids=keep_ids_t, attention_mask=attention_mask)["hate_logits"][0], -1).max())
    comp_list.append(p_orig - p_del)
    suff_list.append(p_orig - p_keep)
results["ig_comprehensiveness_sufficiency"] = {
    "checkpoint": "hx_full",
    "n": len(comp_list),
    "comprehensiveness_mean": round(float(np.mean(comp_list)), 4),
    "comprehensiveness_std": round(float(np.std(comp_list)), 4),
    "sufficiency_mean": round(float(np.mean(suff_list)), 4),
    "sufficiency_std": round(float(np.std(suff_list)), 4),
    "note": "comprehensiveness = p_orig - p_without_evidence (higher = evidence necessary); "
            "sufficiency = p_orig - p_with_only_evidence (closer to 0 = evidence enough)",
}
print(results["ig_comprehensiveness_sufficiency"], flush=True)
del bundle
torch.cuda.empty_cache()

print("=== latency n=100 ===", flush=True)
from app.services.inference import InferenceService  # noqa: E402
svc = InferenceService(settings.model_copy(update={
    "checkpoint_path": str(ROOT / "backend/checkpoints/cc_context/best.pt")
}))
pairs = [
    ("They should all be kicked out.", "I saw a group of immigrants protesting downtown."),
    ("This is disgusting and they should leave.", "Immigrants are ruining everything."),
    ("Great point, I totally agree with you.", "The new policy is a disaster."),
    ("Nobody wants them here anyway.", "They keep taking our jobs."),
    ("Welcome to the neighborhood!", "We just moved in last week."),
]


def bench(fn, n=100, warmup=5):
    for _ in range(warmup):
        fn(pairs[0])
    times = []
    for i in range(n):
        p = pairs[i % len(pairs)]
        t0 = time.perf_counter()
        fn(p)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "n": n,
        "mean_ms": round(sum(times) / len(times), 1),
        "p50_ms": round(times[len(times) // 2], 1),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 1),
        "max_ms": round(times[-1], 1),
    }


svc.predict(*pairs[0])
lat = {
    "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "predict_no_context": bench(lambda p: svc.predict(p[0], None)),
    "predict_with_context": bench(lambda p: svc.predict(p[0], p[1])),
    "explain_ig32_with_context": bench(lambda p: svc.explain(p[0], p[1], target="hate")),
}
results["latency"] = lat
print(json.dumps(lat, indent=2), flush=True)
results["cc_context_vs_cc_baseline"] = {
    "n": 713, "a": "cc_context", "b": "cc_baseline",
    "acc_a": 0.6171, "acc_b": 0.5877, "acc_diff": 0.0295,
    "bootstrap_ci95": [-0.0042, 0.0617],
    "mcnemar_n01_b_wrong_a_right": 84, "mcnemar_n10_a_wrong_b_right": 63,
    "mcnemar_chi2": 2.721, "mcnemar_p": 0.09903,
}
results["hx_full_vs_baseline"] = {
    "n": 1924, "a": "hx_full", "b": "baseline",
    "acc_a": 0.6923, "acc_b": 0.6611, "acc_diff": 0.0312,
    "bootstrap_ci95": [0.012, 0.0499],
    "mcnemar_n01_b_wrong_a_right": 204, "mcnemar_n10_a_wrong_b_right": 144,
    "mcnemar_chi2": 10.003, "mcnemar_p": 0.001563,
}
(OUT / "addenda_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("WROTE addenda_results.json", flush=True)
