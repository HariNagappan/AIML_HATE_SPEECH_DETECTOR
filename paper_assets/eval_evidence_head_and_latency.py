# Evaluate the supervised evidence head vs gold rationales (same protocol as
# evaluate_evidence: macro-averaged per-example token P/R/F1, first 100
# rationale-bearing HateXplain test examples), and measure inference latency
# of the serving checkpoint (predict vs full IG explain).
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
sys.path.insert(0, str(ROOT / "backend"))

import torch  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.datasets.tokenizer import TokenizerWrapper, align_rationales_to_subwords, is_special_token  # noqa: E402
from app.evaluation.rationale import token_binary_metrics  # noqa: E402
from app.services.inference import build_model_from_checkpoint  # noqa: E402
from app.training.configs import TrainConfig  # noqa: E402

settings = get_settings()
setup_logging(settings.log_level)
config = TrainConfig.from_yaml("hx_full")
examples = []
with open(ROOT / "backend/data/processed/hatexplain_test.jsonl", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        if ex.get("rationale_labels") and ex["rationale_labels"] != "None":
            examples.append(ex)
        if len(examples) >= 100:
            break
print(f"evidence-head eval on {len(examples)} rationale-bearing test examples")

bundle = build_model_from_checkpoint(str(ROOT / "backend/checkpoints/hx_full/best.pt"), settings)
model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
model.eval()

per_example = []
with torch.no_grad():
    for ex in examples:
        enc = tokenizer.encode(ex["current_text"], max_length=config.max_seq_length,
                               padding=False, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["rationale_logits"][0]          # [T]
        probs = torch.sigmoid(logits)
        tokens = tokenizer.token_strings(input_ids[0].tolist())
        mask = attention_mask[0].tolist()
        pred = [1 if (probs[i].item() >= 0.5 and mask[i] == 1 and not is_special_token(tokens[i])) else 0
                for i in range(len(tokens))]
        gold = align_rationales_to_subwords(tokenizer, ex["current_text"],
                                            ex["rationale_labels"] or [], max_length=config.max_seq_length)
        per_example.append(token_binary_metrics(gold, pred))

agg = {k: sum(e[k] for e in per_example) / len(per_example) for k in ("precision", "recall", "f1")}
micro_tp = sum(e["tp"] for e in per_example)
micro_fp = sum(e["fp"] for e in per_example)
micro_fn = sum(e["fn"] for e in per_example)
micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
result = {
    "method": "supervised_evidence_head (threshold 0.5)",
    "checkpoint": "hx_full",
    "num_examples": len(per_example),
    "macro_precision": agg["precision"],
    "macro_recall": agg["recall"],
    "macro_f1": agg["f1"],
    "micro_precision": micro_p,
    "micro_recall": micro_r,
    "micro_f1": micro_f1,
}
print(json.dumps(result, indent=2))
out = ROOT / "paper_assets" / "evidence_head_metrics.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ latency
from app.services.inference import InferenceService  # noqa: E402

svc_settings = settings.model_copy(update={"checkpoint_path": str(ROOT / "backend/checkpoints/cc_context/best.pt")})
svc = InferenceService(svc_settings)
pairs = [
    ("They should all be kicked out.", "I saw a group of immigrants protesting downtown."),
    ("This is disgusting and they should leave.", "Immigrants are ruining everything."),
    ("Great point, I totally agree with you.", "The new policy is a disaster."),
    ("Nobody wants them here anyway.", "They keep taking our jobs."),
    ("Welcome to the neighborhood!", "We just moved in last week."),
]


def bench(fn, n=15, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"mean_ms": sum(times) / len(times), "p50_ms": times[len(times) // 2],
            "p95_ms": times[int(len(times) * 0.95) - 1], "min_ms": times[0], "max_ms": times[-1]}


lat = {
    "device": str(next(svc.model.parameters()).device),
    "predict_no_context": bench(lambda: svc.predict(pairs[0][0], None)),
    "predict_with_context": bench(lambda: svc.predict(pairs[1][0], pairs[1][1])),
    "explain_ig32_with_context": bench(lambda: svc.explain(pairs[0][0], pairs[0][1], target="hate")),
}
print(json.dumps(lat, indent=2))
(ROOT / "paper_assets" / "latency.json").write_text(json.dumps(lat, indent=2), encoding="utf-8")
print("done")
