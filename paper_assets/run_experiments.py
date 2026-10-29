# -*- coding: utf-8 -*-
"""Run the reviewer-requested experiments and dump machine-readable results.

Stages (each writes into paper_assets/experiment_results.json):
  eval_existing  - test-split hate metrics for baseline / context / full
  cc             - Counter Context ladder x 3 seeds:
                   no-context | concat | interaction-MLP | interaction+InfoNCE
  xe             - cross-encoder ([CLS] prev [SEP] current [SEP]) x 3 seeds
  hx_seeds       - hx_full with seeds 1 and 2 (seed 42 already exists)
  evidence       - supervised evidence head vs Integrated Gradients (same 100 examples)
  latency        - production predict() latency of the served cc_context checkpoint
  refs           - reference-linker coverage on the Counter Context test set
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

BACKEND = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\backend")
ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
OUT_JSON = ROOT / "paper_assets" / "experiment_results.json"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.core.config import get_settings, resolve_device  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.datasets.collator import build_label_maps  # noqa: E402
from app.datasets.tokenizer import TokenizerWrapper  # noqa: E402
from app.training.configs import TrainConfig  # noqa: E402
from app.training.trainer import Trainer, build_model, set_seed  # noqa: E402
from app.evaluation.ablation import evaluate_checkpoint, evaluate_evidence  # noqa: E402
from app.evaluation.rationale import token_binary_metrics  # noqa: E402
from app.datasets.tokenizer import align_rationales_to_subwords  # noqa: E402
from app.reasoning.context_reasoner import find_reference_links  # noqa: E402
from app.services.inference import InferenceService, build_model_from_checkpoint  # noqa: E402
from app.core.config import get_settings as _gs  # noqa: E402

RESULTS: dict = {}
if OUT_JSON.exists():
    RESULTS = json.loads(OUT_JSON.read_text(encoding="utf-8"))


def save():
    OUT_JSON.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED", OUT_JSON, flush=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_and_eval(cfg: TrainConfig, settings, device) -> dict:
    set_seed(cfg.seed)
    train_examples = cfg.load_examples("train")
    val_examples = cfg.load_examples("val")
    test_examples = cfg.load_examples("test")
    labels = cfg.resolved_label_sets()
    label_maps = build_label_maps(labels["hate"], labels["target"], labels["reason"])
    tokenizer = TokenizerWrapper.from_pretrained(settings.encoder_name)
    model, model_kwargs = build_model(cfg, settings, device)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, train_examples=train_examples,
        val_examples=val_examples, config=cfg, device=device,
        label_maps=label_maps, model_kwargs=model_kwargs,
    )
    t0 = time.perf_counter()
    trainer.train()
    train_min = (time.perf_counter() - t0) / 60
    best_pt = cfg.resolved_output_dir() / "best.pt"
    metrics = evaluate_checkpoint(str(best_pt), test_examples, cfg, settings)
    hate = metrics.get("hate") or {}
    out = {
        "config": cfg.config_name if hasattr(cfg, "config_name") else str(cfg.output_dir),
        "seed": cfg.seed,
        "dataset": cfg.dataset_name,
        "train_minutes": round(train_min, 2),
        "best_epoch": None,
        "val_macro_f1": None,
        "test_accuracy": hate.get("accuracy"),
        "test_macro_f1": hate.get("macro_f1"),
        "test_weighted_f1": hate.get("weighted_f1"),
        "test_per_class_f1": hate.get("per_class_f1"),
        "test_confusion": hate.get("confusion_matrix"),
        "support": hate.get("support"),
        "checkpoint": str(best_pt),
    }
    # pull best-epoch info from training summary if present
    summary = cfg.resolved_output_dir() / "training_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        out["best_epoch"] = s.get("best_epoch")
        out["val_macro_f1"] = s.get("best_metric")
    return out


def make_cfg(template: str, *, seed: int, output_dir: str, dataset: str | None = None,
             interaction_mode: str | None = None, use_contrastive: bool | None = None,
             mode: str | None = None, epochs: int | None = None, batch_size: int | None = None) -> TrainConfig:
    overrides = {"dataset": dataset} if dataset else None
    cfg = TrainConfig.from_yaml(template, overrides=overrides)
    updates: dict = dict(seed=seed, output_dir=output_dir)
    if dataset:
        updates["dataset_name"] = dataset
    if interaction_mode:
        updates["interaction_mode"] = interaction_mode
    if use_contrastive is not None:
        updates["use_contrastive"] = use_contrastive
    if mode:
        updates["mode"] = mode
    if epochs is not None:
        updates["epochs"] = epochs
    if batch_size is not None:
        updates["batch_size"] = batch_size
    return replace(cfg, **updates)


def stage_eval_existing(settings, device):
    jobs = [
        ("baseline", "checkpoints/baseline/best.pt", "baseline", None),
        ("context", "checkpoints/context/best.pt", "context", "counter_context"),
        ("full", "checkpoints/full/best.pt", "full", None),
    ]
    out = {}
    for name, ckpt, cfg_name, ds in jobs:
        if not Path(ckpt).exists():
            out[name] = {"error": "checkpoint missing"}
            continue
        cfg = TrainConfig.from_yaml(cfg_name, overrides={"dataset": ds} if ds else None)
        test = cfg.load_examples("test")
        m = evaluate_checkpoint(ckpt, test, cfg, settings)
        hate = m.get("hate") or {}
        out[name] = {
            "test_accuracy": hate.get("accuracy"),
            "test_macro_f1": hate.get("macro_f1"),
            "test_weighted_f1": hate.get("weighted_f1"),
            "test_per_class_f1": hate.get("per_class_f1"),
            "test_macro_precision": hate.get("macro_precision"),
            "test_macro_recall": hate.get("macro_recall"),
            "support": hate.get("support"),
            "confusion": hate.get("confusion_matrix"),
        }
        log(f"eval_existing {name}: acc={out[name]['test_accuracy']:.4f} macroF1={out[name]['test_macro_f1']:.4f}")
    RESULTS["eval_existing"] = out


def stage_cc(settings, device):
    seeds = [42, 1, 2]
    variants = {
        "cc_baseline": dict(template="baseline", dataset="counter_context", mode="baseline",
                            use_contrastive=False, epochs=4, batch_size=12),
        "cc_concat": dict(template="context", dataset="counter_context", interaction_mode="simple",
                          use_contrastive=False, epochs=4, batch_size=12),
        "cc_inter": dict(template="context", dataset="counter_context", interaction_mode="mlp",
                         use_contrastive=False, epochs=4, batch_size=12),
        "cc_nce": dict(template="cc_context", dataset="counter_context", interaction_mode="mlp",
                       use_contrastive=True, epochs=4, batch_size=12),
    }
    rows = RESULTS.get("cc_ladder", [])
    for name, kw in variants.items():
        for seed in seeds:
            tag = f"{name}_s{seed}"
            if any(r.get("tag") == tag and r.get("test_macro_f1") is not None for r in rows):
                log(f"skip {tag} (done)")
                continue
            log(f"TRAIN {tag}")
            cfg = make_cfg(kw["template"], seed=seed, output_dir=f"checkpoints/{tag}",
                           dataset=kw["dataset"], interaction_mode=kw.get("interaction_mode"),
                           use_contrastive=kw.get("use_contrastive"), mode=kw.get("mode"),
                           epochs=kw["epochs"], batch_size=kw["batch_size"])
            try:
                r = train_and_eval(cfg, settings, device)
                r["tag"] = tag
                r["variant"] = name
                rows.append(r)
                RESULTS["cc_ladder"] = rows
                save()
                log(f"DONE {tag} acc={r['test_accuracy']:.4f} macroF1={r['test_macro_f1']:.4f}")
            except Exception as e:  # noqa: BLE001
                log(f"FAIL {tag}: {e}")
                rows.append({"tag": tag, "variant": name, "seed": seed, "error": str(e)})
                RESULTS["cc_ladder"] = rows
                save()


def stage_xe(settings, device):
    """Cross-encoder baseline: [CLS] previous [SEP] current [SEP]."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from app.datasets.unified import load_unified_jsonl

    model_name = settings.encoder_name
    tok = AutoTokenizer.from_pretrained(model_name)
    label_list = ["hate_speech", "neither", "counter_speech"]
    lab2id = {n: i for i, n in enumerate(label_list)}

    def load(split):
        path = BACKEND / f"data/processed/counter_context_{split}.jsonl"
        return load_unified_jsonl(path)

    def encode_batch(examples):
        prev = [e.context_text or "" for e in examples]
        cur = [e.current_text or "" for e in examples]
        return tok(prev, cur, truncation=True, max_length=128, padding=True, return_tensors="pt")

    def run_seed(seed: int) -> dict:
        set_seed(seed)
        torch.manual_seed(seed)
        train, val, test = load("train"), load("val"), load("test")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        num_steps = 4 * ((len(train) + 11) // 12)
        from transformers import get_linear_schedule_with_warmup
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * num_steps), num_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        from sklearn.metrics import f1_score, accuracy_score

        def predict(examples) -> tuple[list, list]:
            model.eval()
            preds, gold = [], []
            with torch.no_grad():
                for i in range(0, len(examples), 32):
                    chunk = examples[i:i + 32]
                    enc = encode_batch(chunk)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        logits = model(**enc).logits
                    preds += logits.argmax(-1).cpu().tolist()
                    gold += [lab2id[e.hate_label] for e in chunk]
            return gold, preds

        best_f1, best_state, best_epoch = -1.0, None, -1
        rng = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        for epoch in range(4):
            model.train()
            order = torch.randperm(len(train), generator=rng).tolist()
            losses = []
            for i in range(0, len(order), 12):
                chunk = [train[j] for j in order[i:i + 12]]
                enc = encode_batch(chunk)
                enc = {k: v.to(device) for k, v in enc.items()}
                y = torch.tensor([lab2id[e.hate_label] for e in chunk], device=device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = model(**enc, labels=y).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(float(loss))
            gold, preds = predict(val)
            vf1 = f1_score(gold, preds, average="macro")
            log(f"  xe seed{seed} epoch{epoch}: loss={sum(losses)/len(losses):.4f} val_macroF1={vf1:.4f}")
            if vf1 > best_f1:
                best_f1, best_epoch = vf1, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.to(device)
        gold, preds = predict(test)
        minutes = (time.perf_counter() - t0) / 60
        return {
            "seed": seed, "train_minutes": round(minutes, 2), "best_epoch": best_epoch,
            "val_macro_f1": best_f1,
            "test_accuracy": accuracy_score(gold, preds),
            "test_macro_f1": f1_score(gold, preds, average="macro"),
            "test_weighted_f1": f1_score(gold, preds, average="weighted"),
            "test_per_class_f1": {n: f for n, f in zip(label_list, f1_score(gold, preds, average=None))},
            "test_confusion": json.loads(json.dumps(confusion(gold, preds, 3))),
        }

    def confusion(y, p, k):
        cm = [[0] * k for _ in range(k)]
        for a, b in zip(y, p):
            cm[a][b] += 1
        return cm

    rows = RESULTS.get("cross_encoder", [])
    for seed in (42, 1, 2):
        if any(r.get("seed") == seed and r.get("test_macro_f1") is not None for r in rows):
            log(f"skip xe s{seed}")
            continue
        log(f"TRAIN cross-encoder seed={seed}")
        try:
            r = run_seed(seed)
            r["variant"] = "cross_encoder"
            rows.append(r)
            RESULTS["cross_encoder"] = rows
            save()
            log(f"DONE xe s{seed} acc={r['test_accuracy']:.4f} macroF1={r['test_macro_f1']:.4f}")
        except Exception as e:  # noqa: BLE001
            log(f"FAIL xe s{seed}: {e}")
            rows.append({"seed": seed, "error": str(e)})
            RESULTS["cross_encoder"] = rows
            save()


def stage_hx_seeds(settings, device):
    rows = RESULTS.get("hx_seeds", [])
    for seed in (1, 2):
        tag = f"hx_full_s{seed}"
        if any(r.get("tag") == tag and r.get("test_macro_f1") is not None for r in rows):
            log(f"skip {tag}")
            continue
        log(f"TRAIN {tag}")
        cfg = make_cfg("hx_full", seed=seed, output_dir=f"checkpoints/{tag}")
        try:
            r = train_and_eval(cfg, settings, device)
            r["tag"] = tag
            rows.append(r)
            RESULTS["hx_seeds"] = rows
            save()
            log(f"DONE {tag} acc={r['test_accuracy']:.4f} macroF1={r['test_macro_f1']:.4f}")
        except Exception as e:  # noqa: BLE001
            log(f"FAIL {tag}: {e}")
            rows.append({"tag": tag, "seed": seed, "error": str(e)})
            RESULTS["hx_seeds"] = rows
            save()


def stage_evidence(settings, device):
    """Supervised evidence head vs IG on the same 100 rationale-bearing test posts."""
    cfg = TrainConfig.from_yaml("hx_full")
    test = [e for e in cfg.load_examples("test") if e.has_rationale()][:100]
    bundle = build_model_from_checkpoint(str(Path("checkpoints/hx_full/best.pt")), settings)
    model, tok = bundle.model, bundle.tokenizer
    model.eval()

    def head_mask(example):
        from app.datasets.tokenizer import TokenizerWrapper as TW
        enc = tok.encode(example.current_text, max_length=cfg.max_seq_length,
                         padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.to(bundle.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out["rationale_logits"][0]
        probs = torch.sigmoid(logits).cpu().tolist()
        att = enc["attention_mask"][0].cpu().tolist()
        from app.datasets.tokenizer import is_special_token
        mask = []
        for i, p in enumerate(probs):
            special = is_special_token(tok.token_strings([int(enc["input_ids"][0, i])])[0]) if False else False
            mask.append(1 if (att[i] == 1 and p >= 0.5) else 0)
        return mask

    from app.datasets.tokenizer import is_special_token
    def head_mask2(example):
        enc = tok.encode(example.current_text, max_length=cfg.max_seq_length,
                         padding="max_length", truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(bundle.device)
        am = enc["attention_mask"].to(bundle.device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=am)
        probs = torch.sigmoid(out["rationale_logits"][0]).cpu().tolist()
        tokens = tok.token_strings([int(t) for t in ids[0].tolist()])
        pred = [1 if (am[0, i] == 1 and not is_special_token(tokens[i]) and probs[i] >= 0.5) else 0
                for i in range(len(tokens))]
        gold = align_rationales_to_subwords(tok, example.current_text, example.rationale_labels or [],
                                            max_length=cfg.max_seq_length)
        return token_binary_metrics(gold, pred)

    per = [head_mask2(e) for e in test]
    head = {k: sum(x[k] for x in per) / len(per) for k in ("precision", "recall", "f1")}
    head["num_examples"] = len(per)

    ig = evaluate_evidence(str(Path("checkpoints/hx_full/best.pt")), cfg.load_examples("test"),
                           cfg, settings, max_examples=100)
    RESULTS["evidence_head_vs_ig"] = {"supervised_head_macro": head, "ig_macro": ig}
    log(f"evidence head macro F1={head['f1']:.4f} | IG macro F1={(ig or {}).get('f1')}")


def stage_latency(settings, device):
    from app.core.config import replace as _  # noqa: F401
    svc_settings = settings
    svc = InferenceService(svc_settings)
    svc.load()
    import statistics
    from app.datasets.unified import load_unified_jsonl
    pairs = load_unified_jsonl(BACKEND / "data/processed/counter_context_test.jsonl")[:10]
    # warmup
    for e in pairs[:3]:
        svc.predict(e.current_text, e.context_text)
    def bench(with_ctx: bool, n: int = 50):
        lat = []
        for i in range(n):
            e = pairs[i % len(pairs)]
            t0 = time.perf_counter()
            if with_ctx:
                svc.predict(e.current_text, e.context_text)
            else:
                svc.predict(e.current_text, None)
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        return {
            "n": n, "mean_ms": round(statistics.mean(lat), 1),
            "p50_ms": round(lat[len(lat)//2], 1),
            "p95_ms": round(lat[int(len(lat)*0.95)-1], 1),
            "max_ms": round(max(lat), 1),
        }
    dev = str(bundle_device(svc))
    RESULTS["latency"] = {
        "device": dev,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "with_context": bench(True),
        "without_context": bench(False),
        "note": "predict() includes IG attribution (32 steps) + reasoning layer; batch=1",
    }
    log(f"latency ctx={RESULTS['latency']['with_context']}")

def bundle_device(svc):
    return next(svc._require_bundle().model.parameters()).device


def stage_refs(settings, device):
    from app.datasets.unified import load_unified_jsonl
    test = load_unified_jsonl(BACKEND / "data/processed/counter_context_test.jsonl")
    n = len(test)
    n_any = n_ant = n_links = 0
    pronouns = {}
    for e in test:
        links = find_reference_links(e.context_text, e.current_text)
        n_links += len(links)
        if links:
            n_any += 1
        if e.context_text:
            from app.reasoning.context_reasoner import _find_antecedent, _tokens
            if _find_antecedent(_tokens(e.context_text or "")):
                n_ant += 1
        for L in links:
            pronouns[getattr(L, "pronoun", "?")] = pronouns.get(getattr(L, "pronoun", "?"), 0) + 1
    RESULTS["reference_linker"] = {
        "n_pairs": n, "pairs_with_link": n_any, "coverage": n_any / n,
        "total_links": n_links, "links_per_pair": n_links / n,
        "prev_with_antecedent_candidate": n_ant,
        "pronoun_counts": pronouns,
        "note": "heuristic coverage; no gold link annotations exist for this corpus",
    }
    log(f"refs coverage={RESULTS['reference_linker']['coverage']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="eval_existing,cc,xe,hx_seeds,evidence,latency,refs")
    args = ap.parse_args()
    setup_logging("WARNING")
    settings = get_settings()
    device = resolve_device(settings.device)
    log(f"device={device}")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for st in stages:
        log(f"=== STAGE {st} ===")
        try:
            globals()[f"stage_{st}"](settings, device)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"STAGE FAIL {st}: {e}")
        save()
    log("ALL REQUESTED STAGES FINISHED")


if __name__ == "__main__":
    main()
