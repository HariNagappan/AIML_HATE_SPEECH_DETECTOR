# -*- coding: utf-8 -*-
"""Finish reviewer-requested runs and write machine-readable results.

Stages (idempotent; skipped when a tag already has test_macro_f1):
  - evaluate any existing unevaluated cc_inter / cc_nce checkpoints
  - train cc_inter s2, cc_nce s42/s1/s2  (matched 4 epochs / batch 12)
  - train BERT cross-encoder [CLS] prev [SEP] current [SEP] x 3 seeds
  - train hx_full seeds 1 and 2
Results are merged into paper_assets/experiment_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

BACKEND = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\backend")
ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
OUT_JSON = ROOT / "paper_assets" / "experiment_results.json"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from app.core.config import get_settings, resolve_device  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.datasets.collator import build_label_maps  # noqa: E402
from app.datasets.tokenizer import TokenizerWrapper  # noqa: E402
from app.datasets.unified import load_unified_jsonl  # noqa: E402
from app.evaluation.ablation import evaluate_checkpoint  # noqa: E402
from app.training.configs import TrainConfig  # noqa: E402
from app.training.trainer import Trainer, build_model, set_seed  # noqa: E402
from dataclasses import replace  # noqa: E402

RESULTS: dict = {}
if OUT_JSON.exists():
    RESULTS = json.loads(OUT_JSON.read_text(encoding="utf-8"))


def save():
    OUT_JSON.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED", OUT_JSON, flush=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def row_done(tag: str, key: str = "cc_ladder") -> bool:
    return any(
        r.get("tag") == tag and r.get("test_macro_f1") is not None
        for r in RESULTS.get(key, [])
    )


def make_cfg(template: str, *, seed: int, output_dir: str, dataset: str | None = None,
             interaction_mode: str | None = None, use_contrastive: bool | None = None,
             mode: str | None = None, epochs: int | None = None,
             batch_size: int | None = None) -> TrainConfig:
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
        "config": getattr(cfg, "config_name", str(cfg.output_dir)),
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
    summary = cfg.resolved_output_dir() / "training_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        out["best_epoch"] = s.get("best_epoch")
        out["val_macro_f1"] = s.get("best_metric")
    return out


def append_row(key: str, row: dict):
    rows = RESULTS.setdefault(key, [])
    tag = row.get("tag")
    if tag:
        rows[:] = [r for r in rows if r.get("tag") != tag]
    rows.append(row)
    save()


def eval_existing_ckpt(tag: str, template: str, **kw):
    if row_done(tag):
        log(f"skip eval {tag}")
        return
    best = BACKEND / "checkpoints" / tag / "best.pt"
    if not best.exists():
        log(f"missing ckpt {tag}")
        return
    cfg = make_cfg(template, seed=kw.get("seed", 42), output_dir=f"checkpoints/{tag}",
                   dataset=kw.get("dataset", "counter_context"),
                   interaction_mode=kw.get("interaction_mode"),
                   use_contrastive=kw.get("use_contrastive"),
                   epochs=kw.get("epochs", 4), batch_size=kw.get("batch_size", 12))
    test = cfg.load_examples("test")
    m = evaluate_checkpoint(str(best), test, cfg, get_settings())
    hate = m.get("hate") or {}
    summary = BACKEND / "checkpoints" / tag / "training_summary.json"
    val = None
    if summary.exists():
        val = json.loads(summary.read_text(encoding="utf-8")).get("best_metric")
    append_row("cc_ladder", {
        "tag": tag, "variant": kw["variant"], "seed": kw.get("seed", 42),
        "dataset": "counter_context", "best_epoch": None, "val_macro_f1": val,
        "test_accuracy": hate.get("accuracy"), "test_macro_f1": hate.get("macro_f1"),
        "test_weighted_f1": hate.get("weighted_f1"),
        "test_per_class_f1": hate.get("per_class_f1"),
        "test_confusion": hate.get("confusion_matrix"),
        "support": hate.get("support"), "checkpoint": str(best),
        "config": template,
    })
    log(f"EVAL {tag} acc={hate.get('accuracy')} mf1={hate.get('macro_f1')}")


def stage_cc():
    settings = get_settings()
    device = resolve_device(settings.device)
    # evaluate leftover checkpoints
    eval_existing_ckpt("cc_inter_s1", "context", variant="cc_inter", seed=1,
                       interaction_mode="mlp", use_contrastive=False)
    eval_existing_ckpt("cc_inter_s42", "context", variant="cc_inter", seed=42,
                       interaction_mode="mlp", use_contrastive=False)
    # train missing
    jobs = [
        ("cc_inter", "context", dict(interaction_mode="mlp", use_contrastive=False),
         [42, 1, 2]),
        ("cc_nce", "cc_context", dict(interaction_mode="mlp", use_contrastive=True),
         [42, 1, 2]),
    ]
    for name, template, kw, seeds in jobs:
        for seed in seeds:
            tag = f"{name}_s{seed}"
            if row_done(tag):
                log(f"skip train {tag}")
                continue
            log(f"TRAIN {tag}")
            cfg = make_cfg(template, seed=seed, output_dir=f"checkpoints/{tag}",
                           dataset="counter_context", epochs=4, batch_size=12, **kw)
            try:
                r = train_and_eval(cfg, settings, device)
                r["tag"] = tag
                r["variant"] = name
                append_row("cc_ladder", r)
                log(f"DONE {tag} acc={r['test_accuracy']:.4f} mf1={r['test_macro_f1']:.4f}")
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                log(f"FAIL {tag}: {e}")
                append_row("cc_ladder", {"tag": tag, "variant": name, "seed": seed,
                                         "error": str(e)})


def stage_xe():
    settings = get_settings()
    device = resolve_device(settings.device)
    model_name = settings.encoder_name
    tok = AutoTokenizer.from_pretrained(model_name)
    label_list = ["hate_speech", "neither", "counter_speech"]
    lab2id = {n: i for i, n in enumerate(label_list)}

    def load(split):
        return load_unified_jsonl(BACKEND / f"data/processed/counter_context_{split}.jsonl")

    def encode_batch(examples):
        prev = [e.context_text or "" for e in examples]
        cur = [e.current_text or "" for e in examples]
        return tok(prev, cur, truncation=True, max_length=128, padding=True, return_tensors="pt")

    def predict(model, examples):
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

    def run_seed(seed: int) -> dict:
        set_seed(seed)
        torch.manual_seed(seed)
        train, val, test = load("train"), load("val"), load("test")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        num_steps = 4 * ((len(train) + 11) // 12)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * num_steps), num_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
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
            gold, preds = predict(model, val)
            vf1 = f1_score(gold, preds, average="macro")
            log(f"  xe seed{seed} epoch{epoch}: loss={sum(losses)/len(losses):.4f} val_macroF1={vf1:.4f}")
            if vf1 > best_f1:
                best_f1, best_epoch = vf1, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.to(device)
        gold, preds = predict(model, test)
        minutes = (time.perf_counter() - t0) / 60
        cm = [[0] * 3 for _ in range(3)]
        for a, b in zip(gold, preds):
            cm[a][b] += 1
        return {
            "seed": seed, "train_minutes": round(minutes, 2), "best_epoch": best_epoch,
            "val_macro_f1": best_f1,
            "test_accuracy": accuracy_score(gold, preds),
            "test_macro_f1": f1_score(gold, preds, average="macro"),
            "test_weighted_f1": f1_score(gold, preds, average="weighted"),
            "test_per_class_f1": {n: f for n, f in zip(label_list, f1_score(gold, preds, average=None))},
            "test_confusion": cm,
        }

    rows = RESULTS.setdefault("cross_encoder", [])
    for seed in (42, 1, 2):
        if any(r.get("seed") == seed and r.get("test_macro_f1") is not None for r in rows):
            log(f"skip xe s{seed}")
            continue
        log(f"TRAIN cross-encoder seed={seed}")
        try:
            r = run_seed(seed)
            r["variant"] = "cross_encoder"
            r["tag"] = f"xe_s{seed}"
            rows[:] = [x for x in rows if x.get("seed") != seed]
            rows.append(r)
            save()
            log(f"DONE xe s{seed} acc={r['test_accuracy']:.4f} mf1={r['test_macro_f1']:.4f}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"FAIL xe s{seed}: {e}")
            rows.append({"seed": seed, "error": str(e)})
            save()


def stage_hx_seeds():
    settings = get_settings()
    device = resolve_device(settings.device)
    rows = RESULTS.setdefault("hx_seeds", [])
    # include existing seed 42 from hx_full checkpoint for a complete 3-seed set
    if not any(r.get("tag") == "hx_full_s42" and r.get("test_macro_f1") for r in rows):
        cfg = TrainConfig.from_yaml("hx_full")
        test = cfg.load_examples("test")
        m = evaluate_checkpoint(str(BACKEND / "checkpoints/hx_full/best.pt"), test, cfg, settings)
        hate = m.get("hate") or {}
        summary = json.loads((BACKEND / "checkpoints/hx_full/training_summary.json").read_text())
        rows.append({
            "tag": "hx_full_s42", "seed": 42, "dataset": "hatexplain",
            "best_epoch": summary.get("best_epoch"), "val_macro_f1": summary.get("best_metric"),
            "test_accuracy": hate.get("accuracy"), "test_macro_f1": hate.get("macro_f1"),
            "test_weighted_f1": hate.get("weighted_f1"),
            "test_per_class_f1": hate.get("per_class_f1"),
            "test_confusion": hate.get("confusion_matrix"),
            "support": hate.get("support"),
            "checkpoint": str(BACKEND / "checkpoints/hx_full/best.pt"),
            "config": "hx_full", "train_minutes": None,
        })
        save()
        log(f"eval existing hx_full_s42 acc={hate.get('accuracy')}")
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
            rows[:] = [x for x in rows if x.get("tag") != tag]
            rows.append(r)
            save()
            log(f"DONE {tag} acc={r['test_accuracy']:.4f} mf1={r['test_macro_f1']:.4f}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"FAIL {tag}: {e}")
            rows.append({"tag": tag, "seed": seed, "error": str(e)})
            save()


def main():
    setup_logging("WARNING")
    log(f"device candidates ready; cuda={torch.cuda.is_available()}")
    for stage in ("cc", "xe", "hx_seeds"):
        log(f"=== STAGE {stage} ===")
        try:
            globals()[f"stage_{stage}"]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"STAGE FAIL {stage}: {e}")
        save()
    log("ALL STAGES FINISHED")


if __name__ == "__main__":
    main()
