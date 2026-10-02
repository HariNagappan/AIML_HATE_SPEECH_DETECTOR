# Context-Aware Hate Speech Detection with Structured Evidence-Based Reasoning

A Python research backend that classifies hateful content **in conversational
context** and explains its decision with a **structured, evidence-based
reasoning output** — no generative text, no hallucinated reasoning.

Given a current comment and (optionally) the preceding comment(s), the system returns:

| Output | Question it answers |
| --- | --- |
| **Hate prediction** (`hate` / `offensive` / `normal`) | *What?* |
| **Target group** (race, religion, nationality, …) | *Who?* |
| **Reason category** (insult, threat, exclusion, …) | *Why / what type?* |
| **Evidence spans** (actual input phrases, attribution-based) | *Which words support it?* |
| **Context flag** | *Did context affect the decision?* |

> **Research honesty.** The evidence is produced by **post-hoc attribution**
> (Integrated Gradients), not attention weights and not an LLM. The system does
> not claim human-like reasoning; "reasoning" here means a structured,
> checkable explanation. See [Limitations](#limitations).

---

## Architecture

```
                       CURRENT COMMENT
                              │
                              ▼
                        ┌──────────┐
                        │  Shared  │◄─────────── PREVIOUS CONTEXT
                        │   BERT   │             (optional)
                        └────┬─────┘
                             │  E_c = BERT(comment)[CLS]
                             │  E_p = BERT(context)[CLS]   (same weights!)
                             ▼
                   Context Interaction
                    D = |E_c − E_p|
                    M = E_c ⊙ E_p
                   E_int = [E_c ; E_p ; D ; M]   (4 × 768 = 3072)
                             │
                             ▼
                   Interaction Layer (Linear → GELU → Dropout)
                             │
                             ▼
                   Final Representation  ──────────► Contrastive projection
                             │                       (InfoNCE, training only)
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                     ▼
    HATE HEAD           TARGET HEAD           REASON HEAD
        │                    │                     │
        └────────────────────┼─────────────────────┘
                             ▼
                   Evidence Extraction
                   (Integrated Gradients → token scores → merged spans)
                             │
                             ▼
                  Structured Explanation (deterministic builder)
```

Every component is individually switchable (`configs/*.yaml`) so the ablation
experiments can measure the contribution of context, interaction and
contrastive learning.

## Why BERT (and what it actually provides)

BERT is a pretrained bidirectional Transformer encoder. For every input it
produces **contextualised token representations** and a `[CLS]` sentence-level
representation. We use the `[CLS]` embedding as the comment/context embedding
(`E_c`, `E_p`) and the token representations for token-level rationale
supervision. The two inputs share **the same encoder weights**, so both
embeddings live in the same semantic space — this is what makes the
interaction features (`D`, `M`) meaningful.

## Context interaction

```
D = |E_c − E_p|          (absolute difference)
M = E_c ⊙ E_p            (element-wise product)
E_int = [E_c; E_p; D; M] → Linear(4H→1024) → GELU → Dropout
```

`H = 768` for BERT-base → `E_int` is 3072-dimensional. All dimensions are
configurable. When no previous comment is available, the context embedding
falls back to a zero vector (`context_used=false` is reported) — the design
keeps a future *N previous comments* extension open.

## Contrastive learning (training only)

The model learns that the interpretation of a comment depends on its context:

* **positive pair** `(C, P1)` — the comment with its *actual* preceding context
* **contrasting pair** `(C, P2)` — the comment with a *mismatched* context

with an InfoNCE objective over cosine similarities (temperature configurable,
default `0.07`). The anchor is the comment-only representation; the candidates
are the context-aware representations. Contrastive learning is **never** used
at inference — inference only produces the context-aware representation.

## Classification heads

* **Hate head** — `hate` / `offensive` / `normal` (softmax; HateXplain labels).
  A binary mode exists but must be configured explicitly; multiclass labels are
  never silently collapsed. Output includes per-class probabilities.
* **Target head** — race / ethnicity / nationality / religion / gender /
  sexual_orientation / political_group / other / none. Trained only where the
  dataset annotates targets (HateXplain does) via a label-mapping layer.
* **Reason head** — insult / dehumanization / negative_stereotyping / threat /
  exclusion / discrimination / incitement_to_violence / other. The architecture
  always contains this head, but it is **trained only when real reason
  annotations exist** (a reason-annotated dataset or a custom annotation file —
  see "Supervising the reason head" below). It is never trained on fabricated
  labels and is not a keyword lookup.
* **Evidence head (optional)** — supervised token-level rationale prediction
  (BCE over token states), trained when the dataset provides token-level
  rationales (HateXplain does).

### Evidence extraction at inference (attribution, not generation)

`app/reasoning/attribution.py` implements **Integrated Gradients** over the
input embeddings (a `gradient × input` variant is available). Pipeline:
tokenize → forward → select the predicted class → attribute per token →
threshold relative to the strongest token → merge adjacent tokens → decode
subwords back into readable phrases → return top-k spans, each a **verbatim
input span**. Each span also carries character offsets (`start`/`end` into the
original comment, end exclusive) plus inclusive token indices, which the
frontend uses for reliable highlighting.

**Supervised rationale prediction** (evidence head) and **post-hoc
attribution** are different techniques; the project supports comparing them
(see `exp6_evidence`).

### Reason explanation (deterministic)

`app/reasoning/reason_explainer.py` connects the predicted reason category to
the extracted evidence with fixed, auditable templates:

```
reason + target + evidence  →  {summary, details, grounded_in}
```

The explanation is **deterministic and grounded**: it only uses the predicted
reason, the predicted target, and the evidence spans that were actually
extracted — it never adds outside information, never claims to know the
author's intent, and never claims that the context *caused* the prediction
(context is only reported as "included"). When the reason head is
unavailable, no explanation is produced (`reason_explanation: null`).
Templates are defined in code and can be overridden via an optional
`configs/reason_templates.yaml` file.

### Supervising the reason head (own annotations)

The standard splits in this repository contain no reason labels, so the reason
head stays inactive (`reason_available: false`) until you supply real
annotations. The importer converts a small annotation file into
training-ready data — nothing is fabricated and invalid rows are reported:

```bash
# Annotation file (JSONL or CSV): required fields `text` and `reason`;
# optional: `id`, `context`, `target`. Reasons are normalised to the
# canonical categories ("Incitement to violence" → incitement_to_violence).
#   {"text": "Those people are disgusting and should leave.",
#    "reason": "exclusion", "context": "Why are you talking about immigrants?"}

# 1a. standalone dataset (reason head only):
python scripts/import_reason_annotations.py \
    --annotations data/reason_annotations.jsonl

# 1b. or merge into an existing split (keeps hate/evidence labels; matches by
#     `id` first, then by whitespace-normalized text; never overwrites):
python scripts/import_reason_annotations.py \
    --annotations data/reason_annotations.jsonl \
    --merge-into data/processed/counter_context_train.jsonl \
    --output data/processed/counter_context_reason_train.jsonl

# 2. train (configs/cc_reason.yaml already points at the merged files):
python scripts/train.py --config cc_reason

# 3. serve — the API and UI light up automatically once the serving checkpoint
#    reports a trained reason head (point MODEL_CHECKPOINT or
#    checkpoints/default.json at checkpoints/cc_reason/best.pt):
uvicorn app.main:app
```

Options: `--allow-unmatched` appends annotations that match nothing as new
reason-only examples; `--strict` fails without writing when any row is invalid
or conflicts with an existing label. A sample file lives at
`data/reason_annotations.example.jsonl`.

## Structured explanation (the API output)

```json
{
  "prediction": {"label": "hate", "confidence": 0.94, "probabilities": {"hate": 0.94, "offensive": 0.04, "normal": 0.02}},
  "prediction_available": true,
  "target": {"label": "nationality", "confidence": 0.89, "probabilities": {}},
  "target_available": true,
  "reason": {"label": "exclusion", "confidence": 0.87, "probabilities": {}},
  "reason_available": true,
  "evidence": [
    {"text": "kicked out", "start": 20, "end": 30, "token_start": 5, "token_end": 6, "score": 0.82},
    {"text": "disgusting", "start": 8, "end": 18, "token_start": 2, "token_end": 3, "score": 0.76}
  ],
  "reason_explanation": {
    "summary": "The highlighted language expresses exclusion by advocating that members of the targeted group be removed or kept out.",
    "details": "The phrase 'kicked out' advocates removing or keeping out members of the targeted group (predicted target: nationality), which corresponds to the predicted exclusion category.",
    "grounded_in": {"reason": "exclusion", "target": "nationality", "evidence": ["kicked out"]}
  },
  "context_used": true,
  "evidence_available": true
}
```

Untrained/unavailable heads are reported explicitly (`null` +
`*_available: false`) — predictions are never fabricated from untrained heads;
no reason explanation is produced without a trained reason head. (Character
offsets above are illustrative, end-exclusive, and refer to the original
comment: the primary mechanism for frontend highlighting.)

---

## Datasets

Two datasets, **two separate adapters, never merged at the raw-data level**.
Both convert to the shared `UnifiedExample` format, where unavailable
annotations are explicitly `None`.

| Dataset | Used for | Adapter |
| --- | --- | --- |
| **HateXplain** (primary) | hate/offensive/normal classification, target classification, rationale/evidence extraction | `app/datasets/hatexplain.py` |
| **Counter Context** (NAACL 2022, Yu, Blanco & Hong) | parent/current comment relationships, context-dependent classification, contrastive context experiments | `app/datasets/counter_context.py` |

* **HateXplain** — original corpus (`data/dataset.json` + split ids from
  `post_id_divisions.json`) downloaded from
  [hate-alert/HateXplain](https://github.com/hate-alert/HateXplain); labels are
  aggregated by majority vote over annotators, targets are mapped to canonical
  categories (unmapped values are preserved in `metadata`, never guessed), and
  rationales are merged from the majority-label annotators.
* **Counter Context** — JSONL corpus from
  [xinchenyu/counter_context](https://github.com/xinchenyu/counter_context);
  record schema `{"idx", "label", "context", "target"}` where **`context` is
  the preceding comment and `target` is the current comment** (naming quirk —
  `target` is *not* a target-group annotation). Provides `hate` / `counter` /
  `neither` style labels for context experiments and contrastive pairs.
  ⚠️ The numeric label mapping (default `"0"→hate, "1"→counter, "2"→neither`)
  should be **verified against the paper** before productive training; it is
  configurable in `configs/base.yaml`.

```bash
python scripts/download_data.py --dataset all   # raw data → data/raw/
python scripts/preprocess.py                    # → data/processed/*.jsonl
python scripts/inspect_dataset.py --path data/raw/counter_context/gold/train.jsonl
```

## Baseline vs proposed

| Model | Pipeline |
| --- | --- |
| **Baseline** (`configs/baseline.yaml`) | comment → BERT → `[CLS]` → hate head. **No context.** |
| **Context-aware** (`configs/context.yaml`) | + shared-context embedding + interaction |
| **Full** (`configs/full.yaml`) | + contrastive learning + multi-task heads (+ evidence) |

## Training

```
L_total = λ_hate·L_hate + λ_target·L_target + λ_reason·L_reason
          + λ_contrastive·L_contrastive + λ_evidence·L_evidence
```

A loss component is only included when the batch actually contains ground-truth
labels for it (`-100` marks missing labels) — no fake supervision. Every
component is logged per epoch.

```bash
cd backend
pip install -r requirements.txt

python scripts/train.py --config baseline                      # Experiment 1
python scripts/train.py --config context  --dataset counter_context
python scripts/train.py --config full                          # multi-task
python scripts/train.py --config cc_context --resume checkpoints/cc_context/best.pt  # resume a run
```

The trainer supports: train/val/test splits, warmup + linear decay, gradient
accumulation, mixed precision (CUDA only, automatic), gradient clipping,
checkpoints (`last.pt` / `best.pt` with optimizer/scheduler state, config,
label maps, metrics — resumable and reloadable for inference), early stopping
and full reproducibility (`seed` everywhere).

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/cc_context/best.pt
```

* **Hate**: accuracy, macro/weighted precision/recall/F1, per-class F1,
  confusion matrix, ROC-AUC (binary/2-class case).
* **Target / Reason**: accuracy, macro F1, per-class F1 (reason only when
  ground-truth reason labels exist).
* **Evidence**: token-level precision/recall/F1 against HateXplain rationales
  for both the supervised evidence head and the attribution extractor.
  Metrics that cannot be computed for a given dataset are not reported.

## Ablation experiments

`app/evaluation/ablation.py` defines six experiments (baseline → +context →
+interaction → +contrastive → full multi-task → +evidence evaluation). Results
are written to `experiments/*.json` plus a CSV summary. The code reports
numbers only — it never claims an improvement without experimental evidence.

## API

```bash
# serve (auto-loads a trained checkpoint if one exists in checkpoints/)
uvicorn app.main:app --reload

# serve a specific checkpoint explicitly
set MODEL_CHECKPOINT=checkpoints/cc_context/best.pt   # PowerShell: $env:MODEL_CHECKPOINT="..."
uvicorn app.main:app --reload
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness (+ model-loaded flag) |
| `GET /api/v1/model/info` | model name, device, hidden size, label counts, trained status, version |
| `POST /api/v1/predict` | classification + evidence + structured explanation |
| `POST /api/v1/explain` | detailed attribution (per-token scores + spans) for a chosen head |

Example request/response:

```json
POST /api/v1/predict
{"text": "They should all be kicked out.", "context": "Those immigrants are ruining everything."}

{
  "prediction": {"label": "hate", "confidence": 0.94, "probabilities": {"hate": 0.94, "offensive": 0.04, "normal": 0.02}},
  "prediction_available": true,
  "target": {"label": "nationality", "confidence": 0.89, "probabilities": {}},
  "target_available": true,
  "reason": {"label": "exclusion", "confidence": 0.87, "probabilities": {}},
  "reason_available": true,
  "evidence": [
    {"text": "kicked out", "start": 20, "end": 30, "token_start": 5, "token_end": 6, "score": 0.82}
  ],
  "reason_explanation": {
    "summary": "The highlighted language expresses exclusion by advocating that members of the targeted group be removed or kept out.",
    "details": "The phrase 'kicked out' advocates removing or keeping out members of the targeted group (predicted target: nationality), which corresponds to the predicted exclusion category.",
    "grounded_in": {"reason": "exclusion", "target": "nationality", "evidence": ["kicked out"]}
  },
  "context_used": true,
  "evidence_available": true
}
```

The model is loaded **once** at application startup and reused for every
request (`app/services/inference.py`); inference runs under `torch.no_grad`
(attribution temporarily enables gradients for input embeddings only).

**Checkpoint auto-loading.** When `MODEL_CHECKPOINT` is not set, the server
looks for a previously trained model: first the `checkpoints/default.json`
pointer (`{"checkpoint": "checkpoints/<run>/best.pt"}`), then the conventional
paths `checkpoints/cc_context/best.pt` → `checkpoints/hx_full/best.pt` →
`checkpoints/full/best.pt`. If none exists it serves an untrained model and
reports `trained: false`. Set `AUTO_LOAD_CHECKPOINT=false` to disable this.

Training never needs to be repeated unnecessarily: checkpoints persist on disk
(`checkpoints/<config>/best.pt` + `last.pt`, resumable via `--resume`), and
the served model can be switched at any time by setting `MODEL_CHECKPOINT`
or editing `checkpoints/default.json`.

## Project structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI app factory + lifespan model loading
│   ├── api/                    # routes (health, prediction, explanation, model) + schemas
│   ├── core/                   # config (env + YAML), logging
│   ├── models/                 # bert_encoder, context_interaction, contrastive,
│   │                           # hate/target/reason/evidence heads, baseline, full_model
│   ├── reasoning/              # attribution (Integrated Gradients), evidence_extractor,
│   │                           # reason_explainer, structured_explanation
│   ├── datasets/               # preprocessing, tokenizer, unified format, label mapping,
│   │                           # hatexplain + counter_context adapters, collators,
│   │                           # reason annotation importer
│   ├── training/               # trainer, losses, contrastive loss, configs
│   ├── evaluation/             # classification metrics, rationale metrics, ablations
│   └── services/               # inference service (singleton model lifecycle)
├── configs/                    # base + experiment YAML (baseline, context, full,
│                               # cc_context, hx_full, cc_reason)
├── data/                       # raw + processed (git-ignored; see data/README.md)
├── checkpoints/                # saved by training (git-ignored)
├── experiments/                # ablation results (git-ignored)
├── scripts/                    # download_data, preprocess, train, evaluate, explain,
│                               # import_reason_annotations, inspect
├── tests/                      # pytest suite (fast, offline-safe by default)
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## Configuration

* **Runtime** (`app/core/config.py`, `.env` / environment): model name
  (`MODEL_NAME=bert-base-uncased`), `MODEL_CHECKPOINT`, `DEVICE=auto|cuda|cpu`,
  label sets, evidence method/top-k/threshold, IG steps, loss weights,
  contrastive temperature, seeds, CORS. Copy `.env.example` → `.env`.
* **Training** (`configs/*.yaml`): encoder freeze mode, sequence length,
  interaction dims/mode, component switches, lr, batch size, epochs, warmup,
  accumulation, AMP, early stopping, output dir, dataset files, loss weights.
  Nothing about hyper-parameters is hardcoded in model classes.

## Testing

```bash
cd backend
python -m pytest -q                 # fast, offline-safe (small random BERTs)
RUN_INTEGRATION_TESTS=1 python -m pytest -m integration   # real bert-base-uncased (downloads weights)
```

Covered: preprocessing, tokenization + subword alignment, encoder shapes
(including the 768-dim spec checks), interaction math (`D`/`M`/3072),
contrastive loss, all heads, full-model forward + switches, evidence
extraction (merging, ranking, specials, dedupe), dataset adapters, label
mapping, trainer smoke (1 epoch + resume), inference service (untrained +
checkpoint paths), and the API endpoints (including validation errors).

## Docker

```bash
docker build -t hate-speech-backend .
docker run -p 8000:8000 -v /path/to/checkpoints:/srv/backend/checkpoints \
  -e MODEL_CHECKPOINT=/srv/backend/checkpoints/cc_context/best.pt hate-speech-backend
```

## Limitations

* Reported confidence is a model score — not a guarantee of correctness.
* Evidence spans are **attribution-based** (input-grounded but approximate);
  they are not a causal proof of the model's internal decision process.
* The reason head has no training data unless you plug in a reason-annotated
  file (see "Supervising the reason head"); it reports `reason_available: false`
  until then.
* Counter Context numeric label order should be verified against the paper.
* Target-category mapping for HateXplain communities is best-effort and
  configurable; unmapped raw values are always preserved in metadata.
* Running the full training pipeline on CPU is slow — a CUDA GPU is recommended
  (`DEVICE=auto` picks it up automatically).

## Future improvements

* Multi-context aggregation (N previous comments) in the interaction module.
* Collecting reason annotations for the existing splits (the importer and
  `cc_reason` config are ready — see "Supervising the reason head").
* Calibration of confidence scores and richer evidence evaluation
  (comprehensiveness / sufficiency).
* Optional richer frontend surfacing of `/api/v1/explain` (per-token
  attribution view).

## Quick command reference

```bash
pip install -r requirements.txt
python scripts/download_data.py --dataset all
python scripts/preprocess.py
python scripts/train.py --config baseline
python scripts/train.py --config context --dataset counter_context
python scripts/train.py --config full
python scripts/train.py --config cc_context
python scripts/train.py --config hx_full
python scripts/import_reason_annotations.py --annotations data/reason_annotations.jsonl
python scripts/train.py --config cc_reason   # needs the merged reason files
python scripts/evaluate.py --checkpoint checkpoints/cc_context/best.pt
python scripts/export_serving_checkpoint.py --checkpoint checkpoints/cc_context/best.pt
python scripts/explain.py --text "They should all be kicked out." \
  --context "Those immigrants are ruining everything."
uvicorn app.main:app --reload
python -m pytest -q
```
