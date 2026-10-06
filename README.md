# Context-Aware Hate Speech Detection with Structured Evidence-Based Reasoning

A full-stack research prototype: a context-aware BERT classifier that turns a
comment (with optional conversational context) into a **structured, explainable
result** — prediction, target, reason, attribution-based evidence, and a
deterministic reason explanation. No generative LLM anywhere in the pipeline.

```
User input (comment + context)
        ↓
BERT encoder (shared) → context interaction → contrastive representation
        ↓
hate / target / reason heads  →  Integrated-Gradients evidence spans
        ↓
deterministic reason explanation  →  structured JSON  →  React UI
```

## Contents

| Folder | Stack | Docs |
| --- | --- | --- |
| [`backend/`](backend/) | Python · FastAPI · PyTorch · Hugging Face Transformers | [backend/README.md](backend/README.md) |
| [`frontend/`](frontend/) | React · TypeScript · Vite · Tailwind CSS | [frontend/README.md](frontend/README.md) |

## Quick start

**1. Backend** (serves the model + API; auto-loads a trained checkpoint when
one is present in `backend/checkpoints/`):

```bash
cd backend
pip install -r requirements.txt
# first time only — fetch + preprocess the datasets:
python scripts/download_data.py --dataset all
python scripts/preprocess.py
# train (GPU recommended, auto-detected): e.g.
python scripts/train.py --config cc_context
# evaluate the checkpoint (writes checkpoints/cc_context/best.metrics.json,
# which the API serves and the UI shows in the "Evaluation metrics" card):
python scripts/evaluate.py --config cc_context
# serve:
uvicorn app.main:app --port 8001
```

To train **everything** (all configurations across both datasets, full train
splits) on Windows with the CUDA `ml-gpu` conda env — full walkthrough in
[`backend/README.md` → Training](backend/README.md#training):

```powershell
$py = "C:\Users\Haris\anaconda3\envs\ml-gpu\python.exe"
& $py scripts\train.py --config baseline                     # hatexplain, comment-only
& $py scripts\train.py --config context --dataset counter_context
& $py scripts\train.py --config full                         # multi-task
& $py scripts\train.py --config cc_context                   # context-aware
& $py scripts\train.py --config hx_full                      # hate + target + evidence
```

**2. Frontend** (backend URL is configured in `frontend/.env`):

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

## At a glance

* **Structured reasoning UI** — WHAT (prediction) → WHO (target) → WHY (reason) →
  WHICH PARTS (evidence with numbered markers and attribution scores) →
  DID CONTEXT MATTER (context flag). Unavailable sections render an explicit
  "Not available" state (e.g. when the served checkpoint has no target/reason
  head) — never faked.
* **Evaluation metrics in the UI** — `scripts/evaluate.py` writes
  `<checkpoint>.metrics.json`; `/api/v1/model/info` serves the summary and the
  frontend renders an "Evaluation metrics" card (accuracy, macro
  precision/recall/F1) for the served checkpoint — no hardcoded numbers.
* **How reasoning works** — [`REASONING.md`](REASONING.md): the full pipeline
  (prediction → attribution evidence → context-reference analysis →
  explanation), with the honesty rules it follows.
* **Honest by design** — evidence comes from Integrated Gradients over the
  model, explanations are deterministic template composition grounded in the
  predicted fields, and every section renders only from real model outputs
  (`*_available` flags).
* **Tests** — backend: `python -m pytest -q` (offline-safe; integration opt-in
  via `RUN_INTEGRATION_TESTS=1`); frontend: `npm run test`.
* **Reason head** — becomes active when trained on real reason annotations;
  see `backend/README.md` → "Supervising the reason head".
* **Deployment** — `backend/scripts/export_serving_checkpoint.py` produces a
  serve-lean checkpoint (~65% smaller) for containers / transfers.
