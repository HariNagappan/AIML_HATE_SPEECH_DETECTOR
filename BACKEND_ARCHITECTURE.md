# Backend Architecture — from first principles

A companion to [`README.md`](README.md). It explains **only the backend**: how it is put
together, *why* each piece exists, and where it is weak. Written to be read top-to-bottom
by someone who wants to understand the project rather than just run it.

**Full version with diagrams, tables and the complete gap analysis:
[`BACKEND_ARCHITECTURE.html`](BACKEND_ARCHITECTURE.html)** (open it in a browser).

Every number below came from this repository, not from an estimate.

---

## The one mental model

The backend is **two programs stapled together**.

| | Program A — the factory (offline) | Program B — the shop counter (online) |
| --- | --- | --- |
| Lives in | `scripts/`, `app/datasets/`, `app/training/`, `app/models/`, `app/evaluation/` | `app/main.py`, `app/api/`, `app/services/`, `app/reasoning/` |
| Does | downloads data → cleans it → teaches a network → writes `best.pt` | loads those numbers once at boot, answers HTTP in milliseconds |
| Never does | serve requests | train, download, write files |

The bridge between them is a **checkpoint** — one file holding both the learned weights *and*
a written description of how the network was built, so the server can rebuild the identical
shape before loading the weights.

```
PROGRAM A                                   PROGRAM B
data/raw/*.jsonl                            HTTP request
   │ download_data.py · preprocess.py          │
   ▼                                           ▼
UnifiedExample (one shared shape)         api/routes/*.py     (validate, call, wrap errors)
   │ collator.py                               │
   ▼                                           ▼
batches of tensors ─► FullModel           services/inference.py  (the only owner of the model)
   │ trainer.py                                │  1. tokenise
   ▼                                           │  2. one forward pass → class scores
checkpoints/*/best.pt  ◄── THE BRIDGE ──►      │  3. Integrated Gradients → which words mattered
                                               │  4. deterministic templates → English
                                               ▼
                                          api/schemas/*.py    (Pydantic JSON contract)
                                               │
                                               ▼
                                          JSON to the React frontend
```

---

## The first principles, in order

### 1. What BERT actually gives you
A sentence in, vectors out. Word-pieces → 768-number "meaning fingerprints" → run attention
12 times so every piece sees every other piece. Two things get used:

* the `[CLS]` vector — a fingerprint of the **whole sentence** (`E_c`, `E_p`)
* the per-token vectors — a fingerprint of **each word given the others** (for evidence)

The mechanism that makes this project possible: *"kicked"* gets a **different** vector depending
on the words around it. That is how "did the previous comment change the meaning?" becomes
computable — the numbers move. → `app/models/bert_encoder.py`

### 2. Why one encoder for two texts
The **same** BERT weights encode the comment and the context. Not two models. Because both
vectors come out of the same box, they live in the **same coordinate system** — so subtracting
and multiplying them is meaningful. With two separate encoders, comparing them would be
comparing two different vocabularies.

No context? `E_p` is `None` and the interaction layer substitutes zeros. Whole batches are
padded, so the collator also emits a `context_present` 1.0/0.0 mask that zeroes exactly the
rows with no real context.

### 3. The interaction block — how two fingerprints become one
```
D = |E_c − E_p|          "how far apart are the two meanings?"
M = E_c ⊙ E_p            "where do they agree?"
E_int = [E_c ; E_p ; D ; M]   →  4 × 768 = 3072
final = Dropout(GELU(Linear(3072 → 1024)))   →  1024 numbers
```
That single `Linear(3072→1024)` is **3,146,752 parameters** — the biggest block you designed.
A `simple` mode drops `D`/`M` down to plain `[E_c; E_p]` so the ablation study can prove the
engineered features help. → `app/models/context_interaction.py` (83 lines, 5 of them arithmetic)

### 4. Four heads over one shared understanding
| Head | Shape | Answers | Trained when |
| --- | --- | --- | --- |
| hate | 1024→3 | is this hate / offensive / normal? | always |
| target | 1024→9 | who is targeted? (+`none`) | dataset annotates targets |
| reason | 1024→8 | what kind of harm? | **only** with real reason annotations |
| evidence | per-token 768→1 | is this word part of the justification? | dataset has human rationales |

The reason head *always exists in the architecture* but is trained only on real annotations —
never on guessed labels, never as a keyword lookup.

**Do not conflate the two evidence techniques:** the evidence *head* is a learned predictor
trained against human highlights; **Integrated Gradients** is maths computed after the fact.
The UI shows IG. The repo compares the two experimentally.

### 5. Why missing labels never become fake labels
```
L_total = λ_hate·L_hate + λ_target·L_target + λ_reason·L_reason
          + λ_contrastive·L_contrastive + λ_evidence·L_evidence
```
Not every example has every label. Missing labels are stored as **`-100`** (`IGNORE_INDEX`);
PyTorch cross-entropy skips those rows. If every row in a batch is `-100`, the loss returns
`None` and that component is dropped from the sum entirely.

Two consequences worth internalising:
1. The model is never punished for a question it was not asked.
2. The trainer records `trained_heads` by observing **which losses actually fired**. That set
   goes into the checkpoint and is what the API reads back — so `*_available` flags are derived
   from reality, not hand-maintained.

→ `app/training/losses.py`

### 6. Contrastive learning — teaching the model that context matters
Without it the network can cheerfully ignore `E_p` and still score well, because the comment
alone often predicts the label. So: staple a comment to its **real** preceding comment and it
should look similar to the comment-only anchor; staple it to a **wrong**, randomly picked
context and it should look different. Cosine similarity, low temperature (0.07), InfoNCE.

**Training only.** At inference the anchor/positive/negative machinery is never built.

### 7. Evidence — how the model shows its work
The question: the model said "hate", *which words caused it?* A raw gradient is noisy (a big
gradient on a near-zero input means little). **Integrated Gradients** fixes that by walking a
straight line from a baseline (PAD = "nothing") to the real input in 32 steps, averaging
sensitivity along the way, then multiplying by the difference.

Then four passes turn scores into readable text → `app/reasoning/evidence_extractor.py`:
1. keep tokens scoring ≥ 0.15 × the strongest token (a **relative** threshold)
2. merge adjacent tokens into phrases
3. decode `##subword` pieces back into words
4. take the top 5

Every span is a **verbatim slice of the input** and carries `start`/`end` **character offsets**,
which is what the frontend highlights — no re-tokenising, no misalignment bugs.

### 8. The explanation layers — sentences from numbers, deterministically
**No LLM anywhere.** Two template layers:
* `reason_explainer.py` — `reason + target + evidence → {summary, details, grounded_in}`.
  Works only from those fields. Never claims the author's intent; never claims context *caused*
  the prediction (context is only ever reported as "included").
* `context_reasoner.py` — the pronoun→antecedent heuristic that answers *why the previous
  comment mattered*, e.g. `"They"` → `"group of immigrants"`, with offsets **on both sides**.
  Hand-curated vocabulary, fully inspectable. When it cannot establish a relationship it
  **abstains and says so**.

A reasoning failure must never break a classification, so the whole layer is wrapped in
`try/except` in `inference.py` and degrades to `"Reasoning unavailable."`

---

## Layer rules (what makes it maintainable)

```
API          routes/, schemas/    HTTP only. Knows nothing about tensors.
   ↓
Services     inference.py         the ONLY owner of model lifecycle + the pipeline
   ↓                     ↓
ML core      models/ training/    Torch, tensors, losses
             datasets/
Reasoning    reasoning/           pure functions over model OUTPUTS, no training
   ↓
Core         core/config.py       settings, paths, device. Imported by all, imports none.
```

| Rule | Why it earns its keep |
| --- | --- |
| Routes never touch a tensor | swap the model without touching the web layer |
| Model classes never read `.env` | tests can build any shape |
| Reasoning never trains anything | the explanation layer cannot silently change the prediction |
| Config imported by all, imports none | no cycles, one place to change a default |
| Heavy imports (`torch`, `transformers`) are function-local | `import app.main` is cheap |

Routes are ~15 lines each. They validate, call the service, and translate `RuntimeError`
into HTTP 503. **No business logic in a route.**

---

## Learnables — only ~1.4 M of the 113 M are yours

```
207 tensors, 112.96 M params
```

| Component | Parameters | In the served checkpoint |
| --- | --- | --- |
| Shared BERT encoder | ~111.5 M | trained (98.5% of the count is "the English language model") |
| `interaction.projection` (3072→1024) | 3,146,752 | trained — the biggest block you designed |
| `contrastive_projection` | ~328 K | **dead weight at inference** (training only) |
| `hate_head` (1024→3) | 3,075 | trained |
| `target_head` / `reason_head` / `evidence_head` | — | **absent from the file** — `use_*=false` |

That last row is a real design win, in layman terms: a switched-off component leaves a
*visible hole* rather than a random-weight stub. A random head would happily emit a
confident-looking label. Because the head does not exist, the code cannot even ask — so
`target_available: false` is *structurally* guaranteed, not a promise someone must remember.

Verify it yourself:
```bash
python -c "import torch; ck=torch.load('checkpoints/cc_context/best.pt', map_location='cpu', weights_only=False, mmap=True); print(ck['trained_heads'], ck['model_kwargs']['use_reason'])"
```

---

## Checkpoints

| Key | What it is | Who needs it |
| --- | --- | --- |
| `model_kwargs` | the **blueprint** — architecture, label lists, component switches | serving, to rebuild the shell |
| `model_state` | 207 tensors, 112.96 M values | serving |
| `trained_heads` | e.g. `["hate"]` | serving — drives `*_available` |
| `optimizer_state`, `scheduler_state` | momentum + LR position | **training only** — to resume exactly |

Think of optimizer state as the car's current speed and gear: needed to *keep driving*
(resume training), not to *park it in a showroom* (serve predictions). So
`scripts/export_serving_checkpoint.py` writes a copy without them — **~65% smaller** — plus a
`serving_export` provenance note. That is what you ship in a container.

Resolution order at startup: `MODEL_CHECKPOINT` → `checkpoints/default.json` pointer →
conventional paths → else build an **untrained** model that reports every head unavailable.

---

## Testing

```
149 passed, 2 skipped, 1 warning in 47.7s
```

The trick that makes a sub-minute suite possible for a BERT project:
* **no pretrained downloads by default** — tests build BERT shells with
  `BertEncoder.random_init(hidden_size=768, num_layers=2)`; correct *dimensions* tested,
  no 440 MB download
* the real tokenizer fixture **skips cleanly offline** (those are the 2 skips)
* real-model and integration tests sit behind `RUN_INTEGRATION_TESTS=1`

One environment gotcha: if `%TEMP%` is locked you get
`PermissionError: .../Temp/pytest-of-Haris` and 30 errors. That is Windows, not the code —
`python -m pytest --basetemp=.pytest-tmp`.

---

## Read the code in this order

1. `app/datasets/unified.py` — the one shape everything agrees on (note `None` is meaningful)
2. `app/core/config.py` — where every number comes from
3. `app/models/context_interaction.py` — the core idea in 83 lines
4. `app/models/full_model.py` — `encode_pair()` and `forward()`
5. `app/training/losses.py` — the `-100` masking trick
6. `app/training/trainer.py` — the loop, checkpointing, `trained_heads`
7. `app/services/inference.py` → `predict()` — the whole online pipeline in ~90 lines
8. `app/reasoning/attribution.py` — IG, explained in the docstring first
9. `app/reasoning/evidence_extractor.py` — scores → readable spans with offsets
10. `app/api/schemas/prediction.py` — the JSON contract and the `*_available` flags
11. `app/reasoning/context_reasoner.py` → `find_reference_links()`, `_compose_summary()`
12. `app/evaluation/ablation.py` — the six experiments that justify the architecture

---

## Where it breaks (summary — full detail in the HTML)

| Finding | Why it matters |
| --- | --- |
| **IG embedding path may skip position/token-type embeddings** | if so, the evidence describes a slightly *different* function than the one that produced the label — the single most important thing to verify with a numeric parity test |
| **Evidence is only computed for the hate head** | `/predict` hard-codes `target="hate"`, yet the UI can show those spans beside a target/reason label |
| **Contrastive loss is near-decoration** | `negatives_per_anchor=1`; negatives drawn globally at random from a seeded RNG inside `collate_fn`, which runs in parallel DataLoader workers |
| **Untrained checkpoint + no context → 500** | `encode_pair` calls the encoder with `input_ids=None` and only `RuntimeError` is caught at the boundary |
| **`model_info()` is fragile** | unguarded `bundle.model.hate_head`; `BaselineModel` names it `classifier` → `AttributeError` (500) |
| **Reason head has no validation signal** | `cc_reason.yaml` uses an unannotated val split, so it trains blind |
| **Counter Context label order is unverified** | flagged in `base.yaml` itself — if `"0"` is not hate speech, the headline model is trained on mislabelled data |
| **Random splits** | near-duplicate tweets leak between train and test, inflating metrics |
| **Reasoning tone comes from a keyword set** | a hard-coded `_POSITIVE_MARKERS` list, not the model, has the final word on how a decision is described |
| **Deployment gaps** | `CORS_ORIGINS=*`, no rate limit / auth / timeout, unpinned image, no healthcheck, non-root |
| **Honest number** | best val **macro-F1 0.566** over 3 classes (chance ≈ 0.33), selected at epoch 2 of 4 with val loss rising — overfitting, not a working detector |

**If you only fix three things:** (1) the IG parity test, (2) verify the Counter Context label
map against the NAACL 2022 paper, (3) use the datasets' official splits. Those three decide
whether the numbers in your report mean anything.

---

## Deployment notes

```bash
cd backend
pip install -r requirements.txt
python scripts/download_data.py --dataset all
python scripts/preprocess.py
python scripts/train.py --config cc_context          # GPU auto-detected
uvicorn app.main:app --port 8001
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness + `model_loaded` |
| `GET /api/v1/model/info` | model name, device, hidden size, label counts, trained heads |
| `POST /api/v1/predict` | prediction + target + reason + evidence + reasoning |
| `POST /api/v1/explain` | raw per-token attribution for a chosen head |

`/predict` accepts both `{text, context}` and `{current_comment, previous_comment}` (Pydantic
`AliasChoices`) — the frontend was never forced to rename its fields.

The predict endpoint is `def`, not `async def`, so FastAPI runs it in a threadpool — correct
here, since a blocking PyTorch forward pass would otherwise freeze the event loop. But the
inference service is an unsynchronised process-wide singleton (`@lru_cache`) holding a mutable
`_bundle`, with an `assert` that `python -O` strips. Fine for a research prototype; a real
design decision to be aware of, not an accident.
