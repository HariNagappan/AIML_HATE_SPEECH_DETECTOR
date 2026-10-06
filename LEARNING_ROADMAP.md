# Learning Roadmap — from scratch to understanding this project

Companion to [`BACKEND_ARCHITECTURE.md`](BACKEND_ARCHITECTURE.md) / [`.html`](BACKEND_ARCHITECTURE.html).
That pair explains *what the project is*. **This file is what to learn, in what order.**

**Full version with phase cards, tables and a progress tracker:
[`LEARNING_ROADMAP.html`](LEARNING_ROADMAP.html)** (open in a browser).

---

## Three truths that change how you study this

1. **The project reuses ~25 concepts, not ~200.** Almost every file is a permutation of the same
   small set: a dataclass, an `nn.Module`, a `forward()`, a config lookup, a try/except. Learn the
   set once and the codebase becomes readable in an afternoon.
2. **You only need calculus intuition, not calculus.** Autograd computes every derivative for you.
   You need to know *what a gradient is* and *why averaging over a path beats one sample at a
   point*. No proofs.
3. **BERT is a downloaded object, not something you build.** You write
   `AutoModel.from_pretrained("bert-base-uncased")` and get 111.5 M trained parameters. Everything
   you author sits *around* it.

**Sizing, measured from this repo:** 52 Python files / **5,987 lines** in `backend/app`; 18 test
files (149 passing, 47.7 s); 9 files defining an `nn.Module`; exactly **1** file using autograd.
The frontend (47 files, 4,303 lines) is a separate skill and this roadmap stops at the JSON contract.

---

## The 4-layer spine — build toward this

```
LAYER 1 · DATA         "what shape is a training example?"
   UnifiedExample { current_text, context_text, hate/target/reason labels, rationale }
   key idea: None means "we were not told". It does NOT mean "none".
                                │
LAYER 2 · MODEL        "how do texts become a prediction?"      (Phase 2 + 3)
   shared BERT → E_c, E_p → |Ec−Ep|, Ec⊙Ep → [4×768] → Linear → 1024 → heads → logits
   key idea: the SAME weights encode both texts, so the vectors are comparable.
                                │
LAYER 3 · TRAINING     "how does it get good?"                  (Phase 4 + 5)
   masked multi-task loss (−100 = skip) + InfoNCE.
   key idea: one missing label must never become a fake label.
                                │
LAYER 4 · EXPLANATION  "why did it say that?"                   (Phase 6 + 7)
   Integrated Gradients → token scores → merged spans with character offsets
   → deterministic templates → JSON. No LLM writes a single word.
```

---

## Phase 0 — the Python gap (1–2 weeks)

The phase people skip and then drown in Phase 2. **Diagnosis from your machine:** you have
Python 3.14 with `torch 2.14.1+cpu` globally installed, but `python` isn't on your shell PATH and
pytest is fighting Windows temp-folder permissions. That's an *environment* gap, not a knowledge
gap — and it will waste hours if you don't fix it first.

**Step 0.1 — fix the environment today**
- Install Python **3.11 or 3.12** (not 3.14) — matching `backend/README.md`'s "Python 3.11+".
  Wheel availability for torch/transformers is better, and that matters more than being current.
- Tick "Add python.exe to PATH" in the installer.
- Learn virtual environments properly: `python -m venv .venv` → activate → `pip install -r requirements.txt`
- Run: `python -m pytest -q --basetemp=.pytest-tmp` → **149 passed, 2 skipped**.
  That green run is your baseline sanity check for the rest of this roadmap.

**Step 0.2 — the Python this codebase actually uses** (verified by grepping the repo, not invented)
- **Type hints everywhere** — `Optional[str]`, `List[Dict[str, Any]]`, `->`. Read them as docs.
- **Dataclasses** — `@dataclass class UnifiedExample`; auto-generated `__init__`. Used in 9 files.
  Learn `field(default_factory=list)`.
- **Pydantic v2 `BaseModel`** — runtime validation of API bodies. *Different thing from a dataclass:
  it checks at runtime.*
- **Decorators** — `@property`, `@classmethod`, `@staticmethod`, `@lru_cache`,
  `@contextmanager`, `@field_validator`, `@torch.no_grad()`. Five appear here.
- **`Optional` / narrowing** — you'll constantly see `if x is not None:`. Understand *why*.
- **Function-local imports** — `import torch` inside a function, deliberately, to keep imports cheap.
- **Context managers** — `with open(...)`, `with torch.no_grad():`, `with torch.autocast(...)`.
- **Generators / `yield`** — appears once (FastAPI `lifespan`). Recognise it, don't master it.
- **`*args` / `**kwargs`** — `model(**kwargs)` is everywhere. Know the unpacking operator cold.

> **Done when:** the suite runs green from a clean venv and you can read
> `backend/app/datasets/unified.py` with no lookups.

---

## Phase 1 — the maths that matters (1–2 weeks, intuition only)

Four operations are load-bearing. Learn these four and nothing else:

| Operation | Where it appears | Plain meaning |
| --- | --- | --- |
| **Concatenation** `[a; b]` | `[E_c; E_p; D; M]` → 3072 | Stick two lists end to end |
| **Dot product** `a·b` | every `nn.Linear`, cosine similarity | "how aligned are these two?" |
| **Matrix multiply** | `Linear(3072→1024)` | the fundamental layer |
| **Element-wise** `a⊙b`, `\|a−b\|` | `M` and `D` in the interaction block | "where do these agree / differ?" |

Then:
- **Softmax** — turns scores `[3.2, −1.1, 0.4]` into probabilities summing to 1. That's all of
  `softmax_head.py`.
- **Cross-entropy** — "how surprised were you by the truth?" This *is* the classification loss.
- **BCE / sigmoid** — the same ideas for one yes/no output (the evidence head).
- **Temperature** — dividing scores before softmax. 0.07 makes the winner dominate hard.
- **Gradients** — *"if I nudge this one number, which way does the error move, and how much?"*
  That's it. `loss.backward()` does the chain rule for 113 M parameters in one call.
- **Cosine similarity** — the angle between two vectors, ignoring length, −1 to 1. This is why
  vectors are **L2-normalised** first in `contrastive.py`.

> **Done when:** you can look at `nn.Linear(3072, 1024)` and say aloud — "3072 numbers in,
  multiplied by a learned 1024×3072 grid plus a bias of 1024, giving 1024 numbers out, and that
  grid holds 3,146,752 learnable values."

---

## Phase 2 — PyTorch mechanics (2–3 weeks: the big one)

**The `nn.Module` pattern — learn once, recognise 9 times:**

```python
class ContextInteraction(nn.Module):
    def __init__(self, ...):          # 1. declare the learnable pieces
        super().__init__()            #    (must come first)
        self.projection = nn.Linear(3072, 1024)

    def forward(self, e_c, e_p):      # 2. the maths: tensors in → tensors out
        ...
    # 3. you now get .parameters(), .to(device), .train(), .eval() for free
```

That third point is the payoff: you never write code to move a model to a GPU, iterate its weights,
or switch dropout off. You inherited it.

**Tensors and shapes — where beginners actually get stuck**
- Always know your shapes: `[B, T, H]` = batch, sequence length, hidden size.
- Batch dimension first, *always* — that's why `inference.py` builds batch size 1.
- **Broadcasting** — `e_p * mask.unsqueeze(-1)` stretches a `[B]` mask across 768 slots.

**The loop, plus the five things people miscount**

```python
optimizer.zero_grad()          # 1. clear old slopes (they ACCUMULATE by default)
outputs = model(**batch)       # 2. forward
loss = criterion(outputs, y)   # 3. measure wrongness
loss.backward()                # 4. autograd fills .grad on every parameter
optimizer.step()               # 5. nudge every parameter downhill
```

| Concept | One-line explanation |
| --- | --- |
| **AdamW** | The optimizer; smarter downhill walker with momentum. `lr=2e-5` is tiny on purpose. |
| **LR scheduler + warmup** | Ramp up from near-zero, then decay. Prevents a violent start. |
| **Gradient accumulation** | Several small batches, step after collecting all slopes. Fakes a big batch. |
| **Gradient clipping** | Cap slope size so one weird batch can't blow up the run. |
| **AMP + GradScaler** | Mostly 16-bit arithmetic, ~2× faster on GPU. Scaler stops underflow. |
| **`train()` vs `eval()`** | Switches dropout behaviour. Forgetting it is the #1 silent bug. |
| **`no_grad()` / `detach()`** | Stop tracking gradients when only predicting. |
| **`state_dict()`** | A dict of every parameter by name. **This is what a checkpoint is.** |
| **Reproducibility** | Seed `random`, `numpy`, `torch`, `torch.cuda` — all four, or runs differ. |
| **`DataLoader` + `collate_fn`** | Batches examples into tensors. `num_workers>0` runs the collator in separate processes — this matters for a real bug in Phase 5. |

**Then read `app/training/losses.py` twice.** It's short and it contains the project's honesty rule
in code form: `-100` means "no label for this task", PyTorch skips those rows, and if the whole
batch is `-100` the loss is `None` and the component is dropped entirely.

> **Done when:** you can write a small `nn.Module`, train it on toy data in a 20-line AdamW loop,
> and explain why `zero_grad()` is needed. Then read `context_interaction.py` (83 lines) fully.

---

## Phase 3 — Transformers and how BERT works (2 weeks)

Goal: read a sentence and **picture the tensors** flowing through BERT.

Concepts in dependency order: word embeddings → **position embeddings** → **token-type
embeddings** → self-attention → multi-head → feed-forward/residual/LayerNorm → encoder vs decoder
(BERT is encoder-only — that's why no LLM is needed) → `[CLS]`/`[SEP]` → tokenization and the `##`
prefix → pretraining vs fine-tuning → freezing → truncation/padding and **attention masks**.

```python
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased"); model.eval()

enc = tok("They should all be kicked out.", return_tensors="pt")
out = model(**enc)
print(enc["input_ids"].shape)              # [1, T]
print(out.last_hidden_state.shape)         # [1, T, 768]   per-token
print(out.last_hidden_state[:, 0].shape)   # [1, 768]      the [CLS] vector
print(tok.convert_ids_to_tokens(enc["input_ids"][0]))
```

Then inspect `model.embeddings` and find the line where word, position and token-type embeddings
get **added**. That single line is the crux of the biggest bug in this repo — see trap #1.

> **Done when:** you can explain every shape above, and answer "why `[CLS]` and not an average of
> tokens?" Then read `bert_encoder.py` (169 lines) fully.

---

## Phase 4 — multi-task learning and the head pattern (3–4 days)

Small phase, huge payoff — it explains why there are four heads and why the API is full of
`*_available` flags.

- **One body, many mouths.** A shared 1024-vector feeds several small task classifiers.
- **Why share?** Related tasks regularise each other, and you compute the expensive BERT pass *once*.
- **Loss weighting** — `λ_hate`, `λ_target`, `λ_reason`, `λ_contrastive=0.5`.
- **Missing labels** — the central practical problem of multi-task learning. Filling gaps with 0
  teaches the model that everything is class 0. Hence `-100`.
- **`trained_heads` as a derived fact** — the trainer records which losses actually fired; that set
  is written to the checkpoint and read back by the API. It can never drift from reality.

> **Done when:** you can explain why a Counter Context batch produces `hate_loss` and
> `contrastive_loss` but **no** `target_loss`, and why the served checkpoint contains no
> `target_head` keys at all.

---

## Phase 5 — contrastive learning (1 week)

**Plain English:** take comment C with its real preceding comment P1 — the context-aware
representation should look similar to the comment-only anchor. Staple C to a *wrong* P2 — it should
look different. Train until the true pairing beats the false one.

**Why bother?** Without it, the network can ignore the context input entirely and still score well,
because the comment alone often predicts the label. This loss is the pressure that forces it to
actually *use* context.

Learn: anchor/positive/negative · cosine + L2 normalisation · InfoNCE (read it as "softmax over
similarities, and I want the positive to win") · temperature · **how many negatives** (this repo
defaults to **1**, sampled by a `random.Random` inside `collate_fn`, which runs in parallel
DataLoader workers) · train-only components (the projection layer wastes ~328 K parameters in the
shipped file).

> **Done when:** you can explain why 1 negative is a weak signal and describe how you'd fix it.
> You'll need that for Phase 10.

---

## Phase 6 — explainability and Integrated Gradients (1–2 weeks)

The intellectual heart of the project, and where the biggest risk lives.

| Method | What it is | Why it's *not* used here |
| --- | --- | --- |
| Attention weights | where the model attended | Attention ≠ explanation. Explicitly refused. |
| Supervised rationale | train a head on human highlights | Implemented but needs gold labels; a *different technique* |
| LLM writes it | ask a generative model | Would hallucinate. Banned from this project. |
| Gradient × input | one sample at the input | Available as the cheap option. Noisy. |
| **Integrated Gradients** | average sensitivity along a path | — **used** |

**IG, properly understood** (Sundararajan et al., 2017): pick a baseline ("no information" — here
the PAD embedding) → draw a straight line to the real input → sample 32 points → ask at each point
how sensitive the output is to each input slot → average → multiply by `(real_input − baseline)`.
Result: one **signed** score per token. The multiply-by-difference step is why IG satisfies
*completeness*.

**Why it needs `inputs_embeds`:** you cannot interpolate integer token IDs. So IG works in
continuous embedding space.

> **The trap, stated plainly.** BERT's embedding module does three things: look up the word vector,
> **add** a position vector, **add** a token-type vector. If you hand BERT raw `inputs_embeds` and
> it doesn't re-add the last two, IG is explaining **a slightly different function than the one
> that produced the label** — and the evidence section is subtly unsound. This is the #1 thing to
> verify in the whole repository: a numeric parity test, forward-twice-both-ways, assert the
> predictions match.

Also in this phase: relative vs absolute evidence thresholds · character offsets via
`return_offsets_mapping=True` · template composition vs generation (a choice about *trust*) ·
abstention as a feature.

> **Done when:** you can explain completeness in one sentence, why raw gradients mislead, and what
> a parity test would check.

---

## Phase 7 — serving: FastAPI, config, lifecycle (1–2 weeks)

| Concept | Why it's here |
| --- | --- |
| Pydantic `BaseModel` | Runtime validation + free API docs |
| `AliasChoices` | Accept `{text, context}` **and** `{current_comment, previous_comment}` |
| Routers, no business logic | Routes are ~15 lines: validate → call → map errors |
| `lifespan` | Load the 113 M-param model **once** at boot — the most important perf decision here |
| `@lru_cache` singleton | Understand why it must be one instance, and what it costs (shared mutable state) |
| `def` vs `async def` | A forward pass is blocking; `def` makes FastAPI use a threadpool. Correct here. |
| HTTP status semantics | `503` = "model not loaded". Error mapping is design, not boilerplate. |
| `pydantic-settings` + `.env` | Enforces "nothing hardcodes a model name, dimension or path" |
| YAML + `deep_merge` | Two config systems: runtime vs training. Why each experiment is 15 lines. |
| **Checkpoint rebuild pattern** | The checkpoint carries a `model_kwargs` *blueprint*; serving rebuilds the identical architecture before loading weights. **This is the bridge between the two programs.** |
| Function-local imports | Keeps `import app.main` cheap so tests don't pay a 500 MB load |

> **Done when:** you can trace a POST body from JSON through Pydantic into `predict()` and back
> out, and say exactly where a `503` could come from.

---

## Phase 8 — data engineering (1 week)

Unglamorous, and where projects quietly fail.

- **Adapter pattern** — one module per dataset, both emitting one shared dataclass. Never merge raw data.
- **`None` is a first-class value.** `target_label: null` = "not annotated"; `"none"` = "annotated
  as no target". Conflating them trains the model on false negatives.
- **Preserve, don't guess** — unmapped label values go into `metadata`.
- **Label mapping as its own layer** so the model never sees raw corpus weirdness.
- **Word-level → subword alignment** — human highlights are per-word; BERT works in word-pieces.
- **Honest splits** — this repo uses random splits, which leaks near-duplicate tweets. Learn why
  that inflates metrics.
- **Spot the naming quirk:** in Counter Context, `target` is the *current comment* and `context` is
  the *preceding* one. Always verify field semantics against the paper.

---

## Phase 9 — evaluation and ablations (1 week)

- **Accuracy lies on imbalanced data** — always saying "normal" can score 0.90. Hence macro-F1.
- Precision / recall / F1, and **macro vs weighted** averaging.
- Confusion matrix — offensive↔hate is the expected failure mode here.
- Train/val/test discipline — the test set is read **once**.
- **Ablation methodology** — the six experiments: baseline → +context → +interaction → +contrastive
  → full multi-task → +evidence. Each step removes exactly one thing.
- **Honest reporting.** The current served checkpoint's best val macro-F1 is **0.566** over 3
  classes (chance ≈ 0.33), selected at epoch 2 of 4 with val loss rising after. That's overfitting,
  not a working detector. Learn to write that sentence about your own work without flinching.

---

## Phase 10 — rebuild it yourself, smaller (3–4 weeks)

You don't understand an architecture until you've rebuilt a working version. Do it tiny and offline.

```
Dataset:      200 tiny toy sentences you write yourself
Encoder:      your own 2-layer transformer, or random-init tiny BERT
              (hidden 64, 2 layers, 4 heads — iterate in seconds)
Interaction:  D = |E_c − E_p|, M = E_c ⊙ E_p, [E_c;E_p;D;M] → Linear → 64
Heads:        hate (3) + target (3)
Loss:         masked multi-task with -100
Evidence:     Integrated Gradients over YOUR model's embeddings
Serving:      FastAPI, one endpoint, model loaded via lifespan
Tests:        one unit test per component
```

Build in that order, run a test after each step. Then the harder follow-ups:

1. **Write the IG parity test** for the real project — highest-value single action available.
2. **Run the real ablation study** and write up what actually changed.
3. **Fix one real gap end to end** — e.g. raise `negatives_per_anchor` to 4–8 and measure whether
   macro-F1 moves. Report the result either way.
4. **Serve a properly trained multi-task checkpoint** and watch UI sections light up.

---

## Which phase maps to which file

| Phase | Read these, in this order | Lines |
| --- | --- | --- |
| 0 · Python | `datasets/unified.py` → `core/logging.py` | 118 + 25 |
| 1 · Maths | `models/softmax_head.py` → `models/contrastive.py` | 39 + 79 |
| 2 · PyTorch | `models/context_interaction.py` → `training/losses.py` → `training/trainer.py` | 83 + 133 + 537 |
| 3 · Transformers | `models/bert_encoder.py` → `datasets/tokenizer.py` → `datasets/collator.py` | 169 + 162 + 202 |
| 4 · Multi-task | `hate_head.py` → `target_head.py` → `reason_head.py` → `full_model.py` | 83 + 15 + 16 + 234 |
| 5 · Contrastive | `training/contrastive_loss.py` → `ContrastivePairCollator` in `collator.py` | 63 + ~80 |
| 6 · Explainability | `attribution.py` → `evidence_extractor.py` → `reason_explainer.py` → `context_reasoner.py` | 225 + 174 + 239 + 602 |
| 7 · Serving | `core/config.py` → `schemas/prediction.py` → `routes/prediction.py` → `services/inference.py` → `main.py` | 181 + 140 + 20 + 593 + 67 |
| 8 · Data | `hatexplain.py` → `counter_context.py` → `label_mapping.py` → `preprocessing.py` | 208 + 137 + 236 + 136 |
| 9 · Evaluation | `classification.py` → `rationale.py` → `ablation.py` | 77 + 51 + 354 |
| 10 · Rebuild | You write it. Reference `full_model.py` when stuck. | — |

**Leave for last:** `services/inference.py` (593 lines — it's the *integration* of Phases 3–7) and
`reasoning/context_reasoner.py` (~400 of its 602 lines are vocabulary lists — skim those, read
`find_reference_links()` and `_compose_summary()`).

---

## A realistic schedule

| Phase | Time | If you already know… |
| --- | --- | --- |
| 0 · Python gap | 1–2 weeks | solid Python → 2 days |
| 1 · Maths intuition | 1–2 weeks | any linear algebra → 3 days |
| 2 · PyTorch | **2–3 weeks** | any torch → 1 week |
| 3 · Transformers | 2 weeks | used HF before → 3 days |
| 4 · Multi-task | 3–4 days | 1 day |
| 5 · Contrastive | 1 week | 3 days |
| 6 · Explainability | 1–2 weeks | 1 week — **not** skippable |
| 7 · Serving | 1–2 weeks | FastAPI experience → 3 days |
| 8 · Data | 1 week | 3 days |
| 9 · Evaluation | 1 week | 3 days |
| 10 · Rebuild | 3–4 weeks | 1–2 weeks |
| **Total from scratch** | **~4–5 months part-time** | **~2 months if you code in Python** |

**Compression trick:** do Phase 10 *in parallel* with Phases 3–7, not at the end. Rebuild the tiny
version incrementally as you learn. You'll understand twice as fast and have something to show.

---

## What you can safely skip

Writing a transformer from scratch · calculus by hand · distributed training (DDP/FSDP) · LLMs,
RAG, prompt engineering, agents (**explicitly not used** — the project's identity is "no generative
model") · reinforcement learning · graph neural networks · async Python internals (just know `def`
vs `async def` in FastAPI) · Kubernetes/cloud infra (Docker basics suffice) · Pandas beyond basics ·
the whole frontend · the exact contents of the vocabulary lists in `context_reasoner.py`.

---

## The 7 traps — things needing extra care

| # | Trap | What it teaches you | Phase |
| --- | --- | --- | --- |
| 1 | IG path may skip position/token-type embeddings | An explanation is valid only if it explains *the same function* that made the prediction | 6 |
| 2 | Missing label (`None`) vs negative label (`"none"`) | One conflated `None` can corrupt a whole run silently | 8 |
| 3 | InfoNCE with 1 negative, sampled in a parallel worker | A loss can be in the code, logged in the metrics, and still teach almost nothing | 5 |
| 4 | Random splits on duplicate-heavy social data | Leakage inflates metrics; a number can be correct and meaningless | 9 |
| 5 | `encode_pair` with `input_ids=None` → uncaught `ValueError` | Catching `RuntimeError` but not `ValueError` = a 500 instead of a 503 | 7 |
| 6 | `model_info()` assumes `model.hate_head` exists | Reaching into objects by attribute name breaks when the object changes shape | 7 |
| 7 | A keyword list decides how the decision is described | Better than an LLM, but still an unevaluated decision-maker in front of your evaluated one | 6 |

---

## The 20 questions — your exam

Answer all 20 out loud, without notes. Phase in parentheses.

1. Why does the same encoder encode both the comment and the context? (3)
2. What do `D` and `M` mean, and why concatenated rather than added? (2)
3. What happens to `E_p` with no context, and where is that handled? (3)
4. Why `[CLS]` and not an average of token vectors? (3)
5. What does `-100` mean, and what if a whole batch is `-100`? (2)
6. How does `trained_heads` get populated, and why is that better than a config flag? (4)
7. Why can't IG interpolate raw token IDs? (6)
8. Explain the completeness axiom in one sentence. (6)
9. Why is a raw gradient a worse explanation than IG? (6)
10. Why is the evidence threshold *relative*, and what's the risk? (6)
11. Why does the evidence head exist if IG already gives evidence? (4)
12. What does `context_used: true` claim — and what does it *not* claim? (7)
13. Why is contrastive learning never used at inference? (5)
14. Why is the checkpoint's `model_kwargs` as important as its `model_state`? (7)
15. Why are `optimizer_state` / `scheduler_state` dropped on export? (7)
16. Why macro-F1 for early stopping instead of accuracy? (9)
17. Why is the model loaded in `lifespan` rather than in the route handler? (7)
18. Why is the route `def` and not `async def`? (7)
19. Why return `null` **plus** an `*_available: false` flag instead of just omitting the field? (7)
20. What would you change to make the contrastive loss actually teach something? (5, 10)

---

## Progress tracker

```
PHASE 0 · Python gap
  [ ] venv + requirements installed from a clean environment
  [ ] python -m pytest -q --basetemp=.pytest-tmp  → 149 passed, 2 skipped
  [ ] can read unified.py with no lookups

PHASE 1 · Maths
  [ ] explain nn.Linear(3072, 1024) including the parameter count
  [ ] explain softmax → cross-entropy → why it measures surprise
  [ ] explain cosine similarity and why L2 normalisation comes first
  [ ] explain what a gradient IS, in one sentence

PHASE 2 · PyTorch
  [ ] wrote a tiny nn.Module from scratch
  [ ] wrote a 20-line training loop with AdamW, zero_grad, backward, step
  [ ] can explain why zero_grad() is needed
  [ ] can explain gradient accumulation and clipping in my own words
  [ ] can explain state_dict() and why it IS the checkpoint
  [ ] read losses.py twice and understood the -100 trick

PHASE 3 · Transformers
  [ ] ran the AutoTokenizer/AutoModel snippet and explained every shape
  [ ] found where model.embeddings ADDS position and token-type vectors
  [ ] understand the ## subword prefix and offsets_mapping
  [ ] can explain what an attention mask does
  [ ] read bert_encoder.py completely

PHASE 4 · Multi-task
  [ ] explain why there is no target_loss on a Counter Context batch
  [ ] explain why the served checkpoint has no target_head keys
  [ ] explain how trained_heads is derived

PHASE 5 · Contrastive
  [ ] explain anchor / positive / negative
  [ ] explain what temperature does
  [ ] explain why 1 negative is a weak signal
  [ ] can name the DataLoader-worker sampling problem

PHASE 6 · Explainability
  [ ] explain baseline, path, steps, completeness
  [ ] explain why raw gradients mislead
  [ ] describe the inputs_embeds parity test
  [ ] explain relative vs absolute evidence thresholds
  [ ] explain why the context layer abstains instead of guessing

PHASE 7 · Serving
  [ ] traced a POST body from JSON to response
  [ ] can name every source of a 503
  [ ] explain lifespan + the lru_cache singleton
  [ ] explain why the route is def, not async def
  [ ] explain the model_kwargs rebuild pattern

PHASE 8 · Data
  [ ] explain None vs "none" and why it matters
  [ ] explain word→subword rationale alignment
  [ ] explain why random splits inflate metrics

PHASE 9 · Evaluation
  [ ] explain macro vs weighted F1
  [ ] explain how the 6 ablation experiments differ
  [ ] know the honest number: macro-F1 0.566, chosen at epoch 2 of 4

PHASE 10 · Rebuild
  [ ] built a tiny end-to-end version (data → model → loss → IG → API → test)
  [ ] wrote the IG parity test for the real project
  [ ] ran the real ablation study
  [ ] fixed one real gap and measured the effect
  [ ] answered all 20 questions out loud
```
