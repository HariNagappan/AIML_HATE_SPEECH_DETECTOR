# How Reasoning Works

**How this application explains *why* a comment was classified as hate
speech — and what it deliberately does not do.**

This document describes the reasoning pipeline end to end: from the model's
prediction, through evidence extraction and context analysis, to the
structured explanation shown in the UI. Everything described here is
implemented in the codebase and covered by tests.

> **The one rule everything follows:** the explanation is *assembled* from
> real system outputs (the model's prediction, attribution scores, and a
> deterministic analysis of the two comments). No generative model writes
> any part of it, and nothing is invented when the evidence is insufficient.

---

## 1. What "reasoning" means in this project

When the model returns **hate speech**, the application answers four
questions:

| Question | Produced by |
| --- | --- |
| **WHAT?** Which class? (=Hate speech) | The trained **hate head** (the model). |
| **WHICH WORDS?** Which parts of the comment drove that prediction? | **Integrated Gradients attribution** over the model. |
| **WHO / WHAT DOES IT REFER TO?** How does the previous comment matter? | A deterministic **reference analysis** ("They" → "group of immigrants"). |
| **WHY DOES THAT MEAN HARM?** What connects the evidence to the classification? | Fixed **explanation templates**, grounded in the three items above. |

Reasoning here is **not**:

* ❌ a GPT/LLM writing an explanation,
* ❌ attention weights relabeled as "reasoning",
* ❌ a second model guessing a story,
* ❌ keyword rules that fire independently of the model.

It **is** a transparent analysis layer *around* the classifier: the
classification always comes from the trained model; the reasoning layer only
analyses the input that produced it and explains the relationship.

---

## 2. The pipeline, end to end

```
 Previous comment (optional)        Current comment
            │                              │
            └──────────────┬───────────────┘
                           ▼
              Shared BERT encoder + context interaction
              (|Ec−Ep|, Ec⊙Ep — the model really sees the context)
                           ▼
                 hate prediction  (e.g. hate_speech, 0.85)
                           ▼
        Integrated Gradients attribution over the current comment
                           ▼
     evidence spans with character offsets   +   reference analysis
        ("be kicked" 0.68, "they should"…)        ("They" → "group of
                                                    immigrants", offsets)
                           ▼
        deterministic reasoning templates  (label tone + links + evidence)
                           ▼
   structured response:  prediction … + reasoning { summary, links, evidence }
                           ▼
              reasoning UI  ("Why this was classified this way")
```

### Where each stage lives in the code

| Stage | File |
| --- | --- |
| Model forward pass (context-aware) | `backend/app/models/full_model.py` |
| Prediction decoding (label/confidence) | `backend/app/services/inference.py` |
| Attribution (Integrated Gradients) | `backend/app/reasoning/attribution.py` |
| Evidence spans + character offsets | `backend/app/reasoning/evidence_extractor.py` |
| Context reference analysis + reasoning summary | `backend/app/reasoning/context_reasoner.py` |
| Reason-category explanations (optional head) | `backend/app/reasoning/reason_explainer.py` |
| Response assembly | `backend/app/reasoning/structured_explanation.py` |
| API request/response schema | `backend/app/api/schemas/prediction.py` |
| Reasoning UI components | `frontend/src/components/analysis/ReasoningCard.tsx`, `EvidenceViewer.tsx` |

---

## 3. Stage by stage

### 3.1 Prediction — the model's job

The classifier is a fine-tuned BERT with a **context interaction** module:
the previous comment is encoded separately and combined with the current
comment (element-wise difference and product), so the classification is
genuinely context-aware — not just a display feature. The hate head then
outputs a label (`hate_speech` / `counter_speech` / `neither` for the
Counter Context model; `hate` / `offensive` / `normal` for the HateXplain
model) with per-class probabilities.

Everything downstream is gated on what is actually trained: if a head has no
trained weights, the API reports `*_available: false` and the UI hides that
section rather than showing a guess.

### 3.2 Evidence extraction — "which words?"

The strongest question a user asks is *"why?"*. The system answers it with
**Integrated Gradients (IG)** — a standard attribution method:

1. the comment is tokenized and scored token by token: *how much did this
   token contribute to the predicted class?*
2. scores are thresholded relative to the strongest token, adjacent tokens
   are merged, and subwords are joined back into readable phrases;
3. each resulting span is returned **verbatim from the input**, with
   **character offsets** (`start`/`end`) so the UI can highlight it exactly.

Example (real output — previous: *"I saw a group of immigrants protesting
downtown."*, current: *"They should all be kicked out."*):

```json
"evidence": [
  { "text": "be kicked", "score": 0.68, "start": 17, "end": 26 },
  { "text": "they should", "score": 0.68, "start": 0, "end": 11 }
]
```

**What this is:** the text spans that most influenced *this* prediction.
**What it is not:** causal proof that those words *made* the model decide —
attribution is an approximate, input-grounded signal (stated in the UI).

### 3.3 Context-relationship analysis — "who does 'they' mean?"

The example sentence *"They should all be kicked out."* is only fully
interpretable together with the previous comment — "They" alone does not say
who is affected. The reasoning layer therefore runs a small, **auditable
reference analysis** across the two comments:

* it finds referring words in the current comment (`they/them/their`,
  `he/him/his`, `she/her/hers`),
* and links each one to the phrase it points back to in the previous
  comment (the right-most plural noun phrase — e.g. *"group of immigrants" —
  expanded over determiners and prepositions and stripped of leading
  articles),
* returning **character offsets on both sides**:

```json
"links": [
  { "from_text": "They", "from_start": 0, "from_end": 4,
    "to_text": "group of immigrants", "to_start": 8, "to_end": 27,
    "relation": "refers_to" }
]
```

**What this is:** a deliberately simple heuristic — labelled as such in the
response (`method: "deterministic evidence + context-reference analysis"`)
and in the UI ("not model attention"). It makes the *relationship the model
was trained on* visible and checkable. It can miss or pick an imperfect
antecedent in hard cases; it never claims more than a reference reading.

### 3.4 Reason category — "what type of harm?" (optional)

The architecture contains a **reason head** (insult, dehumanization,
negative stereotyping, threat, exclusion, discrimination, incitement to
violence, other). When — and only when — it has been trained on real
reason annotations, the reason prediction is explained through fixed
templates (`reason_explainer.py`):

> *"The highlighted language expresses exclusion by advocating that members
> of the targeted group be removed or kept out."*

No annotations yet → the head reports `reason_available: false` and the
section is hidden (see §7: the app ships an **annotation workspace** to
create such data). The reason head is never trained on synthetic labels.

### 3.5 Composing the explanation — "why is it hate speech?"

`context_reasoner.py` assembles the `reasoning` block using the label's
tone, the found links, and the evidence:

* **Hate-like label + link** (the flagship case):
  > The current comment uses "They" to refer to "group of immigrants" from
  > the previous comment and expresses hostile or exclusionary language
  > toward that target. The contextual reference matters: on its own,
  > "They" does not identify who or what is being referred to. Strongest
  > evidence: the phrase 'be kicked' and 'they should'.

* **Hate-like + context but no reference found:**
  > …The previous comment was included in the analysis, but no explicit
  > reference relationship between the two comments was identified. …
  > The available context does not provide an explicit reference to which
  > group (if any) the statement targets. *(abstention — no invented
  > target)*

* **No previous comment at all:**
  > …based on the current comment alone — no previous comment was provided,
  > so no context relationship could be analysed.

* **`offensive` label (HateXplain policy, insult vs hate):**
  > The model classified the comment as Offensive — abusive or insulting
  > language. The available context does not indicate that the insult
  > targets a protected group.

* **Non-hateful / welcoming:**
  > The comment contains welcoming or supportive language; no hateful or
  > exclusionary language toward a protected group was identified.

* **Counter-speech label:**
  > …language that opposes or rebuts hateful content. No exclusionary
  > relationship toward the previous comment was identified.

Each sentence is a fixed template filled with *actual* values: predicted
label, the link text + offsets, the evidence phrases and scores, and the
context flags.

### 3.6 The structured response

`POST /api/v1/predict` accepts both naming conventions —
`{previous_comment, current_comment}` — and the legacy `{text, context}` —
and answers with classification and reasoning kept strictly separate:

```json
{
  "prediction": { "label": "hate_speech", "confidence": 0.845, "probabilities": {...} },
  "prediction_available": true,
  "evidence": [ ... ],
  "reasoning": {
    "summary": "The current comment uses \"They\" to refer to \"group of immigrants\" ...",
    "method": "deterministic evidence + context-reference analysis",
    "context_used": true,
    "context_available": true,
    "links": [ { "from_text": "They", "to_text": "group of immigrants", ... } ],
    "evidence": [
      { "text": "be kicked", "source": "current_comment",
        "type": "current_span", "reason": "Strongest attribution signal (score 0.68) ..." },
      { "text": "group of immigrants", "source": "previous_comment",
        "type": "context_target", "reason": "Identified by the reference analysis as the phrase \"They\" refers to." }
    ]
  },
  "context_used": true
}
```

Every evidence item says **where it came from** (`source`:
`current_comment` / `previous_comment`), **what kind** it is
(`current_span` / `context_target`), and **why it matters** (`reason`).

---

## 4. How the UI presents it

The result page renders the chain **WHAT → WHO → WHY → WHICH PARTS → CONTEXT**:

1. **Classification** card — label + confidence (from the model).
2. **"Why this was classified this way"** card — the reasoning summary, a
   clickable **connection chip** (`"They" → "group of immigrants"`), and the
   evidence list grouped by source (current / previous comment).
3. **Evidence** card — both comments are shown and highlighted using the
   backend's character offsets: the current comment with its attribution
   spans, and the previous comment with the referenced phrase marked.
   Clicking a connection chip scrolls here and shows the active connection.
4. **Attribution detail** (on demand) — every token with its IG score, for
   readers who want the raw attribution.
5. **"Why this classification?" (reason card)** — appears additionally when
   a reason-trained model is served (see §3.4).

Colour is never the only signal (chips carry text labels), and every card
displays its provenance — e.g. *"…deterministic context analysis — not by a
language model."*

---

## 5. Worked example, start to finish

**Input** (the canonical example):

```
previous: I saw a group of immigrants protesting downtown.
current:  They should all be kicked out.
```

**What happens:**

1. The context-aware model reads both comments and predicts
   **hate_speech (0.845)**.
2. IG attribution scores the tokens of the current comment → spans
   *"they should"* (0.68), *"be kicked"* (0.68), … with offsets.
3. Reference analysis links **"They" (0–4)** → **"group of immigrants"
   (8–27)** in the previous comment.
4. Templates compose the summary (hate + link branch) and assemble the
   evidence list with sources and rationales.
5. The UI highlights *"group of immigrants"* in the previous comment,
   highlights the attribution spans (and the referring "They") in the
   current comment, and shows the connection chip.

A screenshot of this exact flow, produced by a live run against a trained
checkpoint, is in
[`deliverables/hate-speech-walkthrough/09-context-reasoning.png`](deliverables/hate-speech-walkthrough/09-context-reasoning.png).

---

## 6. Honesty rules (built into the code)

* **No generative LLM anywhere.** All explanation text is fixed templates
  + real values; the `method` field says exactly what produced it.
* **No fake explainability.** Attention weights are never presented as
  reasoning; attribution is Integrated Gradients over the actual model, and
  the reference analysis is labelled a heuristic.
* **Abstention over invention.** Insufficient evidence → the summary says
  so. Non-hateful results are never dressed up with hateful evidence.
* **Strict separation.** Classification (model), evidence/context analysis
  (attribution + heuristics), and the natural-language layer (templates)
  are separate fields and separate responsibilities.
* **Graceful degradation.** If reasoning generation fails, the request
  still returns the classification with
  `"reasoning": {"summary": "Reasoning unavailable."}` and the error is
  logged — reasoning never takes the classifier down with it.

**Known limitations** (also listed in the About page): attribution is an
approximate signal, not causal proof; the reference heuristic can miss
antecedents; the reason categories require their own annotated data before
they can appear.

---

## 7. Verify it yourself

```bash
# backend (loads the trained checkpoint from backend/checkpoints/)
cd backend
uvicorn app.main:app --port 8001

# the canonical example
curl -s http://127.0.0.1:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"previous_comment": "I saw a group of immigrants protesting downtown.",
       "current_comment": "They should all be kicked out."}'

# reasoning tests (deterministic layer, reference analysis, API contract)
python -m pytest -q tests/test_context_reasoner.py tests/test_api.py

# frontend
cd ../frontend && npm install && npm run dev   # http://localhost:5173
```

---

*Context-Aware Hate Speech Detection with Structured Evidence-Based
Reasoning — reasoning documentation. See the root `README.md`,
`backend/README.md` ("Context-relationship reasoning") and
`frontend/README.md` for the surrounding system.*
