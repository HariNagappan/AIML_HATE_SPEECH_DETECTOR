# Data directory

Everything under `data/raw/` and `data/processed/` is **git-ignored** — use the
scripts to (re)create it.

```
data/
├── raw/            # downloaded datasets (never edited by hand)
│   ├── hatexplain/               # dataset.json + post_id_divisions.json
│   └── counter_context/          # gold/ and silver/ JSONL files
└── processed/      # UnifiedExample JSONL files produced by scripts/preprocess.py
    ├── hatexplain_{train,val,test}.jsonl
    ├── counter_context_{train,val,test}.jsonl          # gold splits
    └── counter_context_silver_{train,val}.jsonl        # extra (not mixed in)
```

## Download & preprocess

```bash
python scripts/download_data.py --dataset all
python scripts/preprocess.py
python scripts/inspect_dataset.py --path data/raw/counter_context/gold/train.jsonl
```

## Dataset responsibilities (why they are not merged blindly)

* **HateXplain** — hate/offensive/normal labels, target communities,
  token-level rationales. No conversation context, no reason categories
  (those stay `None`).
* **Counter Context** (NAACL 2022, *Hate Speech and Counter Speech Detection:
  Conversational Context Does Matter*) — parent/current comment pairs for
  context-dependent classification and contrastive experiments. No target and
  no rationale annotations (those stay `None`).

⚠️ **Counter Context naming quirk**: in its JSONL records, `context` is the
*preceding comment* and `target` is the *current comment being classified*.
`target` is **not** a hate-target annotation. The adapter documents this and
the unified format renames the fields to `current_text` / `context_text`.

**Label mapping (verified)**: corpus ids map `"0"` → `hate_speech`,
`"1"` → `neither` (neutral), `"2"` → `counter_speech` — verified against the
paper (Yu, Blanco & Hong, NAACL 2022: ~28% / ~49% / ~23% class shares over
6,846 pairs). See `configs/base.yaml` and `app/datasets/label_mapping.py`;
override in the YAML if you ever need a different order.
