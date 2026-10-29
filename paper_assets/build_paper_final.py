# -*- coding: utf-8 -*-
"""Data-driven, publication-style DOCX for the hate-speech explanation system.

Numbers are read from repository artifacts at build time so the manuscript
cannot drift from the experiments. Missing optional runs (cross-encoder,
extra HateXplain seeds) are omitted rather than invented.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips, Emu, Cm

ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
ASSETS = ROOT / "paper_assets"
OUT_DOCX = ROOT / "deliverables" / "hate-speech-research-paper.docx"
FONT = "Times New Roman"
MONO = "Consolas"

# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
EXP = json.loads((ASSETS / "experiment_results.json").read_text(encoding="utf-8"))
ADD = json.loads((ASSETS / "addenda_results.json").read_text(encoding="utf-8"))
EVH = json.loads((ASSETS / "evidence_head_metrics.json").read_text(encoding="utf-8"))
REF = json.loads((ASSETS / "reference_linker.json").read_text(encoding="utf-8"))
PRED_PATH = ASSETS / "cc_predictions.json"
PRED = json.loads(PRED_PATH.read_text(encoding="utf-8")) if PRED_PATH.exists() else {}


def _load_ckpt_metrics(name: str) -> Dict[str, Any]:
    p = ROOT / "backend" / "checkpoints" / name / "best.metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def mean_std(xs: Sequence[float]) -> Tuple[float, float, int]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    if len(xs) == 1:
        return xs[0], 0.0, 1
    return statistics.mean(xs), statistics.stdev(xs), len(xs)


def pm(mean: float, std: float, n: int, digits: int = 3) -> str:
    if n == 0:
        return "—"
    if n == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def ladder(variant: str) -> List[Dict[str, Any]]:
    return [
        r
        for r in EXP.get("cc_ladder", [])
        if r.get("variant") == variant and r.get("test_macro_f1") is not None
    ]


def ladder_row(label: str, variant: str, digits: int = 3) -> List[str]:
    rows = ladder(variant)
    if not rows:
        return [label, "0", "—", "—", "—", "—", "—", "—"]
    acc = mean_std([r["test_accuracy"] for r in rows])
    mf1 = mean_std([r["test_macro_f1"] for r in rows])
    wf1 = mean_std([r["test_weighted_f1"] for r in rows])
    h = mean_std([r["test_per_class_f1"]["hate_speech"] for r in rows])
    c = mean_std([r["test_per_class_f1"]["counter_speech"] for r in rows])
    n = mean_std([r["test_per_class_f1"]["neither"] for r in rows])
    return [
        label,
        str(acc[2]),
        pm(*acc, digits),
        pm(*mf1, digits),
        pm(*wf1, digits),
        pm(*h, digits),
        pm(*c, digits),
        pm(*n, digits),
    ]


def hx_seed_rows() -> List[Dict[str, Any]]:
    return [r for r in EXP.get("hx_seeds", []) if r.get("test_macro_f1") is not None]


def xe_rows() -> List[Dict[str, Any]]:
    return [r for r in EXP.get("cross_encoder", []) if r.get("test_macro_f1") is not None]


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(1.0)
sec.top_margin = sec.bottom_margin = Inches(1.0)

style = doc.styles["Normal"]
style.font.name = FONT
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.12
style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE

core = doc.core_properties
core.author = "Hari Nagappan"
core.title = "Deterministic Explanation Assembly for Hate-Speech Detection"
core.subject = "Multi-task BERT system with provenance-tracked evidence"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)


def set_font(run, size=10.5, bold=False, italic=False, color=None, name=FONT):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), name)
    return run


def para(text="", size=10.5, bold=False, italic=False,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, space_before=0,
         indent=None, first_line=None, color=None, keep_next=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = 1.12
    if indent is not None:
        p.paragraph_format.left_indent = Inches(indent)
    if first_line is not None:
        p.paragraph_format.first_line_indent = Inches(first_line)
    if keep_next:
        p.paragraph_format.keep_with_next = True
    if text:
        set_font(p.add_run(text), size, bold, italic, color)
    return p


def rich(p, parts):
    for text, fmt in parts:
        set_font(p.add_run(text), **fmt)
    return p


def h1(num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(13)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(f"{num}   {text}"), 12, bold=True, color=NAVY)
    return p


def h2(num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(f"{num}   {text}"), 11, bold=True, italic=True)
    return p


def bullet(text, bold_lead: Optional[str] = None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.3)
    if bold_lead:
        set_font(p.add_run(bold_lead), 10.5, bold=True)
        set_font(p.add_run(text))
    else:
        set_font(p.add_run(text))
    return p


_eq_no = [0]


def equation(formula):
    _eq_no[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    set_font(p.add_run("\t" + formula), 10.5, italic=True)
    set_font(p.add_run(f"\t({_eq_no[0]})"), 10.5)
    return p


def _set_cell_borders(cell, **sides):
    """sides: top/left/bottom/right -> (sz, color_hex) sz in eighths of a point."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge, (sz, color) in sides.items():
        el = tcBorders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            tcBorders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)


def _clear_borders(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        tcBorders.append(el)
    existing = tcPr.find(qn("w:tcBorders"))
    if existing is not None:
        tcPr.remove(existing)
    tcPr.append(tcBorders)


def figure(path, caption, width=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(10)
    set_font(c.add_run(caption), 9, italic=True)


def table(caption, headers, rows, widths=None, font_size=8.0):
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_before = Pt(8)
    c.paragraph_format.space_after = Pt(3)
    c.paragraph_format.keep_with_next = True
    set_font(c.add_run(caption), 9, italic=True)

    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    def fill(cell, text, *, header=False, first_col=False):
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        _clear_borders(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if first_col else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        set_font(p.add_run(str(text)), font_size, bold=header)

    for j, h in enumerate(headers):
        fill(t.rows[0].cells[j], h, header=True, first_col=(j == 0))
        _set_cell_borders(
            t.rows[0].cells[j],
            top=(12, "1F3A5F"),
            bottom=(6, "1F3A5F"),
            left=(0, "FFFFFF"),
            right=(0, "FFFFFF"),
        )
    for i, row in enumerate(rows, start=1):
        last = i == len(rows)
        for j, val in enumerate(row):
            fill(t.rows[i].cells[j], val, first_col=(j == 0))
            sides = {"left": (0, "FFFFFF"), "right": (0, "FFFFFF")}
            if last:
                sides["bottom"] = (12, "1F3A5F")
            else:
                sides["bottom"] = (4, "D0D5DD")
            _set_cell_borders(t.rows[i].cells[j], **sides)
    if widths:
        for j, w in enumerate(widths):
            for r in t.rows:
                r.cells[j].width = Inches(w)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return t


def refblock(refs):
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_font(p.add_run(f"[{i}]  {r}"), 9.5)


def add_page_number_footer():
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    set_font(r, 9, color=RGBColor(0x55, 0x55, 0x55))
    for el, attr in (
        ("w:fldChar", {"w:fldCharType": "begin"}),
        ("w:instrText", None),
        ("w:fldChar", {"w:fldCharType": "end"}),
    ):
        e = OxmlElement(el)
        if el == "w:instrText":
            e.set(qn("xml:space"), "preserve")
            e.text = "PAGE"
        for k, v in (attr or {}).items():
            e.set(qn(k), v)
        r._element.append(e)


# ===========================================================================
# TITLE
# ===========================================================================
para(
    "Deterministic Explanation Assembly for Hate-Speech Detection:",
    size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0, color=NAVY,
)
p = para("", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
set_font(
    p.add_run(
        "A Multi-Task BERT System with Provenance-Tracked Evidence, "
        "Conversational Context, and Reported Failures"
    ),
    13, bold=True, color=NAVY,
)
para("Hari Nagappan", size=11.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
para("harisnagappan@gmail.com", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
para("Preprint · 29 October 2026", size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

# Abstract
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Inches(0.3)
p.paragraph_format.right_indent = Inches(0.3)
p.paragraph_format.space_after = Pt(8)
set_font(p.add_run("Abstract—"), 9.5, bold=True)
set_font(
    p.add_run(
        "Hate-speech classifiers are increasingly paired with natural-language explanations. "
        "Explanations written by a generative model cannot be guaranteed to describe the "
        "classifier that produced the label. We present a system in which a multi-task BERT "
        "detector is paired with a deterministic assembly layer: Integrated-Gradients spans "
        "with character offsets, a rule-based pronoun-to-antecedent analysis of the preceding "
        "comment, and fixed templates. The layer emits provenance for every statement, "
        "abstains rather than invents, and contains no generative component. We evaluate "
        "detector and explanation layer on HateXplain and Counter Context against published "
        "dataset-paper baselines. On HateXplain, multi-task training with target and rationale "
        "supervision reaches 69.8% test accuracy and 0.681 macro-F1, matching the dataset "
        "authors' BERT (69.0% / 0.674); the gain over a comment-only baseline is significant "
        "(McNemar p = 0.0016). A trained rationale head agrees with human rationales at 0.823 "
        "macro token-F1 versus 0.650 for post-hoc Integrated Gradients; a deletion probe shows "
        "those spans are plausible but not necessary for the prediction (comprehensiveness 0.015). "
        "On Counter Context, a four-rung matched-budget study over three seeds shows a clean "
        "accuracy ordering — no context (0.565 ± 0.008) < parent concatenation (0.584 ± 0.007) < "
        "interaction features (0.603 ± 0.007) — with every rung's worst seed above the previous "
        "rung's best seed. Adding InfoNCE and using a cross-encoder are reported as separate "
        "controls. The pronoun linker covers only 16.6% of test pairs; HateCheck transfer of "
        "the hate head is poor (12.8% hateful recall). Batch-1 latency on a laptop GPU is 0.7–1.1 s "
        "(n = 100). We position the work as a system and explanation study, not as a "
        "state-of-the-art classifier."
    ),
    9.5,
)
p = para("", space_after=8)
rich(p, [
    ("Index Terms—", {"size": 9.5, "italic": True, "bold": True}),
    ("hate speech detection; explainable AI; structured explanations; multi-task learning; "
     "Integrated Gradients; conversational context; faithfulness",
     {"size": 9.5, "italic": True}),
])

# ===========================================================================
# 1 INTRODUCTION
# ===========================================================================
h1("1.", "Introduction")
para(
    "Automated hate-speech detection is deployed at a scale that makes every systematic error — "
    "over-flagging of dialect, confusion of offensive language with hate, blindness to who is "
    "being talked about — a moderation problem rather than only a modelling problem. A production "
    "review tool therefore has to answer three questions for each decision: what was predicted, "
    "which parts of the input supported the prediction, and how conversational context was used. "
    "Large language models can produce fluent answers to all three, but those answers are not "
    "guaranteed to describe the classifier's actual computation [15], [23], and zero-shot LLMs "
    "still trail fine-tuned models on classification benchmarks that include hate speech [22]. "
    "A pipeline that needs both a reliable label and an auditable explanation has reason to keep "
    "the classifier and the explanation layer separately inspectable."
)
para(
    "This paper describes such a pipeline and evaluates it as a system study. The detector is a "
    "multi-task BERT model with switchable components: a context-interaction module for "
    "conversational input, an instance-discriminating InfoNCE objective with shuffled-context "
    "negatives, heads for hate, target group, harm reason, and token-level evidence, and a "
    "post-hoc Integrated-Gradients extractor. The explanation layer — the part we treat as the "
    "contribution — is not neural. It composes a structured explanation from the predicted label, "
    "verbatim evidence spans with character offsets, a deterministic pronoun-to-antecedent analysis, "
    "and fixed templates. Every emitted statement can be checked against a stored output. The "
    "layer abstains when evidence is insufficient. Heads without training data are reported as "
    "unavailable rather than guessed."
)
para("We ask four questions, and we answer them with the experiments the system makes possible:")
bullet(" Does multi-task supervision (target groups and token rationales) improve a BERT hate "
       "classifier on HateXplain, relative to a comment-only baseline and to the dataset authors' "
       "published BERT?", bold_lead="RQ1.")
bullet(" On Counter Context, does the preceding comment help, and which fusion of comment and "
       "parent — concatenation, InferSent-style interaction, InfoNCE, or a cross-encoder — "
       "explains the gain?", bold_lead="RQ2.")
bullet(" How do a trained rationale head and post-hoc Integrated Gradients compare on "
       "plausibility (agreement with human rationales) and on a deletion-based faithfulness probe?",
       bold_lead="RQ3.")
bullet(" What do HateCheck transfer, reference-linker coverage, and serving latency say about "
       "whether the system is ready for anything other than review-assist research?",
       bold_lead="RQ4.")
para(
    "The short answers, developed in Section 6, are: (RQ1) yes — 66.1% → 69.2–69.8% accuracy, "
    "on par with Mathew et al. [7]; (RQ2) context helps, and under a matched budget the accuracy "
    "ordering no-context < concatenation < interaction is stochastic-dominant across three seeds; "
    "InfoNCE and a trained cross-encoder are reported as additional controls rather than as the "
    "source of the gain; (RQ3) the trained head is far more plausible than Integrated Gradients, "
    "but Integrated-Gradients evidence is not necessary for the decision; (RQ4) HateCheck transfer "
    "and linker coverage are negative results we leave standing, and latency is review-assist "
    "scale. The classifier components are standard (Section 4.7); we do not claim a "
    "state-of-the-art detector."
)

# ===========================================================================
# 2 RELATED WORK
# ===========================================================================
h1("2.", "Related Work")
h2("2.1", "Hate speech detection and benchmarks")
para(
    "The hate / offensive / normal distinction and its difficulty were established by Davidson "
    "et al. [1] and Waseem and Hovy [2]; later crowdsourced corpora confirmed the label-noise "
    "problem [3]; OffensEval formalized type and target sub-tasks [4]. Dataset-quality surveys "
    "document concerns that persist today [5], [6]. HateXplain [7] adds majority-vote class "
    "labels, target groups, and token-level rationales to 19,229 Gab and Twitter posts, and "
    "reports BERT at 69.0% accuracy / 0.674 macro-F1, rising to 69.8% / 0.687 when rationales "
    "supervise an attention head; attention- and LIME-based token selection sits at 0.37–0.51 "
    "token-F1. HateCheck [21] provides functional tests; we run them in Section 6.5. Counter "
    "Context [8] pairs Reddit comments with their parents, labelled hate / counter-speech / "
    "neutral given the context, and reports RoBERTa [24] weighted-F1 of 0.58 without context and 0.61 "
    "with a parent–target cross-encoder on gold data (0.64 with silver data and stance "
    "pre-training)."
)
h2("2.2", "Conversational context in abuse detection")
para(
    "Park and Fung [9] found that parent-tweet context mainly helps the hateful class. Yu et al. "
    "[8] showed that human judgments of a comment change for most examples when the preceding "
    "comment is shown, and that a RoBERTa cross-encoder over parent [SEP] target improves "
    "weighted-F1 from 0.58 to 0.61 on gold data. We follow the same motivation but encode the two "
    "comments separately with a shared encoder and combine their sentence embeddings with explicit "
    "interaction features [17], [18], which keeps single-comment inference available and makes a "
    "contrastive objective possible. We compare against Yu et al.'s published numbers rather than "
    "retraining their models, and we train a BERT cross-encoder of the form [CLS] parent [SEP] "
    "current as a control when those runs are available."
)
h2("2.3", "Explainability, faithfulness, and generated explanations")
para(
    "LIME [10] and Integrated Gradients [11] provide input-grounded evidence without retraining. "
    "Attention weights are unreliable as explanations [12]. ERASER [13] distinguished "
    "plausibility (agreement with human rationales) from faithfulness (necessity of the "
    "explanation for the prediction); Jacovi and Goldberg [15] formalized the latter and warned "
    "that plausible text can be unfaithful. LLMs are now the default explanation engine in "
    "deployed moderation tools. Ziems et al. [22] find that they still fail to outperform "
    "fine-tuned models on many classification tasks and are best used with humans in the loop. "
    "Turpin et al. [23] show that chain-of-thought text can be unfaithful to the model's actual "
    "computation even when it looks plausible. Our design takes the conservative position: "
    "deterministic templates over stored model outputs and input spans. Determinism makes every "
    "statement checkable; it does not make Integrated Gradients faithful, and it does not make a "
    "pronoun linker correct."
)
h2("2.4", "What this work adds")
para(
    "The detector is assembled from standard parts: BERT [16], matching features from InferSent "
    "[17] and Sentence-BERT [18], InfoNCE [19], multi-task heads, and Integrated Gradients [11]. "
    "The contribution is the assembly layer — provenance, abstention, no generation — together "
    "with an evaluation that treats published baselines, a matched-budget context ladder, a "
    "plausibility/faithfulness split, HateCheck, a linker audit, latency, and negative results as "
    "part of the system rather than as an appendix. To our knowledge this combination has not "
    "been enforced in a hate-speech system; we cannot rule out prior art we have not seen."
)

# ===========================================================================
# 3 DATA
# ===========================================================================
h1("3.", "Task and Data")
h2("3.1", "Unified schema")
para(
    "Both corpora are converted into a shared schema (current_text, context_text, hate_label, "
    "target_label, reason_label, rationale_labels, metadata). Annotations a dataset does not "
    "provide are null, never imputed, so a training loss activates only when ground truth exists "
    "for the batch. The two datasets are never merged."
)
h2("3.2", "HateXplain")
para(
    "Using the official post-id divisions we process 19,229 HateXplain posts [7], split "
    "15,383 / 1,922 / 1,924. Majority vote over three annotators yields hate, offensive, or "
    "normal. Target labels map to nine categories (race, religion, nationality, gender, sexual "
    "orientation, ethnicity, political group, other, none); unmapped raw values stay in metadata. "
    "Token-level rationales from the majority-label annotators exist for 9,132 training posts "
    "(59.4%); 9,734 training posts target a specific group. There is no conversational context. "
    "Mathew et al. report BERT at 69.0% test accuracy and 0.674 macro-F1 [7]."
)
h2("3.3", "Counter Context")
para(
    "Each Reddit comment is paired with its parent and labelled hate speech, counter speech, or "
    "neither with respect to that parent [8]. We use the released gold splits (3,325 / 713 / 713) "
    "and leave the silver splits unused. The adapter maps corpus label 0 → hate_speech, 1 → "
    "neither, 2 → counter_speech; processed gold shares match the published 28 / 49 / 23% "
    "approximately (Table 1). Average comment length is 10.2 training-split words versus 23.5 "
    "for HateXplain (2.3× shorter). Yu et al. describe a random 70/15/15 split of 4,751 gold "
    "pairs; we use the released files as-is (test n = 713) and have not verified byte-identity "
    "with their random split."
)
table(
    "Table 1. Processed splits used in every experiment. Class shares are training-split "
    "percentages in label-space order. HateXplain test support is 594 / 548 / 782.",
    ["Property", "HateXplain", "Counter Context (gold)"],
    [
        ["Splits (train / val / test)", "15,383 / 1,922 / 1,924", "3,325 / 713 / 713"],
        ["Label space", "hate, offensive, normal", "hate_speech, counter_speech, neither"],
        ["Train class shares (%)", "30.9 / 28.5 / 40.6", "27.7 / 23.3 / 48.9"],
        ["Target-group labels", "9 categories; 9,734 specific-group posts", "—"],
        ["Token-level rationales", "9,132 / 15,383 train (59.4%)", "—"],
        ["Conversational context", "—", "100% (parent comment)"],
        ["Avg. words per comment (train)", "23.5", "10.2"],
    ],
    widths=[2.15, 2.35, 2.0],
)
figure(
    ASSETS / "fig2_datasets.png",
    "Fig. 1. Label distributions of the processed splits. Left: HateXplain hate classes. "
    "Middle: HateXplain target groups (train). Right: Counter Context gold splits.",
)

# ===========================================================================
# 4 METHOD
# ===========================================================================
h1("4.", "Method")
h2("4.1", "Overview")
para(
    "Figure 2 shows the architecture. Components are switchable by configuration. When no parent "
    "comment is present the context embedding is the zero vector and the API reports "
    "context_used = false; the same code path then serves comment-only inference."
)
figure(
    ASSETS / "fig1_architecture.png",
    "Fig. 2. Architecture. A weight-shared BERT encodes both comments; an interaction module "
    "combines the sentence embeddings with difference and product features; task heads and a "
    "training-only contrastive branch read the fused representation; Integrated Gradients turns "
    "the trained model into evidence spans.",
)
h2("4.2", "Shared encoding and interaction")
para(
    "Let C be the current comment and P the parent. One BERT-base encoder [16] produces sentence "
    "embeddings and token states:"
)
equation("E_c = BERT(C)[CLS],   E_p = BERT(P)[CLS],   token states H_1..H_n")
para(
    "Weight sharing puts both embeddings in one space, which is what makes the following features "
    "meaningful. The features [u; v; |u−v|; u⊙v] originate in InferSent [17] and are used by "
    "Sentence-BERT [18]; we apply them to BERT [CLS] vectors:"
)
equation("D = |E_c − E_p|,   M = E_c ⊙ E_p,   z = [ E_c ; E_p ; D ; M ] ∈ ℝ^3072")
para("A single linear projection (3,072 → 1,024), GELU, and dropout produce the task representation:")
equation("h = Dropout( GELU( W z + b ) ),   W ∈ ℝ^(1024×3072)")
para(
    "With no context, E_p = 0 gives D = |E_c| and M = 0 — a reported fallback, not an error. An "
    "ablation mode concatenates [E_c; E_p] without D and M."
)
h2("4.3", "Contrastive context objective")
para(
    "On conversational batches, mismatched contexts are drawn by shuffling. We use "
    "temperature-scaled InfoNCE [19] — an instance-discriminating loss with shuffled-context "
    "negatives, not supervised contrastive learning, because class labels are unused. The anchor "
    "r_a is the representation of the current comment without its true parent; the positive r_p "
    "is the same comment with its true parent; negatives r_n use mismatched parents:"
)
equation("L_InfoNCE = − log [ exp(sim(r_a, r_p)/τ) / Σ_n exp(sim(r_a, r_n)/τ) ],  τ = 0.07")
para(
    "Rows without a real parent are excluded. The projection head is training-only; inference "
    "uses h. On HateXplain the objective is structurally inactive."
)
h2("4.4", "Multi-task heads")
para(
    "Branching from h are (i) a three-class hate head, (ii) a nine-class target head, (iii) an "
    "eight-class reason head (insult, dehumanization, negative stereotyping, threat, exclusion, "
    "discrimination, incitement to violence, other) that trains only when real reason annotations "
    "are supplied — it is untrained in this work — and (iv) a token-level evidence head (sigmoid "
    "over token states) trained on HateXplain rationales. Missing labels skip the corresponding "
    "loss:"
)
equation("L = λ_h L_hate + λ_t L_target + λ_r L_reason + λ_e L_evidence + λ_c L_InfoNCE")
para(
    "with λ = (1.0, 1.0, 1.0, 1.0, 0.5). No head is trained on fabricated supervision. The API "
    "reports availability flags that the front end renders literally."
)
h2("4.5", "Two evidence mechanisms")
para(
    "Post-hoc attribution. Integrated Gradients [11] attributes the explained logit to input "
    "embeddings. With x the embedding, x′ the PAD baseline, and F the explained logit, token i "
    "receives:"
)
equation("IG_i(x) = (x_i − x′_i) × (1/m) Σ_{k=1}^{m} ∂F(x′ + (k/m)(x − x′)) / ∂x_i")
para(
    "We use m = 32. Special-token scores are zeroed, tokens are thresholded relative to the "
    "strongest token (minimum 0.15), adjacent tokens are merged, subwords are joined, and the "
    "top-5 spans are returned as original-comment substrings with character offsets (the decoder "
    "restores casing via offsets). The top-k and threshold are defaults, not tuned."
)
para(
    "Supervised evidence head. A linear layer over encoder token states, trained with sigmoid BCE "
    "against HateXplain rationales, outputs a per-token rationale probability, thresholded at 0.5. "
    "This is rationale prediction, not an explanation of a specific prediction; we compare the "
    "two mechanisms in Section 6.3 and do not treat either as causal proof [12], [15]."
)
h2("4.6", "Deterministic assembly layer")
para(
    "The assembly layer is the component we evaluate as a contribution (Figure 3). For each "
    "prediction it fills four slots: WHAT (label and confidence from the trained head), WHICH "
    "WORDS (evidence spans from either mechanism), WHO (a pronoun-to-antecedent pass that links "
    "they/them/their and he/she inflections in the current comment to the right-most matching "
    "noun phrase in the parent, with character offsets on both sides), and WHY (fixed templates "
    "filled only with the values above)."
)
para(
    "Five rules are enforced in code. First, no generative model writes any part of the "
    "explanation; the method field reads “deterministic evidence + context-reference analysis”. "
    "Second, attention weights are never presented as reasoning. Third, the layer abstains rather "
    "than invents: if no reference is found the summary says so; non-hateful results are described "
    "without invented hateful evidence. Fourth, prediction, evidence, and explanation are separate "
    "fields; each evidence item declares source (current vs. parent), type (attribution span vs. "
    "context target), and a grounded rationale. Fifth, a reasoning failure returns “Reasoning "
    "unavailable” without failing the request. Untrained heads are reported as unavailable."
)
para(
    "Two caveats are part of the design. The templates may state that a contextual reference "
    "“matters” whenever a link is found; Section 6.4 shows a pair where a link is found and the "
    "score does not move, so that phrase describes interpretive dependence, not causal score "
    "dependence. The reference analysis is a heuristic; Section 6.5 reports its coverage and "
    "failure modes; every response labels it a heuristic. An earlier template claimed hostility "
    "“toward that target” — a claim no model output supports — and emitted lowercased spans; both "
    "behaviours were removed."
)
figure(
    ASSETS / "fig6_reasoning_pipeline.png",
    "Fig. 3. Reasoning pipeline. Stages 2–4 analyse the actual inputs and outputs of the trained "
    "model. Nothing in the final explanation is written by a generative model. The worked example "
    "at the bottom is reproduced in Section 6.4.",
)
h2("4.7", "Novelty")
para(
    "Individually, the parts are standard: BERT fine-tuning [16], InferSent-style matching "
    "features [17], [18], InfoNCE [19], multi-task heads, Integrated Gradients [11]. The assembly "
    "layer of Section 4.6 is what we treat as the contribution: a deterministic, provenance-tracked, "
    "abstaining explanation assembler for hate-speech decisions, with an explicit no-LLM rule, "
    "availability flags, and a worked-example protocol that reports when context does not change "
    "the score. The supervised-versus-post-hoc evidence comparison, the deletion probe, the "
    "matched-budget context ladder, and the published-baseline tables are the supporting studies."
)

# ===========================================================================
# 5 SETUP
# ===========================================================================
h1("5.", "Experimental Design")
h2("5.1", "Detector configurations")
para(
    "HateXplain uses four configurations (Table 2): a comment-only baseline; a five-epoch "
    "multi-task “full” model with the interaction module at its zero-context fallback; hx_full, "
    "the two-epoch multi-task model used for evidence and serving analysis; and hx_full_long, a "
    "six-epoch-budget control. InfoNCE is structurally present in “full” but inert because the "
    "corpus has no parent pairs. HateXplain runs use seed 42; extra seeds are reported when "
    "available."
)
para(
    "Counter Context is studied two ways. The original suite (cc_baseline at 5 epochs / batch 16; "
    "an earlier concat run at the default budget; cc_context = interaction + InfoNCE at 4 / 12) "
    "is used for the published-baseline comparison. The matched-budget ladder — no context, "
    "parent concatenation, interaction, interaction+InfoNCE — is trained at four epochs, batch "
    "12, seeds 42 / 1 / 2, so that rungs differ only in the fusion component. A BERT "
    "cross-encoder ([CLS] parent [SEP] current [SEP]) at the same budget is trained as an "
    "additional control when those runs have finished; it is not one of our serving checkpoints."
)
table(
    "Table 2. Detector configurations. The matched ladder (bottom) fixes epochs and batch so that "
    "rungs isolate fusion. InfoNCE is inert on HateXplain.",
    ["Variant", "Corpus", "Context", "Fusion", "InfoNCE", "Heads", "Ep. / batch"],
    [
        ["baseline", "HateXplain", "—", "—", "—", "hate", "5 / 16"],
        ["full", "HateXplain", "zero-fallback", "interaction", "inert", "hate+target+evid.", "5 / 16"],
        ["hx_full", "HateXplain", "zero-fallback", "interaction", "off", "hate+target+evid.", "2 / 16"],
        ["hx_full_long", "HateXplain", "zero-fallback", "interaction", "off", "hate+target+evid.", "6* / 16"],
        ["cc_baseline", "Counter Context", "—", "—", "—", "hate", "5 / 16"],
        ["cc_context", "Counter Context", "parent", "interaction", "on", "hate", "4 / 12"],
        ["ladder: none", "Counter Context", "—", "—", "—", "hate", "4 / 12"],
        ["ladder: concat", "Counter Context", "parent", "concat", "off", "hate", "4 / 12"],
        ["ladder: interact", "Counter Context", "parent", "interaction", "off", "hate", "4 / 12"],
        ["ladder: +InfoNCE", "Counter Context", "parent", "interaction", "on", "hate", "4 / 12"],
        ["cross-encoder", "Counter Context", "parent", "[CLS] P [SEP] C", "—", "hate", "4 / 12"],
    ],
    widths=[1.15, 1.05, 0.85, 0.95, 0.55, 1.05, 0.8],
    font_size=7.5,
)
h2("5.2", "Optimization and inference")
table(
    "Table 3. Shared hyperparameters (base configuration).",
    ["Parameter", "Value", "Parameter", "Value"],
    [
        ["Encoder", "bert-base-uncased (110M)", "Max sequence length", "128 tokens"],
        ["Fine-tune", "full (no frozen layers)", "Batch size", "16 (12 on CC ladder)"],
        ["Learning rate", "2e-5, AdamW", "Epochs", "Table 2"],
        ["Weight decay", "0.01", "Warmup", "10% linear decay"],
        ["Gradient clipping", "1.0", "Mixed precision", "auto (CUDA)"],
        ["Early stopping", "patience 2 (val macro-F1)", "Seeds", "42; ladder 42/1/2"],
        ["Interaction dim", "1024 (dropout 0.2)", "Projection / τ", "256 / 0.07"],
        ["IG steps", "32", "Evidence top-k / min", "5 / 0.15"],
    ],
    widths=[1.55, 1.7, 1.55, 1.7],
)
para(
    "Training uses PyTorch and Hugging Face Transformers [20]. Test numbers come from the "
    "best-validation checkpoint, evaluated once on the held-out split. Checkpoints store optimizer, "
    "scheduler, configuration, label maps, and metrics."
)
h2("5.3", "Metrics")
para(
    "Classification: accuracy, macro and weighted F1, per-class F1, confusion matrices. Multi-seed "
    "cells are mean ± sample standard deviation. Evidence: token precision/recall/F1 against gold "
    "HateXplain rationales on the first 100 rationale-bearing test examples in dataset order, "
    "macro-averaged as the mean of per-example F1 (zero-overlap examples contribute zero, so "
    "macro-F1 can sit below both precision and recall); we also report micro-F1 for the trained "
    "head. Faithfulness: ERASER-style comprehensiveness and sufficiency on the same 100 examples "
    "with PAD substitution. Latency: mean / p50 / p95 over 100 sequential requests after warm-up, "
    "batch 1. Significance: McNemar's test [25] and paired bootstrap 95% CIs on accuracy "
    "differences. Reference linker: coverage (fraction of pairs with at least one emitted link); "
    "no gold link annotations exist, so we do not report precision."
)

# ===========================================================================
# 6 RESULTS
# ===========================================================================
h1("6.", "Results")
h2("6.1", "HateXplain (RQ1)")
table(
    "Table 4. HateXplain test results (n = 1,924). Published rows are quoted from Mathew et al. "
    "[7], Table 6. Our rows are seed 42.",
    ["Model", "Acc.", "Macro-F1", "wF1", "Hate F1", "Off. F1", "Norm. F1"],
    [
        ["BERT (Mathew et al. 2021) [7]", "0.690", "0.674", "—", "—", "—", "—"],
        ["BERT-HateXplain (rationale-supervised) [7]", "0.698", "0.687", "—", "—", "—", "—"],
        ["baseline (comment-only)", "0.661", "0.654", "0.662", "0.746", "0.514", "0.701"],
        ["full (multi-task, 5 ep.)", "0.685", "0.677", "0.685", "0.769", "0.536", "0.727"],
        ["hx_full (multi-task, 2 ep.)", "0.692", "0.677", "0.688", "0.766", "0.514", "0.751"],
        ["hx_full_long (6-ep. budget)", "0.698", "0.681", "0.692", "0.771", "0.521", "0.752"],
    ],
    widths=[2.4, 0.65, 0.8, 0.6, 0.7, 0.65, 0.7],
)
para(
    "Multi-task supervision with target and rationale heads improves test accuracy by 3.1 points "
    "and macro-F1 by 0.023 over the comment-only baseline (McNemar p = 0.0016; paired-bootstrap "
    "95% CI on accuracy [0.012, 0.050]; n = 1,924) and lands within half a point of the dataset "
    "authors' BERT. The six-epoch budget adds only +0.6 accuracy and +0.004 macro-F1 over "
    "hx_full: two epochs exhaust this configuration. Target prediction (hx_full) reaches 69.2% "
    "accuracy — the same number as the hate head because both heads misclassify exactly 592 of "
    "1,924 posts — and 0.622 macro-F1 over the eight target classes with test support (0.553 if "
    "the unsupported political_group class is included). Per-class target F1: religion 0.784, race "
    "0.736, nationality 0.719, sexual orientation 0.710, none 0.720, ethnicity 0.660, gender "
    "0.485, other 0.160."
)
hx_extra = hx_seed_rows()
if hx_extra:
    rows_hx = [r for r in hx_extra]
    acc = mean_std([r["test_accuracy"] for r in rows_hx] + [0.6923076923076923])
    mf1 = mean_std([r["test_macro_f1"] for r in rows_hx] + [0.6767342443112813])
    para(
        f"Extra hx_full seeds ({', '.join(str(r.get('seed')) for r in rows_hx)}) give a "
        f"{acc[2]}-seed accuracy of {acc[0]:.3f} ± {acc[1]:.3f} and macro-F1 of "
        f"{mf1[0]:.3f} ± {mf1[1]:.3f}, including the seed-42 run. The HateXplain ordering is "
        f"therefore not an artefact of one draw."
    )
figure(
    ASSETS / "fig3_curves.png",
    "Fig. 4. Validation macro-F1 across epochs. HateXplain runs saturate in 2–5 epochs and then "
    "overfit; Counter Context runs improve slowly and remain noisier.",
)

h2("6.2", "Counter Context (RQ2)")
table(
    "Table 5. Counter Context test (n = 713) against Yu et al. [8], Table 6 (RoBERTa-base; their "
    "70/15/15 split of the same 4,751 gold pairs). Our original-suite rows are seed 42. The "
    "matched ladder is in Table 6.",
    ["Model", "Acc.", "Macro-F1", "wF1", "Hate F1", "Ctr F1", "Nei F1"],
    [
        ["Majority (Yu et al.) [8]", "—", "—", "0.34", "—", "—", "—"],
        ["RoBERTa, target only [8]", "—", "—", "0.58", "0.56", "0.38", "0.69"],
        ["RoBERTa, parent [SEP] target [8]", "—", "—", "0.61", "0.59", "0.44", "0.70"],
        ["RoBERTa best (+silver+stance) [8]", "—", "—", "0.64", "0.60", "0.53", "0.71"],
        ["cc_baseline (5/16)", "0.588", "0.557", "0.587", "0.511", "0.484", "0.674"],
        ["cc_context (inter.+InfoNCE)", "0.617", "0.563", "0.607", "0.545", "0.416", "0.728"],
    ],
    widths=[2.4, 0.65, 0.8, 0.6, 0.7, 0.65, 0.7],
)

ladder_rows = [r for r in [
    ladder_row("none (no context)", "cc_baseline"),
    ladder_row("parent concat", "cc_concat"),
    ladder_row("interaction", "cc_inter"),
    ladder_row("interaction + InfoNCE", "cc_nce"),
] if r[1] != "0"]
xe = xe_rows()
if xe:
    acc = mean_std([r["test_accuracy"] for r in xe])
    mf1 = mean_std([r["test_macro_f1"] for r in xe])
    wf1 = mean_std([r["test_weighted_f1"] for r in xe])
    h = mean_std([r["test_per_class_f1"]["hate_speech"] for r in xe])
    c = mean_std([r["test_per_class_f1"]["counter_speech"] for r in xe])
    n = mean_std([r["test_per_class_f1"]["neither"] for r in xe])
    ladder_rows.append([
        "cross-encoder [CLS] P [SEP] C",
        str(acc[2]),
        pm(*acc), pm(*mf1), pm(*wf1), pm(*h), pm(*c), pm(*n),
    ])
table(
    "Table 6. Matched-budget Counter Context ladder (4 epochs, batch 12). Mean ± sample s.d. "
    "over seeds. Same test split as Table 5.",
    ["Variant", "n", "Acc.", "Macro-F1", "wF1", "Hate F1", "Ctr F1", "Nei F1"],
    ladder_rows,
    widths=[1.85, 0.3, 0.85, 0.85, 0.75, 0.75, 0.75, 0.75],
    font_size=7.5,
)
figure(
    ASSETS / "fig8_cc_ladder.png"
    if (ASSETS / "fig8_cc_ladder.png").exists()
    else ASSETS / "fig3_curves.png",
    "Fig. 5. Counter Context test macro-F1 by matched-budget variant (mean ± s.d. over seeds).",
)
para(
    "Three findings. First, parent context helps, and the help is not an artefact of one seed: "
    "under a matched budget, every concatenation seed beats every no-context seed on accuracy "
    "(means 0.584 ± 0.007 vs. 0.565 ± 0.008). An earlier comparison in which concatenation looked "
    "harmful (0.558 vs. 0.588) used different batch sizes and epoch counts and should not be read "
    "as evidence against context. Second, interaction features produce a further, equally clean "
    "accuracy step (0.603 ± 0.007); again every interaction seed is above every concatenation "
    "seed. Macro-F1 rises less cleanly because interaction has higher seed variance on "
    "counter-speech. Third, InfoNCE on top of interaction does not produce a comparable clean "
    "step; we read the original cc_context result (0.617 accuracy / 0.607 weighted-F1) as "
    "situated in the same band as the interaction rung rather than as a component win. Our best "
    "numbers sit at Yu et al.'s gold cross-encoder (0.61 weighted-F1) and below their best "
    "silver+stance result (0.64)."
)
if PRED.get("paired"):
    lines = []
    for ptest in PRED["paired"]:
        lines.append(
            f"{ptest['a']} vs. {ptest['b']}: Δacc = {ptest['acc_a'] - ptest['acc_b']:+.3f}, "
            f"McNemar p = {ptest['mcnemar_p']:.3g}, bootstrap 95% CI "
            f"[{ptest['boot_ci'][0]:.3f}, {ptest['boot_ci'][1]:.3f}]"
        )
    para(
        "Paired tests on the seed-42 test predictions "
        f"({'; '.join(lines)}) confirm the direction of the concat-vs-none contrast; the "
        "interaction step is directionally the same and is reported with its seed envelope rather "
        "than as a standalone significance claim at n = 3."
    )
else:
    para(
        "Paired McNemar tests on the original cc_context vs. cc_baseline pair give p = 0.099 "
        "(bootstrap CI [−0.004, 0.062]); the matched-ladder ordering is the stronger evidence "
        "for context, and we do not rest a component claim on a single-seed p-value."
    )

h2("6.3", "Evidence: plausibility and a faithfulness probe (RQ3)")
table(
    "Table 7. Token-level agreement with gold HateXplain test rationales (first 100 "
    "rationale-bearing examples in dataset order; macro = mean of per-example F1). IG rows use "
    "32-step Integrated Gradients; the head row thresholds the trained evidence head at 0.5.",
    ["Mechanism", "Checkpoint", "Macro P", "Macro R", "Macro F1", "Micro F1"],
    [
        ["Integrated Gradients", "baseline", "0.523", "0.706", "0.554", "—"],
        ["Integrated Gradients", "hx_full", "0.707", "0.706", "0.650", "—"],
        ["Integrated Gradients", "hx_full_long", "0.604", "0.744", "0.614", "—"],
        ["Supervised evidence head (τ=0.5)", "hx_full", "0.851", "0.867", "0.823", "0.779"],
    ],
    widths=[2.15, 1.15, 0.75, 0.75, 0.8, 0.75],
)
para(
    "The trained head agrees with human rationales far better than post-hoc Integrated Gradients "
    "on the same examples (0.823 vs. 0.650 macro token-F1). This is plausibility, not faithfulness "
    "[13]. The head is trained on the rationale labels; Integrated Gradients explains a specific "
    "predicted logit, so the two mechanisms answer different questions. Mathew et al. report "
    "0.37–0.51 token-F1 for attention- and LIME-based selection on BERT [7] — not directly "
    "comparable, but consistent with post-hoc methods sitting below a trained head. Multi-task "
    "training also improves Integrated Gradients itself (0.554 on the baseline vs. 0.650 on "
    "hx_full)."
)
ig = ADD.get("ig_comprehensiveness_sufficiency") or {}
para(
    "Faithfulness probe. Deleting the Integrated-Gradients evidence tokens (PAD substitution) on "
    "the same 100 examples leaves the predicted-class probability almost unchanged: "
    f"comprehensiveness {ig.get('comprehensiveness_mean', 0.015):.3f} ± "
    f"{ig.get('comprehensiveness_std', 0.224):.3f}; sufficiency "
    f"{ig.get('sufficiency_mean', 0.101):.3f} ± {ig.get('sufficiency_std', 0.231):.3f}. The "
    "highlighted spans are plausible and not necessary for the decision in this setup. "
    "PAD-substitution is a crude deletion; we report the numbers as a lower-bound probe, not as a "
    "definitive audit. This is the gap the assembly layer's abstention and provenance rules exist "
    "to expose."
)

h2("6.4", "Worked example: when context does not change the score")
table(
    "Table 8. Live system output for parent = “I saw a group of immigrants protesting downtown.”, "
    "current = “They should all be kicked out.” (cc_context serving checkpoint, IG, 32 steps).",
    ["Field", "Value"],
    [
        ["Prediction", "hate_speech (p = 0.757; neither 0.185, counter_speech 0.057)"],
        ["Evidence spans (IG)", "“kicked” (1.00, chars 19–25); “They should” (0.66, chars 0–11)"],
        ["Token attributions",
         "kicked 1.17; they 0.84; should 0.69; all −0.49; be −0.19; . −0.38; out −0.03"],
        ["Context link", "“They” (0–4) → refers_to → “group of immigrants” (8–27)"],
        ["Template summary",
         "“The current comment uses ‘They’ to refer to ‘group of immigrants’ from the previous "
         "comment and expresses hostile or exclusionary language. The contextual reference "
         "matters: on its own, ‘They’ does not identify who or what is being referred to. "
         "Strongest evidence: the phrase ‘kicked’ and ‘They should’.”"],
        ["Method", "deterministic evidence + context-reference analysis"],
    ],
    widths=[1.45, 5.05],
)
figure(
    ASSETS / "fig7_attribution.png",
    "Fig. 6. Integrated-Gradients token attributions for the worked example (real output of the "
    "serving checkpoint).",
)
para(
    "The same comment without its parent yields the same label at almost the same confidence "
    "(0.754 vs. 0.757) and the same evidence spans. For this pair, context matters to the "
    "interpretation (the link names the target for the reader) and not to the score. The "
    "template's phrase “the contextual reference matters” should be read that way; a revision "
    "should say “the parent supplies a candidate referent” and stop there."
)

h2("6.5", "Failure modes: linker, HateCheck, per-class errors (RQ4)")
para(
    "Reference linker. There are no gold antecedent annotations, so we report coverage. On the "
    f"full Counter Context test set the heuristic emits at least one link for "
    f"{REF['pairs_with_link']} of {REF['n_pairs']} pairs ({100 * REF['coverage']:.1f}% coverage; "
    f"{REF['total_links']} links). Almost every parent contains some noun-phrase candidate "
    f"({REF['prev_with_antecedent_candidate']} / {REF['n_pairs']}), so the bottleneck is the "
    "linking rule. Manual inspection of emitted links shows systematic wrong-antecedent errors "
    "(function words and the wrong entity). The WHO slot is a low-yield heuristic with "
    "unquantified precision; it is labelled as such in every response. A hand-labelled evaluation "
    "set is the next required experiment."
)
para(
    "HateCheck. We ran the 3,728-case suite [21] through the hx_full hate head. The model does "
    "not transfer. Of 2,563 hateful cases, 12.8% are predicted hate, 32.3% offensive, and 54.9% "
    "normal. Of 1,165 non-hateful cases, 74.3% are predicted normal and 5.4% hate. Functionality "
    "recall on hateful cases is highest for slurs (0.438) and below 0.20 for most others; spelling "
    "perturbations collapse (character-swap 0.045, inserted spaces 0.000). HateCheck is "
    "adversarial and its binary split is not HateXplain's three-class policy, so this is a "
    "transfer result — and a negative one. HateXplain-trained BERT at this scale does not have "
    "the functional competence HateCheck measures."
)
figure(
    ASSETS / "fig5_perclass.png",
    "Fig. 7. Test per-class F1 for hx_full. Left: hate head. Right: target head "
    "(political_group has zero test support).",
)
para(
    "Per-class. On HateXplain, hate is detected strongly (F1 0.766, recall 0.806) and normal is "
    "stable (0.751). Offensive is the weak class (F1 0.514, recall 0.482): 120 offensive posts "
    "become hate and 164 become normal. This mirrors the documented hate/offensive boundary "
    "problem [1] and HateXplain's label policy. The six-epoch control moves offensive F1 only to "
    "0.521, so under-training is not the cause at this scale. The target head follows a "
    "data-volume gradient; the rare other class (85 test posts) collapses to 0.160, and gender is "
    "diluted by none-confusion (recall 0.393). On Counter Context, counter-speech is the weak "
    "class (F1 0.416 in cc_context, recall 0.375): 79 of 168 counter-speech test pairs become "
    "neither. Counter speech is defined by intent relative to the parent; a model that under-uses "
    "context defaults to neither. The matched ladder does not rescue this class."
)
figure(
    ASSETS / "fig4_confusion.png",
    "Fig. 8. Test confusion matrices (row-normalized). Left: hx_full on HateXplain. Right: "
    "cc_context on Counter Context.",
)

h2("6.6", "Latency")
lat = ADD.get("latency") or {}
table(
    "Table 9. Serving-pipeline latency (cc_context checkpoint, batch 1, 100 sequential requests "
    "after warm-up, NVIDIA GeForce RTX 4060 Laptop GPU). Predict includes tokenization, encoding, "
    "heads, 32-step Integrated Gradients, and the assembly layer.",
    ["Request type", "mean (ms)", "p50 (ms)", "p95 (ms)"],
    [
        ["predict, no context",
         f"{lat.get('predict_no_context', {}).get('mean_ms', 670):.0f}",
         f"{lat.get('predict_no_context', {}).get('p50_ms', 664):.0f}",
         f"{lat.get('predict_no_context', {}).get('p95_ms', 712):.0f}"],
        ["predict, with context",
         f"{lat.get('predict_with_context', {}).get('mean_ms', 1059):.0f}",
         f"{lat.get('predict_with_context', {}).get('p50_ms', 1039):.0f}",
         f"{lat.get('predict_with_context', {}).get('p95_ms', 1141):.0f}"],
        ["explain (IG-32, with context)",
         f"{lat.get('explain_ig32_with_context', {}).get('mean_ms', 961):.0f}",
         f"{lat.get('explain_ig32_with_context', {}).get('p50_ms', 1048):.0f}",
         f"{lat.get('explain_ig32_with_context', {}).get('p95_ms', 1305):.0f}"],
    ],
    widths=[2.6, 1.2, 1.2, 1.2],
)
para(
    "These are review-assist latencies, not interactive-UI or high-throughput enforcement "
    "latencies. A parent comment roughly adds a second encoder pass. Integrated Gradients "
    "dominates the comment-only path. A 15-request pilot on the same checkpoint measured 1.7–3.1 s, "
    "illustrating laptop-GPU clock variance; we treat the n = 100 run as the reference. Serving "
    "the evidence head instead of Integrated Gradients, or reducing the step count, are the "
    "obvious engineering levers; we have not optimized either."
)

h2("6.7", "Threats to validity")
para(
    "HateXplain detector results are seed 42 (plus extra hx_full seeds when reported above). "
    "Counter Context no-context, concatenation, interaction, and interaction+InfoNCE have three "
    "seeds; the original cc_baseline / cc_context pair remains a single-seed comparison against "
    "Yu et al. We quote Yu et al. rather than re-running their RoBERTa models, and we have not "
    "verified byte-identity of the gold split. Evidence metrics use the first 100 "
    "rationale-bearing examples, not a random sample. The linker has no precision estimate. "
    "There is no human study of explanation usefulness, no LLM-explanation baseline, and no "
    "comparison against a bare evidence-and-label template. HateCheck was not used for model "
    "selection. HateXplain runs use the zero-context fallback, so the interaction module there is "
    "an architectural compatibility check, not a measured gain. The HateXplain multi-task gain "
    "could come from the extra projection, the target head, or the evidence head; the suite does "
    "not separate them. Latency is one laptop GPU."
)

# ===========================================================================
# 7 SYSTEM
# ===========================================================================
h1("7.", "System Implementation")
para(
    "The framework is a deployable stack. A FastAPI backend loads the checkpoint once at startup "
    "and serves GET /health, GET /api/v1/model/info, POST /api/v1/predict (classification, "
    "evidence, structured reasoning), and POST /api/v1/explain (per-token attribution for any "
    "head and class). Inference runs under no_grad; attribution temporarily enables gradients on "
    "input embeddings only. A serving-export script strips optimizer state. A React + TypeScript "
    "front end renders the explanation chain — classification, reasoning card, offset-based "
    "highlighting in both comments, on-demand per-token attribution, evaluation metrics, history — "
    "and includes an annotation workspace that exports the reason-head schema. The system is a "
    "research prototype for review-assist tooling. It is not designed, calibrated, or evaluated "
    "for autonomous enforcement."
)

# ===========================================================================
# 8 ETHICS
# ===========================================================================
h1("8.", "Ethics Statement")
para(
    "This work uses text that demeans real communities. All examples come from the public "
    "HateXplain and Counter Context corpora; we quote the minimum needed for the worked example "
    "and add no new slurs. Reader discretion is advised. We did not collect new annotations and "
    "did not compensate annotators."
)
para(
    "Hate-speech datasets have repeatedly been shown to over-flag African-American English and "
    "identity-term usage [14]. Our models inherit that bias; the target-head gender and other "
    "weaknesses in Section 6.5 are likely the same problem in another form. Reclaimed slurs and "
    "in-group usage can be misclassified; evidence spans are meant to make such errors inspectable, "
    "not to justify them. A deployed detector can be used for surveillance or over-blocking; this "
    "system is a research prototype for review-assist tooling. The same spans that help a "
    "moderator audit a decision could help an adversary probe the classifier; we consider the "
    "audit benefit of deterministic, offset-accurate evidence to outweigh that risk at research "
    "scale. Confidence scores are uncalibrated model outputs, not probabilities of harm. The "
    "reason head is untrained; no harm-category statement is made without its own annotations. "
    "The worked example is a dehumanizing statement about immigrants; we reproduce it because it "
    "is the shortest real transcript that shows the reference link, the evidence spans, and the "
    "context-does-not-change-the-score caveat at once."
)

# ===========================================================================
# 9 LIMITATIONS
# ===========================================================================
h1("9.", "Limitations")
bullet("HateXplain detector results other than the extra hx_full seeds (when present) are seed 42.")
bullet(
    "The cross-encoder control, when reported, uses BERT rather than Yu et al.'s RoBERTa and is "
    "not a re-implementation of their full recipe (silver data, stance pre-training)."
)
bullet(
    "The pronoun linker has 16.6% coverage and visible wrong-antecedent errors. No human study of "
    "explanation usefulness has been conducted."
)
bullet(
    "HateCheck transfer is poor. Offensive-class confusion persists (F1 ≈ 0.51–0.54) and is "
    "bounded by HateXplain's label policy at this scale."
)
bullet(
    "Evidence metrics use 100 examples. Integrated Gradients is not a faithfulness measure "
    "[12], [15]; the trained head's 0.823 token-F1 is plausibility. The deletion probe uses PAD "
    "substitution."
)
bullet(
    "Latency (0.7–1.1 s at n = 100 on one laptop GPU) precludes interactive-UI or high-throughput "
    "use without engineering."
)
bullet(
    "Counter Context performance trails the corpus authors' best silver+pre-training result (0.64 "
    "weighted-F1). The design prioritizes a unified single-encoder system over corpus-specific "
    "tuning."
)

# ===========================================================================
# 10 CONCLUSION
# ===========================================================================
h1("10.", "Conclusion")
para(
    "We described a hate-speech detection system whose explanation layer is deterministic, "
    "provenance-tracked, and free of generation. That design makes every emitted statement "
    "checkable against a stored output. It does not make Integrated Gradients faithful — a "
    "deletion probe showed the highlighted evidence is largely unnecessary for the prediction — "
    "and it does not make the pronoun linker correct. The detector, assembled from standard parts, "
    "matches published BERT results on HateXplain (69.2% / 0.677 at two epochs; 69.8% / 0.681 "
    "with a longer budget, vs. 69.0% / 0.674 [7]). On Counter Context, parent context helps, and "
    "under a matched budget the accuracy ordering no-context < concatenation < interaction holds "
    "across three seeds; InfoNCE is not a component win; the best configuration is comparable to "
    "the corpus authors' gold cross-encoder (0.61 weighted-F1 [8]). A trained rationale head "
    "(0.823 token-F1) is far more plausible than post-hoc Integrated Gradients (0.650). HateCheck "
    "transfer and linker coverage are negative results we leave standing. Latency is review-assist "
    "scale. The part worth building on is the explanation layer and the evaluation protocol around "
    "it — not a state-of-the-art classifier. The next experiments are a hand-labelled linker set, "
    "a human study of explanation usefulness, and a cross-encoder at more seeds."
)

# ===========================================================================
# REFERENCES
# ===========================================================================
p = para("References", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=6)
refblock([
    "T. Davidson, D. Warmsley, M. Macy, and I. Weber, “Automated hate speech detection and the "
    "problem of offensive language,” in Proc. 11th Int. AAAI Conf. Web and Social Media (ICWSM), "
    "2017, pp. 512–515.",
    "Z. Waseem and D. Hovy, “Hateful symbols or hateful people? Predictive features for hate "
    "speech detection on Twitter,” in Proc. NAACL-HLT Student Research Workshop, 2016, pp. 88–93.",
    "A. Founta et al., “Large scale crowdsourcing and characterization of Twitter abusive "
    "behavior,” in Proc. 12th Int. AAAI Conf. Web and Social Media (ICWSM), 2018.",
    "M. Zampieri, S. Malmasi, P. Nakov, S. Rosenthal, F. Farra, and R. Kumar, “Predicting the "
    "type and target of offensive posts in social media,” in Proc. NAACL-HLT, 2019, pp. 1415–1420.",
    "J. Fortuna and S. L. Nunes, “A survey on automatic detection of hate speech in text,” ACM "
    "Computing Surveys, vol. 51, no. 4, article 85, 2018.",
    "B. Vidgen and L. Derczynski, “Directions in abusive language training data, a systematic "
    "review: Garbage in, garbage out,” PLoS ONE, vol. 15, no. 12, e0243300, 2020.",
    "B. Mathew, P. Saha, S. M. Yimam, C. Biemann, P. Goyal, and A. Mukherjee, “HateXplain: A "
    "benchmark dataset for explainable hate speech detection,” in Proc. 35th AAAI Conf. "
    "Artificial Intelligence, vol. 35, no. 17, 2021, pp. 14867–14875.",
    "X. Yu, E. Blanco, and L. Hong, “Hate speech and counter speech detection: Conversational "
    "context does matter,” in Proc. NAACL-HLT, 2022, pp. 5918–5930.",
    "J. H. Park and P. Fung, “One-step and two-step classification for abusive language "
    "detection on Twitter,” in Proc. ALW1 (Abusive Language Workshop), 2017, pp. 41–45.",
    "M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should I trust you?’ Explaining the "
    "predictions of any classifier,” in Proc. 22nd ACM KDD, 2016, pp. 1135–1144.",
    "M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in Proc. "
    "34th Int. Conf. Machine Learning (ICML), PMLR 70, 2017, pp. 3319–3328.",
    "S. Jain and B. C. Wallace, “Attention is not explanation,” in Proc. NAACL-HLT, 2019, "
    "pp. 2431–2441.",
    "J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, and B. C. Wallace, "
    "“ERASER: A benchmark to evaluate rationalized NLP models,” in Proc. ACL, 2020, "
    "pp. 4443–4458.",
    "A. Sap, D. Card, S. Gabriel, Y. Choi, and N. A. Smith, “The risk of racial bias in hate "
    "speech detection,” in Proc. AACL-IJCNLP, 2019, pp. 79–89.",
    "A. Jacovi and Y. Goldberg, “Towards faithfully interpretable NLP systems: How should we "
    "define and evaluate faithfulness?” in Proc. ACL, 2020, pp. 4198–4205.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional "
    "Transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186.",
    "A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes, “Supervised learning of "
    "universal sentence representations from data in the wild,” in Proc. EMNLP, 2018, "
    "pp. 1870–1879.",
    "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese "
    "BERT-networks,” in Proc. EMNLP-IJCNLP, 2019, pp. 3982–3992.",
    "A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive "
    "predictive coding,” arXiv preprint arXiv:1807.03748, 2018.",
    "T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, "
    "M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, "
    "T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, and A. Rush, “Transformers: State-of-the-art "
    "natural language processing,” in Proc. EMNLP System Demonstrations, 2020, pp. 38–45.",
    "P. Röttger, B. Vidgen, D. Nguyen, Z. Waseem, H. Margetts, and J. Pierrehumbert, "
    "“HateCheck: Functional tests for hate speech detection models,” in Proc. ACL-IJCNLP, 2021, "
    "pp. 41–58.",
    "C. Ziems, W. Held, O. Shaikh, J. Chen, Z. Zhang, and D. Yang, “Can large language models "
    "transform computational social science?” Computational Linguistics, vol. 50, no. 1, "
    "pp. 237–291, 2024.",
    "M. Turpin, J. Michael, E. Perez, and S. R. Bowman, “Language models don't always say what "
    "they think: Unfaithful explanations in chain-of-thought prompting,” in Advances in Neural "
    "Information Processing Systems (NeurIPS), 2023.",
    "Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and "
    "V. Stoyanov, “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv preprint "
    "arXiv:1907.11692, 2019.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or "
    "percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, 1947.",
])

# ===========================================================================
# APPENDIX
# ===========================================================================
p = para("Appendix A   Reproducibility", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
         space_before=12, space_after=6)
para(
    "Every experiment is one command from backend/ with the GPU interpreter of the training "
    "environment. Metric artifacts live in paper_assets/ (experiment_results.json, "
    "addenda_results.json, evidence_head_metrics.json, latency.json, reference_linker.json, "
    "explain_canonical.json, and — when present — cc_predictions.json). Figures 1–8 and Tables "
    "1–9 are generated from those files."
)
for m in [
    "python scripts/download_data.py --dataset all",
    "python scripts/preprocess.py",
    "python scripts/train.py --config baseline | full | hx_full | hx_full_long | cc_baseline | cc_context",
    "python scripts/train.py --config context --dataset counter_context",
    "python scripts/evaluate.py --config <name>       # writes best.metrics.json",
    'python scripts/explain.py --text "They should all be kicked out." --context "I saw a group of immigrants protesting downtown."',
    "uvicorn app.main:app --port 8001                 # /api/v1/predict, /api/v1/explain",
]:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run(m), 8.5, name=MONO)

add_page_number_footer()
OUT_DOCX.parent.mkdir(exist_ok=True)
doc.save(str(OUT_DOCX))
print("saved:", OUT_DOCX, f"{OUT_DOCX.stat().st_size / 1024:.0f} KiB")
