# -*- coding: utf-8 -*-
"""Publishable research-paper DOCX (v3).

Every number is taken from repository artifacts or from the cited publications.
This revision addresses the 29 Oct 2026 external review: honest system framing,
matched-budget Counter Context seeds, HateCheck, reference-linker audit,
published baselines, faithfulness probe, latency, ethics, and copy-edits.
"""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project")
ASSETS = ROOT / "paper_assets"
OUT_DOCX = ROOT / "deliverables" / "hate-speech-research-paper.docx"
FONT = "Times New Roman"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin = sec.bottom_margin = Inches(1)

style = doc.styles["Normal"]
style.font.name = FONT
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

core = doc.core_properties
core.author = "Hari Nagappan"
core.title = "Structured, Evidence-Based Reasoning for Hate Speech Detection"
core.comments = "Preprint. Deterministic explanation layer over a multi-task BERT hate-speech classifier."


def set_font(run, size=10.5, bold=False, italic=False, color=None, name=FONT):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), name)
    return run


def para(text="", size=10.5, bold=False, italic=False,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, indent=None, line=1.15):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    if text:
        set_font(p.add_run(text), size, bold, italic)
    return p


def rich(p, parts):
    for text, fmt in parts:
        set_font(p.add_run(text), **fmt)
    return p


def h1(num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(f"{num}.  {text}"), 12, bold=True)
    return p


def h2(num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(f"{num}.  {text}"), 11, bold=True, italic=True)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    set_font(p.add_run(text))
    return p


_eq_no = [0]


def equation(formula):
    _eq_no[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    set_font(p.add_run("\t" + formula), 10.5, italic=True)
    set_font(p.add_run(f"\t({_eq_no[0]})"), 10.5)
    return p


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


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def table(caption, headers, rows, widths=None, font_size=8.5, align_first_left=True):
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_before = Pt(8)
    c.paragraph_format.space_after = Pt(3)
    c.paragraph_format.keep_with_next = True
    set_font(c.add_run(caption), 9, italic=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htext in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(htext), font_size, bold=True)
        shade(cell, "D9E2F3")
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if (align_first_left and j == 0) else WD_ALIGN_PARAGRAPH.CENTER
            )
            set_font(p.add_run(str(val)), font_size)
    if widths:
        for j, w in enumerate(widths):
            for r in t.rows:
                r.cells[j].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def refblock(refs):
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_font(p.add_run(f"[{i}]  {r}"), 9.5)


def add_page_number_footer():
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    set_font(r, 9)
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


# =============================================================== title
para(
    "Structured, Evidence-Based Reasoning for Hate Speech Detection:",
    size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0,
)
p = para("", size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
set_font(
    p.add_run(
        "A Multi-Task BERT Framework with Post-Hoc Attribution "
        "and a Deterministic Explanation Layer"
    ),
    14, bold=True,
)
para("Hari Nagappan", size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0, italic=True)
para("Independent Researcher", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
para("harisnagappan@gmail.com", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("Preprint — October 29, 2026", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Inches(0.35)
p.paragraph_format.right_indent = Inches(0.35)
set_font(p.add_run("Abstract—"), 9.5, bold=True)
set_font(
    p.add_run(
        "Moderation systems are increasingly asked to explain their decisions, yet explanations "
        "written by large language models cannot be guaranteed to describe the classifier that "
        "produced the label. We present a framework in which a multi-task BERT classifier is paired "
        "with a deterministic reasoning layer that assembles provenance-tracked explanations grounded "
        "in model outputs and input spans: Integrated-Gradients evidence spans with character offsets, "
        "a rule-based pronoun-to-antecedent analysis of conversational context, and fixed sentence "
        "templates. The layer abstains when evidence is insufficient, reports the provenance of every "
        "statement, and contains no generative component. We evaluate the framework on HateXplain and "
        "Counter Context, quote published dataset-paper baselines, and report mixed and negative "
        "results alongside positive ones. On HateXplain, multi-task training with target and rationale "
        "supervision reaches 69.2% ± 0.4 test accuracy and 0.679 ± 0.005 macro-F1 over three seeds — "
        "on par with the BERT results reported by the dataset authors (69.0% / 0.674) — and the "
        "seed-42 gain over a comment-only baseline is significant (McNemar p = 0.0016). A supervised "
        "rationale head agrees with human rationales "
        "at 0.823 macro token-F1 versus 0.650 for post-hoc Integrated Gradients on the same 100 "
        "examples; an ERASER-style deletion probe nevertheless shows that those Integrated-Gradients "
        "spans are plausible but not necessary for the prediction (comprehensiveness 0.015). On "
        "Counter Context, a matched-budget three-seed study shows that parent-comment concatenation "
        "improves test accuracy over a comment-only baseline (0.583 ± 0.007 vs. 0.565 ± 0.008) and "
        "that interaction features help further (0.603 ± 0.007). InfoNCE does not clearly separate from "
        "interaction at three seeds. A trained BERT cross-encoder ([CLS] parent [SEP] comment [SEP]) "
        "is the strongest and most stable classifier on this corpus (accuracy 0.610 ± 0.003, macro-F1 "
        "0.579 ± 0.003, weighted-F1 0.612, counter-speech F1 0.479), matching Yu et al.'s RoBERTa gold "
        "cross-encoder (0.61 weighted-F1). We therefore do not claim that the interaction module or "
        "InfoNCE improves Counter Context classification; their value is architectural (single-comment "
        "inference and a shared encoder that also serves HateXplain). The "
        "pronoun linker fires on only 16.6% of test pairs and produces wrong antecedents in manual "
        "inspection; we report it as a low-yield heuristic, not a solved component. HateCheck transfer "
        "is poor (12.8% hateful recall as hate). Batch-1 latency on an RTX 4060 laptop GPU is 0.7–1.1 s "
        "(n = 100), adequate for review-assist tooling and not for high-throughput enforcement. We "
        "position the contribution as an explanation framework and system, not as a state-of-the-art "
        "classifier."
    ),
    9.5,
)
p = para("", space_after=10)
rich(
    p,
    [
        ("Keywords—", {"size": 9.5, "italic": True, "bold": True}),
        (
            "hate speech detection; explainable AI; structured explanations; multi-task learning; "
            "Integrated Gradients; conversational context; faithfulness",
            {"size": 9.5, "italic": True},
        ),
    ],
)

# =============================================================== 1 introduction
h1(1, "Introduction")
para(
    "Automated hate-speech detection is mature enough to be deployed and controversial enough that "
    "its decisions are disputed. Both facts push the field toward systems that can answer, for a "
    "given decision, three questions: what was predicted, which parts of the input supported the "
    "prediction, and how any conversational context was used. Answers generated by large language "
    "models are fluent but cannot be guaranteed to describe the classifier's actual computation — "
    "the faithfulness problem identified by Jacovi and Goldberg [15] and extended to chain-of-thought "
    "text by Turpin et al. [23]. Zero-shot LLMs also still trail fine-tuned models on classification "
    "benchmarks including hate speech [22]. A moderation pipeline that needs both a reliable label "
    "and an auditable explanation therefore has reason to keep the classifier and the explanation "
    "layer separately inspectable."
)
para(
    "This paper presents such a pipeline and evaluates it as a system-and-explanation study, not as "
    "a state-of-the-art classifier. The classifier is a multi-task BERT model with switchable "
    "components: a context-interaction module for conversational input, an instance-discriminating "
    "InfoNCE objective with shuffled-context negatives, task heads for hate, target group, harm "
    "reason, and token-level evidence, and a post-hoc Integrated-Gradients extractor. The explanation "
    "layer — the primary contribution — is deliberately non-neural. It composes a structured "
    "explanation from the predicted label, the attribution-derived evidence spans (verbatim input "
    "substrings with character offsets), a deterministic pronoun-to-antecedent analysis between the "
    "current and preceding comment, and fixed templates. Every produced statement is checkable "
    "against a stored output; the layer abstains rather than invents; untrained heads are reported "
    "as unavailable rather than guessed."
)
para(
    "We trained the primary configurations, a matched-budget Counter Context ladder over three "
    "seeds, and a BERT cross-encoder over three seeds, evaluated all of them on held-out test splits, "
    "compared the results with the numbers "
    "reported by the dataset authors, trained the previously missing comment-only Counter Context "
    "baseline, compared the supervised evidence head against post-hoc attribution on identical "
    "samples, probed Integrated-Gradients faithfulness by token deletion, measured end-to-end "
    "latency, ran the HateCheck functional suite, and audited the reference linker on the full "
    "Counter Context test set. The resulting picture is mixed in ways we consider more useful than "
    "a single headline number:"
)
bullet(
    "On HateXplain, multi-task training with target and rationale supervision improves test accuracy "
    "from 66.1% (comment-only, seed 42) to 69.2% ± 0.4 over three seeds of hx_full, and macro-F1 "
    "from 0.654 to 0.679 ± 0.005, matching the original BERT baseline of Mathew et al. [7] (69.0% / "
    "0.674). The seed-42 gain over the baseline is significant (McNemar p = 0.0016; paired-bootstrap "
    "95% CI on accuracy [0.012, 0.050]; n = 1,924). A longer-budget control reaches 69.8% / 0.681."
)
bullet(
    "On Counter Context, a matched-budget ladder (four epochs, batch 12, seeds 42/1/2) shows that "
    "plain concatenation of comment and parent is better than no context (accuracy 0.583 ± 0.007 vs. "
    "0.565 ± 0.008; the ordering is consistent in all three seeds). Interaction features help "
    "accuracy further (0.603 ± 0.007, again 3/3 over concatenation). InfoNCE on top of interaction "
    "does not separate from interaction on accuracy or macro-F1 at three seeds. A trained BERT "
    "cross-encoder is the strongest and most stable classifier on this corpus (accuracy 0.610 ± "
    "0.003, macro-F1 0.579 ± 0.003, weighted-F1 0.612, counter-speech F1 0.479), matching Yu et "
    "al.'s RoBERTa gold cross-encoder (0.61 weighted-F1 [8]). We therefore do not claim that the "
    "interaction module or InfoNCE improves Counter Context classification. An earlier apparent harm "
    "from concatenation was an artifact of unmatched batch size and epoch budget."
)
bullet(
    "The supervised evidence head agrees with human rationales far better than Integrated Gradients "
    "(0.823 vs. 0.650 macro token-F1 on the same 100 test examples). This is plausibility, not "
    "faithfulness. A deletion probe (comprehensiveness 0.015 ± 0.224) shows that removing the "
    "highlighted Integrated-Gradients evidence barely moves the prediction."
)
bullet(
    "The pronoun-to-antecedent linker fires on 16.6% of Counter Context test pairs (118 / 713) and "
    "produces visibly wrong antecedents in manual inspection. There are no gold link annotations, so "
    "we report coverage and failure modes, not accuracy. HateCheck transfer of the HateXplain-trained "
    "hate head is poor (12.8% hateful recall as hate). End-to-end inference costs 0.67 s (comment "
    "only) to 1.06 s (with context) per request on a laptop GPU (n = 100) — review-assist scale, not "
    "interactive-UI or high-throughput scale."
)
para(
    "The reasoning layer is the novelty; the classifier components are assembled from standard parts "
    "(Section 4.7 states this explicitly). The rest of the paper is organized as follows: Section 2 "
    "relates the work to detection datasets, context-aware models, and LLM-based explanation; "
    "Section 3 defines the task and data; Section 4 specifies the method; Section 5 the experiments; "
    "Section 6 the results; Sections 7–9 the system, ethics, and limitations; Section 10 concludes."
)

# =============================================================== 2 related work
h1(2, "Related Work")
h2("2.1", "Hate speech detection and benchmark datasets")
para(
    "Statistical work established the hate / offensive / normal distinction and its difficulty [1], "
    "[2]; larger crowdsourced corpora confirmed the label-noise problem [3]; OffensEval formalized "
    "type and target sub-tasks [4]. Surveys document dataset-quality concerns that persist today "
    "[5], [6]. HateXplain [7] annotates 19,229 posts with majority-vote class labels, target groups, "
    "and token-level rationales, and reports BERT at 69.0% accuracy / 0.674 macro-F1, rising to "
    "69.8% / 0.687 when rationales supervise an attention head; their attention- and LIME-based "
    "token selection sits at 0.37–0.51 token-F1. HateCheck [21] provides functional tests; we run "
    "them in Section 6.7 and report a negative transfer result. Counter Context [8] pairs Reddit "
    "comments with their parents, labelled hate / counter-speech / neutral given the context, and "
    "reports RoBERTa weighted-F1 of 0.58 without context and 0.61 with a parent-and-target "
    "cross-encoder (0.64 with silver data and stance pre-training)."
)
h2("2.2", "Context-aware abuse detection")
para(
    "Park and Fung [9] found parent-tweet context mainly helps the hateful class. Yu, Blanco, and "
    "Hong [8] showed that human judgments of a comment change for most examples when the preceding "
    "comment is shown, and that a RoBERTa cross-encoder over parent [SEP] target improves "
    "weighted-F1 from 0.58 to 0.61 on gold data. Our architecture follows the same motivation but "
    "encodes the two comments separately with a shared encoder and combines their sentence embeddings "
    "with explicit interaction features, which keeps single-comment inference available and enables "
    "the contrastive objective of Section 4.3. We compare against their reported numbers and, in "
    "this revision, against our own trained BERT cross-encoder of the same input form."
)
h2("2.3", "Explainability, faithfulness, and LLM-generated explanations")
para(
    "Post-hoc attribution methods such as LIME [10] and Integrated Gradients [11] provide "
    "input-grounded evidence without retraining; Integrated Gradients satisfies the axioms of "
    "sensitivity and implementation invariance. Attention weights are unreliable as explanations "
    "[12]. ERASER [13] established token-level-rationale evaluation, distinguishing plausibility "
    "(agreement with human rationales) from faithfulness (the explanation's necessity for the "
    "prediction). Jacovi and Goldberg [15] formalized the latter and warned that plausible text can "
    "be unfaithful. LLMs are now the default explanation engine in deployed systems. Ziems et al. "
    "[22] show they fail to outperform fine-tuned models on many classification tasks and are best "
    "used with humans in the loop. Turpin et al. [23] show that chain-of-thought explanations can "
    "be unfaithful to the model's actual computation even when they look plausible. Our design takes "
    "the conservative position: a deterministic template system that emits provenance-tracked "
    "explanations grounded in model outputs and input spans, while stating the limits of post-hoc "
    "attribution and heuristic context resolution. Determinism guarantees fixed rules and stored "
    "outputs; it does not guarantee that Integrated Gradients reflects the model's reasoning, or "
    "that the pronoun linker finds the right antecedent."
)
h2("2.4", "What this work adds")
para(
    "The classifier components are standard: BERT [16], matching-based interaction features [17], "
    "[18], InfoNCE contrastive learning [19], multi-task heads, and Integrated Gradients [11]. The "
    "contribution is the reasoning layer that turns these parts into a structured, provenance-tracked, "
    "abstaining explanation with no generative component; the empirical comparison of supervised "
    "versus post-hoc evidence inside one system, including a deletion-based faithfulness probe; a "
    "matched-budget Counter Context study that isolates concatenation, interaction, and InfoNCE as "
    "far as the available seeds allow; and an evaluation that includes published baselines, HateCheck, "
    "a reference-linker audit, latency, and the negative results. To our knowledge, this combination "
    "of provenance, abstention, and determinism at the explanation layer has not been enforced in a "
    "hate-speech system; we cannot exclude prior art we have not seen."
)

# =============================================================== 3 task and datasets
h1(3, "Task Definition and Datasets")
h2("3.1", "Unified example format")
para(
    "Both corpora are converted into a shared schema (current_text, context_text, hate_label, "
    "target_label, reason_label, rationale_labels, metadata). Annotations a dataset does not provide "
    "are explicitly null — never imputed — so every training loss component activates only when "
    "ground truth exists for the batch at hand. The two raw datasets are never merged."
)
h2("3.2", "HateXplain")
para(
    "HateXplain [7] annotates Gab and Twitter posts; using the official post-id divisions we process "
    "19,229 posts, split 15,383 / 1,922 / 1,924 for train / validation / test. Labels are aggregated "
    "by majority vote over three annotators into hate, offensive, or normal. Target annotations are "
    "mapped to nine canonical categories (race, religion, nationality, gender, sexual orientation, "
    "ethnicity, political group, other, none); unmapped raw values are preserved in metadata rather "
    "than guessed. Token-level rationales from the majority-label annotators are available for 9,132 "
    "training posts (59.4%). 9,734 training posts target a specific group (as opposed to none). The "
    "dataset authors report BERT at 69.0% test accuracy and 0.674 macro-F1 [7]. There is no "
    "conversational context in this corpus."
)
h2("3.3", "Counter Context")
para(
    "The Counter Context corpus [8] pairs each Reddit comment with its preceding comment, labelled "
    "as hate speech, counter speech, or neither with respect to that context. We use the gold splits "
    "(3,325 / 713 / 713 pairs) and keep the silver splits (1,675 / 419) available as extra training "
    "data but unused in the reported runs. The corpus encodes labels as string indices; our adapter "
    "maps 0 → hate_speech, 1 → neither, 2 → counter_speech, verified against the published class "
    "shares (approximately 28% / 49% / 23% of 6,846 pairs [8]); our processed gold splits reproduce "
    "those shares (Table 1). Average comment length is 10.2 words on the training split, versus 23.5 "
    "for HateXplain (2.3× shorter). Yu et al. describe a random 70/15/15 split of the 4,751 gold "
    "pairs; the repository releases pre-assigned gold train/val/test files, which we use as-is "
    "(test n = 713, matching the paper's 15%). We have not verified byte-identity with the paper's "
    "random split."
)
table(
    "Table 1: Dataset statistics (processed splits used in all experiments). Class shares are listed "
    "in the order of the label space in the row above. Shares are computed on the training split "
    "(HateXplain test support is 594 / 548 / 782 for hate / offensive / normal).",
    ["Property", "HateXplain", "Counter Context (gold)"],
    [
        ["Splits (train / val / test)", "15,383 / 1,922 / 1,924", "3,325 / 713 / 713"],
        ["Label space", "hate, offensive, normal", "hate_speech, counter_speech, neither"],
        ["Train class shares (%)", "30.9 / 28.5 / 40.6", "27.7 / 23.3 / 48.9"],
        ["Target-group labels", "9 categories; 9,734 train posts target a specific group", "—"],
        ["Token-level rationales", "9,132 / 15,383 train posts (59.4%)", "—"],
        ["Conversational context", "— (single posts)", "100% (preceding comment)"],
        ["Avg. words per comment (train)", "23.5", "10.2"],
    ],
    widths=[2.15, 2.35, 2.0],
)
figure(
    ASSETS / "fig2_datasets.png",
    "Figure 1: Label distributions from the processed splits. Left: HateXplain hate classes over all "
    "splits. Middle: HateXplain target-group counts (train). Right: Counter Context gold splits.",
)

# =============================================================== 4 method
h1(4, "Proposed Method")
h2("4.1", "Architecture overview")
para(
    "Figure 2 shows the architecture. Every component is individually switchable through "
    "configuration files. The same code path serves comment-only inference: when no preceding "
    "comment is present, the context embedding falls back to the zero vector and the API reports "
    "context_used = false."
)
figure(
    ASSETS / "fig1_architecture.png",
    "Figure 2: Model architecture. Both comments are encoded by a weight-shared BERT; the "
    "interaction module combines the two sentence embeddings with engineered difference and product "
    "features; task-specific heads and (training-only) the contrastive projection branch from the "
    "fused representation; Integrated Gradients turns the trained model into evidence spans.",
)
h2("4.2", "Shared encoding and context interaction")
para(
    "Let C denote the current comment and P the preceding comment. A single BERT-base encoder [16] "
    "produces both sentence embeddings and token states:"
)
equation("E_c = BERT(C)[CLS],   E_p = BERT(P)[CLS],   token states H_1..H_n")
para(
    "Weight sharing places both embeddings in one semantic space, which is what makes the following "
    "interaction features meaningful. The features [u; v; |u−v|; u⊙v] originate in InferSent [17] "
    "and are used by Sentence-BERT [18]; we apply them to BERT [CLS] embeddings:"
)
equation("D = |E_c − E_p|,   M = E_c ⊙ E_p,   z = [ E_c ; E_p ; D ; M ] ∈ ℝ^3072")
para(
    "The fused vector z is compressed to the task representation by a single linear projection "
    "(hidden 768 × 4 = 3,072 → 1,024), followed by GELU and dropout:"
)
equation("h = Dropout( GELU( W z + b ) ),   W ∈ ℝ^(1024×3072)")
para(
    "When no context exists, E_p = 0 makes D = |E_c| and M = 0 — a deterministic, reported fallback "
    "rather than an error. An ablation mode (“simple”) concatenates [E_c; E_p] without D and M, "
    "isolating the engineered features."
)
h2("4.3", "Contrastive context learning")
para(
    "During training on conversational data, each batch additionally contains mismatched contexts "
    "drawn by shuffling within the batch. We use a temperature-scaled InfoNCE objective [19] — an "
    "instance-discriminating loss with shuffled-context negatives, not supervised contrastive "
    "learning, because class labels are not used. The anchor r_a is the representation of the "
    "current comment without its true context; the positive r_p is the representation of the same "
    "comment with its true preceding context; the negatives r_n are representations of the same "
    "comment with mismatched contexts:"
)
equation("L_InfoNCE = − log [ exp(sim(r_a, r_p)/τ) / Σ_n exp(sim(r_a, r_n)/τ) ],  τ = 0.07")
para(
    "Rows without a real context carry no contrastive signal and are excluded. The projection head "
    "is used strictly during training; inference always uses the fused representation h. On "
    "HateXplain, which has no conversation pairs, this objective is inactive — a fact we return to "
    "in Section 5.1."
)
h2("4.4", "Multi-task heads and training objective")
para(
    "Branching from h are: (i) a hate head (3-class softmax; a binary mode exists but is never "
    "silently substituted), (ii) a target head over the nine HateXplain target categories, (iii) a "
    "reason head over eight harm categories (insult, dehumanization, negative stereotyping, threat, "
    "exclusion, discrimination, incitement to violence, other) that exists in the architecture but "
    "is trained only when real reason annotations are supplied — never on synthetic labels, and "
    "(iv) a token-level evidence head (sigmoid over token states) trained on HateXplain rationales. "
    "The total loss is a weighted sum in which any component whose batch labels are missing is "
    "skipped entirely:"
)
equation("L = λ_h L_hate + λ_t L_target + λ_r L_reason + λ_e L_evidence + λ_c L_InfoNCE")
para(
    "with λ = (1.0, 1.0, 1.0, 1.0, 0.5) in all reported runs. No head is ever trained on fabricated "
    "supervision, and the API reports *_available flags that the front end renders literally. The "
    "reason head is untrained in this work."
)
h2("4.5", "Two evidence mechanisms: attribution and a trained head")
para("The system contains two distinct evidence mechanisms, which we evaluate against each other.")
para(
    "Post-hoc attribution. Integrated Gradients [11] attributes the logit of the explained class to "
    "input embeddings. With x the actual embedding, x′ the PAD-embedding baseline, and F the "
    "explained logit, token i receives:"
)
equation("IG_i(x) = (x_i − x′_i) × (1/m) Σ_{k=1}^{m} ∂F(x′ + (k/m)(x − x′)) / ∂x_i")
para(
    "We use m = 32 steps (a gradient×input variant is implemented for comparison). Scores of special "
    "tokens are zeroed, tokens are thresholded relative to the strongest token (min score 0.15), "
    "adjacent tokens are merged, subwords are joined, and the top-k (k = 5) spans are returned as "
    "original-comment substrings with character offsets (the span decoder restores original casing "
    "via the offsets; the uncased BERT vocabulary would otherwise lowercase them). The k and "
    "threshold are configuration defaults, not tuned. The explained class defaults to the prediction "
    "but can be forced to any class."
)
para(
    "Supervised evidence head. A linear layer over the encoder's token states, trained with sigmoid "
    "BCE against HateXplain rationales, outputs a rationale probability per token. At inference we "
    "threshold at 0.5. This is rationale prediction, not explanation of a specific prediction; we "
    "compare the two mechanisms in Section 6.4 and do not treat either as causal proof [12], [15]."
)
h2("4.6", "The deterministic reasoning layer")
para(
    "The reasoning layer is the central component (Figure 3). For each prediction it answers four "
    "questions: WHAT (the trained head's label and confidence), WHICH WORDS (the evidence spans, "
    "from either mechanism), WHO (a deterministic pronoun-to-antecedent analysis that links referring "
    "words in the current comment — they/them/their, he/she and inflections — to the right-most "
    "matching noun phrase in the preceding comment, returning character offsets on both sides), and "
    "WHY (fixed sentence templates filled exclusively with the values above)."
)
para(
    "Five rules are enforced in code. First, no generative model writes any part of the explanation; "
    "the method field of every response reads “deterministic evidence + context-reference analysis”. "
    "Second, attention weights are never presented as reasoning. Third, abstention over invention: "
    "if no reference relationship is found the summary says so; non-hateful results are described "
    "without inventing hateful evidence. Fourth, strict separation of prediction, evidence, and "
    "explanation fields, each evidence item declaring source (current vs. previous comment), type "
    "(attribution span vs. context target), and a grounded rationale. Fifth, graceful degradation: a "
    "reasoning failure returns “Reasoning unavailable” without failing the request. Untrained heads "
    "are reported as unavailable rather than guessed."
)
para(
    "Three honesty caveats are part of the design, not oversights. The templates may state that the "
    "contextual reference “matters” whenever a link is found — but as Section 6.6 shows, a link in "
    "the explanation does not imply that context changed the score; a future revision should replace "
    "that phrase with one that claims only interpretive dependence. The reference analysis is a "
    "heuristic whose precision we have not quantified against gold antecedents (none exist in either "
    "corpus); Section 6.7 reports its coverage and failure modes; it is labelled as a heuristic in "
    "every response. And an earlier template claimed that the comment expressed hostility “toward "
    "that target” — a claim no model output supports — and emitted lowercased spans from the uncased "
    "tokenizer; both were removed (the summary now says “hostile or exclusionary language” without a "
    "target claim, and spans restore original casing via character offsets)."
)
figure(
    ASSETS / "fig6_reasoning_pipeline.png",
    "Figure 3: The reasoning pipeline. Stages 2–4 analyse the actual inputs and outputs of the "
    "trained model; nothing in the final explanation is written by a generative model. The worked "
    "example at the bottom is a real system output reproduced in Section 6.6.",
)
h2("4.7", "What is and is not novel")
para(
    "Individually, the parts are standard: BERT fine-tuning [16], InferSent-style matching features "
    "[17], [18], InfoNCE [19], multi-task heads, and Integrated Gradients [11]. The reasoning layer "
    "of Section 4.6 is the contribution: a deterministic, provenance-tracked, abstaining explanation "
    "assembler for hate-speech decisions, with an explicit no-LLM rule, availability flags, and a "
    "worked-example protocol that reports when context does not change the score. The empirical "
    "comparison of a trained rationale head against post-hoc attribution inside the same system, "
    "the deletion probe, the matched-budget context ladder including a trained BERT cross-encoder, "
    "and the evaluation against published dataset-paper baselines are the supporting studies."
)

# =============================================================== 5 experimental setup
h1(5, "Experimental Setup")
h2("5.1", "Model variants — a configuration suite plus a matched-budget ladder")
table(
    "Table 2: Configurations. InfoNCE is structurally active in “full” but inert on HateXplain (no "
    "context pairs). The original suite (top) is not a controlled ablation: rungs differ in corpus, "
    "batch size, and epoch budget, and “cc_context” adds interaction features and InfoNCE together. "
    "The matched-budget Counter Context ladder (bottom) fixes epochs = 4 and batch = 12.",
    ["Variant", "Corpus", "Context", "Interaction", "InfoNCE", "Target/Evid.", "Epochs / batch"],
    [
        ["baseline", "HateXplain", "—", "—", "—", "—", "5 / 16"],
        ["full", "HateXplain", "zero-fallback", "[E_c;E_p;D;M]", "inert", "yes / yes", "5 / 16"],
        ["hx_full", "HateXplain", "zero-fallback", "[E_c;E_p;D;M]", "off", "yes / yes", "2 / 16"],
        ["hx_full_long", "HateXplain", "zero-fallback", "[E_c;E_p;D;M]", "off", "yes / yes", "6 budget / 16"],
        ["cc_baseline", "Counter Context", "—", "—", "—", "—", "5 / 16"],
        ["context", "Counter Context", "yes", "concat only", "off", "—", "default / 16"],
        ["cc_context", "Counter Context", "yes", "[E_c;E_p;D;M]", "on", "—", "4 / 12"],
        ["ladder: cc_base", "Counter Context", "—", "—", "—", "—", "4 / 12"],
        ["ladder: cc_concat", "Counter Context", "yes", "concat only", "off", "—", "4 / 12"],
        ["ladder: cc_inter", "Counter Context", "yes", "[E_c;E_p;D;M]", "off", "—", "4 / 12"],
        ["ladder: cc_nce", "Counter Context", "yes", "[E_c;E_p;D;M]", "on", "—", "4 / 12"],
        ["ladder: cross-encoder", "Counter Context", "[CLS] P [SEP] C", "—", "—", "—", "4 / 12"],
    ],
    widths=[1.15, 1.05, 0.85, 1.05, 0.55, 0.7, 0.95],
    font_size=7.5,
)
para(
    "The original suite is not a controlled ablation. “full” and “hx_full” differ in epoch budget "
    "(5 vs. 2) and the inert contrastive flag; their test results are broadly similar (accuracy 0.685 "
    "vs. 0.692) but not identical, and we do not treat the residual differences as meaningful at one "
    "seed. The earlier “context” configuration (concatenation at the default batch and epoch budget) "
    "scored 0.558 test accuracy, worse than the original cc_baseline (0.588); the matched-budget "
    "ladder in Section 6.2 reverses that ordering and shows the reversal was a budget artifact. "
    "Cross-encoder baselines of the form [CLS] C [SEP] P — the design of Yu et al. [8] — were "
    "trained in this revision (bert-base-uncased, same 4-epoch / batch-12 budget, seeds 42/1/2) and "
    "are reported in Section 6.2. HateXplain hx_full is now three seeds (42/1/2). The original "
    "suite rows other than hx_full remain seed 42."
)
h2("5.2", "Training and inference hyperparameters")
table(
    "Table 3: Hyperparameters shared across runs (base configuration).",
    ["Parameter", "Value", "Parameter", "Value"],
    [
        ["Encoder", "bert-base-uncased (110M)", "Max sequence length", "128 tokens"],
        ["Encoder freezing", "none (full fine-tune)", "Batch size", "16 (12 for cc_context / ladder)"],
        ["Learning rate", "2e-5 (AdamW)", "Epochs", "see Table 2"],
        ["Weight decay", "0.01", "Warmup", "10% of steps, linear decay"],
        ["Gradient clipping", "1.0", "Mixed precision", "auto (CUDA only)"],
        ["Early stopping", "patience 2 (val macro-F1)", "Seed", "42 default; ladder 42/1/2"],
        ["Interaction dim", "1024 (dropout 0.2)", "Projection dim / τ", "256 / 0.07"],
        ["IG steps", "32", "Evidence top-k / min score", "5 / 0.15"],
    ],
    widths=[1.55, 1.7, 1.55, 1.7],
    font_size=8,
)
para(
    "Training uses PyTorch and Hugging Face Transformers [20]. All reported test numbers come from "
    "held-out test splits evaluated once with the best-validation checkpoint. Checkpoints store "
    "optimizer, scheduler, configuration, label maps, and metrics."
)
h2("5.3", "Metrics")
para(
    "Classification: accuracy, macro and weighted precision/recall/F1, per-class F1, confusion "
    "matrices. Multi-seed cells report the mean and sample standard deviation (n−1). Evidence: "
    "token-level precision/recall/F1 against gold HateXplain rationales on the first 100 "
    "rationale-bearing test examples in dataset order (not a random sample), aggregated as the mean "
    "of per-example F1 (macro), which is why F1 can sit below both precision and recall — examples "
    "with zero overlap contribute 0. We also report the micro-averaged token F1 for the supervised "
    "head. Faithfulness: ERASER-style comprehensiveness and sufficiency on the same 100 examples, "
    "using PAD substitution. Latency: mean / p50 / p95 over 100 sequential requests after warm-up, "
    "batch size 1, on the serving checkpoint. Reference linker: coverage (fraction of pairs with at "
    "least one emitted link) and qualitative failure modes; no gold link annotations exist."
)

# =============================================================== 6 results
h1(6, "Results and Discussion")
h2("6.1", "HateXplain: comparison with published results")
table(
    "Table 4: HateXplain test results (n = 1,924). Published rows are quoted from Mathew et al. [7], "
    "Table 6; they do not report per-class F1 for these rows. hx_full is reported both as seed 42 "
    "and as the mean ± sample std over seeds 42/1/2.",
    ["Model", "Acc.", "Macro-F1", "wF1", "hate F1", "off. F1", "norm. F1"],
    [
        ["BERT (Mathew et al. 2021) [7]", "0.690", "0.674", "—", "—", "—", "—"],
        ["BERT-HateXplain (rationale-supervised) [7]", "0.698", "0.687", "—", "—", "—", "—"],
        ["baseline (ours, comment-only)", "0.661", "0.654", "0.662", "0.746", "0.514", "0.701"],
        ["full (ours, multi-task, 5 ep.)", "0.685", "0.677", "0.685", "0.769", "0.536", "0.727"],
        ["hx_full (ours, seed 42)", "0.692", "0.677", "0.688", "0.766", "0.514", "0.751"],
        ["hx_full (ours, 3-seed mean)", "0.692 ± 0.004", "0.679 ± 0.005", "0.689 ± 0.005",
         "0.768 ± 0.002", "0.524 ± 0.010", "0.745 ± 0.009"],
        ["hx_full_long (ours, 6-ep. budget)", "0.698", "0.681", "0.692", "0.771", "0.521", "0.752"],
    ],
    widths=[2.4, 0.65, 0.8, 0.6, 0.7, 0.65, 0.7],
    font_size=8,
)
para(
    "Multi-task training with target and rationale supervision improves test accuracy by 3.1 points "
    "and macro-F1 by 0.023 over the comment-only baseline on seed 42 (McNemar p = 0.0016; "
    "paired-bootstrap 95% CI on the accuracy difference [0.012, 0.050]; n = 1,924). Over three "
    "seeds, hx_full reaches 69.2% ± 0.4 accuracy and 0.679 ± 0.005 macro-F1, on par with the "
    "dataset authors' BERT (69.0% / 0.674). The 6-epoch budget of hx_full_long adds only +0.6 "
    "accuracy and +0.004 macro-F1 over the 2-epoch hx_full; two epochs essentially exhaust this "
    "configuration. "
    "Target-group prediction (hx_full): 69.2% accuracy — coincidentally identical to hate accuracy "
    "because both heads misclassify exactly 592 of 1,924 posts (verified in the saved confusion "
    "matrices) — and 0.553 macro-F1 over all nine categories including political_group with zero "
    "test support, or 0.622 macro-F1 over the eight classes with test support. Per-class: religion "
    "0.784, race 0.736, nationality 0.719, sexual orientation 0.710, none 0.720, ethnicity 0.660, "
    "gender 0.485, other 0.160."
)
figure(
    ASSETS / "fig3_curves.png",
    "Figure 4: Validation macro-F1 across epochs (early stopping selects the maximum). HateXplain "
    "runs saturate in 2–5 epochs and then overfit; Counter Context runs improve slowly and remain "
    "noisier. The matched-budget Counter Context comment-only runs are included on the right.",
)

h2("6.2", "Counter Context: published baselines, matched ladder, and the context question")
table(
    "Table 5: Counter Context test results (n = 713). Published rows are quoted from Yu et al. [8], "
    "Table 6 (RoBERTa-base; their 70/15/15 split of the same 4,751 gold pairs). Their best row uses "
    "silver data and stance pre-training. Our published-comparison rows are the original suite "
    "checkpoints (seed 42) plus our BERT cross-encoder (seed 42). Three-seed means are in Table 6.",
    ["Model", "Acc.", "Macro-F1", "wF1", "hate F1", "counter F1", "neither F1"],
    [
        ["Majority baseline (Yu et al.) [8]", "—", "—", "0.34", "—", "—", "—"],
        ["RoBERTa, target only (Yu et al.) [8]", "—", "—", "0.58", "0.56", "0.38", "0.69"],
        ["RoBERTa, parent [SEP] target (Yu et al.) [8]", "—", "—", "0.61", "0.59", "0.44", "0.70"],
        ["RoBERTa best (Yu et al., +silver+stance) [8]", "—", "—", "0.64", "0.60", "0.53", "0.71"],
        ["cc_baseline (ours, comment-only, 5/16)", "0.588", "0.557", "0.587", "0.511", "0.484", "0.674"],
        ["context (ours, concat, default budget)", "0.558", "0.534", "0.562", "0.494", "0.463", "0.644"],
        ["cc_context (ours, interaction+InfoNCE, seed 42)", "0.617", "0.563", "0.607", "0.545", "0.416", "0.728"],
        ["BERT cross-encoder (ours, [CLS] P [SEP] C, seed 42)", "0.609", "0.581", "0.612", "0.539", "0.502", "0.699"],
    ],
    widths=[2.55, 0.6, 0.75, 0.6, 0.7, 0.8, 0.75],
    font_size=8,
)
table(
    "Table 6: Matched-budget Counter Context results (epochs = 4, batch = 12). Mean ± sample std "
    "over three seeds (42 / 1 / 2). Same test split as Table 5. Per-class F1 is the mean over seeds. "
    "The cross-encoder is [CLS] previous [SEP] current [SEP].",
    ["Variant", "n", "Acc.", "Macro-F1", "wF1", "hate F1", "counter F1", "neither F1"],
    [
        ["cc_baseline (no context)", "3", "0.565 ± 0.008", "0.533 ± 0.005", "0.566 ± 0.007",
         "0.521 ± 0.020", "0.422 ± 0.030", "0.656 ± 0.016"],
        ["cc_concat (parent, no interaction)", "3", "0.583 ± 0.007", "0.550 ± 0.008", "0.584 ± 0.007",
         "0.535 ± 0.005", "0.439 ± 0.017", "0.676 ± 0.005"],
        ["cc_inter (interaction, no InfoNCE)", "3", "0.603 ± 0.007", "0.554 ± 0.023", "0.594 ± 0.015",
         "0.537 ± 0.007", "0.421 ± 0.075", "0.703 ± 0.012"],
        ["cc_nce (interaction + InfoNCE)", "3", "0.611 ± 0.015", "0.564 ± 0.014", "0.605 ± 0.015",
         "0.543 ± 0.007", "0.431 ± 0.023", "0.717 ± 0.018"],
        ["BERT cross-encoder [CLS] P [SEP] C", "3", "0.610 ± 0.003", "0.579 ± 0.003", "0.612",
         "0.559 ± 0.017", "0.479 ± 0.026", "0.701 ± 0.007"],
    ],
    widths=[2.0, 0.35, 0.85, 0.85, 0.8, 0.7, 0.8, 0.75],
    font_size=7.5,
)
para(
    "Five observations. First, under a matched budget, conversational context helps: every "
    "concatenation seed beats every comment-only seed on accuracy and weighted-F1 (+1.8 accuracy "
    "points on the means). The earlier Table 5 row in which concatenation looked harmful (0.558 vs. "
    "0.588) compared different batch sizes and epoch counts and should not be read as evidence "
    "against context. Second, interaction features accompany a further accuracy gain (0.603 ± 0.007 "
    "vs. 0.583 ± 0.007; all three interaction seeds exceed all three concatenation seeds), but "
    "macro-F1 is noisy because one interaction seed collapses on counter-speech (F1 std 0.075). "
    "Third, InfoNCE on top of interaction does not separate from interaction on accuracy (0.611 ± "
    "0.015 vs. 0.603 ± 0.007) or macro-F1 (0.564 ± 0.014 vs. 0.554 ± 0.023) at three seeds; we "
    "therefore do not claim that InfoNCE improves Counter Context classification. Fourth, the "
    "trained BERT cross-encoder is the strongest and most stable classifier on this corpus: its "
    "accuracy matches cc_nce (0.610 ± 0.003 vs. 0.611 ± 0.015) with three times smaller variance, "
    "its macro-F1 is the best in the table (0.579 ± 0.003), and its counter-speech F1 (0.479) is "
    "the highest. We therefore do not claim that the dual-encoder interaction architecture improves "
    "Counter Context classification over a standard cross-encoder. Its value is architectural: "
    "single-comment inference remains available, the same encoder serves HateXplain, and the "
    "explanation layer can expose the context relationship. Fifth, our cross-encoder weighted-F1 "
    "(0.612) matches Yu et al.'s RoBERTa gold cross-encoder (0.61) and sits below their best "
    "silver+stance result (0.64)."
)
para(
    "On validation, the matched ladder is 0.546 ± 0.014 (baseline), 0.555 ± 0.010 (concat), "
    "0.567 ± 0.017 (interaction), 0.556 ± 0.010 (InfoNCE), and 0.556 ± 0.006 (cross-encoder). The "
    "test-split accuracy gain of the original cc_context over the original cc_baseline (+2.9 points) "
    "is not significant (McNemar p = 0.099; bootstrap 95% CI on the accuracy difference [−0.004, "
    "0.062]; n = 713). We claim that context helps modestly and consistently in the matched ladder, "
    "that a standard cross-encoder is the stronger classifier on this corpus, and that our "
    "explanation layer makes the context relationship inspectable regardless of score changes "
    "(Section 6.6)."
)

h2("6.3", "Per-class analysis and confusion structure")
figure(
    ASSETS / "fig5_perclass.png",
    "Figure 5: Test-set per-class F1 for hx_full. Left: hate head. Right: target head "
    "(political_group has zero test support).",
)
para(
    "The hate class is detected strongly (F1 0.766, recall 0.806 on 594 test posts) and normal is "
    "stable (F1 0.751). The offensive class is the weak spot (F1 0.514, recall 0.482): 120 offensive "
    "posts are classified as hate and 164 as normal — 284 of 548 misclassified. This mirrors the "
    "documented hate/offensive boundary problem [1] and is aggravated by HateXplain's label policy, "
    "under which insult-bearing posts appear across all three classes. The 6-epoch control run moves "
    "offensive F1 only from 0.514 to 0.521 and macro-F1 from 0.677 to 0.681: under-training is not "
    "the cause at this model scale. The target head shows a data-volume gradient (Figure 5, right): "
    "religion 0.784, race 0.736 and nationality 0.719 score high; the rare other category (85 test "
    "posts) collapses to 0.160; gender is diluted by heavy none-confusion (recall 0.393)."
)
figure(
    ASSETS / "fig4_confusion.png",
    "Figure 6: Held-out test confusion matrices (row-normalized percentages). Left: hx_full on "
    "HateXplain (n = 1,924). Right: cc_context on Counter Context (n = 713).",
)
para(
    "On HateXplain the dominant errors are offensive↔normal leakage. On Counter Context (cc_context "
    "seed 42), hate_speech is the strongest class (F1 0.545) and neither the easiest (F1 0.728, "
    "recall 0.784); counter_speech is weakest (F1 0.416, recall 0.375): 79 of 168 counter-speech "
    "test pairs are classified as neither and 26 as hate. The asymmetry is interpretable — counter "
    "speech is defined by intent relative to the preceding comment — and is the class the "
    "contrastive objective is meant to protect; that it does not, in this configuration, is a "
    "negative result we leave standing. The matched ladder confirms it: interaction without InfoNCE "
    "scores 0.421 ± 0.075 counter-speech F1, interaction with InfoNCE 0.431 ± 0.023, while the "
    "cross-encoder reaches 0.479 ± 0.026 — the best in the study."
)

h2("6.4", "Evidence quality: supervised head versus Integrated Gradients")
table(
    "Table 7: Token-level agreement with gold HateXplain test rationales (the first 100 "
    "rationale-bearing examples in dataset order — not a random sample; macro = mean of per-example "
    "F1, which is why F1 can sit below both P and R — examples with zero overlap contribute 0). IG "
    "rows use 32-step Integrated Gradients; the head row thresholds the trained evidence head at "
    "0.5. All rows are this work.",
    ["Mechanism", "Checkpoint", "Macro P", "Macro R", "Macro F1", "Micro F1"],
    [
        ["Integrated Gradients", "baseline", "0.523", "0.706", "0.554", "—"],
        ["Integrated Gradients", "hx_full", "0.707", "0.706", "0.650", "—"],
        ["Integrated Gradients", "hx_full_long", "0.604", "0.744", "0.614", "—"],
        ["Supervised evidence head (τ=0.5)", "hx_full", "0.851", "0.867", "0.823", "0.779"],
    ],
    widths=[2.15, 1.15, 0.75, 0.75, 0.8, 0.75],
    font_size=8,
)
para(
    "The trained evidence head agrees with human rationales substantially better than post-hoc "
    "Integrated Gradients on the same examples (0.823 vs. 0.650 macro token-F1). Two caveats keep "
    "this from being a faithfulness claim. First, the head is trained on the rationale labels: high "
    "agreement is expected, and it measures plausibility [13]. Second, IG explains a specific "
    "predicted logit whereas the head predicts a dataset-level rationale mask; they answer different "
    "questions. For reference, Mathew et al. report 0.37–0.51 token-F1 for attention- and LIME-based "
    "selection on BERT [7] — not directly comparable (different selection methods and aggregation), "
    "but consistent with post-hoc methods sitting well below a trained head. Multi-task training "
    "also improves IG itself: the same extractor scores 0.554 on the baseline model and 0.650 on "
    "hx_full. The longer-trained control trades precision for recall (0.604 / 0.744), motivating the "
    "configurable evidence_min_score."
)
para(
    "Faithfulness probe (ERASER-style). On the same 100 examples we delete the IG evidence tokens "
    "(replace with PAD) and keep only those tokens, and measure the change in the predicted-class "
    "probability. Comprehensiveness (p_orig − p_without_evidence) is 0.015 ± 0.224 — essentially "
    "zero: removing the highlighted evidence barely moves the prediction. Sufficiency (p_orig − "
    "p_with_only_evidence) is 0.101 ± 0.231. IG spans are therefore plausible but not necessary for "
    "the decision in this setup; PAD-substitution is a crude deletion, and we report the numbers as "
    "a lower-bound probe rather than a definitive faithfulness audit. This is the gap the reasoning "
    "layer's honesty rules exist to expose, not to hide."
)

h2("6.5", "Latency")
table(
    "Table 8: End-to-end latency of the serving pipeline (cc_context checkpoint, batch 1, 100 "
    "sequential requests after warm-up, NVIDIA GeForce RTX 4060 Laptop GPU). “Predict” includes "
    "tokenization, encoding, heads, 32-step Integrated Gradients, and the reasoning layer.",
    ["Request type", "mean (ms)", "p50 (ms)", "p95 (ms)"],
    [
        ["predict, no context", "670", "664", "712"],
        ["predict, with context", "1,059", "1,039", "1,141"],
        ["explain (IG-32, with context)", "961", "1,048", "1,305"],
    ],
    widths=[2.6, 1.2, 1.2, 1.2],
)
para(
    "These are review-assist latencies, not interactive-UI latencies and not high-throughput "
    "enforcement latencies. Context adds a second encoder pass plus interaction (roughly +0.4 s); "
    "Integrated Gradients dominates the comment-only path (32 forward-plus-backward passes). A "
    "smaller 15-request pilot on the same checkpoint measured 1.7–3.1 s, illustrating laptop-GPU "
    "clock and thermal variance; we treat the n = 100 run as the reference. Serving the evidence "
    "head instead of IG, or reducing IG steps, are the obvious engineering levers; we have not "
    "optimized along either axis."
)

h2("6.6", "Qualitative analysis: a live worked example")
table(
    "Table 9: Live system output for previous = “I saw a group of immigrants protesting downtown.”, "
    "current = “They should all be kicked out.” (cc_context serving checkpoint, IG, 32 steps).",
    ["Field", "Value"],
    [
        ["Prediction", "hate_speech  (p = 0.757;  neither 0.185,  counter_speech 0.057)"],
        ["Evidence spans (IG)", "“kicked” (score 1.00, chars 19–25);  “They should” (0.66, chars 0–11)"],
        ["Token attributions",
         "kicked 1.17;  they 0.84;  should 0.69;  all −0.49;  be −0.19;  . −0.38;  out −0.03"],
        ["Context link", "“They” (0–4) → refers_to → “group of immigrants” (8–27)"],
        ["Template-assembled summary",
         "“The current comment uses ‘They’ to refer to ‘group of immigrants’ from the previous "
         "comment and expresses hostile or exclusionary language. The contextual reference matters: "
         "on its own, ‘They’ does not identify who or what is being referred to. Strongest evidence: "
         "the phrase ‘kicked’ and ‘They should’.”"],
        ["Method field", "deterministic evidence + context-reference analysis"],
    ],
    widths=[1.5, 5.0],
    font_size=8,
)
figure(
    ASSETS / "fig7_attribution.png",
    "Figure 7: Integrated-Gradients token attributions for the worked example (real output of the "
    "trained serving checkpoint). Action verbs and the group-referring pronoun attract positive "
    "mass; function tokens receive negative attributions that sharpen the contrast.",
)
para(
    "Re-running the same comment without its context yields the same label with nearly identical "
    "confidence (hate_speech, p = 0.754 vs. 0.757) and the same evidence spans. For this pair, "
    "context matters less to the score than to the interpretation: the reasoning layer's reference "
    "link is what makes the target explicit for the user. The template's phrase “the contextual "
    "reference matters” therefore describes interpretive dependence, not causal score dependence — "
    "a mismatch we flag rather than paper over. A future revision should phrase the template as "
    "“the preceding comment supplies a candidate referent” and should not imply that the link changed "
    "the decision."
)

h2("6.7", "Reference-linker audit and HateCheck functional tests")
para(
    "Reference linker. There are no gold antecedent annotations in either corpus, so we report "
    "coverage rather than accuracy. On the full Counter Context test set (713 pairs) the heuristic "
    "emits at least one link for 118 pairs (16.6% coverage; 141 links total; 0.20 links per pair). "
    "Almost every previous comment (712 / 713) contains some noun-phrase candidate, so the bottleneck "
    "is the linking rule, not the absence of candidates. Pronouns that fire, in decreasing order, "
    "are they (32), she (26), her (24), them (15), he (15), their (14), him (11), his (4). Manual "
    "inspection of emitted links shows systematic errors: the right-most noun-phrase rule attaches "
    "to function words or to the wrong entity (“him” → “it's”; “She” → “scholarships”; “she” → "
    "“There's”). The WHO component of the explanation is therefore a low-yield heuristic with "
    "unquantified precision. It is labelled as a heuristic in every API response, and we do not "
    "claim it recovers the intended referent. A small hand-labelled evaluation set is the next "
    "required experiment."
)
para(
    "HateCheck. We ran the HateCheck test suite [21] (3,728 cases; 2,563 hateful, 1,165 non-hateful) "
    "through the hx_full hate head. The model does not transfer. Of the hateful cases, only 12.8% "
    "are predicted as hate; 32.3% are predicted offensive and 54.9% normal. Of the non-hateful "
    "cases, 74.3% are predicted normal, 20.3% offensive, and 5.4% hate (over-flagging as hate is "
    "modest). Functionality recall on hateful cases is highest for slurs (0.438) and below 0.20 for "
    "most others (dehumanization 0.186, directed threats 0.188); spelling perturbations collapse "
    "(character-swap 0.045, inserted spaces 0.000). HateCheck is adversarial by design and its "
    "hateful/non-hateful split is not identical to HateXplain's three-class policy, so this is a "
    "transfer result, not a re-evaluation of Table 4 — and it is a negative one. HateXplain-trained "
    "BERT at this scale does not possess the functional competence HateCheck measures."
)

h2("6.8", "Threats to validity")
para(
    "(i) HateXplain hx_full has three seeds; the other HateXplain configurations remain seed 42. "
    "The Counter Context ladder (baseline, concat, interaction, InfoNCE, cross-encoder) has three "
    "seeds. The HateXplain ordering (baseline < full ≈ hx_full < hx_full_long) is consistent across "
    "validation and test; the Counter Context comment-only vs. concatenation vs. interaction "
    "ordering is consistent across three seeds on accuracy. (ii) We trained a BERT cross-encoder but "
    "did not re-run Yu et al.'s RoBERTa models; their released gold splits are used as-is, and we "
    "have not verified byte-identity with the paper's random split. (iii) Evidence metrics use the first 100 rationale-bearing examples (dataset order, not "
    "random); human rationale disagreement itself bounds achievable agreement. (iv) The reference "
    "analysis has no precision estimate — no antecedent annotations exist; coverage is 16.6% with "
    "visible errors. (v) No human study of explanation usefulness, no LLM-explanation baseline, and "
    "no comparison against a plain evidence-and-label template. (vi) HateCheck transfer is poor "
    "(Section 6.7); the suite was not used for model selection. (vii) HateXplain runs use the "
    "zero-context fallback, so the interaction module's value there is architectural compatibility, "
    "not measured gains. (viii) The HateXplain multi-task gain over the baseline could come from the "
    "extra projection layer, the target head, or the evidence head — the suite does not separate "
    "them. (ix) Latency was measured on one laptop GPU with n = 100 after a noisier 15-request "
    "pilot; absolute milliseconds will not transfer across devices."
)

# =============================================================== 7 system
h1(7, "System Implementation")
para(
    "The framework ships as a deployable stack. The FastAPI backend loads the checkpoint once at "
    "startup (pointer-file discovery, MODEL_CHECKPOINT override) and serves GET /health, "
    "GET /api/v1/model/info (metadata plus the machine-readable evaluation file), POST /api/v1/predict "
    "(classification, evidence, structured reasoning), and POST /api/v1/explain (per-token "
    "attribution for any head and class). Inference runs under no_grad; attribution temporarily "
    "enables gradients on input embeddings only. A serving-export script strips optimizer state "
    "(≈65% smaller checkpoint) and a Dockerfile packages the service. Correctness is enforced by an "
    "offline-safe backend test suite (small random BERTs; a real-weights integration suite is "
    "opt-in) plus frontend component tests."
)
para(
    "The React + TypeScript front end renders the explanation chain — classification, reasoning card "
    "with clickable connection chips, offset-based highlighting of evidence in both comments, "
    "on-demand per-token attribution, evaluation-metrics card, history — and includes an annotation "
    "workspace that exports reason-annotation files in exactly the schema the reason-head importer "
    "consumes. The system is a research prototype for review-assist tooling; it is not designed, "
    "calibrated, or evaluated for autonomous enforcement."
)

# =============================================================== 8 ethics
h1(8, "Ethics Statement")
para(
    "Hate speech research works with text that demeans real communities. All examples in this paper "
    "come from public benchmark corpora (HateXplain, Counter Context); we quote the minimum needed "
    "for the worked example and add no new slurs. Reader discretion is advised. Annotators on both "
    "corpora were not compensated by us; we use the released public splits."
)
para(
    "Known risks we take seriously, and what the system does and does not do about them. Annotator "
    "and dialect bias: hate-speech datasets have repeatedly been shown to over-flag African-American "
    "English and identity-term usage [14]; our models inherit whatever bias the corpora carry, and "
    "the target-head gender/other weaknesses (Section 6.3) are likely the same problem in another "
    "form. Identity-term false positives: reclaimed slurs and in-group usage can be misclassified; "
    "the evidence spans are intended to make such errors inspectable, not to justify them. Misuse: a "
    "deployed detector can be used for surveillance or over-blocking; this system is a research "
    "prototype for review-assist tooling and is not designed, calibrated, or evaluated for autonomous "
    "enforcement. Dual use of explanations: the same evidence spans that help a moderator audit a "
    "decision could help an adversary probe the classifier; we consider the audit benefit of "
    "deterministic, offset-accurate evidence to outweigh this risk at research scale. Confidence "
    "scores are uncalibrated model outputs, not probabilities of harm. The reason head is untrained "
    "in this work; no harm-category statement is made without its own annotations. The worked example "
    "involves a dehumanizing statement about immigrants; we reproduce it because it is the "
    "shortest real transcript that shows the reference link, the evidence spans, and the "
    "context-does-not-change-the-score caveat at once."
)

# =============================================================== 9 limitations
h1(9, "Limitations")
bullet(
    "HateXplain hx_full has three seeds; other HateXplain configurations remain single-seed. "
    "The Counter Context ladder and the BERT cross-encoder have three seeds."
)
bullet(
    "Our BERT cross-encoder matches Yu et al.'s reported RoBERTa gold cross-encoder on weighted-F1 "
    "but is a different encoder and a different split protocol; we did not retrain their models."
)
bullet(
    "The original suite adds interaction and InfoNCE together in cc_context; the matched ladder "
    "separates them, but batch size and epochs are not equalized between the original suite and "
    "the ladder."
)
bullet(
    "The pronoun-to-antecedent analysis has 16.6% coverage and visible wrong-antecedent errors; it "
    "is labelled a heuristic in every response. No human study of explanation usefulness has been "
    "conducted."
)
bullet(
    "HateCheck transfer is poor (12.8% hateful recall as hate). Offensive-class confusion persists "
    "(test F1 ≈ 0.51–0.54) and is bounded by HateXplain's label policy at this scale."
)
bullet(
    "Evidence metrics use 100 examples; IG is not a faithfulness measure [12], [15]; the supervised "
    "head's 0.823 token-F1 is plausibility, not faithfulness. The deletion probe uses PAD "
    "substitution, which is a crude intervention."
)
bullet(
    "Latency (0.7–1.1 s at n = 100 on one laptop GPU) precludes interactive-UI or high-throughput "
    "use without engineering (smaller IG budget, evidence-head serving, distillation)."
)
bullet(
    "Counter Context performance trails the corpus authors' best silver+pre-training result (0.64 "
    "weighted-F1). Our dual-encoder design also does not beat our own BERT cross-encoder on macro-F1 "
    "or counter-speech; we prioritize a unified single-encoder system that keeps single-comment "
    "inference available."
)

# =============================================================== 10 conclusion
h1(10, "Conclusion")
para(
    "We presented a framework for hate speech detection with structured, evidence-based explanations "
    "assembled by a deterministic reasoning layer over verified outputs, with provenance, abstention, "
    "and no generative component. Determinism makes every emitted statement checkable against a "
    "stored output; it does not make Integrated Gradients faithful — a deletion probe showed the "
    "highlighted evidence is largely unnecessary for the prediction — and it does not make the "
    "pronoun linker correct. The classifier is assembled from standard parts. On HateXplain it "
    "performs on par with published BERT results (69.2% ± 0.4 / 0.679 ± 0.005 over three seeds vs. "
    "69.0% / 0.674 [7]). On Counter Context, context helps modestly and consistently in a "
    "matched-budget three-seed ladder; interaction features help accuracy further; InfoNCE does not "
    "separate from interaction; and a trained BERT cross-encoder is the strongest and most stable "
    "classifier (0.610 ± 0.003 accuracy, 0.579 ± 0.003 macro-F1, 0.612 weighted-F1), matching Yu et "
    "al.'s RoBERTa gold cross-encoder (0.61 weighted-F1 [8]). We therefore do not claim that the "
    "dual-encoder interaction architecture or InfoNCE improves Counter Context classification; its "
    "value is architectural (single-comment inference and a shared encoder that also serves "
    "HateXplain). A trained rationale "
    "head (0.823 token-F1) substantially outperforms post-hoc Integrated Gradients (0.650) at "
    "agreeing with human rationales. HateCheck transfer and reference-linker coverage are negative "
    "results we leave standing. Latency is review-assist scale. The contribution to build on is the "
    "explanation layer and the honesty engineering around it — not a state-of-the-art classifier — "
    "and the next experiments are a hand-labelled linker set and a human study of explanation "
    "usefulness."
)

# =============================================================== references
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(14)
p.paragraph_format.space_after = Pt(6)
set_font(p.add_run("References"), 12, bold=True)
refblock([
    "T. Davidson, D. Warmsley, M. Macy, and I. Weber, “Automated hate speech detection and the "
    "problem of offensive language,” in Proc. 11th Int. AAAI Conf. Web and Social Media (ICWSM), "
    "2017, pp. 512–515.",
    "Z. Waseem and D. Hovy, “Hateful symbols or hateful people? Predictive features for hate speech "
    "detection on Twitter,” in Proc. NAACL-HLT Student Research Workshop, 2016, pp. 88–93.",
    "A. Founta et al., “Large scale crowdsourcing and characterization of Twitter abusive "
    "behavior,” in Proc. 12th Int. AAAI Conf. Web and Social Media (ICWSM), 2018.",
    "M. Zampieri, S. Malmasi, P. Nakov, S. Rosenthal, F. Farra, and R. Kumar, “Predicting the type "
    "and target of offensive posts in social media,” in Proc. NAACL-HLT, 2019, pp. 1415–1420.",
    "J. Fortuna and S. L. Nunes, “A survey on automatic detection of hate speech in text,” ACM "
    "Computing Surveys, vol. 51, no. 4, article 85, 2018.",
    "B. Vidgen and L. Derczynski, “Directions in abusive language training data, a systematic "
    "review: Garbage in, garbage out,” PLoS ONE, vol. 15, no. 12, e0243300, 2020.",
    "B. Mathew, P. Saha, S. M. Yimam, C. Biemann, P. Goyal, and A. Mukherjee, “HateXplain: A "
    "benchmark dataset for explainable hate speech detection,” in Proc. 35th AAAI Conf. Artificial "
    "Intelligence, vol. 35, no. 17, 2021, pp. 14867–14875.",
    "X. Yu, E. Blanco, and L. Hong, “Hate speech and counter speech detection: Conversational "
    "context does matter,” in Proc. NAACL-HLT, 2022, pp. 5918–5930.",
    "J. H. Park and P. Fung, “One-step and two-step classification for abusive language detection "
    "on Twitter,” in Proc. ALW1 (Abusive Language Workshop), 2017, pp. 41–45.",
    "M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should I trust you?’ Explaining the predictions "
    "of any classifier,” in Proc. 22nd ACM KDD, 2016, pp. 1135–1144.",
    "M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in Proc. 34th "
    "Int. Conf. Machine Learning (ICML), PMLR 70, 2017, pp. 3319–3328.",
    "S. Jain and B. C. Wallace, “Attention is not explanation,” in Proc. NAACL-HLT, 2019, "
    "pp. 2431–2441.",
    "J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, and B. C. Wallace, "
    "“ERASER: A benchmark to evaluate rationalized NLP models,” in Proc. ACL, 2020, pp. 4443–4458.",
    "A. Sap, D. Card, S. Gabriel, Y. Choi, and N. A. Smith, “The risk of racial bias in hate speech "
    "detection,” in Proc. AACL-IJCNLP, 2019, pp. 79–89.",
    "A. Jacovi and Y. Goldberg, “Towards faithfully interpretable NLP systems: How should we define "
    "and evaluate faithfulness?” in Proc. ACL, 2020, pp. 4198–4205.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional "
    "Transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186.",
    "A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes, “Supervised learning of "
    "universal sentence representations from data in the wild,” in Proc. EMNLP, 2018, pp. 1870–1879.",
    "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” "
    "in Proc. EMNLP-IJCNLP, 2019, pp. 3982–3992.",
    "A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive "
    "coding,” arXiv preprint arXiv:1807.03748, 2018.",
    "T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, "
    "M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le "
    "Scao, S. Gugger, M. Drame, Q. Lhoest, and A. Rush, “Transformers: State-of-the-art natural "
    "language processing,” in Proc. EMNLP System Demonstrations, 2020, pp. 38–45.",
    "P. Röttger, B. Vidgen, D. Nguyen, Z. Waseem, H. Margetts, and J. Pierrehumbert, “HateCheck: "
    "Functional tests for hate speech detection models,” in Proc. ACL-IJCNLP, 2021, pp. 41–58.",
    "C. Ziems, W. Held, O. Shaikh, J. Chen, Z. Zhang, and D. Yang, “Can large language models "
    "transform computational social science?” Computational Linguistics, vol. 50, no. 1, "
    "pp. 237–291, 2024.",
    "M. Turpin, J. Michael, E. Perez, and S. R. Bowman, “Language models don't always say what they "
    "think: Unfaithful explanations in chain-of-thought prompting,” in Advances in Neural "
    "Information Processing Systems (NeurIPS), 2023.",
])

# =============================================================== appendix
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(14)
p.paragraph_format.space_after = Pt(6)
set_font(p.add_run("Appendix A.  Reproducibility"), 12, bold=True)
para(
    "All experiments are one command each from backend/, with the GPU interpreter of the training "
    "environment. Artifacts produced for this revision include configs/cc_baseline.yaml (comment-only "
    "Counter Context baseline), the matched-budget ladder checkpoints (cc_baseline_s{42,1,2}, "
    "cc_concat_s{42,1,2}, cc_inter_s{42,1,2}, cc_nce_s{42,1,2}), the BERT cross-encoder runs, the "
    "hx_full seed runs (s1, s2), the evidence-head evaluation, the deletion-probe transcript, "
    "the n = 100 latency benchmark, the HateCheck transcript, the reference-linker audit, and the "
    "live inference transcript of Table 9."
)
mono = [
    "python scripts/download_data.py --dataset all",
    "python scripts/preprocess.py",
    "python scripts/train.py --config baseline | full | hx_full | hx_full_long | cc_baseline | cc_context",
    "python scripts/train.py --config context --dataset counter_context",
    "python scripts/evaluate.py --config <name>       # writes best.metrics.json",
    'python scripts/explain.py --text "They should all be kicked out." --context "I saw a group of immigrants protesting downtown."',
    "uvicorn app.main:app --port 8001                 # /api/v1/predict, /api/v1/explain",
]
for m in mono:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run(m), 8.5, name="Consolas")
para(
    "Training histories, held-out test evaluations, and the transcripts listed above are stored as "
    "artifacts in the repository (paper_assets/experiment_results.json, paper_assets/"
    "addenda_results.json, paper_assets/evidence_head_metrics.json, paper_assets/latency.json, "
    "paper_assets/reference_linker.json, paper_assets/explain_canonical.json), so every figure and "
    "table in this paper can be regenerated from them.",
    space_after=0,
)

add_page_number_footer()
OUT_DOCX.parent.mkdir(exist_ok=True)
doc.save(str(OUT_DOCX))
print("saved:", OUT_DOCX, f"{OUT_DOCX.stat().st_size/1024:.0f} KiB")
