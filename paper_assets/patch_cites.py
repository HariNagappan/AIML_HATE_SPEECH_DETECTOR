from pathlib import Path
import re
p = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\paper_assets\build_paper.py")
t = p.read_text(encoding="utf-8")
t = t.replace("well documented [23]", "well documented [20]")
t = t.replace("Llama Guard [24]) and LLM content-censorship evaluations [25]",
              "Llama Guard [23]) and LLM content-censorship evaluations [24]")
t = t.replace("InferSent [26] and Sentence-BERT [27]", "InferSent [18] and Sentence-BERT [19]")
t = t.replace("matching features [26], [27]", "matching features [18], [19]")
t = t.replace("encoders [26], [27]", "encoders [18], [19]")
t = t.replace("explanations [22], [24]", "explanations [22], [23]")
t = t.replace("African-American English [23]", "African-American English [20]")
t = t.replace("subgroup bias audits [23]", "subgroup bias audits [20]")
t = re.sub(r"(fmt_ms\(\*mean_std\([^)]*\)), 0\)", r"\1)", t)
t = t.replace(
    'fmt_ms(*mean_std(xe_rows and [r.get("test_weighted_f1") for r in xe_rows]), 0)',
    'fmt_ms(*mean_std([r.get("test_weighted_f1") for r in xe_rows]))',
)
t = t.replace('in Proc. ACL, 2020, pp. 4198\u20134205.",', 'in Proc. ACL, 2020.",')
start = t.find("# Llama Guard is [23]")
if start != -1:
    end = t.find("# ================================================================= appendix")
    t = t[:start] + t[end:]
p.write_text(t, encoding="utf-8")
print("patched, leftover 26/27:", t.count("[26]"), t.count("[27]"))
for i, line in enumerate(t.splitlines(), 1):
    if ("[26]" in line or "[27]" in line or (", 0)" in line and "mean_std" in line)
            or "[25]" in line or "documented [23]" in line):
        print(i, line.strip()[:130])
