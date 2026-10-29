from pathlib import Path
p = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\paper_assets\build_paper.py")
t = p.read_text(encoding="utf-8")
t = t.replace(
    '"ablation-ordered, seed-replicated experimental suite."), 9.5))',
    '"ablation-ordered, seed-replicated experimental suite."), 9.5)',
)
p.write_text(t, encoding="utf-8")
import py_compile
py_compile.compile(str(p), doraise=True)
print("COMPILES")
