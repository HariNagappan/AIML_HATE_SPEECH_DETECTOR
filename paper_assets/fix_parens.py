from pathlib import Path
import py_compile

p = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\paper_assets\build_paper.py")
t = p.read_text(encoding="utf-8")

a = (
    '                        fmt_ms(*mean_std([r.get("test_per_class_f1", {}).get("hate_speech") for r in xe_rows]),\n'
    '                        fmt_ms(*mean_std([r.get("test_per_class_f1", {}).get("counter_speech") for r in xe_rows]),\n'
    '                        fmt_ms(*mean_std([r.get("test_per_class_f1", {}).get("neither") for r in xe_rows]),\n'
    '                        fmt_ms(*mean_std([r.get("test_weighted_f1") for r in xe_rows]))])'
)
b = (
    '                        fmt_ms(*mean_std([r.get("test_per_class_f1", {}).get("hate_speech") for r in xe_rows])),\n'
    '                        fmt_ms(*mean_std([r.get("test_per_class_f1", {}).get("counter_speech") for r in xe_rows])),\n'
    '                        fmt_ms(*mean_std([r.get("test_per_class_f1", {}).get("neither") for r in xe_rows])),\n'
    '                        fmt_ms(*mean_std([r.get("test_weighted_f1") for r in xe_rows]))])'
)
print("a in t", a in t)
t = t.replace(a, b)

a2 = (
    '                        fmt_ms(*mean_std([p.get("hate_speech") for p in pcs]),\n'
    '                        fmt_ms(*mean_std([p.get("counter_speech") for p in pcs]),\n'
    '                        fmt_ms(*mean_std([p.get("neither") for p in pcs]),\n'
    '                        "\u2014"])'
)
b2 = (
    '                        fmt_ms(*mean_std([p.get("hate_speech") for p in pcs])),\n'
    '                        fmt_ms(*mean_std([p.get("counter_speech") for p in pcs])),\n'
    '                        fmt_ms(*mean_std([p.get("neither") for p in pcs])),\n'
    '                        "\u2014"])'
)
print("a2 in t", a2 in t)
t = t.replace(a2, b2)
p.write_text(t, encoding="utf-8")
try:
    py_compile.compile(str(p), doraise=True)
    print("COMPILES")
except py_compile.PyCompileError as e:
    print("STILL", e)
