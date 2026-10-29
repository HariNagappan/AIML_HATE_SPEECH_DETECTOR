import json, time
from pathlib import Path
p = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\paper_assets\experiment_results.json")
log = Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\paper_assets\exp_run1.log")
deadline = time.time() + 540  # 9 min per poll call
while time.time() < deadline:
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    cc = d.get("cc_ladder", [])
    xe = d.get("cross_encoder", [])
    done = sum(1 for r in cc if r.get("test_macro_f1") is not None) + sum(1 for r in xe if r.get("test_macro_f1") is not None)
    print(f"{time.strftime('%H:%M:%S')} cc_done={sum(1 for r in cc if r.get('test_macro_f1'))}/12 xe_done={sum(1 for r in xe if r.get('test_macro_f1'))}/3", flush=True)
    if done >= 15:
        print("LADDER COMPLETE")
        break
    time.sleep(60)
print("--- recent log ---")
print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]))
