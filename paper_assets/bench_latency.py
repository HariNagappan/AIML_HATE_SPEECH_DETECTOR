import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\backend")
from app.core.config import get_settings
from app.services.inference import InferenceService

settings = get_settings()
svc = InferenceService(settings.model_copy(update={
    "checkpoint_path": r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\backend\checkpoints\cc_context\best.pt"
}))
pairs = [
    ("They should all be kicked out.", "I saw a group of immigrants protesting downtown."),
    ("This is disgusting and they should leave.", "Immigrants are ruining everything."),
    ("Great point, I totally agree with you.", "The new policy is a disaster."),
    ("Nobody wants them here anyway.", "They keep taking our jobs."),
    ("Welcome to the neighborhood!", "We just moved in last week."),
]


def bench(fn, n=15, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean_ms": round(sum(times) / len(times), 1),
        "p50_ms": round(times[len(times) // 2], 1),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 1),
    }


svc.predict(pairs[0][0], pairs[0][1])  # force-load the checkpoint
model = svc._bundle.model
assert model is not None, svc.load_error
lat = {
    "device": str(next(model.parameters()).device),
    "n_requests": 15,
    "predict_no_context": bench(lambda: svc.predict(pairs[0][0], None)),
    "predict_with_context": bench(lambda: svc.predict(pairs[1][0], pairs[1][1])),
    "explain_ig32_with_context": bench(lambda: svc.explain(pairs[0][0], pairs[0][1], target="hate")),
}
print(json.dumps(lat, indent=2))
Path(r"C:\Users\Haris\OneDrive\Desktop\AIML_Project\paper_assets\latency.json").write_text(
    json.dumps(lat, indent=2), encoding="utf-8"
)
