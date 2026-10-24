"""
test_models.py — test the free OpenRouter models IN PARALLEL and print a
ready-to-paste MODELS list for backend/llm.py.

Put this file in the project root (next to .env) and run:
    python test_models.py

It takes about a minute. Each model gets the same small RAG-style question.
Models that answer within TIME_LIMIT seconds are suggested, fastest first.
Free model IDs change often: add or remove IDs in CANDIDATES as you like
(see https://openrouter.ai/models?pricing=free).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit("OPENROUTER_API_KEY not found in .env")

# From your own OpenRouter free list. Left out on purpose:
#   nemotron-3.5-content-safety (a safety classifier, not a chat model),
#   cohere/north-mini-code (code model), nemotron-3.5-super-vl (vision model).
# NOTE: each run uses one free request per model (about 11), so do not
# run this over and over: free accounts have a daily request limit.
CANDIDATES = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/free",
]

TIME_LIMIT = 25   # same as PER_CALL_TIMEOUT in backend/llm.py
WAIT = 60         # how long this script waits for each model

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    max_retries=0,   # no hidden retries, so the timing is honest
)

SYSTEM = (
    "You are a health educator. Answer in English only, strictly from the "
    "context. Cite sources as [Source 1]."
)
USER = (
    "Context:\n[Source 1] (Oral Healthcare for Pregnant Mothers)\n"
    "Pregnant women should brush twice a day with fluoride toothpaste and "
    "visit the dentist during pregnancy.\n\n"
    "Question: What oral care should pregnant women do?\n\n"
    "Answer in 2 sentences (in English):"
)


def test(model):
    start = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            temperature=0,
            max_tokens=600,
            timeout=WAIT,
        )
        secs = time.perf_counter() - start
        if not r.choices:
            extra = getattr(r, "model_extra", None) or {}
            err = extra.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            return model, secs, "error", str(msg)[:160], None
        text = (r.choices[0].message.content or "").strip()
        actual = getattr(r, "model", None) or model
        if not text:
            return model, secs, "empty", "empty answer (reasoning ate the tokens?)", actual
        return model, secs, "ok", text, actual
    except Exception as e:
        secs = time.perf_counter() - start
        return model, secs, "error", f"{type(e).__name__}: {e}"[:160], None


print(f"Testing {len(CANDIDATES)} models in parallel (up to {WAIT}s)...\n")
with ThreadPoolExecutor(max_workers=len(CANDIDATES)) as pool:
    results = list(pool.map(test, CANDIDATES))

good, slow = [], []
for model, secs, status, text, actual in results:
    if status == "ok":
        cites = "cites [Source 1]" if "[Source 1]" in text else "NO citation"
        note = f" (routed to {actual})" if actual and actual != model else ""
        print(f"✅ {model}{note}  {secs:.1f}s  {cites}\n     {text[:160]!r}")
        (good if secs <= TIME_LIMIT else slow).append((secs, model))
    else:
        print(f"❌ {model}  {secs:.1f}s  {status}: {text}")

good.sort()
slow.sort()
print("\n" + "=" * 60)
if good:
    print(f"Working within {TIME_LIMIT}s. Paste this into MODELS in backend/llm.py:\n")
    print("MODELS = [")
    for _, m in good:
        print(f'    "{m}",')
    if "openrouter/free" not in [m for _, m in good]:
        print('    "openrouter/free",   # last resort')
    print("]")
else:
    print(f"No model answered within {TIME_LIMIT}s right now.")
    print("Free models are often busy: wait a few minutes and run again.")
if slow:
    print(f"\nToo slow for the {TIME_LIMIT}s limit (answered, but late):")
    for secs, m in slow:
        print(f"  - {m}: {secs:.0f}s")