"""Quick Health Check for NVIDIA Models - Run before competition"""
import os
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# API KEY — read from environment variable
# Set it before running:
#   Windows:  setx NVIDIA_API_KEY "nvapi-..."
#   Linux/Mac: export NVIDIA_API_KEY="nvapi-..."
# ==========================================
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    print("ERROR: NVIDIA_API_KEY environment variable is not set.")
    print("  Windows:  setx NVIDIA_API_KEY \"nvapi-...\"")
    print("  Linux/Mac: export NVIDIA_API_KEY=\"nvapi-...\"")
    sys.exit(1)
os.environ["NVIDIA_API_KEY"] = api_key

# Request timeout in seconds (prevents hanging for 5+ minutes on slow models)
REQUEST_TIMEOUT = 60

# Known unavailable models (skip these)
SKIP_MODELS = [
    "nvidia/riva-translate-4b-instruct",
    "nvidia/usdcode-llama-3.1-70b-instruct",
    "mistralai/codestral-22b-instruct-v0.1",
    "aisingapore/sea-lion-7b-instruct",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "zyphra/zamba2-7b-instruct",
    "meta/llama-4-scout-17b-16e-instruct",
]

# Models that reject system-role messages — use user-only prompt for these
NO_SYSTEM_ROLE_MODELS = [
    "nvidia/llama3-chatqa-1.5-8b",
    "nvidia/llama3-chatqa-1.5-70b",
    "thudm/chatglm3-6b",
    "baichuan-inc/baichuan2-13b-chat",
    "rakuten/rakutenai-7b-chat",
    "rakuten/rakutenai-7b-instruct",
]

print("=" * 60)
print("  NVIDIA MODEL HEALTH CHECK")
print("=" * 60)
print("\nFetching model list...")

all_models = ChatNVIDIA.get_available_models()
competitors = [
    m.id for m in all_models
    if any(k in m.id.lower() for k in ["instruct", "chat", "coder", "nemotron"])
    and not any(x in m.id.lower() for x in ["embed", "rerank", "vision", "reward", "safety"])
    and m.id not in SKIP_MODELS
]

print(f"Found {len(competitors)} models to test (skipping {len(SKIP_MODELS)} known unavailable)\n")

# Use user-only prompt for models that don't support system role
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("user", "Hello")
])
user_only_prompt = ChatPromptTemplate.from_messages([
    ("user", "You are helpful. Hello")
])

healthy = []
unhealthy = []

for i, model_id in enumerate(competitors, 1):
    print(f"[{i}/{len(competitors)}] {model_id}", end=" ... ", flush=True)

    # Pick the right prompt based on model support
    prompt = user_only_prompt if model_id in NO_SYSTEM_ROLE_MODELS else system_prompt

    try:
        llm = ChatNVIDIA(model=model_id, max_tokens=5)
        chain = prompt | llm
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(chain.invoke, {})
            try:
                future.result(timeout=REQUEST_TIMEOUT)
            except FuturesTimeoutError:
                raise Exception("504 Request timed out")
        print("✓ OK")
        healthy.append(model_id)
    except Exception as e:
        error = str(e)
        if "404" in error:
            status = "404_NOT_FOUND"
        elif "504" in error or "Gateway Timeout" in error:
            status = "504_TIMEOUT"
        elif "503" in error or "Service Unavailable" in error:
            status = "503_UNAVAILABLE"
        elif "422" in error:
            status = "422_VALIDATION"
        elif "429" in error or "Rate Limit" in error:
            status = "429_RATE_LIMIT"
        elif "400" in error:
            status = "400_BAD_REQUEST"
        else:
            status = f"ERROR: {error[:60]}"
        print(f"✗ {status}")
        unhealthy.append((model_id, status))
    time.sleep(0.3)

# Summary
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"✓ Healthy:   {len(healthy)}/{len(competitors)}")
print(f"✗ Unhealthy: {len(unhealthy)}/{len(competitors)}")

if unhealthy:
    print("\nUnhealthy models:")
    for m, status in unhealthy:
        print(f"  - {m}: {status}")

# Save report
report_file = f"health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_file, "w") as f:
    f.write("HEALTH CHECK REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total: {len(competitors)} | Healthy: {len(healthy)} | Unhealthy: {len(unhealthy)}\n\n")
    f.write("HEALTHY MODELS:\n")
    for m in healthy:
        f.write(f"  ✓ {m}\n")
    f.write("\nUNHEALTHY MODELS:\n")
    for m, status in unhealthy:
        f.write(f"  ✗ {m}: {status}\n")

print(f"\nReport saved: {report_file}")
