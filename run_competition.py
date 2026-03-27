"""Jane Street Battle Royale - with Real Data Support and Health Check"""
import os
import sys
import time
import gc
import datetime
import nbformat as nbf
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. API KEY SETUP — read from environment
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

# ==========================================
# 2. TEST MODE CONFIGURATION
# ==========================================
TEST_MODE = False
TEST_LIMIT = 2

# ==========================================
# 2b. SAFETY SETTINGS
# ==========================================
DELAY_SECONDS = 5
ENABLE_GC = True
LOG_ERRORS = True
MAX_RETRIES = 3
RETRY_DELAY = 30
EXPONENTIAL_BACKOFF = True
REQUEST_TIMEOUT = 60  # Hard timeout per request (seconds) — prevents 5-min hangs

# ==========================================
# 2c. HEALTH CHECK SETTINGS
# ==========================================
RUN_HEALTH_CHECK = False  # Set to True to pre-test all models (adds 2-5 min)
HEALTH_CHECK_TIMEOUT = 10
HEALTH_CHECK_PROMPT = "Hi"

# ==========================================
# 2d. SKIP KNOWN UNAVAILABLE MODELS (404 errors)
# ==========================================
SKIP_MODELS = [
    "nvidia/riva-translate-4b-instruct",
    "nvidia/usdcode-llama-3.1-70b-instruct",
    "mistralai/codestral-22b-instruct-v0.1",
    "aisingapore/sea-lion-7b-instruct",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "zyphra/zamba2-7b-instruct",
    "meta/llama-4-scout-17b-16e-instruct",
]

# ==========================================
# 2e. MODELS THAT REJECT SYSTEM ROLE MESSAGES
# These get system instructions merged into the user message instead
# ==========================================
NO_SYSTEM_ROLE_MODELS = [
    "nvidia/llama3-chatqa-1.5-8b",
    "nvidia/llama3-chatqa-1.5-70b",
    "thudm/chatglm3-6b",
    "baichuan-inc/baichuan2-13b-chat",
    "rakuten/rakutenai-7b-chat",
    "rakuten/rakutenai-7b-instruct",
]

# ==========================================
# 3. LOGGING SETUP
# ==========================================
if LOG_ERRORS:
    log_file = f"competition_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_handle = open(log_file, "w", encoding="utf-8")
    def log(msg):
        log_handle.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        log_handle.flush()
else:
    def log(msg): pass

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def get_prompt_template(model_id):
    """Get appropriate prompt template based on model's role support."""
    task_instruction = """Write a complete Python script to solve the Jane Street Market Forecasting challenge.

ENVIRONMENT:
- Data Path: './jane_street_data/train.parquet'
- Libraries: Use `polars` for speed and `xgboost` for modeling.

LOGIC REQUIREMENTS:
1. Load 'train.parquet' using Polars.
2. Feature Engineering: Calculate a global TOP_QUANTILE (top 15%) of 'feature_00'
   relative to 'responder_6' across rolling batches of 'date_id'.
3. Train an XGBoost Regressor on the target 'responder_6'.

OUTPUT:
- Pure Python code only. No markdown. No explanations.
"""
    if model_id in NO_SYSTEM_ROLE_MODELS:
        # Merge system instruction into user message for models that don't support 'system'
        return ChatPromptTemplate.from_messages([
            ("user", f"You are an elite Quant Developer. Output ONLY pure Python code.\n\n{task_instruction}")
        ])
    else:
        return ChatPromptTemplate.from_messages([
            ("system", "You are an elite Quant Developer. Output ONLY pure Python code."),
            ("user", task_instruction)
        ])

def run_health_check(models):
    """Pre-test all models to identify healthy ones before competition."""
    print("\n" + "=" * 60)
    print("  RUNNING HEALTH CHECK ON ALL MODELS")
    print("=" * 60)
    print(f"Testing {len(models)} models with quick ping...\n")

    healthy_models = []
    unhealthy_models = []
    health_results = {}

    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful."),
        ("user", "Hello")
    ])
    user_only_prompt = ChatPromptTemplate.from_messages([
        ("user", "You are helpful. Hello")
    ])

    for i, model_id in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Testing: {model_id}")
        prompt = user_only_prompt if model_id in NO_SYSTEM_ROLE_MODELS else system_prompt
        try:
            llm = ChatNVIDIA(model=model_id, max_tokens=5)
            chain = prompt | llm
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(chain.invoke, {})
                response = future.result(timeout=REQUEST_TIMEOUT)
            health_results[model_id] = "HEALTHY"
            healthy_models.append(model_id)
            print(f"  -> ✓ OK\n")

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "Not Found" in error_str:
                health_results[model_id] = "404_NOT_FOUND"
            elif "504" in error_str or "Gateway Timeout" in error_str:
                health_results[model_id] = "504_TIMEOUT"
            elif "503" in error_str or "Service Unavailable" in error_str:
                health_results[model_id] = "503_UNAVAILABLE"
            elif "422" in error_str:
                health_results[model_id] = "422_VALIDATION_ERROR"
            elif "429" in error_str or "Rate Limit" in error_str:
                health_results[model_id] = "429_RATE_LIMIT"
            elif "400" in error_str:
                # 400 errors often mean model has special requirements — still try in competition
                health_results[model_id] = "400_CHECK_MANUALLY"
                healthy_models.append(model_id)
                print(f"  -> ~ 400 (will try in competition)\n")
                continue
            else:
                health_results[model_id] = "ERROR"
            unhealthy_models.append((model_id, health_results[model_id]))
            print(f"  -> ✗ FAILED: {health_results[model_id]}\n")

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("  HEALTH CHECK SUMMARY")
    print("=" * 60)
    print(f"✓ Healthy:   {len(healthy_models)}/{len(models)}")
    print(f"✗ Unhealthy: {len(unhealthy_models)}/{len(models)}\n")

    if unhealthy_models:
        print("Unhealthy models:")
        for model_id, reason in unhealthy_models:
            print(f"  - {model_id}: {reason}")
        print()

    health_report_file = f"health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(health_report_file, "w", encoding="utf-8") as f:
        f.write("HEALTH CHECK REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Models Tested: {len(models)}\n")
        f.write(f"Healthy: {len(healthy_models)}\n")
        f.write(f"Unhealthy: {len(unhealthy_models)}\n\n")
        f.write("HEALTHY MODELS:\n")
        for m in healthy_models:
            f.write(f"  ✓ {m}\n")
        f.write("\nUNHEALTHY MODELS:\n")
        for m, reason in unhealthy_models:
            f.write(f"  ✗ {m}: {reason}\n")

    print(f"Health report saved to: {health_report_file}\n")
    return healthy_models, health_results

# ==========================================
# 5. MODEL DISCOVERY
# ==========================================
print("Fetching live model list from NVIDIA...")
try:
    all_models = ChatNVIDIA.get_available_models()
    competitors = [
        m.id for m in all_models
        if any(k in m.id.lower() for k in ["instruct", "chat", "coder", "nemotron"])
        and not any(x in m.id.lower() for x in ["embed", "rerank", "vision", "reward", "safety"])
        and m.id not in SKIP_MODELS
    ]

    if TEST_MODE:
        competitors = competitors[:TEST_LIMIT]
        print(f"\n*** TEST MODE: Running {len(competitors)} models ***\n")
    else:
        print(f"\n*** FULL COMPETITION: {len(competitors)} models locked in ***\n")
        print(f"(Skipped {len(SKIP_MODELS)} unavailable models)")

    print("Models in the arena:")
    for i, m in enumerate(competitors, 1):
        print(f"  {i}. {m}")
    print()

except Exception as e:
    print(f"CRITICAL ERROR: Could not fetch models. Check your API Key. Error: {e}")
    sys.exit(1)

# ==========================================
# 6. RUN HEALTH CHECK (optional)
# ==========================================
if RUN_HEALTH_CHECK:
    healthy_models, health_results = run_health_check(competitors)
    unhealthy_count = len(competitors) - len(healthy_models)
    competitors = healthy_models

    for model_id, status in health_results.items():
        if status == "404_NOT_FOUND" and model_id not in SKIP_MODELS:
            SKIP_MODELS.append(model_id)

    print(f"\n*** HEALTH CHECK COMPLETE ***")
    print(f"    Removed {unhealthy_count} unhealthy models")
    print(f"    Competition will run with {len(competitors)} healthy models\n")

    print("=" * 60)
    print(f"Ready to compete with {len(competitors)} healthy models?")
    print("(Press Ctrl+C to cancel, or wait 5 seconds to auto-start...)")
    print("=" * 60)
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCompetition cancelled by user.")
        sys.exit(0)

# ==========================================
# 7. RUN LOOP (40 RPM Safe)
# ==========================================
print("=" * 60)
print("  BATTLE STARTED - MAY THE BEST MODEL WIN!")
print("=" * 60)
print()

success_count = 0
fail_count = 0

for i, model_id in enumerate(competitors, 1):
    print(f"[{i}/{len(competitors)}] Calling: {model_id}")
    log(f"[{i}/{len(competitors)}] Processing: {model_id}")

    prompt_template = get_prompt_template(model_id)
    llm = ChatNVIDIA(model=model_id)
    chain = prompt_template | llm

    response = None
    attempt = 0
    failed = False

    while attempt < MAX_RETRIES:
        try:
            attempt_label = f" (Attempt {attempt+1}/{MAX_RETRIES})" if attempt > 0 else ""
            print(f"  -> Sending request to NVIDIA API...{attempt_label}")
            log(f"  -> Sending API request...{attempt_label}")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(chain.invoke, {})
                try:
                    response = future.result(timeout=REQUEST_TIMEOUT)
                except FuturesTimeoutError:
                    raise Exception(f"504 Request timed out after {REQUEST_TIMEOUT}s")
            print("  -> Response received!")
            log("  -> Response received successfully")
            break

        except Exception as e:
            error_str = str(e)
            if "504" in error_str or "503" in error_str or "Gateway Timeout" in error_str or "Service Unavailable" in error_str:
                attempt += 1
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY * attempt if EXPONENTIAL_BACKOFF else RETRY_DELAY
                    code = "504" if "504" in error_str else "503"
                    print(f"  -> TIMEOUT ({code}). Retrying in {wait_time}s... ({attempt}/{MAX_RETRIES})")
                    log(f"  -> TIMEOUT. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  -> FAILED after {MAX_RETRIES} attempts: {e}\n")
                    log(f"  -> ERROR: {e}")
                    fail_count += 1
                    failed = True
                    break
            elif "429" in error_str or "Rate Limit" in error_str:
                attempt += 1
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY * 2
                    print(f"  -> RATE LIMITED (429). Retrying in {wait_time}s... ({attempt}/{MAX_RETRIES})")
                    log(f"  -> RATE LIMITED. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  -> FAILED after {MAX_RETRIES} attempts: {e}\n")
                    log(f"  -> ERROR: {e}")
                    fail_count += 1
                    failed = True
                    break
            else:
                print(f"  -> FAILED: {e}\n")
                log(f"  -> ERROR: {e}")
                fail_count += 1
                failed = True
                break

    if not failed and response:
        code_content = response.content
        if "```python" in code_content:
            code_content = code_content.split("```python")[1].split("```")[0].strip()
        elif "```" in code_content:
            code_content = code_content.split("```")[1].split("```")[0].strip()
            if code_content.startswith("python"):
                code_content = code_content[6:].strip()

        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# Results for model: {model_id}"))
        nb.cells.append(nbf.v4.new_code_cell(code_content))

        fname = f"{model_id.replace('/', '_').replace(':', '_')}.ipynb"
        with open(fname, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"  -> Created: {fname}\n")
        log(f"  -> Notebook saved: {fname}")
        success_count += 1

    if ENABLE_GC:
        gc.collect()

    print(f"  -> Waiting {DELAY_SECONDS} seconds (Rate Limit Control)...")
    log(f"  -> Waiting {DELAY_SECONDS} seconds before next model...")
    time.sleep(DELAY_SECONDS)

if LOG_ERRORS:
    log_handle.write(f"\n=== SUMMARY ===\n")
    log_handle.write(f"Success: {success_count}/{len(competitors)}\n")
    log_handle.write(f"Failed: {fail_count}/{len(competitors)}\n")
    log_handle.close()
    print(f"\nError log saved to: {log_file}")

print()
print("=" * 60)
print("  BATTLE COMPLETE. Review your .ipynb files.")
print("=" * 60)
