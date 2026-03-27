"""
Hugging Face Quant Wars - Inference API Edition

Uses Hugging Face Inference API with your API token.
No local model download - all inference happens in the cloud.
"""

import os
import sys
import time
import gc
import datetime
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import nbformat as nbf
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ==============================================================================
# 1. ENVIRONMENT LOADING
# ==============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_DIR = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env")

# Hugging Face API token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in .env")
    sys.exit(1)

# Available Hugging Face models (open-weight, good for code generation)
HF_MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "microsoft/Phi-3.5-Mini-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "NousResearch/Hermes-3-Llama-3.1-70B",
    "01-ai/Yi-1.5-34B-Chat",
]

# Test mode
TEST_MODE = False
TEST_LIMIT = 5

# Parallel execution mode
PARALLEL_MODE = False
MAX_WORKERS = 3

# Safety settings
DELAY_SECONDS = 3
ENABLE_GC = True
LOG_ERRORS = True
REQUEST_TIMEOUT = 300
MAX_RETRIES = 3

# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================

if LOG_ERRORS:
    log_file = PLATFORM_DIR / f"hf_competition_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_handle = open(log_file, "w", encoding="utf-8")

    def log(msg):
        log_handle.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        log_handle.flush()
else:
    def log(msg):
        pass

# ==============================================================================
# 3. COLOR-CODED TERMINAL OUTPUT
# ==============================================================================

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.CYAN}ℹ {msg}{Colors.RESET}")

# ==============================================================================
# 4. PROMPT TEMPLATE
# ==============================================================================

def get_prompt():
    return """Write a complete Python script to solve the Jane Street Market Forecasting challenge.

ENVIRONMENT:
- Data Path: './jane-street-real-time-market-data-forecasting/train.parquet'
- Libraries: Use `polars` for speed and `xgboost` for modeling.

LOGIC REQUIREMENTS:
1. Load 'train.parquet' using Polars.
2. Feature Engineering: Calculate a global TOP_QUANTILE (top 15%) of 'feature_00' relative to 'responder_6' across rolling batches of 'date_id'.
3. Train an XGBoost Regressor on the target 'responder_6'.

OUTPUT:
- Pure Python code only. No markdown. No explanations."""

# ==============================================================================
# 5. NOTEBOOK GENERATION
# ==============================================================================

def process_single_model(model_id):
    """Process a single model - for parallel execution"""
    prompt = get_prompt()
    
    print(f"\n{Colors.BLUE}[Parallel] Processing: {model_id}{Colors.RESET}")
    log(f"[Parallel] Processing: {model_id}")
    
    client = InferenceClient(token=HF_TOKEN)
    response = None
    failed = False
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an elite Quant Developer. Output ONLY pure Python code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.1
            )
            break
        except Exception as e:
            error_str = str(e)
            if attempt < MAX_RETRIES - 1:
                log(f"Retry {attempt+1}/{MAX_RETRIES} for {model_id}: {error_str[:80]}")
                time.sleep(2)
            else:
                print_error(f"Failed: {model_id} - {error_str[:80]}")
                log(f"ERROR {model_id}: {error_str}")
                failed = True
    
    if not failed and response:
        try:
            code_content = extract_code_from_response(response.choices[0].message.content)
            nb = create_notebook(code_content, model_id)
            fname = save_notebook(nb, model_id)
            print_success(f"Created: {fname}")
            log(f"Notebook saved: {fname}")
            return (model_id, True, fname)
        except Exception as e:
            print_error(f"Failed to process response: {str(e)[:80]}")
            log(f"Processing error {model_id}: {str(e)}")
            return (model_id, False, str(e))
    
    return (model_id, False, "Failed")


def extract_code_from_response(response_content):
    code_content = response_content
    
    if "```python" in code_content:
        code_content = code_content.split("```python")[1].split("```")[0].strip()
    elif "```" in code_content:
        code_content = code_content.split("```")[1].split("```")[0].strip()
    
    if code_content.startswith("python"):
        code_content = code_content[6:].strip()
    
    return code_content

def create_notebook(code_content, model_id):
    nb = nbf.v4.new_notebook()
    safe_name = model_id.replace("/", "_").replace("-", "_")
    nb.cells.append(nbf.v4.new_markdown_cell(f"# Results for model: {model_id}"))
    nb.cells.append(nbf.v4.new_code_cell(code_content))
    return nb

def save_notebook(nb, model_id, output_dir=None):
    output_dir = Path(output_dir) if output_dir else PLATFORM_DIR / "generated_notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    fname = output_dir / f"{safe_name}.ipynb"
    
    with open(fname, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    return fname

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================

def main():
    global PARALLEL_MODE, TEST_MODE
    
    parser = argparse.ArgumentParser(description='Hugging Face Quant Wars')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel execution')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    PARALLEL_MODE = args.parallel
    TEST_MODE = args.test
    
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.CYAN}HUGGING FACE QUANT WARS{Colors.RESET}")
    print("=" * 70)
    print(f"\n{Colors.WHITE}Total Models: {len(HF_MODELS)}{Colors.RESET}")
    print()
    
    competitors = HF_MODELS.copy()
    
    if TEST_MODE:
        competitors = competitors[:TEST_LIMIT]
        print_warning(f"TEST MODE: Running {len(competitors)} models only")
    else:
        print_success(f"FULL COMPETITION: {len(competitors)} models locked in")
    
    print("\nModels in the arena:")
    for i, m in enumerate(competitors, 1):
        print(f"  {i}. {m}")
    print()
    
    print_info("Validating API token...")
    if HF_TOKEN and HF_TOKEN.startswith("hf_") and len(HF_TOKEN) > 20:
        print_success("HF Token loaded successfully")
    else:
        print_error("Invalid or missing HF token")
        sys.exit(1)
    print()
    
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN}BATTLE STARTED - MAY THE BEST MODEL WIN!{Colors.RESET}")
    print("=" * 70 + "\n")
    
    success_count = 0
    fail_count = 0
    
    if PARALLEL_MODE:
        print(f"{Colors.BOLD}Starting parallel execution with {len(competitors)} models...{Colors.RESET}\n")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_model, model_id): model_id for model_id in competitors}
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    model_id, success, result = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        log(f"Failed: {model_id} - {result}")
                    
                    print(f"{Colors.CYAN}[Progress] {i+1}/{len(competitors)} completed{Colors.RESET}")
                    
                except Exception as e:
                    fail_count += 1
                    print_error(f"Task failed: {str(e)[:80]}")
                    log(f"Task exception: {str(e)}")
        
        time.sleep(DELAY_SECONDS)
    
    else:
        for i, model_id in enumerate(competitors, 1):
            print(f"\n[{i}/{len(competitors)}] Processing: {model_id}")
            log(f"[{i}/{len(competitors)}] Processing: {model_id}")
            
            result = process_single_model(model_id)
            model_id, success, res = result
            
            if success:
                success_count += 1
            else:
                fail_count += 1
            
            if ENABLE_GC:
                gc.collect()
            
            print_info(f"Waiting {DELAY_SECONDS} seconds before next model...")
            time.sleep(DELAY_SECONDS)
    
    if LOG_ERRORS:
        log_handle.write(f"\n{'='*50}\n")
        log_handle.write(f"SUMMARY\n")
        log_handle.write(f"{'='*50}\n")
        log_handle.write(f"Success: {success_count}/{len(competitors)}\n")
        log_handle.write(f"Failed: {fail_count}/{len(competitors)}\n")
        log_handle.close()
        print(f"\nError log saved to: {log_file}")
    
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN}BATTLE COMPLETE!{Colors.RESET}")
    print("=" * 70)
    print(f"\n{Colors.WHITE}Results:{Colors.RESET}")
    print(f"  ✓ Successful: {success_count}/{len(competitors)}")
    print(f"  ✗ Failed: {fail_count}/{len(competitors)}")
    print(f"\n{Colors.CYAN}Review your .ipynb files in 'generated_notebooks' folder.{Colors.RESET}\n")

if __name__ == "__main__":
    main()
