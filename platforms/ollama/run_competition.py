"""
Jane Street Battle Royale - Ollama Cloud API Edition

Uses Ollama Cloud API (https://ollama.com/v1) with your API keys.
No local model download needed - all inference happens in the cloud.
"""

import os
import sys
import time
import gc
import datetime
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import nbformat as nbf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from httpx import HTTPStatusError
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# ==============================================================================
# 1. ENVIRONMENT LOADING
# ==============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_DIR = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env")

# Account rotation configuration
ACCOUNT_ROTATION = [
    {"email": "Account 1", "env_key": "CLOUD_KEY_1"},
    {"email": "Account 2", "env_key": "CLOUD_KEY_2"},
    {"email": "Account 3", "env_key": "CLOUD_KEY_3"},
]

# Ollama Cloud API configuration
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"

# Available Ollama Cloud models
OLLAMA_CLOUD_MODELS = [
    "qwen3-next:80b",
    "deepseek-v3.2",
    "gemma3:27b",
    "gemma3:12b",
    "glm-5",
    "glm-4.6",
    "kimi-k2-thinking",
    "mistral-large-3:675b",
    "ministral-3:14b",
    "ministral-3:8b",
    "nemotron-3-nano:30b",
]

# Test mode
TEST_MODE = False
TEST_LIMIT = 5

# Auto-rotation mode (no manual intervention)
AUTO_ROTATE = False

# Parallel execution mode (use all keys simultaneously)
PARALLEL_MODE = False
MAX_WORKERS = 3

# Single key mode (use only CLOUD_KEY_2)
SINGLE_KEY_MODE = False

# Safety settings
DELAY_SECONDS = 3
ENABLE_GC = True
LOG_ERRORS = True
REQUEST_TIMEOUT = 300  # 5 minutes for cloud model responses
MAX_RETRIES_PER_KEY = 1

# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================

if LOG_ERRORS:
    log_file = PLATFORM_DIR / f"competition_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

def print_rate_limit_warning(current_email, next_email):
    print("\n" + "=" * 70)
    print(f"{Colors.RED}{Colors.BOLD}🚨 RATE LIMIT HIT on [{current_email}]{Colors.RESET}")
    print("=" * 70)
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Please switch to the next account:{Colors.RESET}")
    print(f"\n  👉 {Colors.CYAN}{Colors.BOLD}{next_email}{Colors.RESET}\n")
    print(f"{Colors.MAGENTA}Waiting for you to switch accounts...{Colors.RESET}")
    print(f"{Colors.WHITE}Press Enter ONLY after you have successfully logged in...{Colors.RESET}")
    print("=" * 70 + "\n")

# ==============================================================================
# 4. API ROTATOR
# ==============================================================================

class CloudAPIRotator:
    """Manages Ollama Cloud API key rotation."""

    def __init__(self, accounts, base_url, model_name, account_index=None):
        self.accounts = accounts
        self.base_url = base_url
        self.model_name = model_name
        self.current_index = account_index if account_index is not None else 0
        self.current_key = None
        self.llm = None
        self._load_current_key()

    def _load_current_key(self):
        account = self.accounts[self.current_index]
        self.current_key = os.getenv(account["env_key"])
        if not self.current_key:
            print_error(f"API key not found for {account['email']}")
            sys.exit(1)
        self._create_llm_client()
        if not PARALLEL_MODE:
            print_info(f"Using account: {account['email']}")

    def _create_llm_client(self):
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.current_key,
            base_url=self.base_url,
            temperature=0.1,
            max_tokens=4096,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )

    def get_current_email(self):
        return self.accounts[self.current_index]["email"]

    def get_next_email(self):
        next_index = (self.current_index + 1) % len(self.accounts)
        return self.accounts[next_index]["email"]

    def switch_to_next_account(self):
        old_email = self.get_current_email()
        self.current_index = (self.current_index + 1) % len(self.accounts)
        self._load_current_key()
        print_info(f"Switched from {old_email} to {self.get_current_email()}")

    def invoke_with_rotation(self, prompt_template, inputs, max_total_retries=10):
        total_attempts = 0
        keys_tried = set()

        while total_attempts < max_total_retries:
            try:
                chain = prompt_template | self.llm
                response = chain.invoke(inputs)
                return response

            except Exception as e:
                error_str = str(e)
                total_attempts += 1

                is_rate_limit = (
                    "429" in error_str or
                    "Rate Limit" in error_str or
                    "rate_limit" in error_str
                )

                is_timeout = (
                    "timeout" in error_str.lower() or
                    "timed out" in error_str.lower() or
                    "504" in error_str
                )

                is_auth_error = (
                    "401" in error_str or
                    "403" in error_str or
                    "Unauthorized" in error_str or
                    "Invalid API key" in error_str
                )

                if is_rate_limit or is_timeout or is_auth_error:
                    current_key_idx = self.current_index
                    keys_tried.add(current_key_idx)
                    
                    self.switch_to_next_account()
                    log(f"Auto-switched to {self.get_current_email()}")
                    
                    if len(keys_tried) >= len(self.accounts):
                        log(f"All {len(self.accounts)} keys failed for model {self.model_name}")
                        raise Exception(f"All {len(self.accounts)} API keys failed")
                    
                    if AUTO_ROTATE or PARALLEL_MODE:
                        time.sleep(2)
                        continue
                    else:
                        print_rate_limit_warning(
                            current_email=self.accounts[(self.current_index - 1) % len(self.accounts)]['email'],
                            next_email=self.get_current_email()
                        )
                        input(f"{Colors.BOLD}>>> Press Enter to continue...{Colors.RESET}")
                        time.sleep(2)
                        continue
                else:
                    log(f"Non-rotatable error: {error_str}")
                    raise

        raise Exception(f"Exhausted {max_total_retries} rotation attempts.")

# ==============================================================================
# 5. PROMPT TEMPLATE
# ==============================================================================

def get_prompt_template():
    task_instruction = """Write a complete Python script to solve the Jane Street Market Forecasting challenge.

ENVIRONMENT:
- Data Path: './jane-street-real-time-market-data-forecasting/train.parquet'
- Libraries: Use `polars` for speed and `xgboost` for modeling.

LOGIC REQUIREMENTS:
1. Load 'train.parquet' using Polars.
2. Feature Engineering: Calculate a global TOP_QUANTILE (top 15%) of 'feature_00' relative to 'responder_6' across rolling batches of 'date_id'.
3. Train an XGBoost Regressor on the target 'responder_6'.

OUTPUT:
- Pure Python code only. No markdown. No explanations.
"""

    return ChatPromptTemplate.from_messages([
        ("system", "You are an elite Quant Developer. Output ONLY pure Python code."),
        ("user", task_instruction)
    ])

# ==============================================================================
# 6. NOTEBOOK GENERATION
# ==============================================================================

def process_single_model(args):
    """Process a single model with a specific account - for parallel execution"""
    model_id, account_index = args
    prompt_template = get_prompt_template()
    
    print(f"\n{Colors.BLUE}[Parallel] Processing: {model_id} with {ACCOUNT_ROTATION[account_index]['email']}{Colors.RESET}")
    log(f"[Parallel] Processing: {model_id} with {ACCOUNT_ROTATION[account_index]['email']}")
    
    rotator = CloudAPIRotator(
        accounts=ACCOUNT_ROTATION,
        base_url=OLLAMA_CLOUD_BASE_URL,
        model_name=model_id,
        account_index=account_index
    )
    
    response = None
    failed = False
    
    try:
        response = rotator.invoke_with_rotation(
            prompt_template=prompt_template,
            inputs={},
            max_total_retries=5
        )
    except Exception as e:
        error_str = str(e)
        print_error(f"Failed: {model_id} - {error_str[:80]}")
        log(f"ERROR {model_id}: {error_str}")
        failed = True
    
    if not failed and response:
        try:
            code_content = extract_code_from_response(response.content)
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
    nb.cells.append(nbf.v4.new_markdown_cell(f"# Results for model: {model_id}"))
    nb.cells.append(nbf.v4.new_code_cell(code_content))
    return nb

def save_notebook(nb, model_id, output_dir=None):
    output_dir = Path(output_dir) if output_dir else PLATFORM_DIR / "generated_notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_id.replace("/", "_").replace(":", "_").replace("-", "_")
    fname = output_dir / f"{safe_name}.ipynb"

    with open(fname, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    return fname

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

def main():
    global AUTO_ROTATE, PARALLEL_MODE, SINGLE_KEY_MODE

    parser = argparse.ArgumentParser(description='Jane Street Battle Royale - Ollama Cloud')
    parser.add_argument('--auto-rotate', action='store_true', help='Enable automatic account rotation')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel execution using all API keys')
    parser.add_argument('--single-key', action='store_true', help='Use only CLOUD_KEY_2')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    AUTO_ROTATE = args.auto_rotate
    PARALLEL_MODE = args.parallel
    SINGLE_KEY_MODE = args.single_key

    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.CYAN}JANE STREET BATTLE ROYALE - OLLAMA CLOUD API{Colors.RESET}")
    print("=" * 70)
    print(f"\n{Colors.WHITE}API Base URL: {OLLAMA_CLOUD_BASE_URL}{Colors.RESET}")
    print(f"{Colors.WHITE}Total Cloud Models: {len(OLLAMA_CLOUD_MODELS)}{Colors.RESET}")
    print(f"\n{Colors.BOLD}Account Rotation Order:{Colors.RESET}")
    for i, account in enumerate(ACCOUNT_ROTATION, 1):
        print(f"  {i}. {account['email']} ({account['env_key']})")
    print()

    if SINGLE_KEY_MODE:
        print_info(f"{Colors.GREEN}SINGLE KEY MODE: Using CLOUD_KEY_2 only{Colors.RESET}")
    elif PARALLEL_MODE:
        print_info(f"{Colors.GREEN}PARALLEL MODE: Using all {len(ACCOUNT_ROTATION)} API keys simultaneously{Colors.RESET}")
    elif AUTO_ROTATE:
        print_info(f"{Colors.GREEN}AUTO-ROTATION MODE: Enabled{Colors.RESET}")
    else:
        print_warning("MANUAL MODE: Will pause for account switching")
    print()

    competitors = OLLAMA_CLOUD_MODELS.copy()

    if TEST_MODE or args.test:
        competitors = competitors[:TEST_LIMIT]
        print_warning(f"TEST MODE: Running {len(competitors)} models only")
    else:
        print_success(f"FULL COMPETITION: {len(competitors)} cloud models locked in")

    print("\nModels in the arena:")
    for i, m in enumerate(competitors, 1):
        print(f"  {i}. {m}")
    print()

    print_info("Validating API keys...")
    if SINGLE_KEY_MODE:
        account = ACCOUNT_ROTATION[1]
        key = os.getenv(account["env_key"])
        if key and key != "sk-..." and len(key) > 20:
            print_success(f"{account['email']}: Key loaded")
        else:
            print_error(f"{account['email']}: Invalid or missing key")
            sys.exit(1)
    else:
        for account in ACCOUNT_ROTATION:
            key = os.getenv(account["env_key"])
            if key and key != "sk-..." and len(key) > 20:
                print_success(f"{account['email']}: Key loaded")
            else:
                print_error(f"{account['email']}: Invalid or missing key")
                sys.exit(1)
    print()

    prompt_template = get_prompt_template()

    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN}BATTLE STARTED - MAY THE BEST MODEL WIN!{Colors.RESET}")
    print("=" * 70 + "\n")

    success_count = 0
    fail_count = 0

    if PARALLEL_MODE:
        tasks = []
        for i, model_id in enumerate(competitors):
            account_index = i % len(ACCOUNT_ROTATION)
            tasks.append((model_id, account_index))
        
        print(f"{Colors.BOLD}Starting parallel execution with {len(tasks)} models across {len(ACCOUNT_ROTATION)} accounts...{Colors.RESET}\n")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_model, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    model_id, success, result = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        log(f"Failed: {model_id} - {result}")
                    
                    print(f"{Colors.CYAN}[Progress] {i+1}/{len(tasks)} completed{Colors.RESET}")
                    
                except Exception as e:
                    fail_count += 1
                    print_error(f"Task failed: {str(e)[:80]}")
                    log(f"Task exception: {str(e)}")
        
        time.sleep(DELAY_SECONDS)
    
    else:
        for i, model_id in enumerate(competitors, 1):
            print(f"\n[{i}/{len(competitors)}] Processing: {model_id}")
            log(f"[{i}/{len(competitors)}] Processing: {model_id}")

            account_index = 1 if SINGLE_KEY_MODE else None
            
            rotator = CloudAPIRotator(
                accounts=ACCOUNT_ROTATION,
                base_url=OLLAMA_CLOUD_BASE_URL,
                model_name=model_id,
                account_index=account_index
            )

            response = None
            failed = False

            try:
                response = rotator.invoke_with_rotation(
                    prompt_template=prompt_template,
                    inputs={}
                )

            except Exception as e:
                error_str = str(e)
                print_error(f"Failed after all attempts: {error_str[:100]}")
                log(f"ERROR: {error_str}")
                fail_count += 1
                failed = True

            if not failed and response:
                try:
                    code_content = extract_code_from_response(response.content)
                    nb = create_notebook(code_content, model_id)
                    fname = save_notebook(nb, model_id)

                    print_success(f"Created: {fname}")
                    log(f"Notebook saved: {fname}")
                    success_count += 1

                except Exception as e:
                    print_error(f"Failed to process response: {str(e)[:100]}")
                    log(f"Processing error: {str(e)}")
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
