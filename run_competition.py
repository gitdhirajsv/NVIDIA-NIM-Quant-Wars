import os
import time
import nbformat as nbf
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. API KEY SETUP
# ==========================================
os.environ["NVIDIA_API_KEY"] = "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"

# ==========================================
# 2. TEST MODE CONFIGURATION
# ==========================================
# Set to True for testing (runs only 2 models)
# Set to False for FULL COMPETITION (all 200+ models)
TEST_MODE = False
TEST_LIMIT = 2

# ==========================================
# 3. OFFICIAL MODEL DISCOVERY
# ==========================================
print("Fetching live model list from NVIDIA...")
try:
    all_models = ChatNVIDIA.get_available_models()
    competitors = [
        m.id for m in all_models 
        if any(k in m.id.lower() for k in ["instruct", "chat", "coder", "nemotron"])
        and not any(x in m.id.lower() for x in ["embed", "rerank", "vision", "reward", "safety"])
    ]
    
    if TEST_MODE:
        competitors = competitors[:TEST_LIMIT]
        print(f"\n*** TEST MODE: Running {len(competitors)} models ***\n")
    else:
        print(f"\n*** FULL COMPETITION: {len(competitors)} models locked in ***\n")
    
    print("Models in the arena:")
    for i, m in enumerate(competitors, 1):
        print(f"  {i}. {m}")
    print()

except Exception as e:
    print(f"CRITICAL ERROR: Could not fetch models. Check your API Key. Error: {e}")
    exit()

# ==========================================
# 4. DEFINE PROMPT
# ==========================================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an elite Quant Developer. Output ONLY pure Python code."),
    ("user", """Write a complete Python script to solve the Jane Street Market Forecasting challenge.

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
""")
])

# ==========================================
# 5. RUN LOOP (40 RPM Safe)
# ==========================================
print("=" * 60)
print("  BATTLE STARTED - MAY THE BEST MODEL WIN!")
print("=" * 60)
print()

for model_id in competitors:
    print(f"--- Calling: {model_id} ---")
    try:
        llm = ChatNVIDIA(model=model_id)
        chain = prompt_template | llm
        
        print("  -> Sending request to NVIDIA API...")
        response = chain.invoke({})
        print("  -> Response received!")
        
        code_content = response.content
        if "```" in code_content:
            code_content = code_content.split("```")[1].replace("python", "").split("```")[0].strip()
        
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# Results for model: {model_id}"))
        nb.cells.append(nbf.v4.new_code_cell(code_content))
        
        fname = f"{model_id.replace('/', '_').replace(':', '_')}.ipynb"
        with open(fname, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"  -> Created: {fname}\n")

    except Exception as e:
        print(f"  -> FAILED {model_id}: {e}\n")

    print("Waiting 2 seconds (Rate Limit Control)...")
    time.sleep(2)

print()
print("=" * 60)
print("  BATTLE COMPLETE. Review your .ipynb files.")
print("=" * 60)
