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
# 2. OFFICIAL MODEL DISCOVERY
# ==========================================
print("Fetching live model list from NVIDIA...")
try:
    all_models = ChatNVIDIA.get_available_models()
    competitors = [
        m.id for m in all_models 
        if any(k in m.id.lower() for k in ["instruct", "chat", "coder", "nemotron"])
        and not any(x in m.id.lower() for x in ["embed", "rerank", "vision", "reward", "safety"])
    ]
    print(f"Found {len(competitors)} valid models. Starting Battle...\n")
    
    # Show first 10 models for verification
    print("Top Competitors Found:")
    for m in competitors[:10]:
        print(f" -> {m}")
    print()
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not fetch models. Check your API Key. Error: {e}")
    exit()

# ==========================================
# 3. DEFINE PROMPT
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
# 4. RUN LOOP (40 RPM Safe)
# ==========================================
for model_id in competitors:
    print(f"--- Calling: {model_id} ---")
    try:
        llm = ChatNVIDIA(model=model_id)
        chain = prompt_template | llm
        response = chain.invoke({})
        
        # Clean up code if AI used backticks
        code_content = response.content
        if "```" in code_content:
            code_content = code_content.split("```")[1].replace("python", "").split("```")[0].strip()
        
        # Save as Notebook
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# Results for model: {model_id}"))
        nb.cells.append(nbf.v4.new_code_cell(code_content))
        
        fname = f"{model_id.replace('/', '_').replace(':', '_')}.ipynb"
        with open(fname, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"Created: {fname}")
        
    except Exception as e:
        print(f"Failed {model_id}: {e}")
    
    print("Waiting 2 seconds (Rate Limit Control)...")
    time.sleep(2)

print("\nBATTLE COMPLETE. Review your .ipynb files.")
