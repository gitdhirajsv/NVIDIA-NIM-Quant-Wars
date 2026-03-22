import os
import time
import nbformat as nbf
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. AUTHENTICATION & SETUP
# ==========================================
# IMPORTANT: Set your API key via environment variable or paste below
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY", "PASTE_YOUR_NVIDIA_KEY_HERE")

# ==========================================
# 2. MODEL DISCOVERY (From your CSV)
# ==========================================
competitors = []
try:
    with open('nvidia_nim_models.csv', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 5)
            if len(parts) >= 1:
                model_name = parts[0]
                # Filter for models capable of coding/logic
                if any(k in model_name.lower() for k in ['instruct', 'chat', 'coder', 'nemotron']):
                    competitors.append(model_name)
    print(f"Arena Ready: {len(competitors)} NIM models discovered.")
except FileNotFoundError:
    print("ERROR: nvidia_nim_models.csv not found in root folder.")
    exit()

# ==========================================
# 3. THE QUANT CHALLENGE PROMPT
# ==========================================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an elite Quant Developer. You only output pure Python code."),
    ("user", """
    Write a complete Python script to solve the Jane Street Market Forecasting challenge.

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
# 4. EXECUTION LOOP (40 RPM Safe)
# ==========================================
for model_id in competitors:
    print(f"--- Calling: {model_id} ---")
    try:
        llm = ChatNVIDIA(model=model_id)
        chain = prompt_template | llm
        response = chain.invoke({})
        code_content = response.content

        # Clean up code if AI used backticks
        if "```" in code_content:
            code_content = code_content.split("```")[1].replace("python", "").split("```")[0].strip()

        # Wrap into Jupyter Notebook
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# Results for model: {model_id}"))
        nb.cells.append(nbf.v4.new_code_cell(code_content))

        # Save file
        safe_name = model_id.replace('/', '_').replace(':', '_')
        file_path = f"{safe_name}_solution.ipynb"
        with open(file_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"SUCCESS: Created {file_path}")

    except Exception as e:
        print(f"SKIPPED {model_id}: {str(e)}")

    print("Waiting 2 seconds (Rate Limit Control)...")
    time.sleep(2)

print("\nBATTLE COMPLETE. Review your .ipynb files.")
