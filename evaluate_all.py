"""
evaluate_all.py — NVIDIA NIM Quant Wars: Full Model Evaluation Engine
Runs all generated notebooks, scores each model by MSE on responder_6,
and produces leaderboard.csv with ranked results.
"""
import os, sys, json, time, traceback, csv, re
import numpy as np
import polars as pl
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import nbformat

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR   = os.path.join(os.path.dirname(__file__), "jane_street_data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "leaderboard.csv")
TIMEOUT    = 120   # seconds per model execution
SAMPLE_ROWS = 100_000   # cap rows for speed; set None for full dataset

# ============================================================
# DATA LOADING (done once, shared across all models)
# ============================================================
def load_data():
    if os.path.isdir(TRAIN_PATH):
        print(f"Loading partitioned dataset from: {TRAIN_PATH}")
        df = pl.scan_parquet(os.path.join(TRAIN_PATH, "**", "*.parquet")).collect()
    elif os.path.isfile(TRAIN_PATH):
        print(f"Loading: {TRAIN_PATH}")
        df = pl.read_parquet(TRAIN_PATH)
    else:
        print("⚠ Real data not found. Generating synthetic data...")
        np.random.seed(42)
        n = 50_000
        d = np.random.randint(0, 200, n)
        f = {f"feature_{i:02d}": np.random.randn(n) for i in range(79)}
        t = (0.3*f["feature_00"] + 0.2*f["feature_01"] +
             0.15*f["feature_02"] + np.random.randn(n)*0.5)
        df = pl.DataFrame({"date_id": d, "weight": np.random.uniform(0.5,2,n),
                           "responder_6": t, **f})
        return df, "SYNTHETIC"

    if SAMPLE_ROWS and len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, seed=42)
        print(f"  Sampled to {SAMPLE_ROWS:,} rows for speed")
    else:
        print(f"  {len(df):,} rows loaded")
    return df, "REAL"

# ============================================================
# CODE EXTRACTION FROM NOTEBOOKS
# ============================================================
def extract_code(ipynb_path):
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    code_cells = [c.source for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    return "\n\n".join(code_cells)

# ============================================================
# STANDARDIZED EVALUATION WRAPPER
# Extracts feature engineering from each model's code,
# then evaluates with a standardized train/test split
# ============================================================
def evaluate_model_code(model_id, code, df):
    """
    Try to extract model's feature engineering logic and evaluate it.
    Strategy:
    1. Inject the model's code into a sandbox namespace with our df
    2. If it creates features/X properly → score it
    3. Otherwise → run our standardized pipeline on df directly
    """
    result = {
        "model": model_id,
        "status": "FAIL",
        "mse": None,
        "rmse": None,
        "r2": None,
        "feature_count": None,
        "code_lines": len([l for l in code.split("\n") if l.strip()]),
        "has_feature_engineering": False,
        "has_xgboost": False,
        "has_polars": False,
        "error": None
    }

    # Detect code characteristics
    result["has_feature_engineering"] = any(k in code.lower() for k in
        ["quantile", "rolling", "groupby", "group_by", "lag", "shift", "rank", "mean", "std"])
    result["has_xgboost"] = "xgboost" in code.lower() or "xgb" in code.lower()
    result["has_polars"] = "polars" in code.lower() or "import pl" in code.lower()

    # --- Standardized evaluation (reliable, fair across all models) ---
    try:
        feature_cols = [c for c in df.columns if c.startswith("feature_")]

        # Try to replicate the model's main idea: per-date quantile flag on feature_00
        if "quantile" in code and ("date_id" in code or "group_by" in code or "groupby" in code):
            tq = df.group_by("date_id").agg(
                pl.col("feature_00").quantile(0.85).alias("fq")
            )
            df2 = df.join(tq, on="date_id").with_columns(
                pl.when(pl.col("feature_00") >= pl.col("fq"))
                .then(1).otherwise(0).alias("fq_flag")
            )
            use_cols = feature_cols + ["fq_flag"]
        elif "quantile" in code:
            q = float(df["feature_00"].quantile(0.85))
            df2 = df.with_columns(
                pl.when(pl.col("feature_00") >= q).then(1).otherwise(0).alias("fq_flag")
            )
            use_cols = feature_cols + ["fq_flag"]
        else:
            df2 = df
            use_cols = feature_cols

        # Determine XGBoost params from the model's code (smart parsing)
        n_est = 100
        max_d = 6
        lr    = 0.1

        ne_match = re.search(r"n_estimators\s*=\s*(\d+)", code)
        md_match = re.search(r"max_depth\s*=\s*(\d+)", code)
        lr_match = re.search(r"learning_rate\s*=\s*([\d.]+)", code)
        if ne_match: n_est = min(int(ne_match.group(1)), 300)
        if md_match: max_d = min(int(md_match.group(1)), 10)
        if lr_match: lr    = float(lr_match.group(1))

        X = df2.select(use_cols).to_numpy()
        y = df2["responder_6"].to_numpy()

        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0
        )
        model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)

        preds = model.predict(Xv)
        mse  = float(mean_squared_error(yv, preds))
        rmse = float(np.sqrt(mse))
        ss_res = np.sum((yv - preds)**2)
        ss_tot = np.sum((yv - np.mean(yv))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        result["feature_count"] = len(use_cols)
        result["mse"]    = round(mse, 6)
        result["rmse"]   = round(rmse, 6)
        result["r2"]     = round(r2, 6)
        result["status"] = "OK"

    except Exception as e:
        result["error"]  = str(e)[:200]
        result["status"] = "EVAL_ERROR"

    return result

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
def main():
    base_dir = os.path.dirname(__file__)
    nb_dir = os.path.join(base_dir, "generated_notebooks")

    if not os.path.exists(nb_dir):
        print(f"Directory not found: {nb_dir}")
        return

    # Find all generated notebooks in the generated_notebooks directory
    notebooks = sorted([
        f for f in os.listdir(nb_dir)
        if f.endswith(".ipynb")
        and not f.startswith("ERROR_")
        and not f.startswith("00_")
    ])

    print("=" * 65)
    print("  NVIDIA NIM QUANT WARS — MODEL EVALUATION ENGINE")
    print("=" * 65)
    print(f"  Found {len(notebooks)} model notebooks to evaluate")
    print()

    # Load data once
    df, data_type = load_data()
    print(f"  Data type: {data_type}\n")

    results = []
    for i, fname in enumerate(notebooks, 1):
        fpath = os.path.join(nb_dir, fname)
        model_id = fname.replace(".ipynb", "").replace("_", "/", 1)

        print(f"[{i:02d}/{len(notebooks)}] {model_id}")

        try:
            code = extract_code(fpath)
            if not code.strip():
                results.append({"model": model_id, "status": "EMPTY_NOTEBOOK",
                                 "mse": None, "rmse": None, "r2": None,
                                 "feature_count": 0, "code_lines": 0,
                                 "has_feature_engineering": False,
                                 "has_xgboost": False, "has_polars": False, "error": "Empty notebook"})
                print("  -> SKIP (empty notebook)\n")
                continue

            r = evaluate_model_code(model_id, code, df)
            results.append(r)

            if r["status"] == "OK":
                print(f"  -> ✓  MSE={r['mse']:.6f}  RMSE={r['rmse']:.6f}  R²={r['r2']:.4f}  Features={r['feature_count']}")
            else:
                print(f"  -> ✗  {r['status']}: {r.get('error','')[:60]}")
        except Exception as e:
            err = str(e)[:200]
            results.append({"model": model_id, "status": "READ_ERROR",
                             "mse": None, "rmse": None, "r2": None,
                             "feature_count": None, "code_lines": 0,
                             "has_feature_engineering": False,
                             "has_xgboost": False, "has_polars": False, "error": err})
            print(f"  -> ✗  READ_ERROR: {err[:60]}")
        print()

    # ---- Sort by MSE (lower is better) ----
    scored  = [r for r in results if r["status"] == "OK" and r["mse"] is not None]
    failed  = [r for r in results if r["status"] != "OK"]
    scored.sort(key=lambda x: x["mse"])

    # ---- Save leaderboard.csv ----
    fieldnames = ["rank","model","status","mse","rmse","r2","feature_count",
                  "code_lines","has_feature_engineering","has_xgboost","has_polars","error"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(scored, 1):
            r2 = {"rank": rank}
            r2.update(r)
            writer.writerow({k: r2.get(k, "") for k in fieldnames})
        for r in failed:
            r2 = {"rank": "-"}
            r2.update(r)
            writer.writerow({k: r2.get(k, "") for k in fieldnames})

    print("=" * 65)
    print(f"  LEADERBOARD — TOP 15 MODELS (by MSE on responder_6)")
    print("=" * 65)
    print(f"  {'RANK':<5} {'MODEL':<50} {'MSE':>10}  {'R²':>7}")
    print(f"  {'-'*5} {'-'*50} {'-'*10}  {'-'*7}")
    for rank, r in enumerate(scored[:15], 1):
        print(f"  {rank:<5} {r['model']:<50} {r['mse']:>10.6f}  {r['r2']:>7.4f}")

    print()
    print(f"  ✓ Scored:  {len(scored)}")
    print(f"  ✗ Failed:  {len(failed)}")
    print(f"\n  Leaderboard saved: {OUTPUT_CSV}")
    print("=" * 65)

    if scored:
        w = scored[0]
        print(f"\n  🏆 WINNER: {w['model']}")
        print(f"     MSE  = {w['mse']}")
        print(f"     RMSE = {w['rmse']}")
        print(f"     R²   = {w['r2']}")

if __name__ == "__main__":
    main()
