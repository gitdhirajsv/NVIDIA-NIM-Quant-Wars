"""
Unified evaluation for Jane Street Quant Wars.

Scans notebook reports from the NVIDIA NIM, Ollama Cloud, and Hugging Face
platform folders, scores the generated code on responder_6, and writes both
leaderboard.csv and unified_leaderboard.csv.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import nbformat
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILES = [
    BASE_DIR / "leaderboard.csv",
    BASE_DIR / "unified_leaderboard.csv",
]
SAMPLE_ROWS = 100_000

PLATFORM_NOTEBOOK_DIRS = {
    "nvidia": [
        BASE_DIR / "platforms" / "nvidia" / "generated_notebooks",
        BASE_DIR / "generated_notebooks",
    ],
    "ollama": [
        BASE_DIR / "platforms" / "ollama" / "generated_notebooks",
    ],
    "huggingface": [
        BASE_DIR / "platforms" / "huggingface" / "generated_notebooks",
    ],
}

DATA_PATH_CANDIDATES = [
    BASE_DIR / "jane_street_data" / "train.parquet",
    BASE_DIR.parent / "jane_street_data" / "train.parquet",
    BASE_DIR.parent / "jane-street-real-time-market-data-forecasting" / "train.parquet",
]


def resolve_train_path() -> Path | None:
    for candidate in DATA_PATH_CANDIDATES:
        if candidate.is_file() or candidate.is_dir():
            return candidate
    return None


def load_data():
    train_path = resolve_train_path()

    if train_path is None:
        print("Real data not found. Generating synthetic data for evaluation.")
        np.random.seed(42)
        n = 50_000
        date_ids = np.random.randint(0, 200, n)
        features = {f"feature_{i:02d}": np.random.randn(n) for i in range(79)}
        target = (
            0.3 * features["feature_00"]
            + 0.2 * features["feature_01"]
            + 0.15 * features["feature_02"]
            + np.random.randn(n) * 0.5
        )
        df = pl.DataFrame(
            {
                "date_id": date_ids,
                "weight": np.random.uniform(0.5, 2, n),
                "responder_6": target,
                **features,
            }
        )
        return df, "SYNTHETIC"

    if train_path.is_dir():
        print(f"Loading partitioned dataset from: {train_path}")
        df = pl.scan_parquet(str(train_path / "**" / "*.parquet")).collect()
    else:
        print(f"Loading dataset from: {train_path}")
        df = pl.read_parquet(train_path)

    if SAMPLE_ROWS and len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, seed=42)
        print(f"Sampled to {SAMPLE_ROWS:,} rows for speed")
    else:
        print(f"Loaded {len(df):,} rows")

    return df, "REAL"


def extract_code(ipynb_path: Path) -> str:
    with open(ipynb_path, "r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    code_cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.strip()
    ]
    return "\n\n".join(code_cells)


def evaluate_model_code(platform: str, notebook_path: Path, code: str, df: pl.DataFrame):
    result = {
        "platform": platform,
        "model": notebook_path.stem,
        "source_notebook": str(notebook_path.relative_to(BASE_DIR)),
        "status": "FAIL",
        "mse": None,
        "rmse": None,
        "r2": None,
        "feature_count": None,
        "code_lines": len([line for line in code.splitlines() if line.strip()]),
        "has_feature_engineering": False,
        "has_xgboost": False,
        "has_polars": False,
        "error": None,
    }

    result["has_feature_engineering"] = any(
        token in code.lower()
        for token in ["quantile", "rolling", "groupby", "group_by", "lag", "shift", "rank", "mean", "std"]
    )
    result["has_xgboost"] = "xgboost" in code.lower() or "xgb" in code.lower()
    result["has_polars"] = "polars" in code.lower() or "import pl" in code.lower()

    try:
        feature_cols = [column for column in df.columns if column.startswith("feature_")]

        if "quantile" in code and any(token in code for token in ["date_id", "group_by", "groupby"]):
            thresholds = df.group_by("date_id").agg(
                pl.col("feature_00").quantile(0.85).alias("fq")
            )
            df_eval = df.join(thresholds, on="date_id").with_columns(
                pl.when(pl.col("feature_00") >= pl.col("fq"))
                .then(1)
                .otherwise(0)
                .alias("fq_flag")
            )
            use_cols = feature_cols + ["fq_flag"]
        elif "quantile" in code:
            threshold = float(df["feature_00"].quantile(0.85))
            df_eval = df.with_columns(
                pl.when(pl.col("feature_00") >= threshold)
                .then(1)
                .otherwise(0)
                .alias("fq_flag")
            )
            use_cols = feature_cols + ["fq_flag"]
        else:
            df_eval = df
            use_cols = feature_cols

        n_estimators = 100
        max_depth = 6
        learning_rate = 0.1

        n_estimators_match = re.search(r"n_estimators\s*=\s*(\d+)", code)
        max_depth_match = re.search(r"max_depth\s*=\s*(\d+)", code)
        learning_rate_match = re.search(r"learning_rate\s*=\s*([\d.]+)", code)

        if n_estimators_match:
            n_estimators = min(int(n_estimators_match.group(1)), 300)
        if max_depth_match:
            max_depth = min(int(max_depth_match.group(1)), 10)
        if learning_rate_match:
            learning_rate = float(learning_rate_match.group(1))

        x = df_eval.select(use_cols).to_numpy()
        y = df_eval["responder_6"].to_numpy()

        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

        predictions = model.predict(x_valid)
        mse = float(mean_squared_error(y_valid, predictions))
        rmse = float(np.sqrt(mse))
        ss_res = np.sum((y_valid - predictions) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        result["feature_count"] = len(use_cols)
        result["mse"] = round(mse, 6)
        result["rmse"] = round(rmse, 6)
        result["r2"] = round(r2, 6)
        result["status"] = "OK"
    except Exception as exc:
        result["error"] = str(exc)[:200]
        result["status"] = "EVAL_ERROR"

    return result


def iter_notebooks():
    seen = set()

    for platform, directories in PLATFORM_NOTEBOOK_DIRS.items():
        for directory in directories:
            if not directory.exists():
                continue

            for notebook_path in sorted(directory.glob("*.ipynb")):
                if notebook_path.name.startswith(("ERROR_", "00_")):
                    continue

                resolved = notebook_path.resolve()
                if resolved in seen:
                    continue

                seen.add(resolved)
                yield platform, notebook_path


def write_csv(results):
    fieldnames = [
        "rank",
        "platform",
        "model",
        "status",
        "mse",
        "rmse",
        "r2",
        "feature_count",
        "code_lines",
        "has_feature_engineering",
        "has_xgboost",
        "has_polars",
        "source_notebook",
        "error",
    ]

    scored = [result for result in results if result["status"] == "OK" and result["mse"] is not None]
    failed = [result for result in results if result["status"] != "OK"]
    scored.sort(key=lambda item: item["mse"])

    ordered_results = []
    for rank, result in enumerate(scored, 1):
        row = {"rank": rank}
        row.update(result)
        ordered_results.append(row)

    for result in failed:
        row = {"rank": "-"}
        row.update(result)
        ordered_results.append(row)

    for output_file in OUTPUT_FILES:
        with open(output_file, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in ordered_results:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    return scored, failed


def print_summary(notebooks, scored, failed):
    print("\n" + "=" * 72)
    print("JANE STREET QUANT WARS - UNIFIED EVALUATION")
    print("=" * 72)
    print(f"Notebook reports found: {len(notebooks)}")

    per_platform = {}
    for platform, _path in notebooks:
        per_platform[platform] = per_platform.get(platform, 0) + 1

    for platform in sorted(per_platform):
        print(f"  {platform:<12} {per_platform[platform]} notebooks")

    print(f"\nScored successfully: {len(scored)}")
    print(f"Failed to score:    {len(failed)}")

    if scored:
        print("\nTop 10 by MSE")
        print("-" * 72)
        for rank, result in enumerate(scored[:10], 1):
            print(
                f"{rank:>2}. "
                f"{result['platform']:<12} "
                f"{result['model']:<40} "
                f"MSE={result['mse']:.6f} "
                f"R2={result['r2']:.4f}"
            )

    print("\nWrote:")
    for output_file in OUTPUT_FILES:
        print(f"  {output_file}")


def main():
    notebooks = list(iter_notebooks())
    if not notebooks:
        print("No notebook reports found in platform folders.")
        return

    df, data_type = load_data()
    print(f"Evaluation data type: {data_type}")

    results = []
    for index, (platform, notebook_path) in enumerate(notebooks, 1):
        print(f"[{index:02d}/{len(notebooks)}] {platform}: {notebook_path.name}")

        try:
            code = extract_code(notebook_path)
            if not code.strip():
                results.append(
                    {
                        "platform": platform,
                        "model": notebook_path.stem,
                        "source_notebook": str(notebook_path.relative_to(BASE_DIR)),
                        "status": "EMPTY_NOTEBOOK",
                        "mse": None,
                        "rmse": None,
                        "r2": None,
                        "feature_count": 0,
                        "code_lines": 0,
                        "has_feature_engineering": False,
                        "has_xgboost": False,
                        "has_polars": False,
                        "error": "Empty notebook",
                    }
                )
                continue

            results.append(evaluate_model_code(platform, notebook_path, code, df))
        except Exception as exc:
            results.append(
                {
                    "platform": platform,
                    "model": notebook_path.stem,
                    "source_notebook": str(notebook_path.relative_to(BASE_DIR)),
                    "status": "READ_ERROR",
                    "mse": None,
                    "rmse": None,
                    "r2": None,
                    "feature_count": None,
                    "code_lines": 0,
                    "has_feature_engineering": False,
                    "has_xgboost": False,
                    "has_polars": False,
                    "error": str(exc)[:200],
                }
            )

    scored, failed = write_csv(results)
    print_summary(notebooks, scored, failed)


if __name__ == "__main__":
    main()
