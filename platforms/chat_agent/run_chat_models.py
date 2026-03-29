from __future__ import annotations

import csv
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(r"C:\Users\Administrator\Documents\jane-street-real-time-market-data-forecasting")
RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "agent_leaderboard.csv"

MODEL_NAME_MAP = {
    "claude_opus_thinking_jane_street.py": "Claude Opus (Thinking Model)",
    "claude_sonnet_jane_street.py": "Claude Sonnet 4.6 Max Reasoning",
    "deepseek_r1_jane_street.py": "DeepSeek-R1 (Chat)",
    "gemini_3_1_pro_jane_street.py": "Gemini 3.1 Pro",
    "gemini_3_1_pro_low_jane_street.py": "Gemini 3.1 Pro (Low)",
    "gemini_3_flash_jane_street.py": "Gemini 3 Flash",
    "glm5_turbo_jane_street.py": "GLM-5 Turbo (Chat)",
    "gpt_5_2_jane_street.py": "GPT 5.2 Extra High Codex",
    "gpt_5_3_codex_jane_street.py": "GPT 5.3 Extra High Codex",
    "gpt_5_4_xh_jane_street.py": "GPT 5.4 Extra High Code",
    "kimi2.5_jane_street.py": "Kimi 2.5 Agent",
    "qwen_3_5_jane_street.py": "Qwen 3.5 (Chat)",
}

R2_PATTERN = re.compile(r"Out-of-Sample R2:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
MSE_PATTERN = re.compile(r"Out-of-Sample MSE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def extract_metric(pattern: re.Pattern[str], text: str) -> float:
    match = pattern.search(text)
    if match is None:
        return math.nan
    return float(match.group(1))


def run_script(script_name: str, display_name: str) -> dict[str, object]:
    script_path = ROOT_DIR / script_name
    log_path = RESULTS_DIR / f"{Path(script_name).stem}.log"
    started_at = datetime.now(timezone.utc)

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    combined_output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    log_path.write_text(combined_output, encoding="utf-8")

    finished_at = datetime.now(timezone.utc)
    r2_value = extract_metric(R2_PATTERN, combined_output)
    mse_value = extract_metric(MSE_PATTERN, combined_output)

    return {
        "filename": script_name,
        "model_name": display_name,
        "return_code": completed.returncode,
        "status": "ok" if completed.returncode == 0 and not math.isnan(r2_value) and not math.isnan(mse_value) else "failed",
        "out_of_sample_r2": r2_value,
        "out_of_sample_mse": mse_value,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "log_file": log_path.name,
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "filename",
        "model_name",
        "return_code",
        "status",
        "out_of_sample_r2",
        "out_of_sample_mse",
        "started_at_utc",
        "finished_at_utc",
        "log_file",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, object]] = []

    for script_name, display_name in MODEL_NAME_MAP.items():
        print(f"Running {display_name} from {script_name} ...", flush=True)
        row = run_script(script_name, display_name)
        rows.append(row)
        write_csv(rows)
        print(
            f"Finished {display_name}: status={row['status']} "
            f"r2={row['out_of_sample_r2']} mse={row['out_of_sample_mse']}",
            flush=True,
        )
        time.sleep(30)

    print(f"Wrote isolated leaderboard to {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
