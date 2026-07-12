from pathlib import Path
import subprocess
import re
import csv
import argparse
from datetime import datetime
from urllib.request import urlopen

BASE_URL = "http://localhost:3000/api/admin/export-training"
TOKEN = "toto-edge-admin-2026"
FROM_ROUND = 1506
TO_ROUND = 1633

DATASET_PATH = Path("toto-training-dataset.csv")
EXPERIMENT_LOG = Path("ml/experiments_v2.csv")


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, shell=False)
    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        print(output)
        raise RuntimeError(f"Command failed: {' '.join(command)}")

    return output


def download_dataset():
    url = (
        f"{BASE_URL}"
        f"?from={FROM_ROUND}"
        f"&to={TO_ROUND}"
        f"&league=j"
        f"&token={TOKEN}"
    )

    print("Step1: Download dataset")
    print(url)

    with urlopen(url, timeout=300) as response:
        DATASET_PATH.write_bytes(response.read())

    print(f"Saved: {DATASET_PATH}")
    print()


def extract_float(pattern: str, text: str):
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def extract_draw_accuracy(error_output: str):
    lines = error_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Accuracy by Actual Result":
            for next_line in lines[i + 1 : i + 8]:
                parts = next_line.split()
                if len(parts) == 2 and parts[0] == "D":
                    return parts[1]
    return ""


def append_experiment_log(rf_output: str, error_output: str, memo: str):
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "RandomForest",
        "rows": extract_float(r"Rows:\s*([0-9]+)", rf_output),
        "features": extract_float(r"Features:\s*([0-9]+)", rf_output),
        "mean_accuracy": extract_float(r"Mean Accuracy:\s*([0-9.]+)", rf_output),
        "overall_accuracy": extract_float(r"Overall Accuracy:\s*([0-9.]+)", rf_output),
        "best_window_accuracy": extract_float(r"Best Accuracy:\s*([0-9.]+)", rf_output),
        "worst_window_accuracy": extract_float(r"Worst Accuracy:\s*([0-9.]+)", rf_output),
        "draw_accuracy": extract_draw_accuracy(error_output),
        "memo": memo,
    }

    fieldnames = list(row.keys())
    file_exists = EXPERIMENT_LOG.exists()

    with EXPERIMENT_LOG.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Experiment log saved: {EXPERIMENT_LOG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", default="experiment")
    args = parser.parse_args()

    print("===================================")
    print("Experiment Runner")
    print("===================================")

    download_dataset()

    print("Step2: RandomForest Rolling Backtest")
    rf_output = run_command(["python", "ml/train_random_forest.py"])
    print(rf_output)

    print("Step3: Error Analysis")
    error_output = run_command(["python", "ml/analyze_errors.py"])
    print(error_output)

    print("Step4: Save experiment log")
    append_experiment_log(rf_output, error_output, args.memo)

    print()
    print("Done.")


if __name__ == "__main__":
    main()