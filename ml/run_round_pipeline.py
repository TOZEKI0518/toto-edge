from pathlib import Path
import argparse
import subprocess
from urllib.request import urlopen
from urllib.error import URLError


BASE_URL = (
    "http://localhost:3000"
    "/api/admin/export-training"
)

TOKEN = "toto-edge-admin-2026"

DATASET_PATH = Path(
    "toto-training-dataset.csv"
)

SUPPORTED_MODELS = (
    "rf",
    "lgbm",
    "ensemble",
    "dynamic",
)


def run(
    command: list[str],
) -> None:
    print()
    print("=" * 64)
    print(" ".join(command))
    print("=" * 64)

    subprocess.run(
        command,
        check=True,
    )


def download_dataset(
    from_round: int,
    to_round: int,
) -> None:
    url = (
        f"{BASE_URL}"
        f"?from={from_round}"
        f"&to={to_round}"
        f"&token={TOKEN}"
    )

    print("Download dataset:")
    print(url)

    try:
        with urlopen(
            url,
            timeout=300,
        ) as response:
            DATASET_PATH.write_bytes(
                response.read()
            )

    except URLError as error:
        raise RuntimeError(
            "Next.js server is not running "
            "or the dataset API is unavailable.\n"
            "Open another PowerShell and run:\n"
            "npm run dev"
        ) from error

    print(f"Saved: {DATASET_PATH}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--round",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--from-round",
        type=int,
        default=1506,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="dynamic",
    )

    parser.add_argument(
        "--rf-weight",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--lgbm-weight",
        type=float,
        default=0.3,
    )

    args = parser.parse_args()

    download_dataset(
        from_round=args.from_round,
        to_round=args.round,
    )

    predict_command = [
        "python",
        "ml/predict_round.py",
        "--round",
        str(args.round),
        "--model",
        args.model,
    ]

    if args.model in {
        "ensemble",
        "dynamic",
    }:
        predict_command.extend(
            [
                "--rf-weight",
                str(args.rf_weight),
                "--lgbm-weight",
                str(args.lgbm_weight),
            ]
        )

    run(predict_command)

    run(
        [
            "python",
            "ml/budget_optimizer_beam.py",
            "--budget",
            str(args.budget),
        ]
    )

    run(
        [
            "python",
            "ml/probability_estimator.py",
            "--input",
            "ml/budget_ticket_plan_beam.csv",
        ]
    )


if __name__ == "__main__":
    main()