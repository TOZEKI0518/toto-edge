import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


MODELS = (
    "rf",
    "ensemble",
    "dynamic",
)

PREDICTION_PATH = Path(
    "ml/round_predictions.csv"
)

PLAN_PATH = Path(
    "ml/budget_ticket_plan_beam.csv"
)

PROBABILITY_PATH = Path(
    "ml/probability_summary.json"
)

OUTPUT_PATH = Path(
    "ml/round_model_comparison.csv"
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


def calculate_actual_coverage(
    plan: pd.DataFrame,
) -> tuple[int, int]:
    if "actual" not in plan.columns:
        return 0, 0

    valid = plan[
        plan["actual"].isin(
            ["H", "D", "A"]
        )
    ].copy()

    if valid.empty:
        return 0, 0

    covered = valid.apply(
        lambda row: (
            str(row["actual"])
            in str(row["ticket"])
        ),
        axis=1,
    )

    return (
        int(covered.sum()),
        int(len(valid)),
    )


def calculate_ticket_count(
    plan: pd.DataFrame,
) -> int:
    total = 1

    for ticket in plan["ticket"]:
        size = len(
            {
                character
                for character in str(ticket)
                if character in {
                    "H",
                    "D",
                    "A",
                }
            }
        )

        total *= size

    return total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--round",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
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

    comparison_rows = []

    for model_name in MODELS:
        predict_command = [
            "python",
            "ml/predict_round.py",
            "--round",
            str(args.round),
            "--model",
            model_name,
        ]

        if model_name in {
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
                str(PLAN_PATH),
            ]
        )

        model_directory = Path(
            f"ml/comparison_round_{args.round}"
        )

        model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            PREDICTION_PATH,
            model_directory
            / f"predictions_{model_name}.csv",
        )

        shutil.copy2(
            PLAN_PATH,
            model_directory
            / f"ticket_plan_{model_name}.csv",
        )

        shutil.copy2(
            PROBABILITY_PATH,
            model_directory
            / f"probability_{model_name}.json",
        )

        predictions = pd.read_csv(
            PREDICTION_PATH
        )

        plan = pd.read_csv(
            PLAN_PATH
        )

        probability = json.loads(
            PROBABILITY_PATH.read_text(
                encoding="utf-8"
            )
        )

        total_tickets = (
            calculate_ticket_count(plan)
        )

        investment = total_tickets * 100

        covered_matches, actual_matches = (
            calculate_actual_coverage(plan)
        )

        prediction_accuracy = None

        if (
            "actual" in predictions.columns
            and predictions["actual"].isin(
                ["H", "D", "A"]
            ).any()
        ):
            completed = predictions[
                predictions["actual"].isin(
                    ["H", "D", "A"]
                )
            ]

            prediction_accuracy = float(
                (
                    completed["prediction"]
                    == completed["actual"]
                ).mean()
            )

        comparison_rows.append(
            {
                "roundNo": args.round,
                "model": model_name,
                "tickets": total_tickets,
                "investment": investment,
                "averageCoverage": probability[
                    "averageCoverage"
                ],
                "prize1Probability": probability.get(
                    "prize1Probability"
                ),
                "prize2Probability": probability.get(
                    "prize2Probability"
                ),
                "prize3Probability": probability.get(
                    "prize3Probability"
                ),
                "prize2OrBetter": probability.get(
                    "prize2OrBetterProbability"
                ),
                "prize3OrBetter": probability.get(
                    "prize3OrBetterProbability"
                ),
                "predictionAccuracy": (
                    prediction_accuracy
                ),
                "coveredActualMatches": (
                    covered_matches
                ),
                "actualMatches": (
                    actual_matches
                ),
            }
        )

    result = pd.DataFrame(
        comparison_rows
    )

    result = result.sort_values(
        [
            "prize1Probability",
            "prize3OrBetter",
            "predictionAccuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print(
        f"Round {args.round} "
        "Model Comparison"
    )
    print("=" * 72)

    display_columns = [
        "model",
        "tickets",
        "investment",
        "averageCoverage",
        "prize1Probability",
        "prize2OrBetter",
        "prize3OrBetter",
        "predictionAccuracy",
        "coveredActualMatches",
        "actualMatches",
    ]

    print(
        result[
            display_columns
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()