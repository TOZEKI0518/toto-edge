import json
from pathlib import Path

import pandas as pd


EV = Path(
    "ml/expected_value_summary.json"
)

EXPERIMENTS = Path(
    "ml/experiments_v2.csv"
)

PROBABILITY = Path(
    "ml/probability_summary.json"
)

PREDICTIONS = Path(
    "ml/round_predictions.csv"
)

TICKET_PLAN = Path(
    "ml/budget_ticket_plan_beam.csv"
)


print("=" * 64)
print("                Toto AI Dashboard")
print("=" * 64)
print()


# ----------------------------------------
# Latest Experiment
# ----------------------------------------

if EXPERIMENTS.exists():
    try:
        exp = pd.read_csv(
            EXPERIMENTS,
            on_bad_lines="skip",
        )

        if not exp.empty:
            latest = exp.iloc[-1]

            print("Latest Experiment")
            print("-" * 44)

            for column in latest.index:
                print(
                    f"{column:<24}"
                    f"{latest[column]}"
                )

            print()

        else:
            print(
                "experiments_v2.csv is empty"
            )
            print()

    except Exception as error:
        print(
            "experiments_v2.csv "
            f"read error: {error}"
        )
        print()

else:
    print(
        "experiments_v2.csv not found"
    )
    print()


# ----------------------------------------
# Current Round / Difficulty
# ----------------------------------------

if PREDICTIONS.exists():
    try:
        predictions = pd.read_csv(
            PREDICTIONS
        )

        if not predictions.empty:
            round_number = (
                predictions[
                    "roundNo"
                ].iloc[0]
                if "roundNo"
                in predictions.columns
                else "Unknown"
            )

            model_name = (
                predictions[
                    "model"
                ].iloc[0]
                if "model"
                in predictions.columns
                else "Unknown"
            )

            print("Current Prediction")
            print("-" * 44)

            print(
                f"Round               : "
                f"{round_number}"
            )

            print(
                f"Model               : "
                f"{model_name}"
            )

            print(
                f"Matches             : "
                f"{len(predictions)}"
            )

            if (
                "difficulty"
                in predictions.columns
            ):
                print(
                    f"Average Difficulty  : "
                    f"{predictions['difficulty'].mean():.4f}"
                )

                print()
                print(
                    "Most Difficult Matches"
                )
                print("-" * 44)

                difficult_columns = [
                    "homeTeam",
                    "awayTeam",
                    "difficulty",
                ]

                if (
                    "difficultyStars"
                    in predictions.columns
                ):
                    difficult_columns.append(
                        "difficultyStars"
                    )

                if (
                    "difficultyClass"
                    in predictions.columns
                ):
                    difficult_columns.append(
                        "difficultyClass"
                    )

                difficult = (
                    predictions.sort_values(
                        "difficulty",
                        ascending=False,
                    )[
                        difficult_columns
                    ].head(5)
                )

                print(
                    difficult.to_string(
                        index=False
                    )
                )

            print()

    except Exception as error:
        print(
            "round_predictions.csv "
            f"read error: {error}"
        )
        print()

else:
    print(
        "round_predictions.csv not found"
    )
    print()


# ----------------------------------------
# Ticket Plan
# ----------------------------------------

if TICKET_PLAN.exists():
    try:
        ticket_plan = pd.read_csv(
            TICKET_PLAN
        )

        if not ticket_plan.empty:
            print("Ticket Plan")
            print("-" * 44)

            ticket_counts = (
                ticket_plan[
                    "ticketType"
                ].value_counts()
            )

            for (
                ticket_type,
                count,
            ) in ticket_counts.items():
                print(
                    f"{ticket_type:<20}: "
                    f"{count}"
                )

            if (
                "difficulty"
                in ticket_plan.columns
            ):
                print(
                    f"Average Difficulty  : "
                    f"{ticket_plan['difficulty'].mean():.4f}"
                )

            print()

    except Exception as error:
        print(
            "budget_ticket_plan_beam.csv "
            f"read error: {error}"
        )
        print()


# ----------------------------------------
# Winning Probability
# ----------------------------------------

if PROBABILITY.exists():
    try:
        probability = json.loads(
            PROBABILITY.read_text(
                encoding="utf-8"
            )
        )

        print("Winning Probability")
        print("-" * 44)

        print(
            f"Matches             : "
            f"{probability['matches']}"
        )

        print(
            f"Average Coverage    : "
            f"{probability['averageCoverage']:.2%}"
        )

        if (
            "prize1Probability"
            in probability
        ):
            print(
                f"1st Prize 13/13     : "
                f"{probability['prize1Probability']:.6%}"
            )

            print(
                f"2nd Prize 12/13     : "
                f"{probability['prize2Probability']:.6%}"
            )

            print(
                f"3rd Prize 11/13     : "
                f"{probability['prize3Probability']:.6%}"
            )

            print(
                f"2nd or Better       : "
                f"{probability['prize2OrBetterProbability']:.6%}"
            )

            print(
                f"3rd or Better       : "
                f"{probability['prize3OrBetterProbability']:.6%}"
            )

        elif "prob1" in probability:
            print(
                f"1st Prize           : "
                f"{probability['prob1']:.6%}"
            )

            print(
                f"2nd Prize           : "
                f"{probability['prob2']:.6%}"
            )

            print(
                f"3rd Prize           : "
                f"{probability['prob3']:.6%}"
            )

        print()

    except Exception as error:
        print(
            "probability_summary.json "
            f"read error: {error}"
        )
        print()

else:
    print(
        "probability_summary.json not found"
    )
    print()


# ----------------------------------------
# Expected Value
# ----------------------------------------

if EV.exists():
    try:
        expected_value = json.loads(
            EV.read_text(
                encoding="utf-8"
            )
        )

        print("Expected Value")
        print("-" * 44)

        investment = expected_value.get(
            "investment",
            expected_value.get(
                "budget",
                0,
            ),
        )

        ticket_count = (
            expected_value.get(
                "ticketCount"
            )
        )

        if ticket_count is not None:
            print(
                f"Ticket Count         : "
                f"{ticket_count:,}"
            )

        print(
            f"Investment           : "
            f"{investment:,} yen"
        )

        print(
            f"Expected Return      : "
            f"{expected_value['expectedReturn']:,.0f} yen"
        )

        print(
            f"Expected Profit      : "
            f"{expected_value['expectedProfit']:,.0f} yen"
        )

        print(
            f"Expected ROI         : "
            f"{expected_value['roi']:.2f}%"
        )

        prize_source = (
            expected_value.get(
                "prizeSource",
                "unknown",
            )
        )

        print(
            f"Prize Source         : "
            f"{prize_source}"
        )

        print()
        print(
            f"Recommendation       : "
            f"{expected_value['recommendation']}"
        )

        if expected_value.get(
            "isSimulation",
            False,
        ):
            print()
            print(
                "WARNING: Prize amounts "
                "are temporary assumptions."
            )

        print()

    except Exception as error:
        print(
            "expected_value_summary.json "
            f"read error: {error}"
        )
        print()

else:
    print(
        "expected_value_summary.json not found"
    )
    print()


print("=" * 64)