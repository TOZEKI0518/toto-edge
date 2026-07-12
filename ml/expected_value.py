import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_PLAN = Path(
    "ml/budget_ticket_plan_beam.csv"
)

DEFAULT_PROBABILITY_SUMMARY = Path(
    "ml/probability_summary.json"
)

DEFAULT_EV_SUMMARY = Path(
    "ml/expected_value_summary.json"
)

UNIT_PRICE = 100


def recommendation(
    roi: float,
) -> str:
    if roi >= 150:
        return "★★★★★ BUY"

    if roi >= 120:
        return "★★★★ BUY"

    if roi >= 100:
        return "★★★ HOLD"

    if roi >= 80:
        return "★★ CAUTION"

    return "★ SKIP"


def ticket_size(ticket: str) -> int:
    valid_outcomes = {
        character
        for character in str(ticket).upper()
        if character in {"H", "D", "A"}
    }

    size = len(valid_outcomes)

    if size not in {1, 2, 3}:
        raise ValueError(
            f"Invalid ticket value: {ticket}"
        )

    return size


def calculate_ticket_count(
    df: pd.DataFrame,
) -> int:
    if "ticket" not in df.columns:
        raise ValueError(
            "Ticket plan does not contain "
            "'ticket' column."
        )

    sizes = [
        ticket_size(ticket)
        for ticket in df["ticket"]
    ]

    return math.prod(sizes)


def read_exact_probabilities(
    summary_path: Path,
) -> tuple[float, float, float]:
    if not summary_path.exists():
        raise FileNotFoundError(
            "Probability summary not found: "
            f"{summary_path}\n"
            "Run probability_estimator.py first."
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if int(summary.get("matches", 0)) != 13:
        raise ValueError(
            "Expected value for normal toto "
            "requires exactly 13 matches."
        )

    required_keys = {
        "prize1Probability",
        "prize2Probability",
        "prize3Probability",
    }

    missing = required_keys - set(summary)

    if missing:
        raise ValueError(
            "The probability JSON uses an old "
            "cumulative format.\n"
            "Run the updated command:\n"
            "python ml/probability_estimator.py "
            "--input "
            "ml/budget_ticket_plan_beam.csv"
        )

    return (
        float(
            summary["prize1Probability"]
        ),
        float(
            summary["prize2Probability"]
        ),
        float(
            summary["prize3Probability"]
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_PLAN),
    )

    parser.add_argument(
        "--probability-summary",
        default=str(
            DEFAULT_PROBABILITY_SUMMARY
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_EV_SUMMARY),
    )

    parser.add_argument(
        "--prize1",
        type=int,
        default=18_000_000,
    )

    parser.add_argument(
        "--prize2",
        type=int,
        default=300_000,
    )

    parser.add_argument(
        "--prize3",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--prize-source",
        default="placeholder",
        choices=[
            "placeholder",
            "estimated",
            "official",
        ],
    )

    args = parser.parse_args()

    plan_path = Path(args.input)
    probability_path = Path(
        args.probability_summary
    )
    output_path = Path(args.output)

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Ticket plan not found: {plan_path}"
        )

    df = pd.read_csv(plan_path)

    if df.empty:
        raise ValueError(
            "Ticket plan is empty."
        )

    (
        prize1_probability,
        prize2_probability,
        prize3_probability,
    ) = read_exact_probabilities(
        probability_path
    )

    total_tickets = calculate_ticket_count(
        df
    )

    investment = (
        total_tickets * UNIT_PRICE
    )

    expected1 = (
        prize1_probability
        * args.prize1
    )

    expected2 = (
        prize2_probability
        * args.prize2
    )

    expected3 = (
        prize3_probability
        * args.prize3
    )

    expected_return = (
        expected1
        + expected2
        + expected3
    )

    expected_profit = (
        expected_return
        - investment
    )

    roi = (
        expected_return / investment * 100
        if investment > 0
        else 0.0
    )

    result_recommendation = recommendation(
        roi
    )

    print("=" * 56)
    print("Expected Value Report")
    print("=" * 56)
    print()
    print(
        f"Ticket Count       : "
        f"{total_tickets:,}"
    )
    print(
        f"Investment         : "
        f"{investment:,} yen"
    )
    print(
        f"Prize Source       : "
        f"{args.prize_source}"
    )
    print()

    print(
        f"1st Probability    : "
        f"{prize1_probability:.6%}"
    )
    print(
        f"1st Prize          : "
        f"{args.prize1:,} yen"
    )
    print(
        f"1st Expected       : "
        f"{expected1:,.0f} yen"
    )
    print()

    print(
        f"2nd Probability    : "
        f"{prize2_probability:.6%}"
    )
    print(
        f"2nd Prize          : "
        f"{args.prize2:,} yen"
    )
    print(
        f"2nd Expected       : "
        f"{expected2:,.0f} yen"
    )
    print()

    print(
        f"3rd Probability    : "
        f"{prize3_probability:.6%}"
    )
    print(
        f"3rd Prize          : "
        f"{args.prize3:,} yen"
    )
    print(
        f"3rd Expected       : "
        f"{expected3:,.0f} yen"
    )
    print()

    print("-" * 56)

    print(
        f"Expected Return    : "
        f"{expected_return:,.0f} yen"
    )

    print(
        f"Expected Profit    : "
        f"{expected_profit:,.0f} yen"
    )

    print(
        f"Expected ROI       : "
        f"{roi:.2f}%"
    )

    print()
    print(
        f"Recommendation     : "
        f"{result_recommendation}"
    )

    if args.prize_source == "placeholder":
        print()
        print(
            "WARNING: Prize amounts are "
            "temporary assumptions."
        )
        print(
            "This ROI is a simulation and "
            "must not yet be treated as a "
            "real investment signal."
        )

    summary = {
        "ticketCount": total_tickets,
        "investment": investment,
        "unitPrice": UNIT_PRICE,
        "prizeSource": args.prize_source,
        "prize1Probability": (
            prize1_probability
        ),
        "prize2Probability": (
            prize2_probability
        ),
        "prize3Probability": (
            prize3_probability
        ),
        "prize1": args.prize1,
        "prize2": args.prize2,
        "prize3": args.prize3,
        "expected1": round(
            expected1,
            2,
        ),
        "expected2": round(
            expected2,
            2,
        ),
        "expected3": round(
            expected3,
            2,
        ),
        "expectedReturn": round(
            expected_return,
            2,
        ),
        "expectedProfit": round(
            expected_profit,
            2,
        ),
        "roi": round(
            roi,
            4,
        ),
        "recommendation": (
            result_recommendation
        ),
        "isSimulation": (
            args.prize_source
            == "placeholder"
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()