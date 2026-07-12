import argparse
import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("ml/budget_ticket_plan_beam.csv")
OUTPUT_JSON = Path("ml/probability_summary.json")


def ticket_coverage(row: pd.Series) -> float:
    ticket = str(row["ticket"])

    probability = 0.0

    if "H" in ticket:
        probability += float(row["probH"])

    if "D" in ticket:
        probability += float(row["probD"])

    if "A" in ticket:
        probability += float(row["probA"])

    return min(max(probability, 0.0), 1.0)


def distribution_after_matches(
    coverages: list[float],
) -> list[float]:
    """
    dist[k] = 全試合終了後にk試合をカバーできる確率
    """
    distribution = [1.0]

    for coverage in coverages:
        next_distribution = [0.0] * (
            len(distribution) + 1
        )

        for hits, probability in enumerate(distribution):
            next_distribution[hits] += (
                probability * (1.0 - coverage)
            )

            next_distribution[hits + 1] += (
                probability * coverage
            )

        distribution = next_distribution

    # 浮動小数点誤差による微小な負数を除去
    distribution = [
        max(float(value), 0.0)
        for value in distribution
    ]

    total = sum(distribution)

    if total > 0:
        distribution = [
            value / total
            for value in distribution
        ]

    return distribution


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(INPUT_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_JSON),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Ticket plan not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Ticket plan is empty.")

    required_columns = {
        "ticket",
        "ticketType",
        "probH",
        "probD",
        "probA",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["coverage"] = df.apply(
        ticket_coverage,
        axis=1,
    )

    coverages = df["coverage"].tolist()
    distribution = distribution_after_matches(
        coverages
    )

    matches = len(coverages)
    average_coverage = float(
        df["coverage"].mean()
    )

    print("=" * 48)
    print("Winning Probability Estimator")
    print("=" * 48)
    print(f"Matches          : {matches}")
    print(
        f"Average Coverage : "
        f"{average_coverage:.4%}"
    )
    print()

    print("Ticket Type Summary")
    print(
        df["ticketType"]
        .value_counts()
        .to_string()
    )
    print()

    summary: dict[str, object] = {
        "matches": matches,
        "averageCoverage": average_coverage,
        "hitDistribution": {
            str(hits): float(probability)
            for hits, probability in enumerate(
                distribution
            )
        },
    }

    if matches == 13:
        # 等級別の単独確率
        prize1_probability = distribution[13]
        prize2_probability = distribution[12]
        prize3_probability = distribution[11]

        # 参考表示用の累積確率
        prize2_or_better = (
            prize1_probability
            + prize2_probability
        )

        prize3_or_better = (
            prize1_probability
            + prize2_probability
            + prize3_probability
        )

        summary.update(
            {
                # 新しい正式キー
                "prize1Probability": float(
                    prize1_probability
                ),
                "prize2Probability": float(
                    prize2_probability
                ),
                "prize3Probability": float(
                    prize3_probability
                ),
                "prize2OrBetterProbability": float(
                    prize2_or_better
                ),
                "prize3OrBetterProbability": float(
                    prize3_or_better
                ),

                # 既存コードとの互換用
                "prob1": float(
                    prize1_probability
                ),
                "prob2": float(
                    prize2_probability
                ),
                "prob3": float(
                    prize3_probability
                ),
                "prob2OrBetter": float(
                    prize2_or_better
                ),
                "prob3OrBetter": float(
                    prize3_or_better
                ),
            }
        )

        print(
            f"1st Prize 13/13   : "
            f"{prize1_probability:.6%}"
        )

        print(
            f"2nd Prize 12/13   : "
            f"{prize2_probability:.6%}"
        )

        print(
            f"3rd Prize 11/13   : "
            f"{prize3_probability:.6%}"
        )

        print()
        print(
            f"2nd or Better     : "
            f"{prize2_or_better:.6%}"
        )

        print(
            f"3rd or Better     : "
            f"{prize3_or_better:.6%}"
        )

    else:
        all_correct = distribution[matches]

        summary.update(
            {
                "allCorrectProbability": float(
                    all_correct
                )
            }
        )

        print(
            "13試合ではないため、"
            "通常totoの1等・2等・3等は"
            "計算しません。"
        )

        print(
            f"All Correct {matches}/{matches}: "
            f"{all_correct:.6%}"
        )

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
    print(f"Saved summary: {output_path}")

    print()
    print("Hit Count Distribution")

    for hits, probability in enumerate(
        distribution
    ):
        print(
            f"{hits}/{matches}: "
            f"{probability:.6%}"
        )


if __name__ == "__main__":
    main()