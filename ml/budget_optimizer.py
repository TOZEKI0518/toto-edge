from pathlib import Path
import argparse
import pandas as pd

INPUT_PATH = Path("ml/round_predictions.csv")
OUTPUT_PATH = Path("ml/budget_ticket_plan.csv")

UNIT_PRICE = 100


def ranked_outcomes(row):
    outcomes = [
        ("H", float(row["probH"])),
        ("D", float(row["probD"])),
        ("A", float(row["probA"])),
    ]
    return sorted(outcomes, key=lambda item: item[1], reverse=True)


def ticket_from_size(row, size: int):
    ranked = ranked_outcomes(row)

    if size == 1:
        return ranked[0][0], "Single"

    if size == 2:
        return ranked[0][0] + ranked[1][0], "Double"

    return "HDA", "Triple"


def calc_combinations(sizes):
    total = 1
    for size in sizes:
        total *= size
    return total


def optimize_sizes(df: pd.DataFrame, budget: int):
    max_tickets = max(1, budget // UNIT_PRICE)

    # 最初は全部トリプル
    sizes = [3 for _ in range(len(df))]

    # confidenceが高い試合から削る
    order = (
        df.assign(_index=range(len(df)))
        .sort_values(["confidence", "margin"], ascending=False)["_index"]
        .tolist()
    )

    while calc_combinations(sizes) > max_tickets:
        changed = False

        # Triple -> Double
        for idx in order:
            if sizes[idx] == 3:
                sizes[idx] = 2
                changed = True
                break

        if changed:
            continue

        # Double -> Single
        for idx in order:
            if sizes[idx] == 2:
                sizes[idx] = 1
                changed = True
                break

        if not changed:
            break

    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise ValueError("round_predictions.csv is empty.")

    required = {
        "homeTeam",
        "awayTeam",
        "prediction",
        "probH",
        "probD",
        "probA",
        "confidence",
        "margin",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sizes = optimize_sizes(df, args.budget)

    rows = []

    for size, (_, row) in zip(sizes, df.iterrows()):
        ticket, ticket_type = ticket_from_size(row, size)
        ranked = ranked_outcomes(row)

        rows.append(
            {
                "homeTeam": row["homeTeam"],
                "awayTeam": row["awayTeam"],
                "prediction": row["prediction"],
                "probH": row["probH"],
                "probD": row["probD"],
                "probA": row["probA"],
                "confidence": row["confidence"],
                "margin": row["margin"],
                "ticketType": ticket_type,
                "ticket": ticket,
                "top1": ranked[0][0],
                "top1Prob": ranked[0][1],
                "top2": ranked[1][0],
                "top2Prob": ranked[1][1],
                "top3": ranked[2][0],
                "top3Prob": ranked[2][1],
            }
        )

    out = pd.DataFrame(rows)

    combinations = calc_combinations(sizes)
    cost = combinations * UNIT_PRICE

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 40)
    print("Budget Optimizer")
    print("=" * 40)
    print(f"Budget: {args.budget:,} yen")
    print(f"Total Tickets: {combinations}")
    print(f"Cost: {cost:,} yen")
    print(f"Saved: {OUTPUT_PATH}")
    print()

    print(
        out[
            [
                "homeTeam",
                "awayTeam",
                "probH",
                "probD",
                "probA",
                "confidence",
                "margin",
                "ticketType",
                "ticket",
            ]
        ].to_string(index=False)
    )

    print()
    print("Summary")
    print(out["ticketType"].value_counts().to_string())


if __name__ == "__main__":
    main()