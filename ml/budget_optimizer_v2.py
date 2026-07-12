from pathlib import Path
import argparse
import pandas as pd

INPUT_PATH = Path("ml/round_predictions.csv")
OUTPUT_PATH = Path("ml/budget_ticket_plan_v2.csv")

UNIT_PRICE = 100


def ranked_outcomes(row):
    outcomes = [
        ("H", float(row["probH"])),
        ("D", float(row["probD"])),
        ("A", float(row["probA"])),
    ]
    return sorted(outcomes, key=lambda item: item[1], reverse=True)


def calc_combinations(sizes):
    total = 1
    for size in sizes:
        total *= size
    return total


def ticket_from_size(row, size):
    ranked = ranked_outcomes(row)

    if size == 1:
        return ranked[0][0], "Single", ranked[0][1]

    if size == 2:
        return ranked[0][0] + ranked[1][0], "Double", ranked[0][1] + ranked[1][1]

    return "HDA", "Triple", 1.0


def best_reduction(df, sizes):
    candidates = []

    for idx, (_, row) in enumerate(df.iterrows()):
        ranked = ranked_outcomes(row)

        if sizes[idx] == 3:
            loss = ranked[2][1]
            reduction_ratio = 3 / 2
            efficiency = reduction_ratio / max(loss, 0.0001)

            candidates.append({
                "idx": idx,
                "from": 3,
                "to": 2,
                "loss": loss,
                "efficiency": efficiency,
            })

        elif sizes[idx] == 2:
            loss = ranked[1][1]
            reduction_ratio = 2 / 1
            efficiency = reduction_ratio / max(loss, 0.0001)

            candidates.append({
                "idx": idx,
                "from": 2,
                "to": 1,
                "loss": loss,
                "efficiency": efficiency,
            })

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item["efficiency"], reverse=True)[0]


def optimize_sizes(df, budget):
    max_tickets = max(1, budget // UNIT_PRICE)
    sizes = [3 for _ in range(len(df))]
    history = []

    while calc_combinations(sizes) > max_tickets:
        change = best_reduction(df, sizes)

        if change is None:
            break

        sizes[change["idx"]] = change["to"]

        history.append({
            **change,
            "tickets_after": calc_combinations(sizes),
            "cost_after": calc_combinations(sizes) * UNIT_PRICE,
        })

    return sizes, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)

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

    sizes, history = optimize_sizes(df, args.budget)

    rows = []

    for size, (_, row) in zip(sizes, df.iterrows()):
        ticket, ticket_type, coverage = ticket_from_size(row, size)
        ranked = ranked_outcomes(row)

        rows.append({
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
            "coverage": coverage,
            "top1": ranked[0][0],
            "top1Prob": ranked[0][1],
            "top2": ranked[1][0],
            "top2Prob": ranked[1][1],
            "top3": ranked[2][0],
            "top3Prob": ranked[2][1],
        })

    out = pd.DataFrame(rows)

    combinations = calc_combinations(sizes)
    cost = combinations * UNIT_PRICE
    total_coverage = out["coverage"].prod()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 40)
    print("Budget Optimizer v2")
    print("=" * 40)
    print(f"Budget: {args.budget:,} yen")
    print(f"Total Tickets: {combinations}")
    print(f"Cost: {cost:,} yen")
    print(f"Estimated Coverage: {total_coverage:.4f}")
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
                "ticketType",
                "ticket",
                "coverage",
            ]
        ].to_string(index=False)
    )

    print()
    print("Summary")
    print(out["ticketType"].value_counts().to_string())

    print()
    print("Reduction History")
    for step, item in enumerate(history, start=1):
        row = df.iloc[item["idx"]]
        print(
            f"{step:02d}: {row['homeTeam']} vs {row['awayTeam']} "
            f"{item['from']}→{item['to']} "
            f"loss={item['loss']:.4f} "
            f"eff={item['efficiency']:.2f} "
            f"tickets={item['tickets_after']} "
            f"cost={item['cost_after']:,} yen"
        )


if __name__ == "__main__":
    main()