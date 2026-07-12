from pathlib import Path

import pandas as pd

INPUT = Path("ml/round_predictions.csv")
OUTPUT = Path("ml/ticket_plan.csv")


def second_prediction(row):

    probs = {
        "H": row["probH"],
        "D": row["probD"],
        "A": row["probA"],
    }

    probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    return probs[1][0]


def main():

    df = pd.read_csv(INPUT)

    tickets = []

    for _, row in df.iterrows():

        prediction = row["prediction"]
        confidence = row["confidence"]

        if confidence >= 0.55:

            ticket = prediction
            ticket_type = "Single"

        elif confidence >= 0.45:

            second = second_prediction(row)

            ticket = prediction + second
            ticket_type = "Double"

        else:

            ticket = "HDA"
            ticket_type = "Triple"

        tickets.append({

            "homeTeam": row["homeTeam"],
            "awayTeam": row["awayTeam"],

            "prediction": prediction,

            "confidence": confidence,

            "ticketType": ticket_type,

            "ticket": ticket,

        })

    out = pd.DataFrame(tickets)

    out.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("="*40)
    print("Ticket Optimizer")
    print("="*40)
    print(out.to_string(index=False))
    print()

    print("Summary")

    print(out["ticketType"].value_counts())

    singles = (out["ticketType"]=="Single").sum()
    doubles = (out["ticketType"]=="Double").sum()
    triples = (out["ticketType"]=="Triple").sum()

    combinations = (2**doubles)*(3**triples)

    print()
    print(f"Singles : {singles}")
    print(f"Doubles : {doubles}")
    print(f"Triples : {triples}")

    print()
    print(f"Total Tickets : {combinations}")
    print(f"Cost : {combinations*100:,} yen")


if __name__ == "__main__":
    main()