from pathlib import Path
import pandas as pd

INPUT = Path("ml/round_predictions.csv")
OUTPUT = Path("ml/loss_analysis.csv")


def main():

    df = pd.read_csv(INPUT)

    rows = []

    for _, row in df.iterrows():

        probs = sorted(
            [
                ("H", row["probH"]),
                ("D", row["probD"]),
                ("A", row["probA"]),
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        rows.append(
            {
                "homeTeam": row["homeTeam"],
                "awayTeam": row["awayTeam"],

                "top1": probs[0][0],
                "top1Prob": probs[0][1],

                "top2": probs[1][0],
                "top2Prob": probs[1][1],

                "top3": probs[2][0],
                "top3Prob": probs[2][1],

                "loss3to2": probs[2][1],
                "loss2to1": probs[1][1],
            }
        )

    out = pd.DataFrame(rows)

    out = out.sort_values("loss3to2")

    out.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 40)
    print("Loss Calculator")
    print("=" * 40)
    print(out.to_string(index=False))

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()