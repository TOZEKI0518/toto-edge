import argparse
import subprocess


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--round",
        type=int,
        default=1632,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--from-round",
        type=int,
        default=1506,
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
        choices=[
            "placeholder",
            "estimated",
            "official",
        ],
        default="placeholder",
    )

    args = parser.parse_args()

    pipeline_command = [
        "python",
        "ml/run_round_pipeline.py",
        "--round",
        str(args.round),
        "--from-round",
        str(args.from_round),
        "--budget",
        str(args.budget),
        "--model",
        args.model,
    ]

    if args.model in {
        "ensemble",
        "dynamic",
    }:
        pipeline_command.extend(
            [
                "--rf-weight",
                str(args.rf_weight),
                "--lgbm-weight",
                str(args.lgbm_weight),
            ]
        )

    run(pipeline_command)

    run(
        [
            "python",
            "ml/expected_value.py",
            "--input",
            "ml/budget_ticket_plan_beam.csv",
            "--prize1",
            str(args.prize1),
            "--prize2",
            str(args.prize2),
            "--prize3",
            str(args.prize3),
            "--prize-source",
            args.prize_source,
        ]
    )

    run(
        [
            "python",
            "ml/dashboard.py",
        ]
    )

    print()
    print("=" * 64)
    print("All done.")
    print(
        f"Round={args.round} / "
        f"Model={args.model} / "
        f"Budget limit={args.budget:,} yen"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()