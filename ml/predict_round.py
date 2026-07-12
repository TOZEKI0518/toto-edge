from pathlib import Path
import argparse

import joblib
import pandas as pd

from common.dataset import load_dataset
from common.draw_dynamic import (
    adjust_draw_probability_dynamic,
    create_draw_classifier,
)
from common.features import FEATURE_COLUMNS
from common.match_difficulty import MatchDifficulty
from common.predictor import (
    SUPPORTED_MODELS,
    train_and_predict,
)


OUTPUT_PATH = Path("ml/round_predictions.csv")

DYNAMIC_MODEL_PATH = Path(
    "ml/toto_draw_classifier_dynamic.joblib"
)

MODEL_CHOICES = tuple(
    dict.fromkeys(
        (*SUPPORTED_MODELS, "dynamic")
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--round",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="rf",
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

    target_round = args.round

    df = load_dataset()

    df = df.sort_values(
        "roundNo"
    ).reset_index(drop=True)

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}"
        )

    train_df = df[
        (df["roundNo"] < target_round)
        & df["result"].isin(
            ["H", "D", "A"]
        )
    ].copy()

    target_df = df[
        df["roundNo"] == target_round
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No completed training rows "
            f"before round {target_round}."
        )

    if target_df.empty:
        raise ValueError(
            f"No rows found for round {target_round}."
        )

    # DynamicはEnsembleを基礎確率として利用
    base_model_name = (
        "ensemble"
        if args.model == "dynamic"
        else args.model
    )

    result = train_and_predict(
        train_df=train_df,
        target_df=target_df,
        model_name=base_model_name,
        rf_weight=args.rf_weight,
        lgbm_weight=args.lgbm_weight,
    )

    probabilities = (
        result.probabilities.copy()
    )

    draw_specialist_probabilities = None
    dynamic_alphas = None
    dynamic_config = None

    if args.model == "dynamic":
        if not DYNAMIC_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Dynamic model settings not found: "
                f"{DYNAMIC_MODEL_PATH}\n"
                "Run:\n"
                "python "
                "ml/train_draw_classifier_dynamic.py"
            )

        dynamic_config = joblib.load(
            DYNAMIC_MODEL_PATH
        )

        # 対象回より前だけで再学習し、
        # 過去検証で未来情報が混ざらないようにする
        x_train = train_df[
            FEATURE_COLUMNS
        ].fillna(0)

        x_target = target_df[
            FEATURE_COLUMNS
        ].fillna(0)

        y_train_draw = (
            train_df["result"].astype(str)
            == "D"
        ).astype(int)

        draw_model = create_draw_classifier()

        draw_model.fit(
            x_train,
            y_train_draw,
        )

        draw_specialist_probabilities = (
            draw_model.predict_proba(
                x_target
            )[:, 1]
        )

        probabilities, dynamic_alphas = (
            adjust_draw_probability_dynamic(
                base_probabilities=probabilities,
                specialist_draw_probabilities=(
                    draw_specialist_probabilities
                ),
                low_threshold=float(
                    dynamic_config[
                        "low_threshold"
                    ]
                ),
                high_threshold=float(
                    dynamic_config[
                        "high_threshold"
                    ]
                ),
                low_alpha=float(
                    dynamic_config["low_alpha"]
                ),
                medium_alpha=float(
                    dynamic_config[
                        "medium_alpha"
                    ]
                ),
                high_alpha=float(
                    dynamic_config["high_alpha"]
                ),
            )
        )

    class_indexes = {
        class_name: index
        for index, class_name
        in enumerate(result.classes)
    }

    prediction_indexes = (
        probabilities.argmax(axis=1)
    )

    predictions = [
        result.classes[index]
        for index in prediction_indexes
    ]

    difficulty_engine = MatchDifficulty()

    rows = []

    for position, (_, row) in enumerate(
        target_df.iterrows()
    ):
        row_probabilities = (
            probabilities[position]
        )

        prob_h = float(
            row_probabilities[
                class_indexes["H"]
            ]
        )

        prob_d = float(
            row_probabilities[
                class_indexes["D"]
            ]
        )

        prob_a = float(
            row_probabilities[
                class_indexes["A"]
            ]
        )

        difficulty = (
            difficulty_engine.calculate(
                prob_h=prob_h,
                prob_d=prob_d,
                prob_a=prob_a,
            )
        )

        output_row = {
            "roundNo": row["roundNo"],
            "snapshotDate": row.get(
                "snapshotDate",
                "",
            ),
            "league": row.get(
                "league",
                "",
            ),
            "homeTeam": row["homeTeam"],
            "awayTeam": row["awayTeam"],
            "model": args.model,
            "prediction": predictions[position],
            "actual": row.get(
                "result",
                "",
            ),
            "probH": prob_h,
            "probD": prob_d,
            "probA": prob_a,
            "confidence": (
                difficulty.confidence
            ),
            "margin": difficulty.margin,
            "entropy": difficulty.entropy,
            "difficulty": (
                difficulty.difficulty
            ),
            "difficultyClass": (
                difficulty.difficulty_class
            ),
            "difficultyStars": (
                difficulty.difficulty_stars
            ),
        }

        if args.model == "dynamic":
            output_row[
                "drawSpecialistProb"
            ] = float(
                draw_specialist_probabilities[
                    position
                ]
            )

            output_row[
                "dynamicAlpha"
            ] = float(
                dynamic_alphas[position]
            )

        rows.append(output_row)

    out = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 40)
    print("Round Prediction")
    print("=" * 40)
    print(f"Round: {target_round}")
    print(f"Model: {args.model}")

    if args.model in (
        "ensemble",
        "dynamic",
    ):
        print(
            f"Weights: RF "
            f"{args.rf_weight:.2f} / "
            f"LightGBM "
            f"{args.lgbm_weight:.2f}"
        )

    if args.model == "dynamic":
        print(
            "Dynamic config: "
            f"thresholds "
            f"{dynamic_config['low_threshold']:.2f}/"
            f"{dynamic_config['high_threshold']:.2f}, "
            f"alphas "
            f"{dynamic_config['low_alpha']:.2f}/"
            f"{dynamic_config['medium_alpha']:.2f}/"
            f"{dynamic_config['high_alpha']:.2f}"
        )

        print(
            f"Average Alpha: "
            f"{dynamic_alphas.mean():.4f}"
        )

        high_alpha = float(
            dynamic_config["high_alpha"]
        )

        print(
            "High Alpha Matches: "
            f"{(dynamic_alphas == high_alpha).sum()}"
        )

    print(
        f"Training Rows: {len(train_df)}"
    )

    print(
        f"Target Rows: {len(out)}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    print(f"Saved: {OUTPUT_PATH}")
    print()

    display_columns = [
        "homeTeam",
        "awayTeam",
        "model",
        "prediction",
        "actual",
        "probH",
        "probD",
        "probA",
        "confidence",
        "margin",
        "entropy",
        "difficulty",
        "difficultyStars",
        "difficultyClass",
    ]

    if args.model == "dynamic":
        display_columns.extend(
            [
                "drawSpecialistProb",
                "dynamicAlpha",
            ]
        )

    print(
        out[
            display_columns
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()