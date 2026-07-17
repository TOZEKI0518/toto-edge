from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


LOGGER = logging.getLogger("rf_v3_backtest")

PLAYER_FEATURES: Final[list[str]] = [
    "homePlayerCoreScore",
    "awayPlayerCoreScore",
    "playerCoreScoreDiff",
    "homePlayerMomentum",
    "awayPlayerMomentum",
    "playerMomentumDiff",
    "homeTeamStability",
    "awayTeamStability",
    "teamStabilityDiff",
    "homeCorePlayerCount",
    "awayCorePlayerCount",
    "corePlayerCountDiff",
    "homeAvailablePlayerCount",
    "awayAvailablePlayerCount",
    "availablePlayerCountDiff",
]

TARGET_CANDIDATES: Final[list[str]] = [
    "actual",
    "result",
    "target",
    "label",
]

ROUND_CANDIDATES: Final[list[str]] = [
    "round",
    "roundNo",
    "totoRound",
    "toto_round",
]

NON_FEATURE_COLUMNS: Final[set[str]] = {
    "actual",
    "result",
    "target",
    "label",
    "prediction",
    "model",
    "homeTeam",
    "awayTeam",
    "home_team",
    "away_team",
    "match_card_id",
    "matchId",
    "match_id",
    "date",
    "matchDate",
    "match_date",
    "competition",
    "section",
}


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    rows: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float
    draw_precision: float
    draw_recall: float
    draw_f1: float
    home_recall: float
    away_recall: float


@dataclass(frozen=True)
class WindowMetrics:
    window: int
    model: str
    train_round_from: int
    train_round_to: int
    test_round_from: int
    test_round_to: int
    train_rows: int
    test_rows: int
    accuracy: float
    macro_f1: float
    log_loss: float
    draw_recall: float


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match

    raise ValueError(
        f"Could not find {label} column. Candidates: {candidates}"
    )


def prepare_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, list[str], list[str]]:
    target_column = find_column(frame, TARGET_CANDIDATES, "target")
    round_column = find_column(frame, ROUND_CANDIDATES, "round")

    prepared = frame.copy()
    prepared[target_column] = prepared[target_column].astype(str).str.strip()
    prepared[round_column] = pd.to_numeric(
        prepared[round_column],
        errors="coerce",
    )

    prepared = prepared.loc[
        prepared[target_column].isin(["H", "D", "A"])
        & prepared[round_column].notna()
    ].copy()

    prepared[round_column] = prepared[round_column].astype(int)

    missing_player_features = [
        column for column in PLAYER_FEATURES if column not in prepared.columns
    ]
    if missing_player_features:
        raise ValueError(
            "Missing Player Intelligence columns: "
            f"{missing_player_features}"
        )

    numeric_columns = prepared.select_dtypes(include="number").columns.tolist()

    v3_features = [
        column
        for column in numeric_columns
        if column not in NON_FEATURE_COLUMNS
        and column != round_column
    ]
    v2_features = [
        column for column in v3_features if column not in PLAYER_FEATURES
    ]

    if not v2_features:
        raise ValueError("No baseline numeric features were found.")

    player_available = (
        prepared["homeAvailablePlayerCount"].gt(0)
        | prepared["awayAvailablePlayerCount"].gt(0)
    )
    prepared = prepared.loc[player_available].copy()

    if prepared.empty:
        raise ValueError(
            "No rows contain Player Intelligence data."
        )

    for column in v3_features:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return prepared, target_column, round_column, v2_features, v3_features


def make_model(
    random_state: int,
    n_estimators: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def aligned_probabilities(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    raw = model.predict_proba(features)
    result = np.zeros((len(features), len(labels)), dtype=float)

    class_to_index = {
        str(label): index for index, label in enumerate(model.classes_)
    }

    for output_index, label in enumerate(labels):
        model_index = class_to_index.get(label)
        if model_index is not None:
            result[:, output_index] = raw[:, model_index]

    row_sum = result.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0

    return result / row_sum


def calculate_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> ModelMetrics:
    precision, recall, f1_values, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    label_index = {label: index for index, label in enumerate(labels)}

    def value(values: np.ndarray, label: str) -> float:
        index = label_index.get(label)
        return float(values[index]) if index is not None else 0.0

    return ModelMetrics(
        model=model_name,
        rows=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        log_loss=float(log_loss(y_true, probabilities, labels=labels)),
        draw_precision=value(precision, "D"),
        draw_recall=value(recall, "D"),
        draw_f1=value(f1_values, "D"),
        home_recall=value(recall, "H"),
        away_recall=value(recall, "A"),
    )


def rolling_backtest(
    frame: pd.DataFrame,
    target_column: str,
    round_column: str,
    feature_sets: dict[str, list[str]],
    min_train_rounds: int,
    test_window: int,
    n_estimators: int,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, ModelMetrics],
    dict[str, np.ndarray],
]:
    rounds = sorted(frame[round_column].unique().tolist())

    if len(rounds) <= min_train_rounds:
        raise ValueError(
            f"Only {len(rounds)} rounds available; "
            f"more than {min_train_rounds} are required."
        )

    labels = ["A", "D", "H"]
    window_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    all_true: dict[str, list[str]] = {name: [] for name in feature_sets}
    all_pred: dict[str, list[str]] = {name: [] for name in feature_sets}
    all_prob: dict[str, list[np.ndarray]] = {name: [] for name in feature_sets}
    importance_totals: dict[str, np.ndarray] = {
        name: np.zeros(len(features), dtype=float)
        for name, features in feature_sets.items()
    }
    importance_counts = {name: 0 for name in feature_sets}

    window_number = 0
    start_index = min_train_rounds

    while start_index < len(rounds):
        test_rounds = rounds[start_index : start_index + test_window]
        if not test_rounds:
            break

        train_rounds = rounds[:start_index]

        train = frame.loc[frame[round_column].isin(train_rounds)]
        test = frame.loc[frame[round_column].isin(test_rounds)]

        if train.empty or test.empty:
            start_index += test_window
            continue

        window_number += 1

        print(
            f"\nWindow {window_number}: "
            f"Train {train_rounds[0]}-{train_rounds[-1]} "
            f"({len(train)} rows) -> "
            f"Test {test_rounds[0]}-{test_rounds[-1]} "
            f"({len(test)} rows)"
        )

        y_train = train[target_column]
        y_test = test[target_column]

        for model_name, features in feature_sets.items():
            model = make_model(
                random_state=random_state,
                n_estimators=n_estimators,
            )
            model.fit(train[features], y_train)

            predictions = model.predict(test[features])
            probabilities = aligned_probabilities(
                model,
                test[features],
                labels,
            )
            metrics = calculate_metrics(
                model_name,
                y_test,
                predictions,
                probabilities,
                labels,
            )

            window = WindowMetrics(
                window=window_number,
                model=model_name,
                train_round_from=int(train_rounds[0]),
                train_round_to=int(train_rounds[-1]),
                test_round_from=int(test_rounds[0]),
                test_round_to=int(test_rounds[-1]),
                train_rows=len(train),
                test_rows=len(test),
                accuracy=metrics.accuracy,
                macro_f1=metrics.macro_f1,
                log_loss=metrics.log_loss,
                draw_recall=metrics.draw_recall,
            )
            window_rows.append(asdict(window))

            print(
                f"  {model_name:<18} "
                f"Accuracy={metrics.accuracy:.4f} "
                f"MacroF1={metrics.macro_f1:.4f} "
                f"DrawRecall={metrics.draw_recall:.4f} "
                f"LogLoss={metrics.log_loss:.4f}"
            )

            all_true[model_name].extend(y_test.tolist())
            all_pred[model_name].extend(predictions.tolist())
            all_prob[model_name].append(probabilities)

            importance_totals[model_name] += model.feature_importances_
            importance_counts[model_name] += 1

            for row_index, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "round": int(row[round_column]),
                        "actual": str(y_test.iloc[row_index]),
                        "prediction": str(predictions[row_index]),
                        "probA": probabilities[row_index, 0],
                        "probD": probabilities[row_index, 1],
                        "probH": probabilities[row_index, 2],
                    }
                )

        start_index += test_window

    overall: dict[str, ModelMetrics] = {}
    average_importances: dict[str, np.ndarray] = {}

    for model_name in feature_sets:
        if not all_true[model_name]:
            raise RuntimeError(f"No predictions produced for {model_name}.")

        probabilities = np.vstack(all_prob[model_name])
        overall[model_name] = calculate_metrics(
            model_name,
            pd.Series(all_true[model_name]),
            np.asarray(all_pred[model_name]),
            probabilities,
            labels,
        )
        average_importances[model_name] = (
            importance_totals[model_name]
            / max(importance_counts[model_name], 1)
        )

    return (
        pd.DataFrame(window_rows),
        pd.DataFrame(prediction_rows),
        overall,
        average_importances,
    )


def print_overall(
    overall: dict[str, ModelMetrics],
    predictions: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("Overall Results - Matched Player Data Only")
    print("=" * 72)

    for model_name, metrics in overall.items():
        model_predictions = predictions.loc[
            predictions["model"] == model_name
        ]

        print(f"\n{model_name}")
        print("-" * 40)
        print(f"Rows           : {metrics.rows}")
        print(f"Accuracy       : {metrics.accuracy:.4f}")
        print(f"Macro F1       : {metrics.macro_f1:.4f}")
        print(f"Weighted F1    : {metrics.weighted_f1:.4f}")
        print(f"Log Loss       : {metrics.log_loss:.4f}")
        print(f"Draw Precision : {metrics.draw_precision:.4f}")
        print(f"Draw Recall    : {metrics.draw_recall:.4f}")
        print(f"Draw F1        : {metrics.draw_f1:.4f}")
        print(f"Away Recall    : {metrics.away_recall:.4f}")
        print(f"Home Recall    : {metrics.home_recall:.4f}")

        print("\nClassification Report")
        print(
            classification_report(
                model_predictions["actual"],
                model_predictions["prediction"],
                labels=["A", "D", "H"],
                zero_division=0,
            )
        )

        print("Confusion Matrix [A, D, H]")
        print(
            confusion_matrix(
                model_predictions["actual"],
                model_predictions["prediction"],
                labels=["A", "D", "H"],
            )
        )


def save_feature_importance(
    feature_sets: dict[str, list[str]],
    importances: dict[str, np.ndarray],
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_name, features in feature_sets.items():
        for feature, importance in zip(
            features,
            importances[model_name],
            strict=True,
        ):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance": float(importance),
                    "is_player_feature": feature in PLAYER_FEATURES,
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["model", "importance"],
        ascending=[True, False],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    return result


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Fair rolling comparison of RandomForest v2 and "
            "Player Intelligence v3."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "ml" / "training_dataset_v3.csv",
    )
    parser.add_argument("--min-train-rounds", type=int, default=20)
    parser.add_argument("--test-window", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    raw = pd.read_csv(args.input)
    (
        frame,
        target_column,
        round_column,
        v2_features,
        v3_features,
    ) = prepare_dataset(raw)

    feature_sets = {
        "RandomForest v2": v2_features,
        "RandomForest v3": v3_features,
    }

    print("=" * 72)
    print("RandomForest v2 vs v3 Rolling Backtest")
    print("=" * 72)
    print(f"Rows with Player data : {len(frame)}")
    print(
        f"Rounds                : "
        f"{frame[round_column].min()} - {frame[round_column].max()}"
    )
    print(f"Baseline features     : {len(v2_features)}")
    print(f"V3 features           : {len(v3_features)}")
    print(f"Player features       : {len(PLAYER_FEATURES)}")
    print(f"Minimum train rounds  : {args.min_train_rounds}")
    print(f"Test window           : {args.test_window}")

    (
        windows,
        predictions,
        overall,
        importances,
    ) = rolling_backtest(
        frame=frame,
        target_column=target_column,
        round_column=round_column,
        feature_sets=feature_sets,
        min_train_rounds=args.min_train_rounds,
        test_window=args.test_window,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print_overall(overall, predictions)

    output_dir = project_root() / "ml"
    windows_path = output_dir / "rf_v3_windows.csv"
    predictions_path = output_dir / "rf_v3_predictions.csv"
    summary_path = output_dir / "rf_v3_summary.csv"
    importance_path = output_dir / "rf_v3_feature_importance.csv"
    json_path = output_dir / "rf_v3_summary.json"

    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [asdict(metrics) for metrics in overall.values()]
    ).sort_values("accuracy", ascending=False)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    importance = save_feature_importance(
        feature_sets,
        importances,
        importance_path,
    )

    json_path.write_text(
        json.dumps(
            {
                "rows": len(frame),
                "round_from": int(frame[round_column].min()),
                "round_to": int(frame[round_column].max()),
                "baseline_features": len(v2_features),
                "v3_features": len(v3_features),
                "models": {
                    name: asdict(metrics)
                    for name, metrics in overall.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Comparison")
    print("=" * 72)
    print(summary.to_string(index=False))

    print("\nTop 30 v3 Features")
    print("-" * 72)
    print(
        importance.loc[
            importance["model"] == "RandomForest v3"
        ].head(30).to_string(index=False)
    )

    v2_accuracy = overall["RandomForest v2"].accuracy
    v3_accuracy = overall["RandomForest v3"].accuracy

    print("\nAccuracy change")
    print("-" * 72)
    print(
        f"{v2_accuracy:.4f} -> {v3_accuracy:.4f} "
        f"({v3_accuracy - v2_accuracy:+.4f})"
    )

    print("\nSaved")
    print("-" * 72)
    for path in (
        windows_path,
        predictions_path,
        summary_path,
        importance_path,
        json_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
