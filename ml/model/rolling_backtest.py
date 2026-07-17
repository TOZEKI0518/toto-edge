from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .feature_selector import FeatureSelector
from .metrics import (
    ClassificationMetrics,
    align_probabilities,
    calculate_metrics,
)
from .random_forest_model import RandomForestModel


@dataclass(frozen=True)
class WindowResult:
    window: int
    train_round_from: int
    train_round_to: int
    test_round_from: int
    test_round_to: int
    train_rows: int
    test_rows: int
    selected_features: int
    accuracy: float
    macro_f1: float
    log_loss: float
    draw_recall: float


@dataclass
class BacktestResult:
    windows: pd.DataFrame
    predictions: pd.DataFrame
    overall: ClassificationMetrics
    importance: pd.DataFrame
    selected_frequency: pd.DataFrame


def rolling_backtest(
    frame: pd.DataFrame,
    target_column: str,
    round_column: str,
    feature_columns: list[str],
    model_factory: Callable[[], RandomForestModel],
    selector_factory: Callable[[], FeatureSelector],
    min_train_rounds: int = 20,
    test_window: int = 5,
    labels: list[str] | None = None,
) -> BacktestResult:
    labels = labels or ["A", "D", "H"]
    rounds = sorted(frame[round_column].unique().tolist())

    if len(rounds) <= min_train_rounds:
        raise ValueError(
            f"Only {len(rounds)} rounds available; "
            f"more than {min_train_rounds} are required."
        )

    window_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    all_true: list[str] = []
    all_pred: list[str] = []
    all_probabilities: list[np.ndarray] = []
    importance_sum: dict[str, float] = {}
    selection_count: dict[str, int] = {}
    completed_windows = 0

    start = min_train_rounds

    while start < len(rounds):
        test_rounds = rounds[start : start + test_window]
        train_rounds = rounds[:start]

        train = frame.loc[frame[round_column].isin(train_rounds)]
        test = frame.loc[frame[round_column].isin(test_rounds)]

        if train.empty or test.empty:
            start += test_window
            continue

        completed_windows += 1
        selector = selector_factory()
        x_train = selector.fit_transform(train[feature_columns])
        x_test = selector.transform(test[feature_columns])

        model = model_factory().fit(x_train, train[target_column])
        predictions = model.predict(x_test)
        probabilities = align_probabilities(
            model.classes_,
            model.predict_proba(x_test),
            labels,
        )

        metrics = calculate_metrics(
            "RandomForest v4",
            test[target_column],
            predictions,
            probabilities,
            labels,
        )

        window_rows.append(
            asdict(
                WindowResult(
                    window=completed_windows,
                    train_round_from=int(train_rounds[0]),
                    train_round_to=int(train_rounds[-1]),
                    test_round_from=int(test_rounds[0]),
                    test_round_to=int(test_rounds[-1]),
                    train_rows=len(train),
                    test_rows=len(test),
                    selected_features=len(selector.selected_features_),
                    accuracy=metrics.accuracy,
                    macro_f1=metrics.macro_f1,
                    log_loss=metrics.log_loss,
                    draw_recall=metrics.draw_recall,
                )
            )
        )

        for feature, importance in zip(
            selector.selected_features_,
            model.feature_importances_,
            strict=True,
        ):
            importance_sum[feature] = (
                importance_sum.get(feature, 0.0) + float(importance)
            )
            selection_count[feature] = selection_count.get(feature, 0) + 1

        for row_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "round": int(row[round_column]),
                    "actual": str(test[target_column].iloc[row_index]),
                    "prediction": str(predictions[row_index]),
                    "probA": probabilities[row_index, 0],
                    "probD": probabilities[row_index, 1],
                    "probH": probabilities[row_index, 2],
                }
            )

        all_true.extend(test[target_column].astype(str).tolist())
        all_pred.extend(predictions.astype(str).tolist())
        all_probabilities.append(probabilities)

        print(
            f"Window {completed_windows}: "
            f"Train {train_rounds[0]}-{train_rounds[-1]} "
            f"({len(train)}) -> Test {test_rounds[0]}-{test_rounds[-1]} "
            f"({len(test)}) | Features={len(selector.selected_features_)} "
            f"Accuracy={metrics.accuracy:.4f} "
            f"MacroF1={metrics.macro_f1:.4f} "
            f"DrawRecall={metrics.draw_recall:.4f} "
            f"LogLoss={metrics.log_loss:.4f}"
        )

        start += test_window

    if not all_true:
        raise RuntimeError("Rolling backtest produced no predictions.")

    probability_matrix = np.vstack(all_probabilities)
    overall = calculate_metrics(
        "RandomForest v4",
        pd.Series(all_true),
        np.asarray(all_pred),
        probability_matrix,
        labels,
    )

    importance = pd.DataFrame(
        [
            {
                "feature": feature,
                "mean_importance": total
                / max(selection_count.get(feature, 1), 1),
                "selected_windows": selection_count.get(feature, 0),
                "selection_rate": selection_count.get(feature, 0)
                / completed_windows,
            }
            for feature, total in importance_sum.items()
        ]
    ).sort_values("mean_importance", ascending=False)

    frequency = importance[
        ["feature", "selected_windows", "selection_rate"]
    ].sort_values(
        ["selection_rate", "feature"],
        ascending=[False, True],
    )

    return BacktestResult(
        windows=pd.DataFrame(window_rows),
        predictions=pd.DataFrame(prediction_rows),
        overall=overall,
        importance=importance,
        selected_frequency=frequency,
    )
