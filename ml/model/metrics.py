from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    model: str
    rows: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float
    draw_precision: float
    draw_recall: float
    draw_f1: float
    away_recall: float
    home_recall: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def align_probabilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Align model probabilities to a fixed class order."""
    result = np.zeros((len(probabilities), len(labels)), dtype=float)
    index = {str(label): i for i, label in enumerate(classes)}

    for target_index, label in enumerate(labels):
        source_index = index.get(label)
        if source_index is not None:
            result[:, target_index] = probabilities[:, source_index]

    sums = result.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return result / sums


def calculate_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> ClassificationMetrics:
    precision, recall, f1_values, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    label_index = {label: i for i, label in enumerate(labels)}

    def metric(values: np.ndarray, label: str) -> float:
        idx = label_index.get(label)
        return float(values[idx]) if idx is not None else 0.0

    return ClassificationMetrics(
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
        draw_precision=metric(precision, "D"),
        draw_recall=metric(recall, "D"),
        draw_f1=metric(f1_values, "D"),
        away_recall=metric(recall, "A"),
        home_recall=metric(recall, "H"),
    )
