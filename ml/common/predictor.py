from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.features import FEATURE_COLUMNS
from common.models import create_random_forest


SUPPORTED_MODELS = ("rf", "lgbm", "ensemble")


@dataclass
class PredictionResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    classes: list[str]


def create_lightgbm():
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def align_probabilities(
    model_classes,
    probabilities: np.ndarray,
    target_classes: list[str],
) -> np.ndarray:
    aligned = np.zeros(
        (probabilities.shape[0], len(target_classes)),
        dtype=float,
    )

    model_class_list = list(model_classes)

    for target_index, target_class in enumerate(target_classes):
        if target_class in model_class_list:
            source_index = model_class_list.index(target_class)
            aligned[:, target_index] = probabilities[:, source_index]

    return aligned


def validate_model_name(model_name: str) -> str:
    normalized = model_name.lower().strip()

    if normalized not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: {', '.join(SUPPORTED_MODELS)}"
        )

    return normalized


def train_and_predict(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model_name: str,
    rf_weight: float = 0.7,
    lgbm_weight: float = 0.3,
) -> PredictionResult:
    model_name = validate_model_name(model_name)

    if train_df.empty:
        raise ValueError("Training data is empty.")

    if target_df.empty:
        raise ValueError("Target data is empty.")

    if abs((rf_weight + lgbm_weight) - 1.0) > 1e-9:
        raise ValueError("RF and LightGBM weights must total 1.0.")

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in train_df.columns or feature not in target_df.columns
    ]

    if missing:
        raise ValueError(f"Missing feature columns: {sorted(set(missing))}")

    x_train = train_df[FEATURE_COLUMNS].fillna(0)
    x_target = target_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["result"].astype(str)

    target_classes = ["A", "D", "H"]

    if model_name == "rf":
        rf = create_random_forest()
        rf.fit(x_train, y_train)

        probabilities = align_probabilities(
            rf.classes_,
            rf.predict_proba(x_target),
            target_classes,
        )

    elif model_name == "lgbm":
        lgbm = create_lightgbm()
        lgbm.fit(x_train, y_train)

        probabilities = align_probabilities(
            lgbm.classes_,
            lgbm.predict_proba(x_target),
            target_classes,
        )

    else:
        rf = create_random_forest()
        lgbm = create_lightgbm()

        rf.fit(x_train, y_train)
        lgbm.fit(x_train, y_train)

        rf_probabilities = align_probabilities(
            rf.classes_,
            rf.predict_proba(x_target),
            target_classes,
        )

        lgbm_probabilities = align_probabilities(
            lgbm.classes_,
            lgbm.predict_proba(x_target),
            target_classes,
        )

        probabilities = (
            rf_probabilities * rf_weight
            + lgbm_probabilities * lgbm_weight
        )

    prediction_indexes = probabilities.argmax(axis=1)
    predictions = np.array(
        [target_classes[index] for index in prediction_indexes]
    )

    return PredictionResult(
        probabilities=probabilities,
        predictions=predictions,
        classes=target_classes,
    )