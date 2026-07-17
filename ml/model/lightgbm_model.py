from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class LightGBMModel:
    """Thin wrapper around LightGBM's multiclass classifier."""

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.80,
        colsample_bytree: float = 0.80,
        reg_alpha: float = 0.10,
        reg_lambda: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.estimator = LGBMClassifier(
            objective="multiclass",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=5,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    @property
    def classes_(self) -> np.ndarray:
        return self.estimator.classes_

    @property
    def feature_importances_(self) -> np.ndarray:
        raw = self.estimator.feature_importances_.astype(float)
        total = raw.sum()
        return raw / total if total > 0 else raw

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "LightGBMModel":
        self.estimator.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(features)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        instance = cls()
        instance.estimator = joblib.load(path)
        return instance
