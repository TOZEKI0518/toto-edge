from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    """Thin wrapper around RandomForestClassifier."""

    def __init__(
        self,
        n_estimators: int = 900,
        max_depth: int | None = 14,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
        max_features: str | float = "sqrt",
        random_state: int = 42,
    ) -> None:
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    @property
    def classes_(self) -> np.ndarray:
        return self.estimator.classes_

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator.feature_importances_

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "RandomForestModel":
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
    def load(cls, path: Path) -> "RandomForestModel":
        instance = cls()
        instance.estimator = joblib.load(path)
        return instance
