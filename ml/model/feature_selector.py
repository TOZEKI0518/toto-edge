from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RemovedFeature:
    feature: str
    reason: str
    detail: float | str


class FeatureSelector:
    """
    Lightweight train-only feature selector.

    Removes:
    - excessive missing-rate columns
    - constant/near-constant columns
    - one feature from highly correlated pairs
    """

    def __init__(
        self,
        missing_threshold: float = 0.50,
        variance_threshold: float = 1e-10,
        correlation_threshold: float = 0.98,
        protected_features: set[str] | None = None,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.protected_features = protected_features or set()

        self.selected_features_: list[str] = []
        self.removed_features_: list[RemovedFeature] = []
        self.fill_values_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "FeatureSelector":
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        candidates = list(numeric.columns)
        removed: dict[str, RemovedFeature] = {}

        for column in candidates:
            missing_rate = float(numeric[column].isna().mean())
            if (
                missing_rate > self.missing_threshold
                and column not in self.protected_features
            ):
                removed[column] = RemovedFeature(
                    column, "missing_rate", missing_rate
                )

        remaining = [c for c in candidates if c not in removed]

        for column in remaining:
            variance = float(numeric[column].var(skipna=True) or 0.0)
            unique_count = int(numeric[column].nunique(dropna=True))
            if (
                (unique_count <= 1 or variance <= self.variance_threshold)
                and column not in self.protected_features
            ):
                removed[column] = RemovedFeature(
                    column, "low_variance", variance
                )

        remaining = [c for c in remaining if c not in removed]

        if len(remaining) > 1:
            correlation = numeric[remaining].corr().abs()
            for i, left in enumerate(remaining):
                if left in removed:
                    continue
                for right in remaining[i + 1 :]:
                    if right in removed:
                        continue
                    value = correlation.at[left, right]
                    if pd.isna(value) or value < self.correlation_threshold:
                        continue

                    if left in self.protected_features:
                        drop = right
                    elif right in self.protected_features:
                        drop = left
                    else:
                        left_missing = numeric[left].isna().mean()
                        right_missing = numeric[right].isna().mean()
                        drop = right if right_missing >= left_missing else left

                    removed[drop] = RemovedFeature(
                        drop,
                        "high_correlation",
                        f"{left}|{right}|{value:.6f}",
                    )

        self.selected_features_ = [
            c for c in candidates if c not in removed
        ]
        self.removed_features_ = list(removed.values())
        self.fill_values_ = {
            column: float(numeric[column].median())
            if numeric[column].notna().any()
            else 0.0
            for column in self.selected_features_
        }

        if not self.selected_features_:
            raise ValueError("Feature selection removed every feature.")

        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("FeatureSelector has not been fitted.")

        missing = set(self.selected_features_) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Input is missing selected features: {sorted(missing)}"
            )

        result = frame[self.selected_features_].copy()

        for column in self.selected_features_:
            result[column] = (
                pd.to_numeric(result[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.fill_values_[column])
            )

        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def selected_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.selected_features_, "selected": True}
        )

    def removed_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": item.feature,
                    "reason": item.reason,
                    "detail": item.detail,
                }
                for item in self.removed_features_
            ]
        )
