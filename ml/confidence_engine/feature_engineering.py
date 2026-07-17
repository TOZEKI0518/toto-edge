from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml.confidence_engine.config import ConfidenceEngineConfig
from ml.confidence_engine.metrics import (
    best_edge_metrics,
    jensen_shannon_distance,
    maximum_probability_gap,
    probability_shape_metrics,
    roi_priority_score,
    top_pick_agreement,
)
from ml.confidence_engine.models import (
    ConfidenceLevel,
    EdgeLevel,
    TicketRecommendation,
)
from ml.confidence_engine.validator import ConfidenceInputValidator

_OUTCOME_ORDER: Final[tuple[str, str, str]] = ("A", "D", "H")


@dataclass(frozen=True, slots=True)
class ConfidenceFeatureFrame:
    """Container for engineered confidence features."""

    frame: pd.DataFrame
    probability_columns: tuple[str, str, str]
    model_probability_groups: Mapping[str, tuple[str, str, str]]
    market_probability_columns: tuple[str, str, str] | None


class ConfidenceFeatureEngineer:
    """
    Build confidence, agreement, recommendation, and market-value features.

    The class preserves the current rule-based Version 3 design while keeping
    the implementation independent from exporters and orchestration code.
    """

    def __init__(
        self,
        config: ConfidenceEngineConfig | None = None,
        validator: ConfidenceInputValidator | None = None,
    ) -> None:
        """
        Initialize the feature engineer.

        Args:
            config: Confidence-engine configuration.
            validator: Optional shared validator instance.
        """
        self._config = config or ConfidenceEngineConfig()
        self._validator = validator or ConfidenceInputValidator(
            probability_tolerance=self._config.probability_tolerance,
            allow_zero_probability=self._config.allow_zero_probability,
        )

    @property
    def config(self) -> ConfidenceEngineConfig:
        """Return the active immutable configuration."""
        return self._config

    def build(
        self,
        predictions: pd.DataFrame,
        *,
        market: pd.DataFrame | None = None,
        copy: bool = True,
    ) -> ConfidenceFeatureFrame:
        """
        Build the complete feature frame.

        Args:
            predictions: Prediction DataFrame containing ensemble, Random
                Forest, and LightGBM probability columns.
            market: Optional market probability DataFrame.
            copy: Whether to preserve the original prediction DataFrame.

        Returns:
            A ``ConfidenceFeatureFrame``.

        Raises:
            ValueError: If validation fails.
        """
        prediction_report = self._validator.validate_prediction_frame(
            predictions,
            key_columns=self._config.key_columns,
            probability_groups=self._config.prediction_probability_groups,
            text_columns=self._config.team_columns,
            frame_name="predictions",
            require_normalized_probabilities=True,
        )
        prediction_report.raise_for_errors()

        result = predictions.copy(deep=True) if copy else predictions

        result = self._add_probability_shape_features(result)
        result = self._add_model_agreement_features(result)
        result = self._add_rule_based_confidence(result)
        result = self._add_confidence_labels(result)
        result = self._add_ticket_recommendations(result)

        market_columns: tuple[str, str, str] | None = None
        if market is not None and self._config.market.enabled:
            result = self._merge_market_features(result, market)
            market_columns = self._config.market_probability_columns

        return ConfidenceFeatureFrame(
            frame=result,
            probability_columns=self._config.ensemble_probabilities.as_tuple(),
            model_probability_groups=self._config.prediction_probability_groups,
            market_probability_columns=market_columns,
        )

    def _add_probability_shape_features(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add top probabilities, rank order, margin, and entropy features."""
        columns = self._config.ensemble_probabilities.as_tuple()
        matrix = self._validator.probability_matrix(frame, columns)
        metrics = probability_shape_metrics(matrix)

        frame["top_probability"] = metrics.top_probability
        frame["second_probability"] = metrics.second_probability
        frame["third_probability"] = metrics.third_probability
        frame["probability_margin"] = metrics.probability_margin
        frame["normalized_entropy"] = metrics.normalized_entropy
        frame["entropy_certainty"] = metrics.certainty_from_entropy
        frame["top_outcome_index"] = metrics.top_outcome_index
        frame["second_outcome_index"] = metrics.second_outcome_index
        frame["top_outcome"] = self._indices_to_outcomes(
            metrics.top_outcome_index
        )
        frame["second_outcome"] = self._indices_to_outcomes(
            metrics.second_outcome_index
        )

        return frame

    def _add_model_agreement_features(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add Random Forest versus LightGBM agreement diagnostics."""
        rf_columns = self._config.random_forest_probabilities.as_tuple()
        lgbm_columns = self._config.lightgbm_probabilities.as_tuple()

        rf_matrix = self._validator.probability_matrix(frame, rf_columns)
        lgbm_matrix = self._validator.probability_matrix(
            frame,
            lgbm_columns,
        )

        same_top_pick = top_pick_agreement(rf_matrix, lgbm_matrix)
        js_distance = jensen_shannon_distance(rf_matrix, lgbm_matrix)
        max_gap = maximum_probability_gap(rf_matrix, lgbm_matrix)

        frame["model_same_top_pick"] = same_top_pick
        frame["model_agreement_score"] = same_top_pick.astype(np.float64)
        frame["model_js_distance"] = js_distance
        frame["model_similarity_score"] = 1.0 - js_distance
        frame["model_max_probability_gap"] = max_gap

        return frame

    def _add_rule_based_confidence(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate the current weighted confidence score."""
        weights = self._config.weights

        score = (
            weights.top_probability * frame["top_probability"].to_numpy(
                dtype=np.float64
            )
            + weights.probability_margin
            * frame["probability_margin"].to_numpy(dtype=np.float64)
            + weights.entropy_certainty
            * frame["entropy_certainty"].to_numpy(dtype=np.float64)
            + weights.model_agreement
            * frame["model_agreement_score"].to_numpy(dtype=np.float64)
        )

        frame["confidence_score"] = np.clip(score, 0.0, 1.0)
        return frame

    def _add_confidence_labels(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign HIGH, MEDIUM, or LOW confidence levels."""
        thresholds = self._config.thresholds
        scores = frame["confidence_score"].to_numpy(dtype=np.float64)

        levels = np.select(
            [
                scores >= thresholds.high_confidence,
                scores >= thresholds.medium_confidence,
            ],
            [
                ConfidenceLevel.HIGH.value,
                ConfidenceLevel.MEDIUM.value,
            ],
            default=ConfidenceLevel.LOW.value,
        )

        frame["confidence_level"] = levels
        return frame

    def _add_ticket_recommendations(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign SINGLE, DOUBLE, or TRIPLE recommendation and coverage."""
        thresholds = self._config.thresholds

        top_probability = frame["top_probability"].to_numpy(
            dtype=np.float64
        )
        second_probability = frame["second_probability"].to_numpy(
            dtype=np.float64
        )
        margin = frame["probability_margin"].to_numpy(dtype=np.float64)

        single_mask = (
            (top_probability >= thresholds.single_min_top_probability)
            & (margin >= thresholds.single_min_margin)
        )
        double_mask = (
            ~single_mask
            & (
                top_probability + second_probability
                >= thresholds.double_min_combined_probability
            )
        )

        recommendation = np.select(
            [single_mask, double_mask],
            [
                TicketRecommendation.SINGLE.value,
                TicketRecommendation.DOUBLE.value,
            ],
            default=TicketRecommendation.TRIPLE.value,
        )

        frame["ticket_recommendation"] = recommendation
        frame["coverage_count"] = np.select(
            [single_mask, double_mask],
            [1, 2],
            default=3,
        ).astype(np.int64)

        frame["covered_outcomes"] = [
            self._coverage_string(
                top_outcome=str(top),
                second_outcome=str(second),
                recommendation=str(rec),
            )
            for top, second, rec in zip(
                frame["top_outcome"],
                frame["second_outcome"],
                frame["ticket_recommendation"],
                strict=True,
            )
        ]

        return frame

    def _merge_market_features(
        self,
        predictions: pd.DataFrame,
        market: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate, merge, and engineer market-relative features."""
        market_frame = market.copy(deep=True)

        if self._config.market.normalize_probabilities:
            normalized = self._validator.normalize_probability_columns(
                market_frame,
                self._config.market_probability_columns,
                frame_name="market",
                copy=False,
            )
            market_frame = normalized.frame
        else:
            market_report = self._validator.validate_market_frame(
                market_frame,
                key_columns=self._config.key_columns,
                market_probability_columns=(
                    self._config.market_probability_columns
                ),
                text_columns=self._config.team_columns,
                frame_name="market",
                require_normalized_probabilities=True,
            )
            market_report.raise_for_errors()

        key_report = self._validator.validate_matching_keys(
            predictions,
            market_frame,
            key_columns=self._config.key_columns,
            left_name="predictions",
            right_name="market",
            require_exact_match=(
                self._config.market.require_exact_match_keys
            ),
        )
        key_report.raise_for_errors()

        merge_columns = [
            *self._config.key_columns,
            *self._config.market_probability_columns,
        ]
        market_subset = market_frame.loc[:, merge_columns].copy()

        merged = predictions.merge(
            market_subset,
            how="left",
            on=list(self._config.key_columns),
            validate="one_to_one",
        )

        market_probability_columns = (
            self._config.market_probability_columns
        )
        missing_market_mask = merged.loc[
            :,
            list(market_probability_columns),
        ].isna().any(axis=1)

        if missing_market_mask.any():
            missing_rows = np.flatnonzero(
                missing_market_mask.to_numpy(dtype=bool)
            ).tolist()
            raise ValueError(
                "Market probabilities are missing after merge for rows: "
                f"{missing_rows}"
            )

        model_matrix = self._validator.probability_matrix(
            merged,
            self._config.ensemble_probabilities.as_tuple(),
        )
        market_matrix = self._validator.probability_matrix(
            merged,
            market_probability_columns,
        )

        best_edge, best_ratio, best_indices = best_edge_metrics(
            model_matrix,
            market_matrix,
        )

        merged["market_best_outcome_index"] = best_indices
        merged["market_best_outcome"] = self._indices_to_outcomes(
            best_indices
        )
        merged["market_best_edge"] = best_edge
        merged["market_best_value_ratio"] = best_ratio
        merged["market_edge_level"] = self._edge_levels(best_edge)
        merged["roi_priority_score"] = roi_priority_score(
            merged["confidence_score"].to_numpy(dtype=np.float64),
            best_edge,
            best_ratio,
            value_ratio_cap=self._config.market.value_ratio_cap,
        )

        ensemble_top_indices = merged["top_outcome_index"].to_numpy(
            dtype=np.int64
        )
        merged["ai_market_same_top_pick"] = (
            ensemble_top_indices
            == np.argmax(market_matrix, axis=1)
        )

        row_indices = np.arange(len(merged))
        merged["market_probability_for_ai_pick"] = market_matrix[
            row_indices,
            ensemble_top_indices,
        ]
        merged["market_edge_for_ai_pick"] = (
            model_matrix[row_indices, ensemble_top_indices]
            - market_matrix[row_indices, ensemble_top_indices]
        )
        merged["market_value_ratio_for_ai_pick"] = np.divide(
            model_matrix[row_indices, ensemble_top_indices],
            np.clip(
                market_matrix[row_indices, ensemble_top_indices],
                1e-15,
                None,
            ),
        )

        return merged

    def _edge_levels(
        self,
        best_edge: NDArray[np.float64],
    ) -> NDArray[np.str_]:
        """Convert edge values into configured edge categories."""
        thresholds = self._config.thresholds

        return np.select(
            [
                best_edge >= thresholds.extreme_edge,
                best_edge >= thresholds.strong_edge,
                best_edge > thresholds.positive_edge,
            ],
            [
                EdgeLevel.EXTREME.value,
                EdgeLevel.STRONG.value,
                EdgeLevel.POSITIVE.value,
            ],
            default=EdgeLevel.NONE.value,
        )

    @staticmethod
    def _indices_to_outcomes(
        indices: NDArray[np.int64],
    ) -> NDArray[np.str_]:
        """Convert zero-based class indices into A, D, H labels."""
        outcome_array = np.asarray(_OUTCOME_ORDER)
        if ((indices < 0) | (indices >= len(outcome_array))).any():
            raise ValueError("Outcome index is outside the valid range.")
        return outcome_array[indices]

    @staticmethod
    def _coverage_string(
        *,
        top_outcome: str,
        second_outcome: str,
        recommendation: str,
    ) -> str:
        """Build a stable comma-separated outcome coverage string."""
        if recommendation == TicketRecommendation.SINGLE.value:
            return top_outcome
        if recommendation == TicketRecommendation.DOUBLE.value:
            return f"{top_outcome},{second_outcome}"
        return ",".join(_OUTCOME_ORDER)
