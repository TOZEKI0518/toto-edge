from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Mapping

import pandas as pd

from ml.confidence_engine.config import ConfidenceEngineConfig
from ml.confidence_engine.exporter import ConfidenceExporter
from ml.confidence_engine.feature_engineering import (
    ConfidenceFeatureEngineer,
    ConfidenceFeatureFrame,
)
from ml.confidence_engine.models import (
    ConfidenceLevel,
    ConfidenceRunResult,
    EdgeLevel,
    MarketEdgeMetrics,
    MatchConfidenceResult,
    ModelAgreementMetrics,
    ProbabilityVector,
    TicketRecommendation,
)
from ml.confidence_engine.repository import (
    ConfidenceDataRepository,
    ConfidenceInputBundle,
    ConfidenceInputPaths,
)
from ml.confidence_engine.validator import ConfidenceInputValidator

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


class ConfidenceEngineBuilder:
    """
    Orchestrate Confidence Engine Version 3 processing.

    The builder coordinates repository loading, input validation, feature
    engineering, typed result construction, summary generation, and optional
    output export. Calculation details remain delegated to focused modules.
    """

    def __init__(
        self,
        *,
        config: ConfidenceEngineConfig | None = None,
        validator: ConfidenceInputValidator | None = None,
        repository: ConfidenceDataRepository | None = None,
        feature_engineer: ConfidenceFeatureEngineer | None = None,
        exporter: ConfidenceExporter | None = None,
    ) -> None:
        """
        Initialize the confidence-engine builder.

        Args:
            config: Immutable engine configuration.
            validator: Optional shared input validator.
            repository: Optional shared data repository.
            feature_engineer: Optional shared feature engineer.
            exporter: Optional shared exporter.
        """
        self._config = config or ConfidenceEngineConfig()

        self._validator = validator or ConfidenceInputValidator(
            probability_tolerance=self._config.probability_tolerance,
            allow_zero_probability=self._config.allow_zero_probability,
        )
        self._repository = repository or ConfidenceDataRepository(
            config=self._config,
            validator=self._validator,
        )
        self._feature_engineer = (
            feature_engineer
            or ConfidenceFeatureEngineer(
                config=self._config,
                validator=self._validator,
            )
        )
        self._exporter = exporter or ConfidenceExporter(
            config=self._config,
            repository=self._repository,
        )

    @property
    def config(self) -> ConfidenceEngineConfig:
        """Return the active immutable configuration."""
        return self._config

    def build_from_paths(
        self,
        paths: ConfidenceInputPaths,
        *,
        export: bool = False,
        prediction_output_path: str | Path | None = None,
        summary_output_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConfidenceRunResult:
        """
        Load input files and execute the confidence pipeline.

        Args:
            paths: Prediction, market, and optional actual-result paths.
            export: Whether to write CSV and JSON outputs.
            prediction_output_path: Optional prediction CSV override.
            summary_output_path: Optional summary JSON override.
            metadata: Optional run-level metadata.

        Returns:
            Complete confidence run result.
        """
        bundle = self._repository.load_inputs(paths, validate=True)

        run_metadata = {
            "prediction_source": str(paths.predictions),
            "market_source": (
                str(paths.market) if paths.market is not None else None
            ),
            "actual_results_source": (
                str(paths.actual_results)
                if paths.actual_results is not None
                else None
            ),
            **dict(metadata or {}),
        }

        return self.build_from_bundle(
            bundle,
            export=export,
            prediction_output_path=prediction_output_path,
            summary_output_path=summary_output_path,
            metadata=run_metadata,
        )

    def build_from_bundle(
        self,
        bundle: ConfidenceInputBundle,
        *,
        export: bool = False,
        prediction_output_path: str | Path | None = None,
        summary_output_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConfidenceRunResult:
        """
        Execute the pipeline from an already-loaded input bundle.

        Args:
            bundle: Loaded input DataFrames.
            export: Whether to write CSV and JSON outputs.
            prediction_output_path: Optional prediction CSV override.
            summary_output_path: Optional summary JSON override.
            metadata: Optional run-level metadata.

        Returns:
            Complete confidence run result.
        """
        return self.build_from_frames(
            predictions=bundle.predictions,
            market=bundle.market,
            actual_results=bundle.actual_results,
            export=export,
            prediction_output_path=prediction_output_path,
            summary_output_path=summary_output_path,
            metadata=metadata,
        )

    def build_from_frames(
        self,
        *,
        predictions: pd.DataFrame,
        market: pd.DataFrame | None = None,
        actual_results: pd.DataFrame | None = None,
        export: bool = False,
        prediction_output_path: str | Path | None = None,
        summary_output_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConfidenceRunResult:
        """
        Execute the complete confidence pipeline from DataFrames.

        Args:
            predictions: Model prediction probabilities.
            market: Optional market probabilities.
            actual_results: Optional actual outcomes used for diagnostics.
            export: Whether to write output files.
            prediction_output_path: Optional prediction CSV override.
            summary_output_path: Optional summary JSON override.
            metadata: Optional run-level metadata.

        Returns:
            Complete confidence run result.
        """
        started_at = datetime.now().astimezone()
        LOGGER.info(
            "Starting Confidence Engine V3: prediction_rows=%d, market=%s",
            len(predictions),
            market is not None,
        )

        engineered = self._feature_engineer.build(
            predictions,
            market=market,
            copy=True,
        )

        result_frame = engineered.frame
        if actual_results is not None:
            result_frame = self._attach_actual_results(
                result_frame,
                actual_results,
            )

        match_results = self._build_match_results(result_frame)

        run_metadata = {
            "engine_version": "3",
            "generated_at": started_at.isoformat(),
            "market_enabled": market is not None and self._config.market.enabled,
            "actual_results_attached": actual_results is not None,
            **dict(metadata or {}),
        }

        run_result = ConfidenceRunResult.from_matches(
            match_results,
            metadata=run_metadata,
        )

        export_frame = self._combine_typed_and_engineered_frames(
            typed_frame=run_result.dataframe,
            engineered_frame=result_frame,
        )

        run_result = ConfidenceRunResult(
            matches=run_result.matches,
            summary=run_result.summary,
            dataframe=export_frame,
        )

        if export:
            written = self._exporter.export_run(
                run_result,
                prediction_path=prediction_output_path,
                summary_path=summary_output_path,
            )
            LOGGER.info(
                "Confidence Engine V3 outputs written: %s",
                written,
            )

        LOGGER.info(
            "Completed Confidence Engine V3: rows=%d, high=%d, "
            "single=%d, mean_score=%.4f",
            run_result.summary.total_matches,
            run_result.summary.high_confidence_matches,
            run_result.summary.single_recommendations,
            run_result.summary.mean_confidence_score,
        )

        return run_result

    def _attach_actual_results(
        self,
        frame: pd.DataFrame,
        actual_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and merge actual match outcomes."""
        self._repository.validate_actual_results(
            actual_results
        ).raise_for_errors()

        key_report = self._validator.validate_matching_keys(
            frame,
            actual_results,
            key_columns=self._config.key_columns,
            left_name="predictions",
            right_name="actual_results",
            require_exact_match=True,
        )
        key_report.raise_for_errors()

        result_column = self._config.actual_result_column
        actual_subset = actual_results.loc[
            :,
            [*self._config.key_columns, result_column],
        ].copy()
        actual_subset[result_column] = (
            actual_subset[result_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        merged = frame.merge(
            actual_subset,
            how="left",
            on=list(self._config.key_columns),
            validate="one_to_one",
        )

        merged["prediction_correct"] = (
            merged["top_outcome"] == merged[result_column]
        )
        merged["actual_result_probability"] = self._actual_probabilities(
            merged
        )

        return merged

    def _actual_probabilities(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Return model probability assigned to each actual outcome."""
        result_column = self._config.actual_result_column
        probability_columns = (
            self._config.ensemble_probabilities.as_mapping()
        )

        values = []
        for actual_result, row in zip(
            frame[result_column],
            frame.to_dict(orient="records"),
            strict=True,
        ):
            column = probability_columns[str(actual_result)]
            values.append(float(row[column]))

        return pd.Series(values, index=frame.index, dtype="float64")

    def _build_match_results(
        self,
        frame: pd.DataFrame,
    ) -> tuple[MatchConfidenceResult, ...]:
        """Convert engineered rows into typed match-level results."""
        results: list[MatchConfidenceResult] = []

        for row in frame.to_dict(orient="records"):
            probabilities = ProbabilityVector(
                away=float(
                    row[self._config.ensemble_probabilities.away]
                ),
                draw=float(
                    row[self._config.ensemble_probabilities.draw]
                ),
                home=float(
                    row[self._config.ensemble_probabilities.home]
                ),
            )

            model_agreement = ModelAgreementMetrics(
                same_top_pick=bool(row["model_same_top_pick"]),
                jensen_shannon_distance=float(
                    row["model_js_distance"]
                ),
                maximum_probability_gap=float(
                    row["model_max_probability_gap"]
                ),
            )

            market_edge: MarketEdgeMetrics | None = None
            if "market_best_outcome" in row:
                market_edge = MarketEdgeMetrics(
                    best_outcome=str(row["market_best_outcome"]),
                    best_edge=float(row["market_best_edge"]),
                    best_value_ratio=float(
                        row["market_best_value_ratio"]
                    ),
                    edge_level=EdgeLevel(
                        str(row["market_edge_level"])
                    ),
                )

            metadata = self._result_metadata(row)

            results.append(
                MatchConfidenceResult(
                    match_id=str(row[self._config.match_id_column]),
                    home_team=str(row[self._config.home_team_column]),
                    away_team=str(row[self._config.away_team_column]),
                    probabilities=probabilities,
                    top_outcome=str(row["top_outcome"]),
                    second_outcome=str(row["second_outcome"]),
                    top_probability=float(row["top_probability"]),
                    second_probability=float(
                        row["second_probability"]
                    ),
                    third_probability=float(row["third_probability"]),
                    probability_margin=float(
                        row["probability_margin"]
                    ),
                    normalized_entropy=float(
                        row["normalized_entropy"]
                    ),
                    entropy_certainty=float(
                        row["entropy_certainty"]
                    ),
                    confidence_score=float(row["confidence_score"]),
                    confidence_level=ConfidenceLevel(
                        str(row["confidence_level"])
                    ),
                    ticket_recommendation=TicketRecommendation(
                        str(row["ticket_recommendation"])
                    ),
                    covered_outcomes=tuple(
                        str(row["covered_outcomes"]).split(",")
                    ),
                    model_agreement=model_agreement,
                    market_edge=market_edge,
                    metadata=metadata,
                )
            )

        return tuple(results)

    def _result_metadata(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Extract optional row-level fields for typed result metadata."""
        candidate_columns = (
            "coverage_count",
            "model_agreement_score",
            "model_similarity_score",
            "market_best_outcome_index",
            "ai_market_same_top_pick",
            "market_probability_for_ai_pick",
            "market_edge_for_ai_pick",
            "market_value_ratio_for_ai_pick",
            "roi_priority_score",
            self._config.actual_result_column,
            "prediction_correct",
            "actual_result_probability",
        )

        return {
            column: row[column]
            for column in candidate_columns
            if column in row
        }

    @staticmethod
    def _combine_typed_and_engineered_frames(
        *,
        typed_frame: pd.DataFrame,
        engineered_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preserve all engineered source columns while prioritizing typed values.
        """
        result = engineered_frame.copy(deep=True)

        for column in typed_frame.columns:
            result[column] = typed_frame[column].to_numpy(copy=True)

        preferred_front = (
            "match_id",
            "home_team",
            "away_team",
            "top_outcome",
            "top_probability",
            "confidence_score",
            "confidence_level",
            "ticket_recommendation",
            "covered_outcomes",
            "market_best_edge",
            "market_edge_level",
            "roi_priority_score",
        )

        front = [
            column for column in preferred_front if column in result.columns
        ]
        remaining = [
            column for column in result.columns if column not in front
        ]
        return result.loc[:, [*front, *remaining]].copy()
