"""
Confidence Engine Version 3 package.

This package provides validation, feature engineering, typed models,
repository I/O, exporting, and orchestration for Project Alpha confidence
processing.
"""

from ml.confidence_engine.builder import ConfidenceEngineBuilder
from ml.confidence_engine.config import (
    CalibrationConfig,
    ConfidenceEngineConfig,
    ConfidenceThresholdConfig,
    ConfidenceWeightConfig,
    MarketConfig,
    OutputConfig,
    ProbabilityColumnConfig,
)
from ml.confidence_engine.exporter import ConfidenceExporter
from ml.confidence_engine.feature_engineering import (
    ConfidenceFeatureEngineer,
    ConfidenceFeatureFrame,
)
from ml.confidence_engine.metrics import (
    CalibrationMetrics,
    ProbabilityShapeMetrics,
    best_edge_metrics,
    calibration_errors,
    calibration_metrics,
    calibration_table,
    jensen_shannon_distance,
    maximum_probability_gap,
    multiclass_brier_score,
    multiclass_log_loss,
    normalized_entropy,
    outcome_indices,
    probability_edge,
    probability_shape_metrics,
    probability_value_ratio,
    roi_priority_score,
    top_pick_agreement,
)
from ml.confidence_engine.models import (
    ConfidenceLevel,
    ConfidenceRunResult,
    ConfidenceSummary,
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
from ml.confidence_engine.validator import (
    ConfidenceInputValidator,
    ProbabilityValidationResult,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "CalibrationConfig",
    "CalibrationMetrics",
    "ConfidenceDataRepository",
    "ConfidenceEngineBuilder",
    "ConfidenceEngineConfig",
    "ConfidenceExporter",
    "ConfidenceFeatureEngineer",
    "ConfidenceFeatureFrame",
    "ConfidenceInputBundle",
    "ConfidenceInputPaths",
    "ConfidenceInputValidator",
    "ConfidenceLevel",
    "ConfidenceRunResult",
    "ConfidenceSummary",
    "ConfidenceThresholdConfig",
    "ConfidenceWeightConfig",
    "EdgeLevel",
    "MarketConfig",
    "MarketEdgeMetrics",
    "MatchConfidenceResult",
    "ModelAgreementMetrics",
    "OutputConfig",
    "ProbabilityColumnConfig",
    "ProbabilityShapeMetrics",
    "ProbabilityValidationResult",
    "ProbabilityVector",
    "TicketRecommendation",
    "ValidationIssue",
    "ValidationReport",
    "best_edge_metrics",
    "calibration_errors",
    "calibration_metrics",
    "calibration_table",
    "jensen_shannon_distance",
    "maximum_probability_gap",
    "multiclass_brier_score",
    "multiclass_log_loss",
    "normalized_entropy",
    "outcome_indices",
    "probability_edge",
    "probability_shape_metrics",
    "probability_value_ratio",
    "roi_priority_score",
    "top_pick_agreement",
]

__version__ = "3.0.0"
