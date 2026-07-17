from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_ALLOWED_OUTCOMES: Final[tuple[str, str, str]] = ("A", "D", "H")


class ConfidenceLevel(StrEnum):
    """Confidence category assigned to a match prediction."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TicketRecommendation(StrEnum):
    """Recommended ticket coverage for one match."""

    SINGLE = "SINGLE"
    DOUBLE = "DOUBLE"
    TRIPLE = "TRIPLE"


class EdgeLevel(StrEnum):
    """Market-edge category assigned to a prediction."""

    NONE = "NONE"
    POSITIVE = "POSITIVE"
    STRONG = "STRONG"
    EXTREME = "EXTREME"


@dataclass(frozen=True, slots=True)
class ProbabilityVector:
    """
    Immutable three-way probability distribution.

    Probabilities follow the canonical Project Alpha order:

    - ``A``: away win;
    - ``D``: draw;
    - ``H``: home win.
    """

    away: float
    draw: float
    home: float

    def __post_init__(self) -> None:
        """Validate probability values and normalize floating-point noise."""
        values = np.asarray(self.as_tuple(), dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError("ProbabilityVector contains NaN or infinity.")
        if (values < 0.0).any():
            raise ValueError("ProbabilityVector contains negative values.")

        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("ProbabilityVector total must be positive.")

        normalized = values / total
        object.__setattr__(self, "away", float(normalized[0]))
        object.__setattr__(self, "draw", float(normalized[1]))
        object.__setattr__(self, "home", float(normalized[2]))

    def as_tuple(self) -> tuple[float, float, float]:
        """Return probabilities in A, D, H order."""
        return self.away, self.draw, self.home

    def as_mapping(self) -> dict[str, float]:
        """Return outcome-to-probability mapping."""
        return {
            "A": self.away,
            "D": self.draw,
            "H": self.home,
        }

    def top_outcome(self) -> str:
        """Return the outcome with the highest probability."""
        index = int(np.argmax(self.as_tuple()))
        return _ALLOWED_OUTCOMES[index]

    def top_probability(self) -> float:
        """Return the highest probability."""
        return float(max(self.as_tuple()))

    def sorted_outcomes(self) -> tuple[str, str, str]:
        """Return outcome labels ordered from highest to lowest probability."""
        values = np.asarray(self.as_tuple(), dtype=np.float64)
        order = np.argsort(-values, kind="stable")
        return tuple(_ALLOWED_OUTCOMES[int(index)] for index in order)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float],
    ) -> "ProbabilityVector":
        """
        Construct from a mapping containing ``A``, ``D``, and ``H``.

        Args:
            values: Outcome-to-probability mapping.

        Returns:
            Validated ``ProbabilityVector``.
        """
        normalized = {
            str(key).strip().upper(): float(value)
            for key, value in values.items()
        }
        missing = [outcome for outcome in _ALLOWED_OUTCOMES if outcome not in normalized]
        if missing:
            raise ValueError(
                "Probability mapping is missing outcomes: "
                + ", ".join(missing)
            )

        return cls(
            away=normalized["A"],
            draw=normalized["D"],
            home=normalized["H"],
        )

    @classmethod
    def from_sequence(
        cls,
        values: Sequence[float],
    ) -> "ProbabilityVector":
        """
        Construct from a three-value sequence in A, D, H order.

        Args:
            values: Probability sequence.

        Returns:
            Validated ``ProbabilityVector``.
        """
        if len(values) != 3:
            raise ValueError("Probability sequence must contain exactly 3 values.")
        return cls(
            away=float(values[0]),
            draw=float(values[1]),
            home=float(values[2]),
        )


@dataclass(frozen=True, slots=True)
class ModelAgreementMetrics:
    """Agreement diagnostics between two prediction models."""

    same_top_pick: bool
    jensen_shannon_distance: float
    maximum_probability_gap: float

    def __post_init__(self) -> None:
        """Validate agreement metrics."""
        if not 0.0 <= self.jensen_shannon_distance <= 1.0:
            raise ValueError(
                "jensen_shannon_distance must be in [0, 1]."
            )
        if not 0.0 <= self.maximum_probability_gap <= 1.0:
            raise ValueError(
                "maximum_probability_gap must be in [0, 1]."
            )


@dataclass(frozen=True, slots=True)
class MarketEdgeMetrics:
    """Market-relative value diagnostics for one prediction."""

    best_outcome: str
    best_edge: float
    best_value_ratio: float
    edge_level: EdgeLevel = EdgeLevel.NONE

    def __post_init__(self) -> None:
        """Validate market edge values."""
        outcome = self.best_outcome.strip().upper()
        if outcome not in _ALLOWED_OUTCOMES:
            raise ValueError(
                "best_outcome must be one of: "
                + ", ".join(_ALLOWED_OUTCOMES)
            )
        if not np.isfinite(self.best_edge):
            raise ValueError("best_edge must be finite.")
        if not np.isfinite(self.best_value_ratio):
            raise ValueError("best_value_ratio must be finite.")
        if self.best_value_ratio < 0.0:
            raise ValueError("best_value_ratio must be non-negative.")

        object.__setattr__(self, "best_outcome", outcome)


@dataclass(frozen=True, slots=True)
class MatchConfidenceResult:
    """
    Complete confidence-engine result for one match.

    This object is intentionally independent of DataFrame implementation
    details and can be reused by exporters, optimizers, APIs, and tests.
    """

    match_id: str
    home_team: str
    away_team: str

    probabilities: ProbabilityVector
    top_outcome: str
    second_outcome: str

    top_probability: float
    second_probability: float
    third_probability: float
    probability_margin: float
    normalized_entropy: float
    entropy_certainty: float

    confidence_score: float
    confidence_level: ConfidenceLevel
    ticket_recommendation: TicketRecommendation
    covered_outcomes: tuple[str, ...]

    model_agreement: ModelAgreementMetrics | None = None
    market_edge: MarketEdgeMetrics | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate match-level confidence output."""
        if not str(self.match_id).strip():
            raise ValueError("match_id must not be blank.")
        if not self.home_team.strip():
            raise ValueError("home_team must not be blank.")
        if not self.away_team.strip():
            raise ValueError("away_team must not be blank.")

        top_outcome = self.top_outcome.strip().upper()
        second_outcome = self.second_outcome.strip().upper()
        covered_outcomes = tuple(
            str(outcome).strip().upper()
            for outcome in self.covered_outcomes
        )

        if top_outcome not in _ALLOWED_OUTCOMES:
            raise ValueError("top_outcome must be A, D, or H.")
        if second_outcome not in _ALLOWED_OUTCOMES:
            raise ValueError("second_outcome must be A, D, or H.")
        if top_outcome == second_outcome:
            raise ValueError(
                "top_outcome and second_outcome must be different."
            )
        if not covered_outcomes:
            raise ValueError("covered_outcomes must contain at least one outcome.")
        if any(outcome not in _ALLOWED_OUTCOMES for outcome in covered_outcomes):
            raise ValueError("covered_outcomes contains an invalid outcome.")
        if len(set(covered_outcomes)) != len(covered_outcomes):
            raise ValueError("covered_outcomes must not contain duplicates.")
        if top_outcome not in covered_outcomes:
            raise ValueError("covered_outcomes must include top_outcome.")

        probability_values = {
            "top_probability": self.top_probability,
            "second_probability": self.second_probability,
            "third_probability": self.third_probability,
            "probability_margin": self.probability_margin,
            "normalized_entropy": self.normalized_entropy,
            "entropy_certainty": self.entropy_certainty,
            "confidence_score": self.confidence_score,
        }

        for name, value in probability_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

        if self.top_probability < self.second_probability:
            raise ValueError(
                "top_probability must be greater than or equal to "
                "second_probability."
            )
        if self.second_probability < self.third_probability:
            raise ValueError(
                "second_probability must be greater than or equal to "
                "third_probability."
            )

        expected_margin = self.top_probability - self.second_probability
        if abs(self.probability_margin - expected_margin) > 1e-9:
            raise ValueError(
                "probability_margin must equal "
                "top_probability - second_probability."
            )

        expected_certainty = 1.0 - self.normalized_entropy
        if abs(self.entropy_certainty - expected_certainty) > 1e-9:
            raise ValueError(
                "entropy_certainty must equal 1 - normalized_entropy."
            )

        expected_coverage = {
            TicketRecommendation.SINGLE: 1,
            TicketRecommendation.DOUBLE: 2,
            TicketRecommendation.TRIPLE: 3,
        }[self.ticket_recommendation]
        if len(covered_outcomes) != expected_coverage:
            raise ValueError(
                "covered_outcomes length does not match "
                f"{self.ticket_recommendation.value}."
            )

        object.__setattr__(self, "match_id", str(self.match_id).strip())
        object.__setattr__(self, "top_outcome", top_outcome)
        object.__setattr__(self, "second_outcome", second_outcome)
        object.__setattr__(self, "covered_outcomes", covered_outcomes)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_record(self) -> dict[str, Any]:
        """
        Convert this result into a flat export-ready dictionary.

        Returns:
            Flat record suitable for CSV or JSON export.
        """
        record: dict[str, Any] = {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prob_a": self.probabilities.away,
            "prob_d": self.probabilities.draw,
            "prob_h": self.probabilities.home,
            "top_outcome": self.top_outcome,
            "second_outcome": self.second_outcome,
            "top_probability": self.top_probability,
            "second_probability": self.second_probability,
            "third_probability": self.third_probability,
            "probability_margin": self.probability_margin,
            "normalized_entropy": self.normalized_entropy,
            "entropy_certainty": self.entropy_certainty,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "ticket_recommendation": self.ticket_recommendation.value,
            "covered_outcomes": ",".join(self.covered_outcomes),
        }

        if self.model_agreement is not None:
            record.update(
                {
                    "model_same_top_pick": self.model_agreement.same_top_pick,
                    "model_js_distance": (
                        self.model_agreement.jensen_shannon_distance
                    ),
                    "model_max_probability_gap": (
                        self.model_agreement.maximum_probability_gap
                    ),
                }
            )

        if self.market_edge is not None:
            record.update(
                {
                    "market_best_outcome": self.market_edge.best_outcome,
                    "market_best_edge": self.market_edge.best_edge,
                    "market_best_value_ratio": (
                        self.market_edge.best_value_ratio
                    ),
                    "market_edge_level": self.market_edge.edge_level.value,
                }
            )

        record.update(dict(self.metadata))
        return record


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    """Aggregate summary for one confidence-engine run."""

    total_matches: int
    high_confidence_matches: int
    medium_confidence_matches: int
    low_confidence_matches: int

    single_recommendations: int
    double_recommendations: int
    triple_recommendations: int

    mean_confidence_score: float
    mean_top_probability: float
    mean_probability_margin: float
    mean_normalized_entropy: float

    positive_edge_matches: int = 0
    strong_edge_matches: int = 0
    extreme_edge_matches: int = 0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate aggregate counts and averages."""
        count_fields = {
            "total_matches": self.total_matches,
            "high_confidence_matches": self.high_confidence_matches,
            "medium_confidence_matches": self.medium_confidence_matches,
            "low_confidence_matches": self.low_confidence_matches,
            "single_recommendations": self.single_recommendations,
            "double_recommendations": self.double_recommendations,
            "triple_recommendations": self.triple_recommendations,
            "positive_edge_matches": self.positive_edge_matches,
            "strong_edge_matches": self.strong_edge_matches,
            "extreme_edge_matches": self.extreme_edge_matches,
        }

        for name, value in count_fields.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

        confidence_total = (
            self.high_confidence_matches
            + self.medium_confidence_matches
            + self.low_confidence_matches
        )
        if confidence_total != self.total_matches:
            raise ValueError(
                "Confidence-level counts must sum to total_matches."
            )

        recommendation_total = (
            self.single_recommendations
            + self.double_recommendations
            + self.triple_recommendations
        )
        if recommendation_total != self.total_matches:
            raise ValueError(
                "Recommendation counts must sum to total_matches."
            )

        average_fields = {
            "mean_confidence_score": self.mean_confidence_score,
            "mean_top_probability": self.mean_top_probability,
            "mean_probability_margin": self.mean_probability_margin,
            "mean_normalized_entropy": self.mean_normalized_entropy,
        }

        for name, value in average_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_results(
        cls,
        results: Iterable[MatchConfidenceResult],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ConfidenceSummary":
        """
        Build an aggregate summary from match-level results.

        Args:
            results: Match confidence results.
            metadata: Optional run-level metadata.

        Returns:
            Calculated ``ConfidenceSummary``.

        Raises:
            ValueError: If no results are supplied.
        """
        result_list = list(results)
        if not result_list:
            raise ValueError("At least one result is required.")

        confidence_scores = np.asarray(
            [result.confidence_score for result in result_list],
            dtype=np.float64,
        )
        top_probabilities = np.asarray(
            [result.top_probability for result in result_list],
            dtype=np.float64,
        )
        margins = np.asarray(
            [result.probability_margin for result in result_list],
            dtype=np.float64,
        )
        entropies = np.asarray(
            [result.normalized_entropy for result in result_list],
            dtype=np.float64,
        )

        def count_confidence(level: ConfidenceLevel) -> int:
            return sum(
                result.confidence_level is level
                for result in result_list
            )

        def count_recommendation(
            recommendation: TicketRecommendation,
        ) -> int:
            return sum(
                result.ticket_recommendation is recommendation
                for result in result_list
            )

        edge_levels = [
            result.market_edge.edge_level
            for result in result_list
            if result.market_edge is not None
        ]

        return cls(
            total_matches=len(result_list),
            high_confidence_matches=count_confidence(
                ConfidenceLevel.HIGH
            ),
            medium_confidence_matches=count_confidence(
                ConfidenceLevel.MEDIUM
            ),
            low_confidence_matches=count_confidence(
                ConfidenceLevel.LOW
            ),
            single_recommendations=count_recommendation(
                TicketRecommendation.SINGLE
            ),
            double_recommendations=count_recommendation(
                TicketRecommendation.DOUBLE
            ),
            triple_recommendations=count_recommendation(
                TicketRecommendation.TRIPLE
            ),
            mean_confidence_score=float(confidence_scores.mean()),
            mean_top_probability=float(top_probabilities.mean()),
            mean_probability_margin=float(margins.mean()),
            mean_normalized_entropy=float(entropies.mean()),
            positive_edge_matches=sum(
                level in {
                    EdgeLevel.POSITIVE,
                    EdgeLevel.STRONG,
                    EdgeLevel.EXTREME,
                }
                for level in edge_levels
            ),
            strong_edge_matches=sum(
                level in {EdgeLevel.STRONG, EdgeLevel.EXTREME}
                for level in edge_levels
            ),
            extreme_edge_matches=sum(
                level is EdgeLevel.EXTREME
                for level in edge_levels
            ),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceRunResult:
    """Container holding all outputs produced by one engine run."""

    matches: tuple[MatchConfidenceResult, ...]
    summary: ConfidenceSummary
    dataframe: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate run-result consistency."""
        if not self.matches:
            raise ValueError("matches must contain at least one result.")
        if self.summary.total_matches != len(self.matches):
            raise ValueError(
                "summary.total_matches must equal len(matches)."
            )
        if len(self.dataframe) != len(self.matches):
            raise ValueError(
                "dataframe row count must equal len(matches)."
            )

    @classmethod
    def from_matches(
        cls,
        matches: Iterable[MatchConfidenceResult],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ConfidenceRunResult":
        """
        Build run output from match-level results.

        Args:
            matches: Match confidence results.
            metadata: Optional run metadata passed to the summary.

        Returns:
            Complete ``ConfidenceRunResult``.
        """
        match_tuple = tuple(matches)
        if not match_tuple:
            raise ValueError("At least one match result is required.")

        summary = ConfidenceSummary.from_results(
            match_tuple,
            metadata=metadata,
        )
        dataframe = pd.DataFrame(
            [match.to_record() for match in match_tuple]
        )

        return cls(
            matches=match_tuple,
            summary=summary,
            dataframe=dataframe,
        )
