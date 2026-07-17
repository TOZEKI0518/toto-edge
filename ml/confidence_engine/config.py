from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

_DEFAULT_OUTCOME_ORDER: Final[tuple[str, str, str]] = ("A", "D", "H")


@dataclass(frozen=True, slots=True)
class ProbabilityColumnConfig:
    """Column names for one three-way probability distribution."""

    away: str
    draw: str
    home: str

    def __post_init__(self) -> None:
        """Validate configured column names."""
        values = self.as_tuple()

        if any(not value.strip() for value in values):
            raise ValueError("Probability column names must not be blank.")
        if len(set(values)) != len(values):
            raise ValueError(
                "Probability column names must be unique within a group."
            )

    def as_tuple(self) -> tuple[str, str, str]:
        """
        Return columns in canonical Project Alpha order: Away, Draw, Home.

        Returns:
            Tuple containing away, draw, and home probability columns.
        """
        return self.away, self.draw, self.home

    def as_mapping(self) -> dict[str, str]:
        """
        Return outcome-to-column mapping.

        Returns:
            Mapping using ``A``, ``D``, and ``H`` outcome labels.
        """
        return {
            "A": self.away,
            "D": self.draw,
            "H": self.home,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceWeightConfig:
    """Weights used by the current rule-based confidence score."""

    top_probability: float = 0.35
    probability_margin: float = 0.30
    entropy_certainty: float = 0.20
    model_agreement: float = 0.15

    def __post_init__(self) -> None:
        """Validate weight values and total."""
        values = (
            self.top_probability,
            self.probability_margin,
            self.entropy_certainty,
            self.model_agreement,
        )

        if any(value < 0.0 for value in values):
            raise ValueError("Confidence weights must be non-negative.")

        total = sum(values)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Confidence weights must sum to 1.0; "
                f"received {total:.12f}."
            )

    def as_mapping(self) -> dict[str, float]:
        """
        Return the confidence weights as a dictionary.

        Returns:
            Feature-name to weight mapping.
        """
        return {
            "top_probability": self.top_probability,
            "probability_margin": self.probability_margin,
            "entropy_certainty": self.entropy_certainty,
            "model_agreement": self.model_agreement,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceThresholdConfig:
    """Thresholds used for confidence labels and ticket recommendations."""

    high_confidence: float = 0.70
    medium_confidence: float = 0.50
    single_min_top_probability: float = 0.50
    single_min_margin: float = 0.10
    double_min_combined_probability: float = 0.72
    positive_edge: float = 0.00
    strong_edge: float = 0.08
    extreme_edge: float = 0.15

    def __post_init__(self) -> None:
        """Validate threshold ranges and ordering."""
        probability_values = {
            "high_confidence": self.high_confidence,
            "medium_confidence": self.medium_confidence,
            "single_min_top_probability": self.single_min_top_probability,
            "single_min_margin": self.single_min_margin,
            "double_min_combined_probability": (
                self.double_min_combined_probability
            ),
        }

        for name, value in probability_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

        if self.high_confidence < self.medium_confidence:
            raise ValueError(
                "high_confidence must be greater than or equal to "
                "medium_confidence."
            )

        if not (
            self.positive_edge
            <= self.strong_edge
            <= self.extreme_edge
        ):
            raise ValueError(
                "Edge thresholds must satisfy "
                "positive_edge <= strong_edge <= extreme_edge."
            )


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Settings used when integrating market probabilities."""

    enabled: bool = True
    require_exact_match_keys: bool = True
    normalize_probabilities: bool = True
    value_ratio_cap: float = 3.0

    def __post_init__(self) -> None:
        """Validate market-related settings."""
        if self.value_ratio_cap <= 0.0:
            raise ValueError("value_ratio_cap must be positive.")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Configuration for probability-calibration diagnostics."""

    enabled: bool = True
    n_bins: int = 15
    method: str = "isotonic"

    def __post_init__(self) -> None:
        """Validate calibration settings."""
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2.")

        allowed_methods = {"isotonic", "sigmoid", "none"}
        normalized_method = self.method.strip().lower()

        if normalized_method not in allowed_methods:
            raise ValueError(
                "method must be one of: "
                + ", ".join(sorted(allowed_methods))
            )

        object.__setattr__(self, "method", normalized_method)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output paths and file-generation settings."""

    output_directory: Path = Path("ml/output/confidence_engine")
    prediction_filename: str = "current_round_confidence_v3.csv"
    summary_filename: str = "current_round_confidence_summary_v3.json"
    calibration_filename: str = "confidence_calibration_v3.csv"
    write_index: bool = False
    create_directories: bool = True

    def __post_init__(self) -> None:
        """Normalize paths and validate output filenames."""
        object.__setattr__(
            self,
            "output_directory",
            Path(self.output_directory),
        )

        filenames = (
            self.prediction_filename,
            self.summary_filename,
            self.calibration_filename,
        )

        if any(not filename.strip() for filename in filenames):
            raise ValueError("Output filenames must not be blank.")

        if Path(self.prediction_filename).suffix.lower() != ".csv":
            raise ValueError("prediction_filename must use the .csv suffix.")
        if Path(self.summary_filename).suffix.lower() != ".json":
            raise ValueError("summary_filename must use the .json suffix.")
        if Path(self.calibration_filename).suffix.lower() != ".csv":
            raise ValueError("calibration_filename must use the .csv suffix.")

    @property
    def prediction_path(self) -> Path:
        """Return the complete prediction-output path."""
        return self.output_directory / self.prediction_filename

    @property
    def summary_path(self) -> Path:
        """Return the complete summary-output path."""
        return self.output_directory / self.summary_filename

    @property
    def calibration_path(self) -> Path:
        """Return the complete calibration-output path."""
        return self.output_directory / self.calibration_filename


@dataclass(frozen=True, slots=True)
class ConfidenceEngineConfig:
    """
    Top-level configuration for Project Alpha Confidence Engine Version 3.

    Defaults preserve the current rule-based architecture while allowing the
    builder, validator, exporter, optimizer, and future learned confidence
    model to share one immutable configuration object.
    """

    outcome_order: tuple[str, str, str] = _DEFAULT_OUTCOME_ORDER

    match_id_column: str = "match_id"
    home_team_column: str = "home_team"
    away_team_column: str = "away_team"
    actual_result_column: str = "actual_result"

    ensemble_probabilities: ProbabilityColumnConfig = field(
        default_factory=lambda: ProbabilityColumnConfig(
            away="ensemble_prob_a",
            draw="ensemble_prob_d",
            home="ensemble_prob_h",
        )
    )
    random_forest_probabilities: ProbabilityColumnConfig = field(
        default_factory=lambda: ProbabilityColumnConfig(
            away="rf_prob_a",
            draw="rf_prob_d",
            home="rf_prob_h",
        )
    )
    lightgbm_probabilities: ProbabilityColumnConfig = field(
        default_factory=lambda: ProbabilityColumnConfig(
            away="lgbm_prob_a",
            draw="lgbm_prob_d",
            home="lgbm_prob_h",
        )
    )
    market_probabilities: ProbabilityColumnConfig = field(
        default_factory=lambda: ProbabilityColumnConfig(
            away="market_prob_a",
            draw="market_prob_d",
            home="market_prob_h",
        )
    )

    weights: ConfidenceWeightConfig = field(
        default_factory=ConfidenceWeightConfig
    )
    thresholds: ConfidenceThresholdConfig = field(
        default_factory=ConfidenceThresholdConfig
    )
    market: MarketConfig = field(default_factory=MarketConfig)
    calibration: CalibrationConfig = field(
        default_factory=CalibrationConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)

    probability_tolerance: float = 1e-8
    allow_zero_probability: bool = True

    def __post_init__(self) -> None:
        """Validate top-level configuration consistency."""
        normalized_order = tuple(
            str(label).strip().upper()
            for label in self.outcome_order
        )
        if normalized_order != _DEFAULT_OUTCOME_ORDER:
            raise ValueError(
                "outcome_order must be exactly ('A', 'D', 'H') to match "
                "Project Alpha probability-column order."
            )

        object.__setattr__(self, "outcome_order", normalized_order)

        name_columns = (
            self.match_id_column,
            self.home_team_column,
            self.away_team_column,
            self.actual_result_column,
        )
        if any(not column.strip() for column in name_columns):
            raise ValueError("Configured core column names must not be blank.")

        if self.probability_tolerance <= 0.0:
            raise ValueError("probability_tolerance must be positive.")

        self._validate_probability_column_uniqueness()

    @property
    def key_columns(self) -> tuple[str, ...]:
        """
        Return the preferred match key columns.

        Returns:
            Tuple containing the configured match ID column.
        """
        return (self.match_id_column,)

    @property
    def team_columns(self) -> tuple[str, str]:
        """Return home-team and away-team columns."""
        return self.home_team_column, self.away_team_column

    @property
    def prediction_probability_groups(
        self,
    ) -> dict[str, tuple[str, str, str]]:
        """
        Return model probability groups for validator integration.

        Returns:
            Model-name to ordered probability-column mapping.
        """
        return {
            "ensemble": self.ensemble_probabilities.as_tuple(),
            "random_forest": self.random_forest_probabilities.as_tuple(),
            "lightgbm": self.lightgbm_probabilities.as_tuple(),
        }

    @property
    def market_probability_columns(self) -> tuple[str, str, str]:
        """Return market probability columns in A, D, H order."""
        return self.market_probabilities.as_tuple()

    @property
    def required_prediction_columns(self) -> tuple[str, ...]:
        """
        Return all mandatory prediction columns.

        Returns:
            Deduplicated ordered tuple of required columns.
        """
        columns = [
            self.match_id_column,
            self.home_team_column,
            self.away_team_column,
        ]

        for probability_group in self.prediction_probability_groups.values():
            columns.extend(probability_group)

        return tuple(dict.fromkeys(columns))

    @property
    def required_market_columns(self) -> tuple[str, ...]:
        """
        Return all mandatory market columns.

        Returns:
            Deduplicated ordered tuple of required columns.
        """
        columns = [
            self.match_id_column,
            self.home_team_column,
            self.away_team_column,
            *self.market_probability_columns,
        ]
        return tuple(dict.fromkeys(columns))

    def ensure_output_directory(self) -> Path:
        """
        Create and return the configured output directory when enabled.

        Returns:
            Configured output directory.

        Raises:
            FileNotFoundError: If directory creation is disabled and the
                directory does not exist.
        """
        directory = self.output.output_directory

        if self.output.create_directories:
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {directory}"
            )

        return directory

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete configuration into JSON-compatible data.

        Returns:
            Nested dictionary representation.
        """
        raw = asdict(self)
        output = dict(raw["output"])
        output["output_directory"] = str(output["output_directory"])
        raw["output"] = output
        return raw

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "ConfidenceEngineConfig":
        """
        Construct configuration from a nested mapping.

        This helper is suitable for later YAML or JSON loading while retaining
        typed dataclass validation.

        Args:
            values: Nested configuration mapping.

        Returns:
            Validated ``ConfidenceEngineConfig`` instance.
        """
        data = dict(values)

        def build_probability_config(
            key: str,
            default: ProbabilityColumnConfig,
        ) -> ProbabilityColumnConfig:
            raw = data.pop(key, None)
            if raw is None:
                return default
            if isinstance(raw, ProbabilityColumnConfig):
                return raw
            return ProbabilityColumnConfig(**dict(raw))

        def build_nested(
            key: str,
            config_type: type[Any],
            default: Any,
        ) -> Any:
            raw = data.pop(key, None)
            if raw is None:
                return default
            if isinstance(raw, config_type):
                return raw
            return config_type(**dict(raw))

        default_config = cls()

        return cls(
            ensemble_probabilities=build_probability_config(
                "ensemble_probabilities",
                default_config.ensemble_probabilities,
            ),
            random_forest_probabilities=build_probability_config(
                "random_forest_probabilities",
                default_config.random_forest_probabilities,
            ),
            lightgbm_probabilities=build_probability_config(
                "lightgbm_probabilities",
                default_config.lightgbm_probabilities,
            ),
            market_probabilities=build_probability_config(
                "market_probabilities",
                default_config.market_probabilities,
            ),
            weights=build_nested(
                "weights",
                ConfidenceWeightConfig,
                default_config.weights,
            ),
            thresholds=build_nested(
                "thresholds",
                ConfidenceThresholdConfig,
                default_config.thresholds,
            ),
            market=build_nested(
                "market",
                MarketConfig,
                default_config.market,
            ),
            calibration=build_nested(
                "calibration",
                CalibrationConfig,
                default_config.calibration,
            ),
            output=build_nested(
                "output",
                OutputConfig,
                default_config.output,
            ),
            **data,
        )

    def _validate_probability_column_uniqueness(self) -> None:
        """Ensure probability-column names do not overlap across groups."""
        groups = {
            "ensemble": self.ensemble_probabilities.as_tuple(),
            "random_forest": self.random_forest_probabilities.as_tuple(),
            "lightgbm": self.lightgbm_probabilities.as_tuple(),
            "market": self.market_probabilities.as_tuple(),
        }

        ownership: dict[str, str] = {}
        duplicates: list[str] = []

        for group_name, columns in groups.items():
            for column in columns:
                previous_group = ownership.get(column)
                if previous_group is not None:
                    duplicates.append(
                        f"{column} ({previous_group}, {group_name})"
                    )
                else:
                    ownership[column] = group_name

        if duplicates:
            raise ValueError(
                "Probability column names must be unique across groups: "
                + ", ".join(duplicates)
            )
