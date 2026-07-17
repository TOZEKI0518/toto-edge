from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping

import pandas as pd

from ml.confidence_engine.config import ConfidenceEngineConfig
from ml.confidence_engine.validator import (
    ConfidenceInputValidator,
    ValidationReport,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfidenceInputPaths:
    """Filesystem paths used by the confidence-engine repository."""

    predictions: Path
    market: Path | None = None
    actual_results: Path | None = None

    def __post_init__(self) -> None:
        """Normalize all configured paths."""
        object.__setattr__(self, "predictions", Path(self.predictions))

        if self.market is not None:
            object.__setattr__(self, "market", Path(self.market))
        if self.actual_results is not None:
            object.__setattr__(
                self,
                "actual_results",
                Path(self.actual_results),
            )


@dataclass(frozen=True, slots=True)
class ConfidenceInputBundle:
    """Loaded input DataFrames for one confidence-engine run."""

    predictions: pd.DataFrame
    market: pd.DataFrame | None = None
    actual_results: pd.DataFrame | None = None


class ConfidenceDataRepository:
    """
    Read and write confidence-engine data using filesystem-backed CSV files.

    The repository owns file I/O only. Validation logic remains delegated to
    ``ConfidenceInputValidator``, while feature generation and export formatting
    remain outside this class.
    """

    def __init__(
        self,
        *,
        config: ConfidenceEngineConfig | None = None,
        validator: ConfidenceInputValidator | None = None,
        encoding: str = "utf-8-sig",
    ) -> None:
        """
        Initialize the repository.

        Args:
            config: Confidence-engine configuration.
            validator: Optional shared validator.
            encoding: CSV encoding used for reads and writes.

        Raises:
            ValueError: If ``encoding`` is blank.
        """
        if not encoding.strip():
            raise ValueError("encoding must not be blank.")

        self._config = config or ConfidenceEngineConfig()
        self._validator = validator or ConfidenceInputValidator(
            probability_tolerance=self._config.probability_tolerance,
            allow_zero_probability=self._config.allow_zero_probability,
        )
        self._encoding = encoding

    @property
    def config(self) -> ConfidenceEngineConfig:
        """Return the active configuration."""
        return self._config

    @property
    def encoding(self) -> str:
        """Return the configured CSV encoding."""
        return self._encoding

    def load_inputs(
        self,
        paths: ConfidenceInputPaths,
        *,
        validate: bool = True,
        normalize_market_probabilities: bool | None = None,
    ) -> ConfidenceInputBundle:
        """
        Load all configured confidence-engine inputs.

        Args:
            paths: Input file paths.
            validate: Whether to validate loaded DataFrames.
            normalize_market_probabilities: Optional override controlling market
                probability normalization. ``None`` uses configuration.

        Returns:
            Loaded ``ConfidenceInputBundle``.

        Raises:
            FileNotFoundError: If a required configured file does not exist.
            ValueError: If validation fails.
        """
        predictions = self.load_predictions(
            paths.predictions,
            validate=validate,
        )

        market: pd.DataFrame | None = None
        if paths.market is not None:
            market = self.load_market(
                paths.market,
                validate=validate,
                normalize_probabilities=normalize_market_probabilities,
            )

        actual_results: pd.DataFrame | None = None
        if paths.actual_results is not None:
            actual_results = self.load_actual_results(
                paths.actual_results,
                validate=validate,
            )

        if validate and market is not None:
            self._validator.validate_matching_keys(
                predictions,
                market,
                key_columns=self._config.key_columns,
                left_name="predictions",
                right_name="market",
                require_exact_match=(
                    self._config.market.require_exact_match_keys
                ),
            ).raise_for_errors()

        return ConfidenceInputBundle(
            predictions=predictions,
            market=market,
            actual_results=actual_results,
        )

    def load_predictions(
        self,
        path: str | Path,
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load prediction input from CSV.

        Args:
            path: Prediction CSV path.
            validate: Whether to validate required fields and probabilities.

        Returns:
            Loaded prediction DataFrame.
        """
        frame = self.read_csv(path, frame_name="predictions")

        if validate:
            self.validate_predictions(frame).raise_for_errors()

        return frame

    def load_market(
        self,
        path: str | Path,
        *,
        validate: bool = True,
        normalize_probabilities: bool | None = None,
    ) -> pd.DataFrame:
        """
        Load market probabilities from CSV.

        Args:
            path: Market CSV path.
            validate: Whether to validate market input.
            normalize_probabilities: Whether to normalize market probabilities
                row by row. ``None`` uses configuration.

        Returns:
            Loaded market DataFrame.
        """
        frame = self.read_csv(path, frame_name="market")
        should_normalize = (
            self._config.market.normalize_probabilities
            if normalize_probabilities is None
            else normalize_probabilities
        )

        if should_normalize:
            normalized = self._validator.normalize_probability_columns(
                frame,
                self._config.market_probability_columns,
                frame_name="market",
                copy=False,
            )
            frame = normalized.frame
            if normalized.normalized_rows:
                LOGGER.info(
                    "Normalized market probabilities for %d row(s).",
                    len(normalized.normalized_rows),
                )

        if validate:
            self.validate_market(frame).raise_for_errors()

        return frame

    def load_actual_results(
        self,
        path: str | Path,
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load actual match results from CSV.

        Args:
            path: Actual-result CSV path.
            validate: Whether to validate key and result columns.

        Returns:
            Loaded actual-result DataFrame.
        """
        frame = self.read_csv(path, frame_name="actual_results")

        if validate:
            self.validate_actual_results(frame).raise_for_errors()

        return frame

    def read_csv(
        self,
        path: str | Path,
        *,
        frame_name: str = "frame",
        dtype: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Read a CSV file with consistent repository defaults.

        Args:
            path: CSV path.
            frame_name: Human-readable source name for logging and errors.
            dtype: Optional pandas dtype mapping.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the path does not exist.
            IsADirectoryError: If the path points to a directory.
            ValueError: If the CSV cannot be parsed into a non-empty DataFrame.
        """
        csv_path = self._require_file(path, frame_name=frame_name)

        LOGGER.info("Loading %s from %s", frame_name, csv_path)

        try:
            frame = pd.read_csv(
                csv_path,
                encoding=self._encoding,
                dtype=dtype,
                low_memory=False,
            )
        except UnicodeDecodeError:
            LOGGER.warning(
                "Failed to read %s with encoding %s; retrying with utf-8.",
                csv_path,
                self._encoding,
            )
            frame = pd.read_csv(
                csv_path,
                encoding="utf-8",
                dtype=dtype,
                low_memory=False,
            )
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{frame_name} CSV is empty: {csv_path}") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Failed to parse {frame_name} CSV: {csv_path}"
            ) from exc

        if frame.empty:
            raise ValueError(
                f"{frame_name} CSV contains no data rows: {csv_path}"
            )

        LOGGER.info(
            "Loaded %s: rows=%d, columns=%d",
            frame_name,
            len(frame),
            len(frame.columns),
        )
        return frame

    def save_csv(
        self,
        frame: pd.DataFrame,
        path: str | Path,
        *,
        index: bool = False,
        atomic: bool = True,
    ) -> Path:
        """
        Save a DataFrame to CSV.

        Args:
            frame: DataFrame to save.
            path: Destination path.
            index: Whether to write the pandas index.
            atomic: Whether to use a temporary file and atomic replacement.

        Returns:
            Final destination path.

        Raises:
            ValueError: If ``frame`` is not a DataFrame.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("frame must be a pandas DataFrame.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if atomic:
            self._atomic_write_csv(
                frame,
                destination,
                index=index,
            )
        else:
            frame.to_csv(
                destination,
                index=index,
                encoding=self._encoding,
            )

        LOGGER.info(
            "Saved CSV: path=%s, rows=%d, columns=%d",
            destination,
            len(frame),
            len(frame.columns),
        )
        return destination

    def validate_predictions(
        self,
        frame: pd.DataFrame,
    ) -> ValidationReport:
        """
        Validate a prediction DataFrame.

        Args:
            frame: Prediction DataFrame.

        Returns:
            Validation report.
        """
        return self._validator.validate_prediction_frame(
            frame,
            key_columns=self._config.key_columns,
            probability_groups=self._config.prediction_probability_groups,
            text_columns=self._config.team_columns,
            frame_name="predictions",
            require_normalized_probabilities=True,
        )

    def validate_market(
        self,
        frame: pd.DataFrame,
    ) -> ValidationReport:
        """
        Validate a market DataFrame.

        Args:
            frame: Market DataFrame.

        Returns:
            Validation report.
        """
        return self._validator.validate_market_frame(
            frame,
            key_columns=self._config.key_columns,
            market_probability_columns=(
                self._config.market_probability_columns
            ),
            text_columns=self._config.team_columns,
            frame_name="market",
            require_normalized_probabilities=True,
        )

    def validate_actual_results(
        self,
        frame: pd.DataFrame,
    ) -> ValidationReport:
        """
        Validate actual-result input.

        Args:
            frame: Actual-result DataFrame.

        Returns:
            Validation report.
        """
        required = (
            *self._config.key_columns,
            self._config.actual_result_column,
        )
        issues = list(
            self._validator.validate_non_empty(
                frame,
                frame_name="actual_results",
            ).issues
        )
        issues.extend(
            self._validator.validate_required_columns(
                frame,
                required,
                frame_name="actual_results",
            ).issues
        )

        if issues:
            return ValidationReport(
                is_valid=False,
                issues=tuple(issues),
            )

        issues.extend(
            self._validator.validate_unique_key(
                frame,
                self._config.key_columns,
                frame_name="actual_results",
            ).issues
        )
        issues.extend(
            self._validator.validate_text_columns(
                frame,
                (self._config.actual_result_column,),
                frame_name="actual_results",
            ).issues
        )

        normalized_results = (
            frame[self._config.actual_result_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        invalid_mask = ~normalized_results.isin(self._config.outcome_order)

        if invalid_mask.any():
            from ml.confidence_engine.validator import ValidationIssue

            issues.append(
                ValidationIssue(
                    code="INVALID_ACTUAL_RESULT",
                    message=(
                        "actual_results contains outcomes outside "
                        f"{self._config.outcome_order}."
                    ),
                    rows=tuple(
                        int(position)
                        for position in invalid_mask.to_numpy().nonzero()[0]
                    ),
                )
            )

        return ValidationReport(
            is_valid=not issues,
            issues=tuple(issues),
        )

    @staticmethod
    def select_columns(
        frame: pd.DataFrame,
        columns: Iterable[str],
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Select a stable ordered subset of columns.

        Args:
            frame: Source DataFrame.
            columns: Columns to select.
            copy: Whether to return a copied DataFrame.

        Returns:
            Selected DataFrame.

        Raises:
            ValueError: If any requested column is missing.
        """
        selected = tuple(dict.fromkeys(str(column) for column in columns))
        missing = [column for column in selected if column not in frame.columns]
        if missing:
            raise ValueError(
                "Missing requested columns: " + ", ".join(missing)
            )

        result = frame.loc[:, list(selected)]
        return result.copy(deep=True) if copy else result

    @staticmethod
    def _require_file(
        path: str | Path,
        *,
        frame_name: str,
    ) -> Path:
        """Validate and resolve an input file path."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"{frame_name} file does not exist: {file_path}"
            )
        if file_path.is_dir():
            raise IsADirectoryError(
                f"{frame_name} path points to a directory: {file_path}"
            )

        return file_path

    def _atomic_write_csv(
        self,
        frame: pd.DataFrame,
        destination: Path,
        *,
        index: bool,
    ) -> None:
        """Write CSV through a temporary file and atomically replace."""
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding=self._encoding,
                newline="",
                suffix=".tmp",
                prefix=f".{destination.stem}_",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                frame.to_csv(
                    temporary_file,
                    index=index,
                )

            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
