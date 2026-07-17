from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_DEFAULT_TOLERANCE: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Single validation issue discovered in an input DataFrame."""

    code: str
    message: str
    rows: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured result returned by non-raising validation methods."""

    is_valid: bool
    issues: tuple[ValidationIssue, ...]

    def raise_for_errors(self) -> None:
        """
        Raise a consolidated ``ValueError`` when validation failed.

        Raises:
            ValueError: If one or more validation issues exist.
        """
        if self.is_valid:
            return

        lines = ["Input validation failed:"]
        for issue in self.issues:
            row_text = f" Rows: {list(issue.rows)}." if issue.rows else ""
            lines.append(f"- [{issue.code}] {issue.message}{row_text}")

        raise ValueError("\n".join(lines))


@dataclass(frozen=True, slots=True)
class ProbabilityValidationResult:
    """Validated and normalized probability data."""

    frame: pd.DataFrame
    probability_columns: tuple[str, ...]
    normalized_rows: tuple[int, ...]


class ConfidenceInputValidator:
    """
    Validate prediction and market inputs used by the confidence engine.

    The validator is intentionally independent of concrete model names.
    Probability columns are supplied explicitly, making the class reusable for
    Random Forest, LightGBM, ensemble, market, and future model outputs.
    """

    def __init__(
        self,
        *,
        probability_tolerance: float = _DEFAULT_TOLERANCE,
        allow_zero_probability: bool = True,
    ) -> None:
        """
        Initialize the validator.

        Args:
            probability_tolerance: Absolute tolerance used when checking whether
                probability rows sum to one.
            allow_zero_probability: Whether individual class probabilities may
                equal zero.

        Raises:
            ValueError: If ``probability_tolerance`` is not positive.
        """
        if probability_tolerance <= 0.0:
            raise ValueError("probability_tolerance must be positive.")

        self._probability_tolerance = float(probability_tolerance)
        self._allow_zero_probability = bool(allow_zero_probability)

    @property
    def probability_tolerance(self) -> float:
        """Return the configured probability-sum tolerance."""
        return self._probability_tolerance

    def validate_required_columns(
        self,
        frame: pd.DataFrame,
        required_columns: Iterable[str],
        *,
        frame_name: str = "frame",
    ) -> ValidationReport:
        """
        Validate that all required columns exist.

        Args:
            frame: DataFrame to inspect.
            required_columns: Column names that must exist.
            frame_name: Human-readable source name used in messages.

        Returns:
            A ``ValidationReport``.
        """
        issues: list[ValidationIssue] = []

        if not isinstance(frame, pd.DataFrame):
            return ValidationReport(
                is_valid=False,
                issues=(
                    ValidationIssue(
                        code="NOT_DATAFRAME",
                        message=f"{frame_name} must be a pandas DataFrame.",
                    ),
                ),
            )

        required = tuple(dict.fromkeys(str(column) for column in required_columns))
        missing = tuple(column for column in required if column not in frame.columns)

        if missing:
            issues.append(
                ValidationIssue(
                    code="MISSING_COLUMNS",
                    message=(
                        f"{frame_name} is missing required columns: "
                        + ", ".join(missing)
                    ),
                )
            )

        return ValidationReport(is_valid=not issues, issues=tuple(issues))

    def validate_non_empty(
        self,
        frame: pd.DataFrame,
        *,
        frame_name: str = "frame",
    ) -> ValidationReport:
        """
        Validate that a DataFrame contains at least one row.

        Args:
            frame: DataFrame to inspect.
            frame_name: Human-readable source name used in messages.

        Returns:
            A ``ValidationReport``.
        """
        if not isinstance(frame, pd.DataFrame):
            return ValidationReport(
                is_valid=False,
                issues=(
                    ValidationIssue(
                        code="NOT_DATAFRAME",
                        message=f"{frame_name} must be a pandas DataFrame.",
                    ),
                ),
            )

        if frame.empty:
            return ValidationReport(
                is_valid=False,
                issues=(
                    ValidationIssue(
                        code="EMPTY_FRAME",
                        message=f"{frame_name} must contain at least one row.",
                    ),
                ),
            )

        return ValidationReport(is_valid=True, issues=())

    def validate_unique_key(
        self,
        frame: pd.DataFrame,
        key_columns: Sequence[str],
        *,
        frame_name: str = "frame",
    ) -> ValidationReport:
        """
        Validate that a key-column combination is unique and non-null.

        Args:
            frame: DataFrame to inspect.
            key_columns: One or more columns forming the key.
            frame_name: Human-readable source name used in messages.

        Returns:
            A ``ValidationReport``.
        """
        required_report = self.validate_required_columns(
            frame,
            key_columns,
            frame_name=frame_name,
        )
        if not required_report.is_valid:
            return required_report

        issues: list[ValidationIssue] = []
        columns = list(key_columns)

        null_mask = frame[columns].isna().any(axis=1)
        if null_mask.any():
            issues.append(
                ValidationIssue(
                    code="NULL_KEY",
                    message=(
                        f"{frame_name} contains null values in key columns: "
                        + ", ".join(columns)
                    ),
                    rows=self._row_positions(null_mask),
                )
            )

        duplicate_mask = frame.duplicated(subset=columns, keep=False)
        if duplicate_mask.any():
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_KEY",
                    message=(
                        f"{frame_name} contains duplicate key values for: "
                        + ", ".join(columns)
                    ),
                    rows=self._row_positions(duplicate_mask),
                )
            )

        return ValidationReport(is_valid=not issues, issues=tuple(issues))

    def validate_text_columns(
        self,
        frame: pd.DataFrame,
        text_columns: Sequence[str],
        *,
        frame_name: str = "frame",
    ) -> ValidationReport:
        """
        Validate that selected text columns are non-null and non-blank.

        Args:
            frame: DataFrame to inspect.
            text_columns: Columns expected to contain meaningful text.
            frame_name: Human-readable source name used in messages.

        Returns:
            A ``ValidationReport``.
        """
        required_report = self.validate_required_columns(
            frame,
            text_columns,
            frame_name=frame_name,
        )
        if not required_report.is_valid:
            return required_report

        issues: list[ValidationIssue] = []

        for column in text_columns:
            values = frame[column]
            invalid_mask = values.isna() | values.astype(str).str.strip().eq("")
            if invalid_mask.any():
                issues.append(
                    ValidationIssue(
                        code="BLANK_TEXT",
                        message=(
                            f"{frame_name}.{column} contains null or blank values."
                        ),
                        rows=self._row_positions(invalid_mask),
                    )
                )

        return ValidationReport(is_valid=not issues, issues=tuple(issues))

    def validate_numeric_columns(
        self,
        frame: pd.DataFrame,
        numeric_columns: Sequence[str],
        *,
        frame_name: str = "frame",
    ) -> ValidationReport:
        """
        Validate that selected columns can be converted to finite numbers.

        Args:
            frame: DataFrame to inspect.
            numeric_columns: Columns expected to contain numeric values.
            frame_name: Human-readable source name used in messages.

        Returns:
            A ``ValidationReport``.
        """
        required_report = self.validate_required_columns(
            frame,
            numeric_columns,
            frame_name=frame_name,
        )
        if not required_report.is_valid:
            return required_report

        issues: list[ValidationIssue] = []

        for column in numeric_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            invalid_mask = numeric.isna() | ~np.isfinite(
                numeric.to_numpy(dtype=np.float64)
            )
            if invalid_mask.any():
                issues.append(
                    ValidationIssue(
                        code="INVALID_NUMERIC",
                        message=(
                            f"{frame_name}.{column} contains non-numeric, "
                            "NaN, or infinite values."
                        ),
                        rows=self._row_positions(invalid_mask),
                    )
                )

        return ValidationReport(is_valid=not issues, issues=tuple(issues))

    def validate_probability_columns(
        self,
        frame: pd.DataFrame,
        probability_columns: Sequence[str],
        *,
        frame_name: str = "frame",
        require_sum_to_one: bool = True,
    ) -> ValidationReport:
        """
        Validate a group of class-probability columns.

        Checks include:

        - required columns;
        - finite numeric values;
        - permitted probability range;
        - positive row totals;
        - row sums equal to one when requested.

        Args:
            frame: DataFrame containing probability columns.
            probability_columns: Ordered class-probability columns.
            frame_name: Human-readable source name used in messages.
            require_sum_to_one: Whether each row must already sum to one.

        Returns:
            A ``ValidationReport``.
        """
        columns = tuple(probability_columns)
        if len(columns) < 2:
            return ValidationReport(
                is_valid=False,
                issues=(
                    ValidationIssue(
                        code="INSUFFICIENT_CLASSES",
                        message=(
                            "At least two probability columns are required."
                        ),
                    ),
                ),
            )

        required_report = self.validate_required_columns(
            frame,
            columns,
            frame_name=frame_name,
        )
        if not required_report.is_valid:
            return required_report

        issues: list[ValidationIssue] = []
        numeric = frame.loc[:, columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        matrix = numeric.to_numpy(dtype=np.float64)

        non_finite_mask = ~np.isfinite(matrix)
        if non_finite_mask.any():
            row_mask = pd.Series(
                non_finite_mask.any(axis=1),
                index=frame.index,
            )
            issues.append(
                ValidationIssue(
                    code="NON_FINITE_PROBABILITY",
                    message=(
                        f"{frame_name} contains non-numeric, NaN, or infinite "
                        "probability values."
                    ),
                    rows=self._row_positions(row_mask),
                )
            )

        finite_matrix = np.where(np.isfinite(matrix), matrix, 0.0)
        lower_bound = 0.0 if self._allow_zero_probability else 0.0
        if self._allow_zero_probability:
            invalid_range = finite_matrix < lower_bound
        else:
            invalid_range = finite_matrix <= lower_bound
        invalid_range |= finite_matrix > 1.0

        if invalid_range.any():
            row_mask = pd.Series(
                invalid_range.any(axis=1),
                index=frame.index,
            )
            comparator = "[0, 1]" if self._allow_zero_probability else "(0, 1]"
            issues.append(
                ValidationIssue(
                    code="PROBABILITY_OUT_OF_RANGE",
                    message=(
                        f"{frame_name} probability values must lie in "
                        f"{comparator}."
                    ),
                    rows=self._row_positions(row_mask),
                )
            )

        row_sums = finite_matrix.sum(axis=1)
        zero_sum_mask = row_sums <= self._probability_tolerance
        if zero_sum_mask.any():
            issues.append(
                ValidationIssue(
                    code="ZERO_PROBABILITY_SUM",
                    message=(
                        f"{frame_name} contains probability rows whose total "
                        "is zero."
                    ),
                    rows=tuple(
                        int(position)
                        for position in np.flatnonzero(zero_sum_mask)
                    ),
                )
            )

        if require_sum_to_one:
            invalid_sum_mask = ~np.isclose(
                row_sums,
                1.0,
                rtol=0.0,
                atol=self._probability_tolerance,
            )
            if invalid_sum_mask.any():
                issues.append(
                    ValidationIssue(
                        code="INVALID_PROBABILITY_SUM",
                        message=(
                            f"{frame_name} probability rows must sum to one "
                            f"within tolerance {self._probability_tolerance:g}."
                        ),
                        rows=tuple(
                            int(position)
                            for position in np.flatnonzero(invalid_sum_mask)
                        ),
                    )
                )

        return ValidationReport(is_valid=not issues, issues=tuple(issues))

    def normalize_probability_columns(
        self,
        frame: pd.DataFrame,
        probability_columns: Sequence[str],
        *,
        frame_name: str = "frame",
        copy: bool = True,
    ) -> ProbabilityValidationResult:
        """
        Validate and normalize class-probability columns row by row.

        This method permits probability rows that do not initially sum to one,
        provided all values are finite, non-negative, no greater than one, and
        the row total is positive.

        Args:
            frame: DataFrame containing probability columns.
            probability_columns: Ordered class-probability columns.
            frame_name: Human-readable source name used in messages.
            copy: Whether to return a copied DataFrame.

        Returns:
            A ``ProbabilityValidationResult`` containing normalized data.

        Raises:
            ValueError: If the probability data cannot be normalized safely.
        """
        columns = tuple(probability_columns)
        report = self.validate_probability_columns(
            frame,
            columns,
            frame_name=frame_name,
            require_sum_to_one=False,
        )
        report.raise_for_errors()

        result = frame.copy(deep=True) if copy else frame
        matrix = result.loc[:, columns].apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=np.float64)
        row_sums = matrix.sum(axis=1)

        normalized_rows_mask = ~np.isclose(
            row_sums,
            1.0,
            rtol=0.0,
            atol=self._probability_tolerance,
        )
        normalized = matrix / row_sums[:, np.newaxis]
        result.loc[:, columns] = normalized

        final_report = self.validate_probability_columns(
            result,
            columns,
            frame_name=frame_name,
            require_sum_to_one=True,
        )
        final_report.raise_for_errors()

        return ProbabilityValidationResult(
            frame=result,
            probability_columns=columns,
            normalized_rows=tuple(
                int(position)
                for position in np.flatnonzero(normalized_rows_mask)
            ),
        )

    def validate_prediction_frame(
        self,
        frame: pd.DataFrame,
        *,
        key_columns: Sequence[str],
        probability_groups: Mapping[str, Sequence[str]],
        text_columns: Sequence[str] = (),
        frame_name: str = "predictions",
        require_normalized_probabilities: bool = True,
    ) -> ValidationReport:
        """
        Validate a complete prediction input DataFrame.

        Args:
            frame: Prediction DataFrame.
            key_columns: Columns uniquely identifying each match.
            probability_groups: Mapping of model name to ordered probability
                columns, for example
                ``{"ensemble": ("prob_a", "prob_d", "prob_h")}``.
            text_columns: Optional team-name or label columns.
            frame_name: Human-readable source name used in messages.
            require_normalized_probabilities: Whether each probability group
                must already sum to one.

        Returns:
            Consolidated ``ValidationReport``.
        """
        issues: list[ValidationIssue] = []

        issues.extend(
            self.validate_non_empty(
                frame,
                frame_name=frame_name,
            ).issues
        )
        if issues:
            return ValidationReport(is_valid=False, issues=tuple(issues))

        issues.extend(
            self.validate_unique_key(
                frame,
                key_columns,
                frame_name=frame_name,
            ).issues
        )

        if text_columns:
            issues.extend(
                self.validate_text_columns(
                    frame,
                    text_columns,
                    frame_name=frame_name,
                ).issues
            )

        if not probability_groups:
            issues.append(
                ValidationIssue(
                    code="NO_PROBABILITY_GROUPS",
                    message=(
                        f"{frame_name} must define at least one probability "
                        "group."
                    ),
                )
            )
        else:
            for group_name, columns in probability_groups.items():
                issues.extend(
                    self.validate_probability_columns(
                        frame,
                        columns,
                        frame_name=f"{frame_name}.{group_name}",
                        require_sum_to_one=require_normalized_probabilities,
                    ).issues
                )

        return ValidationReport(is_valid=not issues, issues=tuple(issues))

    def validate_market_frame(
        self,
        frame: pd.DataFrame,
        *,
        key_columns: Sequence[str],
        market_probability_columns: Sequence[str],
        text_columns: Sequence[str] = (),
        frame_name: str = "market",
        require_normalized_probabilities: bool = True,
    ) -> ValidationReport:
        """
        Validate market probability input.

        Args:
            frame: Market DataFrame.
            key_columns: Columns uniquely identifying each match.
            market_probability_columns: Ordered market probability columns.
            text_columns: Optional team-name or label columns.
            frame_name: Human-readable source name used in messages.
            require_normalized_probabilities: Whether market probabilities must
                already sum to one.

        Returns:
            Consolidated ``ValidationReport``.
        """
        return self.validate_prediction_frame(
            frame,
            key_columns=key_columns,
            probability_groups={
                "probabilities": tuple(market_probability_columns),
            },
            text_columns=text_columns,
            frame_name=frame_name,
            require_normalized_probabilities=require_normalized_probabilities,
        )

    def validate_matching_keys(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        key_columns: Sequence[str],
        left_name: str = "left",
        right_name: str = "right",
        require_exact_match: bool = True,
    ) -> ValidationReport:
        """
        Validate consistency of match keys across two DataFrames.

        Args:
            left: First DataFrame.
            right: Second DataFrame.
            key_columns: Shared key columns.
            left_name: Name used for the first DataFrame.
            right_name: Name used for the second DataFrame.
            require_exact_match: Whether both key sets must be identical. When
                false, only duplicate/null key validation is performed.

        Returns:
            A ``ValidationReport``.
        """
        issues: list[ValidationIssue] = []

        left_key_report = self.validate_unique_key(
            left,
            key_columns,
            frame_name=left_name,
        )
        right_key_report = self.validate_unique_key(
            right,
            key_columns,
            frame_name=right_name,
        )
        issues.extend(left_key_report.issues)
        issues.extend(right_key_report.issues)

        if issues or not require_exact_match:
            return ValidationReport(is_valid=not issues, issues=tuple(issues))

        left_keys = self._key_tuples(left, key_columns)
        right_keys = self._key_tuples(right, key_columns)

        missing_from_right = left_keys - right_keys
        missing_from_left = right_keys - left_keys

        if missing_from_right:
            issues.append(
                ValidationIssue(
                    code="KEYS_MISSING_FROM_RIGHT",
                    message=(
                        f"{right_name} is missing {len(missing_from_right)} "
                        f"key(s) present in {left_name}: "
                        f"{self._preview_keys(missing_from_right)}"
                    ),
                )
            )

        if missing_from_left:
            issues.append(
                ValidationIssue(
                    code="KEYS_MISSING_FROM_LEFT",
                    message=(
                        f"{left_name} is missing {len(missing_from_left)} "
                        f"key(s) present in {right_name}: "
                        f"{self._preview_keys(missing_from_left)}"
                    ),
                )
            )

        return ValidationReport(is_valid=not issues, issues=tuple(issues))

    @staticmethod
    def probability_matrix(
        frame: pd.DataFrame,
        probability_columns: Sequence[str],
    ) -> NDArray[np.float64]:
        """
        Extract probability columns as a ``float64`` NumPy matrix.

        This helper assumes validation has already succeeded.

        Args:
            frame: Source DataFrame.
            probability_columns: Ordered class-probability columns.

        Returns:
            Two-dimensional NumPy probability matrix.
        """
        return frame.loc[:, list(probability_columns)].to_numpy(
            dtype=np.float64,
            copy=True,
        )

    @staticmethod
    def _row_positions(mask: pd.Series) -> tuple[int, ...]:
        """Convert a boolean Series into zero-based row positions."""
        return tuple(
            int(position)
            for position in np.flatnonzero(mask.to_numpy(dtype=bool))
        )

    @staticmethod
    def _key_tuples(
        frame: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> set[tuple[object, ...]]:
        """Convert DataFrame key columns into a set of tuples."""
        return {
            tuple(row)
            for row in frame.loc[:, list(key_columns)].itertuples(
                index=False,
                name=None,
            )
        }

    @staticmethod
    def _preview_keys(
        keys: set[tuple[object, ...]],
        *,
        limit: int = 5,
    ) -> str:
        """Return a stable preview of missing keys for error messages."""
        preview = sorted((repr(key) for key in keys))[:limit]
        suffix = " ..." if len(keys) > limit else ""
        return ", ".join(preview) + suffix
