from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

_EPSILON: Final[float] = 1e-15


@dataclass(frozen=True, slots=True)
class ProbabilityShapeMetrics:
    """Probability-shape metrics calculated for each prediction row."""

    top_probability: NDArray[np.float64]
    second_probability: NDArray[np.float64]
    third_probability: NDArray[np.float64]
    probability_margin: NDArray[np.float64]
    normalized_entropy: NDArray[np.float64]
    certainty_from_entropy: NDArray[np.float64]
    top_outcome_index: NDArray[np.int64]
    second_outcome_index: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Aggregate calibration diagnostics for probabilistic predictions."""

    brier_score: float
    multiclass_log_loss: float
    expected_calibration_error: float
    maximum_calibration_error: float
    mean_confidence: float
    accuracy: float
    observations: int


def _as_probability_matrix(
    values: ArrayLike,
    *,
    name: str,
    require_normalized: bool = True,
) -> NDArray[np.float64]:
    """Return a validated two-dimensional probability matrix."""
    matrix = np.asarray(values, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional matrix; received shape {matrix.shape}."
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two classes.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if (matrix < 0.0).any():
        raise ValueError(f"{name} contains negative probabilities.")

    row_sums = matrix.sum(axis=1)
    if (row_sums <= 0.0).any():
        raise ValueError(f"{name} contains a row whose sum is zero.")

    if require_normalized and not np.allclose(
        row_sums,
        1.0,
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            f"{name} rows must sum to one. Normalize probabilities first."
        )

    return matrix


def normalized_entropy(probabilities: ArrayLike) -> NDArray[np.float64]:
    """Calculate Shannon entropy normalized to the range ``[0, 1]``."""
    matrix = _as_probability_matrix(probabilities, name="probabilities")
    clipped = np.clip(matrix, _EPSILON, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return np.asarray(entropy / math.log(matrix.shape[1]), dtype=np.float64)


def jensen_shannon_distance(
    first: ArrayLike,
    second: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate normalized Jensen-Shannon distance for each row."""
    first_matrix = _as_probability_matrix(first, name="first")
    second_matrix = _as_probability_matrix(second, name="second")

    if first_matrix.shape != second_matrix.shape:
        raise ValueError(
            "first and second must have identical shapes; "
            f"received {first_matrix.shape} and {second_matrix.shape}."
        )

    midpoint = 0.5 * (first_matrix + second_matrix)
    first_clipped = np.clip(first_matrix, _EPSILON, 1.0)
    second_clipped = np.clip(second_matrix, _EPSILON, 1.0)
    midpoint_clipped = np.clip(midpoint, _EPSILON, 1.0)

    kl_first = np.sum(
        first_clipped * np.log(first_clipped / midpoint_clipped),
        axis=1,
    )
    kl_second = np.sum(
        second_clipped * np.log(second_clipped / midpoint_clipped),
        axis=1,
    )
    divergence = 0.5 * (kl_first + kl_second)
    normalized = divergence / math.log(2.0)

    return np.sqrt(np.clip(normalized, 0.0, 1.0)).astype(
        np.float64,
        copy=False,
    )


def maximum_probability_gap(
    first: ArrayLike,
    second: ArrayLike,
) -> NDArray[np.float64]:
    """Return the largest absolute class-probability gap per row."""
    first_matrix = _as_probability_matrix(first, name="first")
    second_matrix = _as_probability_matrix(second, name="second")

    if first_matrix.shape != second_matrix.shape:
        raise ValueError(
            "first and second must have identical shapes; "
            f"received {first_matrix.shape} and {second_matrix.shape}."
        )

    return np.max(np.abs(first_matrix - second_matrix), axis=1).astype(
        np.float64,
        copy=False,
    )


def top_pick_agreement(
    first: ArrayLike,
    second: ArrayLike,
) -> NDArray[np.bool_]:
    """Return whether two models select the same top outcome per row."""
    first_matrix = _as_probability_matrix(first, name="first")
    second_matrix = _as_probability_matrix(second, name="second")

    if first_matrix.shape != second_matrix.shape:
        raise ValueError(
            "first and second must have identical shapes; "
            f"received {first_matrix.shape} and {second_matrix.shape}."
        )

    return np.argmax(first_matrix, axis=1) == np.argmax(second_matrix, axis=1)


def probability_shape_metrics(
    probabilities: ArrayLike,
) -> ProbabilityShapeMetrics:
    """Calculate ranking, margin, entropy, and certainty metrics."""
    matrix = _as_probability_matrix(probabilities, name="probabilities")
    if matrix.shape[1] < 3:
        raise ValueError(
            "probability_shape_metrics requires at least three classes."
        )

    order = np.argsort(-matrix, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(matrix, order, axis=1)
    entropy = normalized_entropy(matrix)

    return ProbabilityShapeMetrics(
        top_probability=sorted_probabilities[:, 0].astype(
            np.float64, copy=False
        ),
        second_probability=sorted_probabilities[:, 1].astype(
            np.float64, copy=False
        ),
        third_probability=sorted_probabilities[:, 2].astype(
            np.float64, copy=False
        ),
        probability_margin=(
            sorted_probabilities[:, 0] - sorted_probabilities[:, 1]
        ).astype(np.float64, copy=False),
        normalized_entropy=entropy,
        certainty_from_entropy=(1.0 - entropy).astype(
            np.float64, copy=False
        ),
        top_outcome_index=order[:, 0].astype(np.int64, copy=False),
        second_outcome_index=order[:, 1].astype(np.int64, copy=False),
    )


def probability_edge(
    model_probabilities: ArrayLike,
    market_probabilities: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate model-minus-market probability edge."""
    model = _as_probability_matrix(
        model_probabilities, name="model_probabilities"
    )
    market = _as_probability_matrix(
        market_probabilities, name="market_probabilities"
    )

    if model.shape != market.shape:
        raise ValueError(
            "model_probabilities and market_probabilities must have "
            f"identical shapes; received {model.shape} and {market.shape}."
        )

    return (model - market).astype(np.float64, copy=False)


def probability_value_ratio(
    model_probabilities: ArrayLike,
    market_probabilities: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate model-to-market probability ratios."""
    model = _as_probability_matrix(
        model_probabilities, name="model_probabilities"
    )
    market = _as_probability_matrix(
        market_probabilities, name="market_probabilities"
    )

    if model.shape != market.shape:
        raise ValueError(
            "model_probabilities and market_probabilities must have "
            f"identical shapes; received {model.shape} and {market.shape}."
        )

    return np.divide(model, np.clip(market, _EPSILON, None)).astype(
        np.float64,
        copy=False,
    )


def best_edge_metrics(
    model_probabilities: ArrayLike,
    market_probabilities: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Return best edge, corresponding value ratio, and outcome index."""
    edges = probability_edge(model_probabilities, market_probabilities)
    ratios = probability_value_ratio(
        model_probabilities,
        market_probabilities,
    )
    best_indices = np.argmax(edges, axis=1).astype(np.int64)
    row_indices = np.arange(edges.shape[0])

    return (
        edges[row_indices, best_indices].astype(np.float64, copy=False),
        ratios[row_indices, best_indices].astype(np.float64, copy=False),
        best_indices,
    )


def roi_priority_score(
    confidence_score: ArrayLike,
    best_edge: ArrayLike,
    best_value_ratio: ArrayLike,
    *,
    value_ratio_cap: float = 3.0,
) -> NDArray[np.float64]:
    """Calculate the current rule-based ROI-priority score."""
    confidence = np.asarray(confidence_score, dtype=np.float64)
    edge = np.asarray(best_edge, dtype=np.float64)
    ratio = np.asarray(best_value_ratio, dtype=np.float64)

    if not (confidence.shape == edge.shape == ratio.shape):
        raise ValueError(
            "confidence_score, best_edge, and best_value_ratio must "
            "have identical shapes."
        )
    if value_ratio_cap <= 0.0:
        raise ValueError("value_ratio_cap must be positive.")
    if not (
        np.isfinite(confidence).all()
        and np.isfinite(edge).all()
        and np.isfinite(ratio).all()
    ):
        raise ValueError("ROI-priority inputs contain NaN or infinite values.")

    return (
        np.clip(confidence, 0.0, 1.0)
        * np.clip(edge, 0.0, None)
        * np.clip(ratio, 0.0, value_ratio_cap)
    ).astype(np.float64, copy=False)


def multiclass_brier_score(
    probabilities: ArrayLike,
    actual_class_indices: ArrayLike,
) -> float:
    """Calculate the multiclass Brier score."""
    matrix = _as_probability_matrix(probabilities, name="probabilities")
    actual = np.asarray(actual_class_indices, dtype=np.int64)

    if actual.ndim != 1 or len(actual) != len(matrix):
        raise ValueError(
            "actual_class_indices must be one-dimensional and match "
            "the number of probability rows."
        )
    if ((actual < 0) | (actual >= matrix.shape[1])).any():
        raise ValueError("actual_class_indices contains an invalid class.")

    target = np.zeros_like(matrix)
    target[np.arange(len(matrix)), actual] = 1.0
    return float(np.mean(np.sum((matrix - target) ** 2, axis=1)))


def multiclass_log_loss(
    probabilities: ArrayLike,
    actual_class_indices: ArrayLike,
) -> float:
    """Calculate multiclass logarithmic loss."""
    matrix = _as_probability_matrix(probabilities, name="probabilities")
    actual = np.asarray(actual_class_indices, dtype=np.int64)

    if actual.ndim != 1 or len(actual) != len(matrix):
        raise ValueError(
            "actual_class_indices must be one-dimensional and match "
            "the number of probability rows."
        )
    if ((actual < 0) | (actual >= matrix.shape[1])).any():
        raise ValueError("actual_class_indices contains an invalid class.")

    selected = matrix[np.arange(len(matrix)), actual]
    return float(-np.mean(np.log(np.clip(selected, _EPSILON, 1.0))))


def calibration_errors(
    probabilities: ArrayLike,
    actual_class_indices: ArrayLike,
    *,
    n_bins: int = 15,
) -> tuple[float, float]:
    """Calculate expected and maximum calibration error."""
    matrix = _as_probability_matrix(probabilities, name="probabilities")
    actual = np.asarray(actual_class_indices, dtype=np.int64)

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if actual.ndim != 1 or len(actual) != len(matrix):
        raise ValueError(
            "actual_class_indices must be one-dimensional and match "
            "the number of probability rows."
        )
    if ((actual < 0) | (actual >= matrix.shape[1])).any():
        raise ValueError("actual_class_indices contains an invalid class.")

    predicted = np.argmax(matrix, axis=1)
    confidence = np.max(matrix, axis=1)
    correct = (predicted == actual).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    expected_error = 0.0
    maximum_error = 0.0

    for bin_index in range(n_bins):
        lower = boundaries[bin_index]
        upper = boundaries[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        observations = int(mask.sum())
        if observations == 0:
            continue

        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidence[mask].mean())
        gap = abs(bin_accuracy - bin_confidence)
        expected_error += (observations / len(matrix)) * gap
        maximum_error = max(maximum_error, gap)

    return float(expected_error), float(maximum_error)


def calibration_metrics(
    probabilities: ArrayLike,
    actual_class_indices: ArrayLike,
    *,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Calculate aggregate calibration diagnostics."""
    matrix = _as_probability_matrix(probabilities, name="probabilities")
    actual = np.asarray(actual_class_indices, dtype=np.int64)

    if actual.ndim != 1 or len(actual) != len(matrix):
        raise ValueError(
            "actual_class_indices must be one-dimensional and match "
            "the number of probability rows."
        )

    predicted = np.argmax(matrix, axis=1)
    confidence = np.max(matrix, axis=1)
    expected_error, maximum_error = calibration_errors(
        matrix,
        actual,
        n_bins=n_bins,
    )

    return CalibrationMetrics(
        brier_score=multiclass_brier_score(matrix, actual),
        multiclass_log_loss=multiclass_log_loss(matrix, actual),
        expected_calibration_error=expected_error,
        maximum_calibration_error=maximum_error,
        mean_confidence=float(confidence.mean()),
        accuracy=float((predicted == actual).mean()),
        observations=len(matrix),
    )


def outcome_indices(
    outcomes: Iterable[str],
    *,
    class_order: Sequence[str] = ("A", "D", "H"),
) -> NDArray[np.int64]:
    """Convert outcome labels into zero-based class indices."""
    labels = tuple(str(label).upper() for label in class_order)
    if len(labels) != len(set(labels)):
        raise ValueError("class_order must contain unique labels.")

    lookup = {label: index for index, label in enumerate(labels)}
    normalized = [str(value).upper() for value in outcomes]
    unknown = sorted({value for value in normalized if value not in lookup})

    if unknown:
        raise ValueError("Unknown outcome labels: " + ", ".join(unknown))

    return np.asarray([lookup[value] for value in normalized], dtype=np.int64)


def calibration_table(
    probabilities: ArrayLike,
    actual_class_indices: ArrayLike,
    *,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Build a reliability-table DataFrame for diagnostics and charting."""
    matrix = _as_probability_matrix(probabilities, name="probabilities")
    actual = np.asarray(actual_class_indices, dtype=np.int64)

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if actual.ndim != 1 or len(actual) != len(matrix):
        raise ValueError(
            "actual_class_indices must be one-dimensional and match "
            "the number of probability rows."
        )
    if ((actual < 0) | (actual >= matrix.shape[1])).any():
        raise ValueError("actual_class_indices contains an invalid class.")

    predicted = np.argmax(matrix, axis=1)
    confidence = np.max(matrix, axis=1)
    correct = (predicted == actual).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []

    for bin_index in range(n_bins):
        lower = float(boundaries[bin_index])
        upper = float(boundaries[bin_index + 1])
        if bin_index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        observations = int(mask.sum())
        mean_confidence = (
            float(confidence[mask].mean()) if observations else math.nan
        )
        accuracy = float(correct[mask].mean()) if observations else math.nan
        gap = abs(accuracy - mean_confidence) if observations else math.nan

        rows.append(
            {
                "bin": bin_index + 1,
                "lower_bound": lower,
                "upper_bound": upper,
                "observations": observations,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )

    return pd.DataFrame(rows)
