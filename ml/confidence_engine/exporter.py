from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ml.confidence_engine.config import ConfidenceEngineConfig
from ml.confidence_engine.models import (
    ConfidenceRunResult,
    ConfidenceSummary,
    MatchConfidenceResult,
)
from ml.confidence_engine.repository import ConfidenceDataRepository

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


class ConfidenceExporter:
    """
    Export confidence-engine outputs to CSV and JSON.

    The exporter is responsible only for output formatting and persistence.
    Confidence calculation, validation, and file loading remain delegated to
    other modules.
    """

    def __init__(
        self,
        *,
        config: ConfidenceEngineConfig | None = None,
        repository: ConfidenceDataRepository | None = None,
        json_indent: int = 2,
    ) -> None:
        """
        Initialize the exporter.

        Args:
            config: Confidence-engine configuration.
            repository: Optional shared repository used for CSV persistence.
            json_indent: Number of spaces used for formatted JSON.

        Raises:
            ValueError: If ``json_indent`` is negative.
        """
        if json_indent < 0:
            raise ValueError("json_indent must be non-negative.")

        self._config = config or ConfidenceEngineConfig()
        self._repository = repository or ConfidenceDataRepository(
            config=self._config
        )
        self._json_indent = int(json_indent)

    @property
    def config(self) -> ConfidenceEngineConfig:
        """Return the active immutable configuration."""
        return self._config

    def export_run(
        self,
        run_result: ConfidenceRunResult,
        *,
        prediction_path: str | Path | None = None,
        summary_path: str | Path | None = None,
    ) -> dict[str, Path]:
        """
        Export one complete confidence-engine run.

        Args:
            run_result: Match-level results, summary, and export DataFrame.
            prediction_path: Optional CSV destination override.
            summary_path: Optional JSON destination override.

        Returns:
            Mapping of logical output names to written paths.
        """
        prediction_destination = Path(
            prediction_path or self._config.output.prediction_path
        )
        summary_destination = Path(
            summary_path or self._config.output.summary_path
        )

        written_predictions = self.export_predictions(
            run_result.dataframe,
            prediction_destination,
        )
        written_summary = self.export_summary(
            run_result.summary,
            summary_destination,
        )

        return {
            "predictions": written_predictions,
            "summary": written_summary,
        }

    def export_predictions(
        self,
        data: pd.DataFrame | Iterable[MatchConfidenceResult],
        path: str | Path | None = None,
    ) -> Path:
        """
        Export match-level confidence results to CSV.

        Args:
            data: Export-ready DataFrame or match result objects.
            path: Optional destination override.

        Returns:
            Written CSV path.
        """
        frame = self._prediction_frame(data)
        destination = Path(path or self._config.output.prediction_path)

        ordered = self._order_prediction_columns(frame)

        return self._repository.save_csv(
            ordered,
            destination,
            index=self._config.output.write_index,
            atomic=True,
        )

    def export_summary(
        self,
        summary: ConfidenceSummary | Mapping[str, Any],
        path: str | Path | None = None,
    ) -> Path:
        """
        Export aggregate confidence summary to JSON.

        Args:
            summary: Summary dataclass or mapping.
            path: Optional destination override.

        Returns:
            Written JSON path.
        """
        destination = Path(path or self._config.output.summary_path)

        if isinstance(summary, ConfidenceSummary):
            payload: Mapping[str, Any] = summary.to_dict()
        else:
            payload = dict(summary)

        return self._write_json_atomic(payload, destination)

    def export_calibration(
        self,
        calibration: pd.DataFrame,
        path: str | Path | None = None,
    ) -> Path:
        """
        Export calibration diagnostics to CSV.

        Args:
            calibration: Calibration table.
            path: Optional destination override.

        Returns:
            Written CSV path.

        Raises:
            ValueError: If the supplied table is not a DataFrame.
        """
        if not isinstance(calibration, pd.DataFrame):
            raise ValueError("calibration must be a pandas DataFrame.")

        destination = Path(path or self._config.output.calibration_path)

        return self._repository.save_csv(
            calibration,
            destination,
            index=self._config.output.write_index,
            atomic=True,
        )

    def export_metadata(
        self,
        metadata: Mapping[str, Any],
        path: str | Path,
    ) -> Path:
        """
        Export arbitrary run metadata to JSON.

        Args:
            metadata: Metadata mapping.
            path: JSON destination.

        Returns:
            Written JSON path.
        """
        return self._write_json_atomic(dict(metadata), Path(path))

    def build_summary_payload(
        self,
        summary: ConfidenceSummary,
        *,
        generated_at: datetime | None = None,
        additional_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build an enriched summary payload without writing it.

        Args:
            summary: Aggregate run summary.
            generated_at: Optional generation timestamp.
            additional_metadata: Optional metadata merged under ``metadata``.

        Returns:
            JSON-compatible summary dictionary.
        """
        payload = summary.to_dict()
        payload["generated_at"] = (
            generated_at or datetime.now().astimezone()
        ).isoformat()

        metadata = dict(payload.get("metadata", {}))
        metadata.update(dict(additional_metadata or {}))
        payload["metadata"] = metadata

        return self._json_compatible(payload)

    def _prediction_frame(
        self,
        data: pd.DataFrame | Iterable[MatchConfidenceResult],
    ) -> pd.DataFrame:
        """Convert supported prediction data into a DataFrame."""
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("Prediction DataFrame must not be empty.")
            return data.copy(deep=True)

        records = [result.to_record() for result in data]
        if not records:
            raise ValueError("At least one match result is required.")

        return pd.DataFrame(records)

    @staticmethod
    def _order_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Return a stable user-facing column order.

        Unknown or future columns are retained after the preferred columns.
        """
        preferred_columns = (
            "match_id",
            "home_team",
            "away_team",
            "prob_a",
            "prob_d",
            "prob_h",
            "top_outcome",
            "second_outcome",
            "top_probability",
            "second_probability",
            "third_probability",
            "probability_margin",
            "normalized_entropy",
            "entropy_certainty",
            "model_same_top_pick",
            "model_agreement_score",
            "model_js_distance",
            "model_similarity_score",
            "model_max_probability_gap",
            "confidence_score",
            "confidence_level",
            "ticket_recommendation",
            "coverage_count",
            "covered_outcomes",
            "market_best_outcome",
            "market_best_edge",
            "market_best_value_ratio",
            "market_edge_level",
            "ai_market_same_top_pick",
            "market_probability_for_ai_pick",
            "market_edge_for_ai_pick",
            "market_value_ratio_for_ai_pick",
            "roi_priority_score",
        )

        existing_preferred = [
            column for column in preferred_columns if column in frame.columns
        ]
        remaining = [
            column
            for column in frame.columns
            if column not in existing_preferred
        ]

        return frame.loc[:, [*existing_preferred, *remaining]].copy()

    def _write_json_atomic(
        self,
        payload: Mapping[str, Any] | Sequence[Any],
        destination: Path,
    ) -> Path:
        """Write JSON through a temporary file and atomic replacement."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None

        normalized_payload = self._json_compatible(payload)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                suffix=".tmp",
                prefix=f".{destination.stem}_",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    normalized_payload,
                    temporary_file,
                    ensure_ascii=False,
                    indent=self._json_indent,
                    sort_keys=False,
                )
                temporary_file.write("\n")

            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        LOGGER.info("Saved JSON: path=%s", destination)
        return destination

    @classmethod
    def _json_compatible(cls, value: Any) -> Any:
        """
        Recursively convert common scientific-Python objects to JSON values.
        """
        if is_dataclass(value):
            return cls._json_compatible(asdict(value))

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, Mapping):
            return {
                str(key): cls._json_compatible(item)
                for key, item in value.items()
            }

        if isinstance(value, tuple):
            return [cls._json_compatible(item) for item in value]

        if isinstance(value, list):
            return [cls._json_compatible(item) for item in value]

        if isinstance(value, np.ndarray):
            return [
                cls._json_compatible(item)
                for item in value.tolist()
            ]

        if isinstance(value, np.generic):
            return value.item()

        if pd.isna(value):
            return None

        return value
