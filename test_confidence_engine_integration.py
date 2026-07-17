from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.confidence_engine import (
    ConfidenceEngineBuilder,
    ConfidenceEngineConfig,
    ConfidenceInputPaths,
    OutputConfig,
)

LOGGER = logging.getLogger(__name__)


def _normalize(values: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a three-way probability tuple."""
    array = np.asarray(values, dtype=np.float64)
    array = array / array.sum()
    return float(array[0]), float(array[1]), float(array[2])


def build_sample_predictions() -> pd.DataFrame:
    """Create deterministic prediction data for 13 toto matches."""
    ensemble = [
        (0.16, 0.20, 0.64),
        (0.58, 0.24, 0.18),
        (0.25, 0.46, 0.29),
        (0.34, 0.33, 0.33),
        (0.20, 0.28, 0.52),
        (0.44, 0.31, 0.25),
        (0.29, 0.28, 0.43),
        (0.18, 0.22, 0.60),
        (0.36, 0.39, 0.25),
        (0.49, 0.27, 0.24),
        (0.27, 0.26, 0.47),
        (0.31, 0.37, 0.32),
        (0.22, 0.19, 0.59),
    ]

    rf = [
        (0.15, 0.19, 0.66),
        (0.61, 0.23, 0.16),
        (0.24, 0.48, 0.28),
        (0.36, 0.32, 0.32),
        (0.19, 0.29, 0.52),
        (0.46, 0.29, 0.25),
        (0.30, 0.26, 0.44),
        (0.17, 0.21, 0.62),
        (0.34, 0.41, 0.25),
        (0.51, 0.25, 0.24),
        (0.28, 0.25, 0.47),
        (0.33, 0.36, 0.31),
        (0.21, 0.18, 0.61),
    ]

    lgbm = [
        (0.17, 0.21, 0.62),
        (0.55, 0.25, 0.20),
        (0.26, 0.44, 0.30),
        (0.32, 0.34, 0.34),
        (0.21, 0.27, 0.52),
        (0.42, 0.33, 0.25),
        (0.28, 0.30, 0.42),
        (0.19, 0.23, 0.58),
        (0.38, 0.37, 0.25),
        (0.47, 0.29, 0.24),
        (0.26, 0.27, 0.47),
        (0.29, 0.38, 0.33),
        (0.23, 0.20, 0.57),
    ]

    rows: list[dict[str, object]] = []
    for index, (ensemble_row, rf_row, lgbm_row) in enumerate(
        zip(ensemble, rf, lgbm, strict=True),
        start=1,
    ):
        ea, ed, eh = _normalize(ensemble_row)
        ra, rd, rh = _normalize(rf_row)
        la, ld, lh = _normalize(lgbm_row)

        rows.append(
            {
                "match_id": f"TOTO-{index:02d}",
                "home_team": f"Home Team {index:02d}",
                "away_team": f"Away Team {index:02d}",
                "ensemble_prob_a": ea,
                "ensemble_prob_d": ed,
                "ensemble_prob_h": eh,
                "rf_prob_a": ra,
                "rf_prob_d": rd,
                "rf_prob_h": rh,
                "lgbm_prob_a": la,
                "lgbm_prob_d": ld,
                "lgbm_prob_h": lh,
                "source_round": 9999,
            }
        )

    return pd.DataFrame(rows)


def build_sample_market(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic market probabilities with useful edge variation."""
    market_values = [
        (0.24, 0.24, 0.52),
        (0.48, 0.29, 0.23),
        (0.31, 0.39, 0.30),
        (0.38, 0.31, 0.31),
        (0.27, 0.31, 0.42),
        (0.37, 0.34, 0.29),
        (0.35, 0.30, 0.35),
        (0.25, 0.27, 0.48),
        (0.34, 0.34, 0.32),
        (0.40, 0.31, 0.29),
        (0.33, 0.31, 0.36),
        (0.35, 0.34, 0.31),
        (0.28, 0.25, 0.47),
    ]

    rows: list[dict[str, object]] = []
    for prediction_row, values in zip(
        predictions.to_dict(orient="records"),
        market_values,
        strict=True,
    ):
        market_a, market_d, market_h = _normalize(values)
        rows.append(
            {
                "match_id": prediction_row["match_id"],
                "home_team": prediction_row["home_team"],
                "away_team": prediction_row["away_team"],
                "market_prob_a": market_a,
                "market_prob_d": market_d,
                "market_prob_h": market_h,
            }
        )

    return pd.DataFrame(rows)


def build_sample_actual_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create actual outcomes for integration diagnostics."""
    outcomes = ("H", "A", "D", "H", "H", "D", "H", "H", "D", "A", "A", "D", "H")

    return pd.DataFrame(
        {
            "match_id": predictions["match_id"],
            "actual_result": outcomes,
        }
    )


def run_integration_test() -> None:
    """Execute an end-to-end Confidence Engine V3 integration test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    predictions = build_sample_predictions()
    market = build_sample_market(predictions)
    actual_results = build_sample_actual_results(predictions)

    with tempfile.TemporaryDirectory(prefix="confidence_engine_v3_") as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        prediction_path = input_dir / "predictions.csv"
        market_path = input_dir / "market.csv"
        actual_path = input_dir / "actual_results.csv"

        predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        market.to_csv(market_path, index=False, encoding="utf-8-sig")
        actual_results.to_csv(actual_path, index=False, encoding="utf-8-sig")

        config = ConfidenceEngineConfig(
            output=OutputConfig(
                output_directory=output_dir,
                prediction_filename="integration_confidence_v3.csv",
                summary_filename="integration_summary_v3.json",
                calibration_filename="integration_calibration_v3.csv",
            )
        )

        builder = ConfidenceEngineBuilder(config=config)
        result = builder.build_from_paths(
            ConfidenceInputPaths(
                predictions=prediction_path,
                market=market_path,
                actual_results=actual_path,
            ),
            export=True,
            metadata={
                "test_name": "confidence_engine_v3_integration",
                "round_id": 9999,
            },
        )

        prediction_output = config.output.prediction_path
        summary_output = config.output.summary_path

        assert result.summary.total_matches == 13
        assert len(result.matches) == 13
        assert len(result.dataframe) == 13

        required_columns = {
            "confidence_score",
            "confidence_level",
            "ticket_recommendation",
            "covered_outcomes",
            "market_best_edge",
            "market_best_value_ratio",
            "market_edge_level",
            "roi_priority_score",
            "actual_result",
            "prediction_correct",
            "actual_result_probability",
        }
        missing = required_columns.difference(result.dataframe.columns)
        assert not missing, f"Missing output columns: {sorted(missing)}"

        probability_columns = [
            "ensemble_prob_a",
            "ensemble_prob_d",
            "ensemble_prob_h",
        ]
        probability_sums = result.dataframe[probability_columns].sum(axis=1)
        assert np.allclose(probability_sums, 1.0)

        assert result.dataframe["confidence_score"].between(0.0, 1.0).all()
        assert result.dataframe["normalized_entropy"].between(0.0, 1.0).all()
        assert result.dataframe["actual_result_probability"].between(
            0.0,
            1.0,
        ).all()

        assert set(result.dataframe["confidence_level"]).issubset(
            {"HIGH", "MEDIUM", "LOW"}
        )
        assert set(result.dataframe["ticket_recommendation"]).issubset(
            {"SINGLE", "DOUBLE", "TRIPLE"}
        )
        assert set(result.dataframe["market_edge_level"]).issubset(
            {"NONE", "POSITIVE", "STRONG", "EXTREME"}
        )

        assert prediction_output.exists()
        assert summary_output.exists()

        exported = pd.read_csv(prediction_output, encoding="utf-8-sig")
        assert len(exported) == 13
        assert "source_round" in exported.columns

        with summary_output.open("r", encoding="utf-8") as file:
            summary_payload = json.load(file)

        assert summary_payload["total_matches"] == 13
        assert summary_payload["metadata"]["round_id"] == 9999
        assert (
            summary_payload["metadata"]["test_name"]
            == "confidence_engine_v3_integration"
        )

        print()
        print("=== Confidence Engine V3 Integration Test ===")
        print(f"Total matches       : {result.summary.total_matches}")
        print(
            "Confidence H/M/L    : "
            f"{result.summary.high_confidence_matches}/"
            f"{result.summary.medium_confidence_matches}/"
            f"{result.summary.low_confidence_matches}"
        )
        print(
            "Ticket S/D/T        : "
            f"{result.summary.single_recommendations}/"
            f"{result.summary.double_recommendations}/"
            f"{result.summary.triple_recommendations}"
        )
        print(
            "Mean confidence     : "
            f"{result.summary.mean_confidence_score:.4f}"
        )
        print(
            "Mean top probability: "
            f"{result.summary.mean_top_probability:.4f}"
        )
        print(
            "Prediction CSV      : "
            f"{prediction_output}"
        )
        print(
            "Summary JSON        : "
            f"{summary_output}"
        )
        print("INTEGRATION TEST OK")


if __name__ == "__main__":
    run_integration_test()
