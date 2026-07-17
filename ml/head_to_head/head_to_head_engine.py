from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    ML_ROOT = Path(__file__).resolve().parents[1]
    if str(ML_ROOT) not in sys.path:
        sys.path.insert(0, str(ML_ROOT))

from head_to_head.feature_builder import HeadToHeadFeatureBuilder
from head_to_head.models import (
    DEFAULT_OUTPUT_COLUMNS,
    HeadToHeadEngineConfig,
)
from head_to_head.repository import HeadToHeadRepository, RepositoryConfig

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def default_config() -> HeadToHeadEngineConfig:
    """Return project-relative default paths."""
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / "player_engine" / "data" / "processed"
    diagnostics_dir = ml_root / "diagnostics" / "head_to_head"

    return HeadToHeadEngineConfig(
        input_matches_csv=processed_dir / "matches.csv",
        output_features_csv=processed_dir / "historical_h2h_features.csv",
        diagnostics_dir=diagnostics_dir,
    )


def run(config: HeadToHeadEngineConfig) -> dict[str, Any]:
    """Generate H2H features and diagnostics."""
    config.validate()
    config.output_features_csv.parent.mkdir(parents=True, exist_ok=True)
    config.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    repository = HeadToHeadRepository(
        RepositoryConfig(matches_csv=config.input_matches_csv)
    )
    repository.load()

    builder = HeadToHeadFeatureBuilder(
        repository=repository,
        config=config,
    )

    rows = [asdict(builder.build(match)) for match in repository.matches]
    features = pd.DataFrame(rows, columns=DEFAULT_OUTPUT_COLUMNS)

    _validate_output(features, expected_rows=len(repository.matches))
    features.to_csv(config.output_features_csv, index=False, encoding="utf-8-sig")

    missing_report = _build_missing_report(features)
    missing_path = config.diagnostics_dir / "h2h_feature_missing.csv"
    missing_report.to_csv(missing_path, index=False, encoding="utf-8-sig")

    distribution = (
        features["h2h_matches"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("h2h_matches")
        .reset_index(name="row_count")
    )
    distribution["row_ratio"] = (
        distribution["row_count"] / len(features)
        if len(features)
        else 0.0
    )
    distribution_path = (
        config.diagnostics_dir / "h2h_sample_size_distribution.csv"
    )
    distribution.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = _build_summary(
        features=features,
        input_path=config.input_matches_csv,
        output_path=config.output_features_csv,
    )
    summary_path = config.diagnostics_dir / "h2h_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    diagnostics_copy = (
        config.diagnostics_dir / "historical_h2h_features.csv"
    )
    features.to_csv(
        diagnostics_copy,
        index=False,
        encoding="utf-8-sig",
    )

    LOGGER.info(
        "Generated %d H2H feature rows: %s",
        len(features),
        config.output_features_csv,
    )
    return summary


def _validate_output(features: pd.DataFrame, expected_rows: int) -> None:
    """Validate output integrity before writing."""
    if len(features) != expected_rows:
        raise ValueError(
            f"Output row count mismatch: expected={expected_rows}, "
            f"actual={len(features)}"
        )

    duplicate_ids = int(features["match_card_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"Output contains {duplicate_ids} duplicate match_card_id values."
        )

    missing_columns = sorted(set(DEFAULT_OUTPUT_COLUMNS) - set(features.columns))
    if missing_columns:
        raise ValueError(
            "Output is missing expected columns: " + ", ".join(missing_columns)
        )

    missing_values = int(features.isna().sum().sum())
    if missing_values:
        raise ValueError(
            f"Output contains {missing_values} missing feature values."
        )


def _build_missing_report(features: pd.DataFrame) -> pd.DataFrame:
    """Return one row per feature with missing-value diagnostics."""
    report = pd.DataFrame(
        {
            "column": features.columns,
            "missing_count": [
                int(features[column].isna().sum())
                for column in features.columns
            ],
        }
    )
    report["missing_ratio"] = (
        report["missing_count"] / len(features)
        if len(features)
        else 0.0
    )
    return report


def _build_summary(
    features: pd.DataFrame,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create a compact machine-readable diagnostics summary."""
    h2h_matches = features["h2h_matches"]

    return {
        "input_matches_csv": str(input_path),
        "output_features_csv": str(output_path),
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "duplicate_match_card_ids": int(
            features["match_card_id"].duplicated().sum()
        ),
        "missing_values": int(features.isna().sum().sum()),
        "rows_without_history": int((h2h_matches == 0).sum()),
        "rows_with_history": int((h2h_matches > 0).sum()),
        "history_match_count_mean": float(h2h_matches.mean()),
        "history_match_count_median": float(h2h_matches.median()),
        "history_match_count_max": int(h2h_matches.max()),
        "confidence_mean": float(features["h2h_confidence"].mean()),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    defaults = default_config()
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe historical H2H features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=defaults.input_matches_csv,
        help="Path to processed matches.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output_features_csv,
        help="Path to historical_h2h_features.csv.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=defaults.diagnostics_dir,
        help="Directory for H2H diagnostics.",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=defaults.decay_half_life_days,
    )
    parser.add_argument(
        "--max-history-matches",
        type=int,
        default=defaults.max_history_matches,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    config = HeadToHeadEngineConfig(
        input_matches_csv=args.input,
        output_features_csv=args.output,
        diagnostics_dir=args.diagnostics_dir,
        decay_half_life_days=args.half_life_days,
        max_history_matches=args.max_history_matches,
    )
    summary = run(config)

    print(
        "Head-to-Head feature generation completed\n"
        f"Rows             : {summary['rows']}\n"
        f"Columns          : {summary['columns']}\n"
        f"Duplicate IDs    : {summary['duplicate_match_card_ids']}\n"
        f"Missing values   : {summary['missing_values']}\n"
        f"Rows with history: {summary['rows_with_history']}\n"
        f"Output           : {config.output_features_csv}"
    )


if __name__ == "__main__":
    main()
