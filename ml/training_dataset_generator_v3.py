from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

KEY_COLUMN = "match_card_id"
H2H_PREFIX = "h2h_"

REQUIRED_H2H_COLUMNS = {
    "match_card_id",
    "h2h_matches",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_home_points_per_game",
    "h2h_away_points_per_game",
    "h2h_average_home_goals",
    "h2h_average_away_goals",
    "h2h_average_goal_difference",
    "h2h_btts_rate",
    "h2h_over_2_5_rate",
    "h2h_last3_home_win_rate",
    "h2h_last3_draw_rate",
    "h2h_last3_away_win_rate",
    "h2h_last3_goal_difference",
    "h2h_last5_home_win_rate",
    "h2h_last5_draw_rate",
    "h2h_last5_away_win_rate",
    "h2h_last5_goal_difference",
    "h2h_same_venue_matches",
    "h2h_same_venue_home_win_rate",
    "h2h_same_venue_draw_rate",
    "h2h_same_venue_away_win_rate",
    "h2h_same_venue_goal_difference",
    "h2h_weighted_home_win_rate",
    "h2h_weighted_draw_rate",
    "h2h_weighted_away_win_rate",
    "h2h_weighted_goal_difference",
    "h2h_days_since_last_match",
    "h2h_home_unbeaten_streak",
    "h2h_away_unbeaten_streak",
    "h2h_confidence",
}


@dataclass(frozen=True, slots=True)
class GeneratorPaths:
    """Input, output and diagnostics paths for dataset version 3."""

    training_v2_csv: Path
    h2h_features_csv: Path
    output_csv: Path
    diagnostics_dir: Path

    def validate(self) -> None:
        """Validate all required input paths."""
        for label, path in (
            ("training_dataset_v2.csv", self.training_v2_csv),
            ("historical_h2h_features.csv", self.h2h_features_csv),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} was not found: {path}")
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """Summary of the version 3 dataset build."""

    source_rows: int
    source_columns: int
    h2h_rows: int
    h2h_columns: int
    output_rows: int
    output_columns: int
    added_h2h_features: int
    duplicate_match_ids: int
    unmatched_h2h_rows: int
    h2h_missing_values: int
    total_missing_values: int
    infinite_numeric_values: int


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def default_paths() -> GeneratorPaths:
    """Build paths relative to the ml directory containing this script."""
    ml_root = Path(__file__).resolve().parent
    return GeneratorPaths(
        training_v2_csv=ml_root / "training_dataset_v2.csv",
        h2h_features_csv=(
            ml_root
            / "player_engine"
            / "data"
            / "processed"
            / "historical_h2h_features.csv"
        ),
        output_csv=ml_root / "training_dataset_v3.csv",
        diagnostics_dir=(
            ml_root / "diagnostics" / "training_dataset_v3"
        ),
    )


def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load one CSV and report its dimensions."""
    dataframe = pd.read_csv(path)
    LOGGER.info(
        "Loaded %s: rows=%d columns=%d path=%s",
        label,
        len(dataframe),
        len(dataframe.columns),
        path,
    )
    return dataframe


def validate_key(
    dataframe: pd.DataFrame,
    label: str,
    *,
    require_unique: bool = True,
) -> None:
    """Validate the dataset join key."""
    if KEY_COLUMN not in dataframe.columns:
        raise ValueError(
            f"{label} is missing required column: {KEY_COLUMN}"
        )

    missing_keys = int(dataframe[KEY_COLUMN].isna().sum())
    if missing_keys:
        raise ValueError(
            f"{label} contains {missing_keys} missing {KEY_COLUMN} values."
        )

    if require_unique:
        duplicates = int(dataframe[KEY_COLUMN].duplicated().sum())
        if duplicates:
            duplicate_examples = (
                dataframe.loc[
                    dataframe[KEY_COLUMN].duplicated(keep=False),
                    KEY_COLUMN,
                ]
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"{label} contains {duplicates} duplicate {KEY_COLUMN} "
                f"values. Examples: {duplicate_examples}"
            )


def validate_h2h_schema(h2h: pd.DataFrame) -> list[str]:
    """Validate H2H columns and return feature columns in source order."""
    missing_columns = sorted(REQUIRED_H2H_COLUMNS - set(h2h.columns))
    if missing_columns:
        raise ValueError(
            "H2H feature CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    feature_columns = [
        column
        for column in h2h.columns
        if column.startswith(H2H_PREFIX)
    ]
    if not feature_columns:
        raise ValueError("No h2h_ feature columns were found.")

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(h2h[column])
    ]
    if non_numeric:
        raise ValueError(
            "H2H feature columns must be numeric: "
            + ", ".join(non_numeric)
        )

    return feature_columns


def build_dataset(
    training_v2: pd.DataFrame,
    h2h: pd.DataFrame,
) -> tuple[pd.DataFrame, GenerationReport, list[str]]:
    """Merge H2H features onto v2 without changing existing v2 columns."""
    validate_key(training_v2, "training_dataset_v2")
    validate_key(h2h, "historical_h2h_features")
    h2h_columns = validate_h2h_schema(h2h)

    overlapping_features = sorted(
        set(h2h_columns).intersection(training_v2.columns)
    )
    if overlapping_features:
        raise ValueError(
            "training_dataset_v2 already contains H2H columns: "
            + ", ".join(overlapping_features)
        )

    source_rows = len(training_v2)
    source_columns = list(training_v2.columns)

    h2h_for_merge = h2h[[KEY_COLUMN, *h2h_columns]].copy()

    merged = training_v2.merge(
        h2h_for_merge,
        on=KEY_COLUMN,
        how="left",
        validate="one_to_one",
        indicator="_h2h_merge",
        sort=False,
    )

    if len(merged) != source_rows:
        raise ValueError(
            "Row count changed during H2H merge: "
            f"before={source_rows}, after={len(merged)}"
        )

    unmatched_h2h_rows = int(
        (merged["_h2h_merge"] != "both").sum()
    )
    if unmatched_h2h_rows:
        unmatched_examples = (
            merged.loc[
                merged["_h2h_merge"] != "both",
                KEY_COLUMN,
            ]
            .head(20)
            .tolist()
        )
        raise ValueError(
            f"{unmatched_h2h_rows} training rows did not match H2H features. "
            f"Examples: {unmatched_examples}"
        )

    merged = merged.drop(columns="_h2h_merge")

    if list(merged.columns[: len(source_columns)]) != source_columns:
        raise ValueError(
            "Existing v2 column order changed during the merge."
        )

    h2h_missing_values = int(
        merged[h2h_columns].isna().sum().sum()
    )
    if h2h_missing_values:
        raise ValueError(
            f"Merged H2H columns contain {h2h_missing_values} missing values."
        )

    duplicate_ids = int(merged[KEY_COLUMN].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"Output contains {duplicate_ids} duplicate match_card_id values."
        )

    numeric = merged.select_dtypes(include="number")
    infinite_numeric_values = int(
        np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()
    )
    if infinite_numeric_values:
        raise ValueError(
            f"Output contains {infinite_numeric_values} infinite numeric values."
        )

    report = GenerationReport(
        source_rows=source_rows,
        source_columns=len(training_v2.columns),
        h2h_rows=len(h2h),
        h2h_columns=len(h2h.columns),
        output_rows=len(merged),
        output_columns=len(merged.columns),
        added_h2h_features=len(h2h_columns),
        duplicate_match_ids=duplicate_ids,
        unmatched_h2h_rows=unmatched_h2h_rows,
        h2h_missing_values=h2h_missing_values,
        total_missing_values=int(merged.isna().sum().sum()),
        infinite_numeric_values=infinite_numeric_values,
    )
    return merged, report, h2h_columns


def save_outputs(
    training_v3: pd.DataFrame,
    report: GenerationReport,
    h2h_columns: list[str],
    paths: GeneratorPaths,
) -> None:
    """Save the dataset and diagnostics."""
    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    training_v3.to_csv(
        paths.output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        **asdict(report),
        "generator_version": "3.0",
        "created_at": datetime.now().isoformat(),
        "training_v2_csv": str(paths.training_v2_csv),
        "h2h_features_csv": str(paths.h2h_features_csv),
        "output_csv": str(paths.output_csv),
    }
    (
        paths.diagnostics_dir
        / "training_dataset_v3_summary.json"
    ).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "feature": h2h_columns,
            "dtype": [
                str(training_v3[column].dtype)
                for column in h2h_columns
            ],
            "missing_count": [
                int(training_v3[column].isna().sum())
                for column in h2h_columns
            ],
            "minimum": [
                float(training_v3[column].min())
                for column in h2h_columns
            ],
            "maximum": [
                float(training_v3[column].max())
                for column in h2h_columns
            ],
            "mean": [
                float(training_v3[column].mean())
                for column in h2h_columns
            ],
        }
    ).to_csv(
        paths.diagnostics_dir / "h2h_feature_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "column": training_v3.columns,
            "missing_count": [
                int(training_v3[column].isna().sum())
                for column in training_v3.columns
            ],
        }
    ).sort_values(
        ["missing_count", "column"],
        ascending=[False, True],
    ).to_csv(
        paths.diagnostics_dir / "missing_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_ids = pd.Index(
        pd.read_csv(
            paths.training_v2_csv,
            usecols=[KEY_COLUMN],
        )[KEY_COLUMN]
    )
    h2h_ids = pd.Index(
        pd.read_csv(
            paths.h2h_features_csv,
            usecols=[KEY_COLUMN],
        )[KEY_COLUMN]
    )
    unmatched = pd.DataFrame(
        {
            KEY_COLUMN: source_ids.difference(h2h_ids),
        }
    )
    unmatched.to_csv(
        paths.diagnostics_dir / "unmatched_match_card_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    LOGGER.info("Saved training dataset v3: %s", paths.output_csv)
    LOGGER.info("Saved diagnostics: %s", paths.diagnostics_dir)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Merge historical H2H features into Project Alpha "
            "training_dataset_v2.csv."
        )
    )
    parser.add_argument(
        "--training-v2",
        type=Path,
        default=defaults.training_v2_csv,
    )
    parser.add_argument(
        "--h2h-features",
        type=Path,
        default=defaults.h2h_features_csv,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output_csv,
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=defaults.diagnostics_dir,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    paths = GeneratorPaths(
        training_v2_csv=args.training_v2,
        h2h_features_csv=args.h2h_features,
        output_csv=args.output,
        diagnostics_dir=args.diagnostics_dir,
    )
    paths.validate()

    training_v2 = load_csv(
        paths.training_v2_csv,
        "training_dataset_v2",
    )
    h2h = load_csv(
        paths.h2h_features_csv,
        "historical_h2h_features",
    )

    training_v3, report, h2h_columns = build_dataset(
        training_v2,
        h2h,
    )
    save_outputs(
        training_v3,
        report,
        h2h_columns,
        paths,
    )

    print("=" * 72)
    print("Project Alpha Training Dataset Generator v3")
    print("=" * 72)
    print(f"Source rows                 : {report.source_rows}")
    print(f"Source columns              : {report.source_columns}")
    print(f"H2H rows                    : {report.h2h_rows}")
    print(f"Added H2H features          : {report.added_h2h_features}")
    print(f"Output rows                 : {report.output_rows}")
    print(f"Output columns              : {report.output_columns}")
    print(f"Duplicate match IDs         : {report.duplicate_match_ids}")
    print(f"Unmatched H2H rows          : {report.unmatched_h2h_rows}")
    print(f"H2H missing values          : {report.h2h_missing_values}")
    print(f"Total missing values        : {report.total_missing_values}")
    print(f"Infinite numeric values     : {report.infinite_numeric_values}")
    print(f"Saved                       : {paths.output_csv}")
    print(f"Diagnostics                 : {paths.diagnostics_dir}")


if __name__ == "__main__":
    main()
