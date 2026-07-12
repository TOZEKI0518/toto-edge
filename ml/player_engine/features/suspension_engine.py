from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("suspension_engine")

DEFAULT_COLUMNS = [
    "player_key",
    "player_name",
    "team",
    "target_match_card_id",
    "reason",
    "matches_remaining",
    "is_suspended",
]

OUTPUT_COLUMNS = [
    "player_key",
    "player_name",
    "team",
    "target_match_card_id",
    "reason",
    "matches_remaining",
    "is_suspended",
    "core_score",
    "suspension_impact",
]


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def default_processed_dir() -> Path:
    """Return the Project Alpha processed-data directory."""
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def ensure_template(path: Path) -> None:
    """Create an empty suspension input template when it does not exist."""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=DEFAULT_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )
    LOGGER.info("Created suspension template: %s", path)


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV file, returning an empty frame when the file is absent."""
    if not path.exists():
        return pd.DataFrame(columns=columns)

    return pd.read_csv(path)


def validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    """Raise a clear error when required columns are missing."""
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing)}"
        )


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean representations."""
    true_values = {"1", "true", "t", "yes", "y", "on"}
    false_values = {"0", "false", "f", "no", "n", "off", ""}

    def convert(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)

        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False

        raise ValueError(f"Invalid boolean value in is_suspended: {value!r}")

    return series.map(convert)


def normalize_suspensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize official/manual suspension rows."""
    if frame.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    validate_columns(
        frame,
        {"player_name", "team", "reason", "is_suspended"},
        "suspensions.csv",
    )

    result = frame.copy()

    for column in DEFAULT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result["player_name"] = result["player_name"].astype(str).str.strip()
    result["team"] = result["team"].astype(str).str.strip()
    result["reason"] = result["reason"].fillna("").astype(str).str.strip()
    result["is_suspended"] = normalize_boolean(result["is_suspended"])

    result["matches_remaining"] = pd.to_numeric(
        result["matches_remaining"],
        errors="coerce",
    ).fillna(0).clip(lower=0).astype(int)

    result["target_match_card_id"] = pd.to_numeric(
        result["target_match_card_id"],
        errors="coerce",
    ).astype("Int64")

    missing_key = (
        result["player_key"].isna()
        | result["player_key"].astype(str).str.strip().eq("")
    )
    result.loc[missing_key, "player_key"] = (
        result.loc[missing_key, "team"]
        + "::"
        + result.loc[missing_key, "player_name"]
    )

    result = result.loc[
        result["player_name"].ne("")
        & result["team"].ne("")
    ]

    return result[DEFAULT_COLUMNS].drop_duplicates(
        subset=["player_key", "target_match_card_id"],
        keep="last",
    )


def load_core_scores(path: Path) -> pd.DataFrame:
    """Load optional Core Player Score data."""
    frame = read_csv_or_empty(
        path,
        ["player_key", "player_name", "team", "core_score"],
    )

    if frame.empty:
        return frame

    validate_columns(
        frame,
        {"player_name", "team", "core_score"},
        "player_core_scores.csv",
    )

    result = frame.copy()

    if "player_key" not in result.columns:
        result["player_key"] = (
            result["team"].astype(str).str.strip()
            + "::"
            + result["player_name"].astype(str).str.strip()
        )

    result["core_score"] = pd.to_numeric(
        result["core_score"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    return result[
        ["player_key", "core_score"]
    ].drop_duplicates("player_key", keep="last")


def build_suspension_features(
    suspensions: pd.DataFrame,
    core_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build player-level suspension features.

    suspension_impact equals core_score for suspended players and zero
    otherwise. Team/position weighting is intentionally deferred to
    team_feature_generator.py.
    """
    result = normalize_suspensions(suspensions)

    if result.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if core_scores.empty:
        result["core_score"] = 0.0
    else:
        result = result.merge(
            core_scores,
            on="player_key",
            how="left",
            validate="many_to_one",
        )
        result["core_score"] = result["core_score"].fillna(0.0)

    result["suspension_impact"] = (
        result["core_score"]
        * result["is_suspended"].astype(float)
    )

    return result[OUTPUT_COLUMNS].sort_values(
        ["is_suspended", "suspension_impact", "team", "player_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    processed_dir = default_processed_dir()

    parser = argparse.ArgumentParser(
        description="Build lightweight player suspension features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=processed_dir / "suspensions.csv",
        help="Official/manual suspension CSV.",
    )
    parser.add_argument(
        "--core-scores",
        type=Path,
        default=processed_dir / "player_core_scores.csv",
        help="Optional Core Player Score CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_dir / "player_suspension_features.csv",
        help="Output feature CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the suspension feature pipeline."""
    args = parse_args()
    configure_logging(args.verbose)

    ensure_template(args.input)

    suspensions = read_csv_or_empty(args.input, DEFAULT_COLUMNS)
    core_scores = load_core_scores(args.core_scores)

    features = build_suspension_features(
        suspensions=suspensions,
        core_scores=core_scores,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    suspended_count = int(features["is_suspended"].sum()) if not features.empty else 0

    LOGGER.info("Rows: %s", len(features))
    LOGGER.info("Suspended players: %s", suspended_count)
    LOGGER.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
