from pathlib import Path
import pandas as pd

FEATURE_REGISTRY_PATH = Path("ml/feature_registry.csv")

DEFAULT_FEATURES = [
    "homeAttackRating",
    "defenseRatingDiff",
    "awayAttackRating",
    "homeAwayDiff",
    "homeRank",
    "awayAwayGoalsAgainstPerMatch",
    "rankDiff",
    "homeHomeElo",
    "homeHomeGoalsPerMatch",
    "absEloDiff",
    "absRankDiff",
    "absPointsDiff",
    "absAttackDiff",
    "balanceScore",
    "homeLast3Points",
    "awayLast3Points",
    "last3PointsDiff",
    "homeLast3Wins",
    "awayLast3Wins",
    "last3WinsDiff",
    "homeUnbeatenStreak",
    "awayUnbeatenStreak",
    "unbeatenStreakDiff",
    "homeLosingStreak",
    "awayLosingStreak",
    "losingStreakDiff",
]


def load_feature_columns() -> list[str]:
    if not FEATURE_REGISTRY_PATH.exists():
        return DEFAULT_FEATURES

    df = pd.read_csv(FEATURE_REGISTRY_PATH)

    required = {"feature", "enabled"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"feature_registry.csv missing columns: {missing}")

    enabled = df[df["enabled"].astype(str).str.upper().isin(["TRUE", "1", "YES"])]

    features = enabled["feature"].dropna().astype(str).tolist()

    if not features:
        raise ValueError("No enabled features in feature_registry.csv")

    return features


FEATURE_COLUMNS = load_feature_columns()