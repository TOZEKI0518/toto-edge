from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("feature_importance_analyzer")


@dataclass(frozen=True)
class AnalyzerConfig:
    random_forest_csv: Path
    lightgbm_csv: Path
    output_dir: Path
    top_n: int = 30
    bottom_n: int = 30
    low_importance_quantile: float = 0.25
    disagreement_threshold: float = 0.35


@dataclass(frozen=True)
class AnalysisSummary:
    total_features: int
    top_consensus_features: int
    low_importance_features: int
    high_disagreement_features: int
    random_forest_only_features: int
    lightgbm_only_features: int
    common_features: int


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths() -> tuple[Path, Path, Path]:
    root = project_root()
    return (
        root / "ml" / "rf_v5" / "randomforest_v5_feature_importance.csv",
        root / "ml" / "lgbm_v5" / "lightgbm_v5_feature_importance.csv",
        root / "ml" / "feature_importance_analysis",
    )


def load_importance(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} importance file not found: {path}")

    frame = pd.read_csv(path)
    required = {"feature", "importance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{model_name} importance file missing columns: {sorted(missing)}"
        )

    result = frame[["feature", "importance"]].copy()
    result["feature"] = result["feature"].astype(str).str.strip()
    result["importance"] = pd.to_numeric(
        result["importance"],
        errors="coerce",
    )
    result = result.dropna(subset=["feature", "importance"])
    result = (
        result.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    total = float(result["importance"].sum())
    if total > 0:
        result["importance"] = result["importance"] / total

    result[f"{model_name}_rank"] = (
        result["importance"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    result = result.rename(
        columns={"importance": f"{model_name}_importance"}
    )

    LOGGER.info("%s features: %s", model_name, len(result))
    return result


def classify_feature_family(feature: str) -> str:
    lower = feature.lower()

    if "momentum" in lower:
        return "Momentum"
    if "core" in lower:
        return "Core player strength"
    if "starter" in lower:
        return "Starter availability"
    if "available" in lower or "availability" in lower:
        return "Availability"
    if any(token in lower for token in ("gk", "df", "mf", "fw")):
        return "Position-specific"
    if "balance" in lower:
        return "Squad balance"
    if "stability" in lower:
        return "Team stability"
    if "suspension" in lower:
        return "Suspension"
    if "rank" in lower:
        return "Ranking"
    if "elo" in lower:
        return "Elo"
    if "attack" in lower:
        return "Attack"
    if "defense" in lower:
        return "Defense"
    if "goal" in lower:
        return "Goals"
    if "points" in lower:
        return "Points"
    if "winrate" in lower or "win_rate" in lower:
        return "Win rate"
    if feature.endswith("Diff"):
        return "Difference"
    if feature.endswith("Ratio"):
        return "Ratio"
    return "Other"


def feature_side(feature: str) -> str:
    if feature.startswith("home"):
        return "Home"
    if feature.startswith("away"):
        return "Away"
    if feature.endswith("Diff"):
        return "Difference"
    if feature.endswith("Ratio"):
        return "Ratio"
    return "Neutral"


def feature_position(feature: str) -> str:
    upper = feature.upper()
    for position in ("GK", "DF", "MF", "FW"):
        if re.search(rf"(^|_){position}($|_)", upper) or position in upper:
            return position
    return "Team-level"


def build_comparison(
    random_forest: pd.DataFrame,
    lightgbm: pd.DataFrame,
) -> pd.DataFrame:
    comparison = random_forest.merge(
        lightgbm,
        on="feature",
        how="outer",
    )

    comparison["random_forest_importance"] = comparison[
        "random_forest_importance"
    ].fillna(0.0)
    comparison["lightgbm_importance"] = comparison[
        "lightgbm_importance"
    ].fillna(0.0)

    max_rank = len(comparison) + 1
    comparison["random_forest_rank"] = comparison[
        "random_forest_rank"
    ].fillna(max_rank).astype(int)
    comparison["lightgbm_rank"] = comparison[
        "lightgbm_rank"
    ].fillna(max_rank).astype(int)

    comparison["mean_importance"] = comparison[
        ["random_forest_importance", "lightgbm_importance"]
    ].mean(axis=1)

    comparison["geometric_mean_importance"] = np.sqrt(
        comparison["random_forest_importance"]
        * comparison["lightgbm_importance"]
    )

    comparison["mean_rank"] = comparison[
        ["random_forest_rank", "lightgbm_rank"]
    ].mean(axis=1)

    comparison["rank_gap"] = (
        comparison["random_forest_rank"]
        - comparison["lightgbm_rank"]
    ).abs()

    denominator = comparison[
        ["random_forest_importance", "lightgbm_importance"]
    ].max(axis=1)
    comparison["relative_disagreement"] = np.where(
        denominator > 0,
        (
            comparison["random_forest_importance"]
            - comparison["lightgbm_importance"]
        ).abs()
        / denominator,
        0.0,
    )

    comparison["consensus_score"] = (
        0.55 * comparison["mean_importance"]
        + 0.45 * comparison["geometric_mean_importance"]
    )

    comparison["family"] = comparison["feature"].map(
        classify_feature_family
    )
    comparison["side"] = comparison["feature"].map(feature_side)
    comparison["position"] = comparison["feature"].map(feature_position)

    comparison["present_in_random_forest"] = (
        comparison["random_forest_importance"] > 0
    )
    comparison["present_in_lightgbm"] = (
        comparison["lightgbm_importance"] > 0
    )

    comparison["consensus_rank"] = (
        comparison["consensus_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return comparison.sort_values(
        ["consensus_score", "mean_rank"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_family_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby("family", as_index=False)
        .agg(
            features=("feature", "count"),
            random_forest_importance=(
                "random_forest_importance",
                "sum",
            ),
            lightgbm_importance=("lightgbm_importance", "sum"),
            consensus_importance=("consensus_score", "sum"),
            average_rank=("consensus_rank", "mean"),
        )
        .sort_values("consensus_importance", ascending=False)
        .reset_index(drop=True)
    )


def build_position_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby("position", as_index=False)
        .agg(
            features=("feature", "count"),
            random_forest_importance=(
                "random_forest_importance",
                "sum",
            ),
            lightgbm_importance=("lightgbm_importance", "sum"),
            consensus_importance=("consensus_score", "sum"),
        )
        .sort_values("consensus_importance", ascending=False)
        .reset_index(drop=True)
    )


def build_side_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby("side", as_index=False)
        .agg(
            features=("feature", "count"),
            random_forest_importance=(
                "random_forest_importance",
                "sum",
            ),
            lightgbm_importance=("lightgbm_importance", "sum"),
            consensus_importance=("consensus_score", "sum"),
        )
        .sort_values("consensus_importance", ascending=False)
        .reset_index(drop=True)
    )


def build_recommendations(
    comparison: pd.DataFrame,
    config: AnalyzerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    low_threshold = comparison["consensus_score"].quantile(
        config.low_importance_quantile
    )

    low_importance = comparison.loc[
        comparison["consensus_score"] <= low_threshold
    ].copy()
    low_importance["recommendation"] = (
        "Candidate for removal; validate with backtest ablation."
    )

    disagreement = comparison.loc[
        comparison["relative_disagreement"]
        >= config.disagreement_threshold
    ].copy()
    disagreement["recommendation"] = (
        "Model disagreement; inspect leakage, non-linearity, and stability."
    )

    keep = comparison.head(config.top_n).copy()
    keep["recommendation"] = (
        "High consensus importance; retain in the core feature set."
    )

    return keep, low_importance, disagreement



def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table without tabulate."""
    columns = [str(column) for column in df.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]

    for row in df.itertuples(index=False, name=None):
        values: list[str] = []

        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def save_markdown_report(
    path: Path,
    summary: AnalysisSummary,
    top_features: pd.DataFrame,
    family_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    low_importance: pd.DataFrame,
    disagreement: pd.DataFrame,
) -> None:
    """Save a Markdown analysis report without optional dependencies."""
    top_table = top_features[
        [
            "consensus_rank",
            "feature",
            "consensus_score",
            "random_forest_rank",
            "lightgbm_rank",
            "family",
            "position",
            "side",
        ]
    ]

    low_table = low_importance[
        [
            "feature",
            "consensus_rank",
            "consensus_score",
            "random_forest_rank",
            "lightgbm_rank",
            "family",
        ]
    ].head(30)

    disagreement_table = disagreement[
        [
            "feature",
            "random_forest_importance",
            "lightgbm_importance",
            "random_forest_rank",
            "lightgbm_rank",
            "relative_disagreement",
        ]
    ].head(30)

    lines = [
        "# Project Alpha Feature Importance Analysis",
        "",
        "## Summary",
        "",
        f"- Total features: {summary.total_features}",
        f"- Common features: {summary.common_features}",
        f"- RandomForest-only features: "
        f"{summary.random_forest_only_features}",
        f"- LightGBM-only features: "
        f"{summary.lightgbm_only_features}",
        f"- Low-importance candidates: "
        f"{summary.low_importance_features}",
        f"- High-disagreement features: "
        f"{summary.high_disagreement_features}",
        "",
        "## Top Consensus Features",
        "",
        dataframe_to_markdown(top_table),
        "",
        "## Feature Family Summary",
        "",
        dataframe_to_markdown(family_summary),
        "",
        "## Position Summary",
        "",
        dataframe_to_markdown(position_summary),
        "",
        "## Home/Away/Difference Summary",
        "",
        dataframe_to_markdown(side_summary),
        "",
        "## Low-Importance Removal Candidates",
        "",
        dataframe_to_markdown(low_table),
        "",
        "## High Model Disagreement",
        "",
        dataframe_to_markdown(disagreement_table),
        "",
        "## Next Validation Step",
        "",
        "Run an ablation backtest after removing low-importance features. "
        "Do not remove features solely from importance scores.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(config: AnalyzerConfig) -> AnalysisSummary:
    random_forest = load_importance(
        config.random_forest_csv,
        "random_forest",
    )
    lightgbm = load_importance(
        config.lightgbm_csv,
        "lightgbm",
    )

    comparison = build_comparison(random_forest, lightgbm)
    family_summary = build_family_summary(comparison)
    position_summary = build_position_summary(comparison)
    side_summary = build_side_summary(comparison)

    keep, low_importance, disagreement = build_recommendations(
        comparison,
        config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        config.output_dir / "feature_importance_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    keep.to_csv(
        config.output_dir / "recommended_core_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    low_importance.to_csv(
        config.output_dir / "low_importance_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    disagreement.to_csv(
        config.output_dir / "high_disagreement_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    family_summary.to_csv(
        config.output_dir / "feature_family_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    position_summary.to_csv(
        config.output_dir / "feature_position_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    side_summary.to_csv(
        config.output_dir / "feature_side_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = AnalysisSummary(
        total_features=len(comparison),
        top_consensus_features=len(keep),
        low_importance_features=len(low_importance),
        high_disagreement_features=len(disagreement),
        random_forest_only_features=int(
            (
                comparison["present_in_random_forest"]
                & ~comparison["present_in_lightgbm"]
            ).sum()
        ),
        lightgbm_only_features=int(
            (
                ~comparison["present_in_random_forest"]
                & comparison["present_in_lightgbm"]
            ).sum()
        ),
        common_features=int(
            (
                comparison["present_in_random_forest"]
                & comparison["present_in_lightgbm"]
            ).sum()
        ),
    )

    (config.output_dir / "analysis_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_markdown_report(
        config.output_dir / "feature_importance_report.md",
        summary,
        keep,
        family_summary,
        position_summary,
        side_summary,
        low_importance,
        disagreement,
    )

    print("=" * 72)
    print("Project Alpha Feature Importance Analyzer")
    print("=" * 72)
    print(f"Total features              : {summary.total_features}")
    print(f"Common to both models       : {summary.common_features}")
    print(f"Low-importance candidates   : {summary.low_importance_features}")
    print(f"High-disagreement features  : {summary.high_disagreement_features}")
    print()
    print("Top 20 Consensus Features")
    print("-" * 72)
    print(
        comparison[
            [
                "consensus_rank",
                "feature",
                "consensus_score",
                "random_forest_rank",
                "lightgbm_rank",
                "family",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print()
    print("Feature Families")
    print("-" * 72)
    print(family_summary.to_string(index=False))
    print()
    print(f"Saved                       : {config.output_dir}")

    return summary


def parse_args() -> argparse.Namespace:
    random_forest, lightgbm, output = default_paths()

    parser = argparse.ArgumentParser(
        description="Compare RandomForest and LightGBM feature importance."
    )
    parser.add_argument(
        "--random-forest",
        type=Path,
        default=random_forest,
    )
    parser.add_argument(
        "--lightgbm",
        type=Path,
        default=lightgbm,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output,
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--bottom-n", type=int, default=30)
    parser.add_argument(
        "--low-importance-quantile",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--disagreement-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = AnalyzerConfig(
        random_forest_csv=args.random_forest,
        lightgbm_csv=args.lightgbm,
        output_dir=args.output_dir,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        low_importance_quantile=args.low_importance_quantile,
        disagreement_threshold=args.disagreement_threshold,
    )
    run_analysis(config)


if __name__ == "__main__":
    main()
