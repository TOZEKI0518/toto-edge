from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("v6_confidence_diagnostics")
CLASS_ORDER = ("A", "D", "H")
PROBABILITY_COLUMNS = ("prob_away", "prob_draw", "prob_home")


class ConfidenceDiagnosticsError(RuntimeError):
    """Raised when confidence diagnostics cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    matches_csv: Path
    output_dir: Path
    target_match_count: int = 13
    fixed_count: int = 8

    def validate(self) -> None:
        if not self.matches_csv.is_file():
            raise FileNotFoundError(f"Backtest match file not found: {self.matches_csv}")
        if self.target_match_count <= 0:
            raise ValueError("target_match_count must be positive.")
        if not 0 < self.fixed_count < self.target_match_count:
            raise ValueError("fixed_count must be between 1 and target_match_count - 1.")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths() -> dict[str, Path]:
    root = project_root()
    return {
        "matches": root / "ml" / "true_backtest_v6" / "v6_true_backtest_matches.csv",
        "output": root / "ml" / "confidence_diagnostics_v6",
    }


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_outcome(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return {
        "A": "A", "D": "D", "H": "H", "0": "D", "1": "H", "2": "A",
        "AWAY": "A", "DRAW": "D", "HOME": "H",
    }.get(str(value).strip().upper())


def load_matches(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"round_no", "toto_match_no", "fixture_actual", *PROBABILITY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ConfidenceDiagnosticsError(f"Input CSV is missing columns: {missing}")

    frame = frame.copy()
    frame["round_no"] = pd.to_numeric(frame["round_no"], errors="coerce")
    frame["toto_match_no"] = pd.to_numeric(frame["toto_match_no"], errors="coerce")
    frame["fixture_actual"] = frame["fixture_actual"].map(normalize_outcome)
    probs = frame[list(PROBABILITY_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    invalid = (
        frame["round_no"].isna()
        | frame["toto_match_no"].isna()
        | ~frame["fixture_actual"].isin(CLASS_ORDER)
        | probs.isna().any(axis=1)
        | (probs < 0).any(axis=1)
        | (probs.sum(axis=1) <= 0)
    )
    if invalid.any():
        LOGGER.warning("Dropping %d invalid rows.", int(invalid.sum()))
    frame = frame.loc[~invalid].copy()
    probs = probs.loc[frame.index]
    frame[list(PROBABILITY_COLUMNS)] = probs.div(probs.sum(axis=1), axis=0)
    frame["round_no"] = frame["round_no"].astype(int)
    frame["toto_match_no"] = frame["toto_match_no"].astype(int)

    if frame.duplicated(["round_no", "toto_match_no"]).any():
        raise ConfidenceDiagnosticsError("Duplicate round_no/toto_match_no rows detected.")
    return frame


def add_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    probs = out[list(PROBABILITY_COLUMNS)].to_numpy(float)
    order = np.argsort(probs, axis=1)[:, ::-1]
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    classes = np.asarray(CLASS_ORDER)

    out["top1_pick"] = classes[order[:, 0]]
    out["top2_pick"] = classes[order[:, 1]]
    out["third_pick"] = classes[order[:, 2]]
    out["top1_probability"] = sorted_probs[:, 0]
    out["top2_probability"] = sorted_probs[:, 1]
    out["third_probability"] = sorted_probs[:, 2]
    out["top1_margin"] = sorted_probs[:, 0] - sorted_probs[:, 1]
    out["top2_coverage_probability"] = sorted_probs[:, 0] + sorted_probs[:, 1]

    safe = np.clip(probs, 1e-15, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1) / math.log(3)
    out["normalized_entropy"] = entropy
    out["certainty_score"] = (
        0.45 * out["top1_probability"]
        + 0.35 * out["top1_margin"]
        + 0.20 * (1.0 - out["normalized_entropy"])
    )
    out["top1_hit"] = out["top1_pick"] == out["fixture_actual"]
    out["top2_hit"] = (
        (out["top1_pick"] == out["fixture_actual"])
        | (out["top2_pick"] == out["fixture_actual"])
    )
    return out


def select_complete_rounds(frame: pd.DataFrame, match_count: int) -> pd.DataFrame:
    sizes = frame.groupby("round_no").size()
    rounds = sizes.loc[sizes == match_count].index
    selected = frame.loc[frame["round_no"].isin(rounds)].copy()
    if selected.empty:
        raise ConfidenceDiagnosticsError(f"No complete {match_count}-match rounds found.")
    return selected


def rank_within_round(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("round_no", sort=True):
        ranked = group.sort_values(
            ["certainty_score", "top1_margin", "top1_probability", "toto_match_no"],
            ascending=[False, False, False, True],
            kind="stable",
        ).copy()
        ranked["confidence_rank"] = np.arange(1, len(ranked) + 1)
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True)


def rank_summary(ranked: pd.DataFrame, match_count: int) -> pd.DataFrame:
    rows = []
    for rank in range(1, match_count + 1):
        x = ranked.loc[ranked["confidence_rank"] == rank]
        rows.append({
            "confidence_rank": rank,
            "rows": len(x),
            "top1_accuracy": float(x["top1_hit"].mean()),
            "top2_coverage": float(x["top2_hit"].mean()),
            "mean_top1_probability": float(x["top1_probability"].mean()),
            "mean_top1_margin": float(x["top1_margin"].mean()),
            "mean_certainty_score": float(x["certainty_score"].mean()),
        })
    return pd.DataFrame(rows)


def top_k_summary(ranked: pd.DataFrame, match_count: int) -> pd.DataFrame:
    rows = []
    round_count = ranked["round_no"].nunique()
    for k in range(1, match_count + 1):
        x = ranked.loc[ranked["confidence_rank"] <= k]
        by_round = x.groupby("round_no")["top1_hit"].sum().astype(int)
        misses = k - by_round
        rows.append({
            "top_k": k,
            "rounds": round_count,
            "top1_accuracy": float(x["top1_hit"].mean()),
            "mean_hits": float(by_round.mean()),
            "all_correct_rounds": int((misses == 0).sum()),
            "all_correct_rate": float((misses == 0).mean()),
            "one_miss_or_better_rounds": int((misses <= 1).sum()),
            "one_miss_or_better_rate": float((misses <= 1).mean()),
            "two_miss_or_better_rounds": int((misses <= 2).sum()),
            "two_miss_or_better_rate": float((misses <= 2).mean()),
        })
    return pd.DataFrame(rows)


def strategy_by_round(ranked: pd.DataFrame, fixed_count: int, match_count: int) -> pd.DataFrame:
    double_count = match_count - fixed_count
    rows: list[dict[str, Any]] = []
    for round_no, group in ranked.groupby("round_no", sort=True):
        fixed = group.loc[group["confidence_rank"] <= fixed_count]
        doubles = group.loc[group["confidence_rank"] > fixed_count]
        fixed_hits = int(fixed["top1_hit"].sum())
        double_hits = int(doubles["top2_hit"].sum())
        fixed_misses = fixed_count - fixed_hits
        double_misses = double_count - double_hits
        rows.append({
            "round_no": int(round_no),
            "fixed_count": fixed_count,
            "double_count": double_count,
            "base_ticket_count": 2 ** double_count,
            "fixed_hits": fixed_hits,
            "fixed_misses": fixed_misses,
            "double_top2_hits": double_hits,
            "double_top2_misses": double_misses,
            "strategy_best_hits": fixed_hits + double_hits,
            "strategy_best_misses": fixed_misses + double_misses,
            "fixed_all_correct": fixed_misses == 0,
            "fixed_one_miss_or_better": fixed_misses <= 1,
            "doubles_all_covered": double_misses == 0,
            "exact_result_covered": fixed_misses == 0 and double_misses == 0,
        })
    return pd.DataFrame(rows).sort_values("round_no", ascending=False)


def threshold_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.34, 0.36, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60]:
        x = ranked.loc[ranked["top1_probability"] >= threshold]
        rows.append({
            "top1_probability_threshold": threshold,
            "rows": len(x),
            "share_of_matches": len(x) / len(ranked),
            "realized_top1_accuracy": float(x["top1_hit"].mean()) if len(x) else None,
            "realized_top2_coverage": float(x["top2_hit"].mean()) if len(x) else None,
        })
    return pd.DataFrame(rows)


def build_summary(ranked: pd.DataFrame, strategy: pd.DataFrame, config: DiagnosticsConfig) -> dict[str, Any]:
    fixed = ranked.loc[ranked["confidence_rank"] <= config.fixed_count]
    doubles = ranked.loc[ranked["confidence_rank"] > config.fixed_count]
    return {
        "configuration": {
            **asdict(config),
            "matches_csv": str(config.matches_csv),
            "output_dir": str(config.output_dir),
        },
        "rounds_evaluated": int(ranked["round_no"].nunique()),
        "matches_evaluated": len(ranked),
        "overall_top1_accuracy": float(ranked["top1_hit"].mean()),
        "overall_top2_coverage": float(ranked["top2_hit"].mean()),
        "fixed_strategy": {
            "fixed_count": config.fixed_count,
            "double_count": config.target_match_count - config.fixed_count,
            "base_ticket_count": 2 ** (config.target_match_count - config.fixed_count),
            "fixed_match_accuracy": float(fixed["top1_hit"].mean()),
            "fixed_all_correct_rounds": int(strategy["fixed_all_correct"].sum()),
            "fixed_all_correct_rate": float(strategy["fixed_all_correct"].mean()),
            "fixed_one_miss_or_better_rounds": int(strategy["fixed_one_miss_or_better"].sum()),
            "fixed_one_miss_or_better_rate": float(strategy["fixed_one_miss_or_better"].mean()),
            "double_match_top2_coverage": float(doubles["top2_hit"].mean()),
            "doubles_all_covered_rounds": int(strategy["doubles_all_covered"].sum()),
            "doubles_all_covered_rate": float(strategy["doubles_all_covered"].mean()),
            "exact_result_covered_rounds": int(strategy["exact_result_covered"].sum()),
            "exact_result_covered_rate": float(strategy["exact_result_covered"].mean()),
            "average_strategy_best_hits": float(strategy["strategy_best_hits"].mean()),
            "one_miss_or_better_rounds": int((strategy["strategy_best_misses"] <= 1).sum()),
            "two_miss_or_better_rounds": int((strategy["strategy_best_misses"] <= 2).sum()),
        },
    }


def save_outputs(output_dir: Path, **frames: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def print_summary(summary: dict[str, Any]) -> None:
    s = summary["fixed_strategy"]
    print("=" * 88)
    print("Project Alpha v6 Confidence Diagnostics")
    print("=" * 88)
    print(f"Complete rounds evaluated     : {summary['rounds_evaluated']}")
    print(f"Matches evaluated             : {summary['matches_evaluated']}")
    print(f"Overall top-1 accuracy        : {summary['overall_top1_accuracy']:.4f}")
    print(f"Overall top-2 coverage        : {summary['overall_top2_coverage']:.4f}")
    print("-" * 88)
    print(f"Fixed / double design         : {s['fixed_count']} singles + {s['double_count']} doubles")
    print(f"Base ticket count             : {s['base_ticket_count']}")
    print(f"Top fixed-match accuracy      : {s['fixed_match_accuracy']:.4f}")
    print(f"Fixed all-correct rounds      : {s['fixed_all_correct_rounds']} ({s['fixed_all_correct_rate']:.4f})")
    print(f"Fixed <= 1 miss rounds        : {s['fixed_one_miss_or_better_rounds']} ({s['fixed_one_miss_or_better_rate']:.4f})")
    print(f"Double-match top-2 coverage   : {s['double_match_top2_coverage']:.4f}")
    print(f"All doubles covered rounds    : {s['doubles_all_covered_rounds']} ({s['doubles_all_covered_rate']:.4f})")
    print(f"Exact result covered          : {s['exact_result_covered_rounds']} ({s['exact_result_covered_rate']:.4f})")
    print(f"Average strategy best hits    : {s['average_strategy_best_hits']:.4f}")
    print(f"One-miss-or-better rounds     : {s['one_miss_or_better_rounds']}")
    print(f"Two-miss-or-better rounds     : {s['two_miss_or_better_rounds']}")
    print(f"Output                        : {summary['configuration']['output_dir']}")


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Diagnose confidence ranking and an 8-single/5-double toto strategy."
    )
    parser.add_argument("--matches-csv", type=Path, default=defaults["matches"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output"])
    parser.add_argument("--target-match-count", type=int, default=13)
    parser.add_argument("--fixed-count", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config = DiagnosticsConfig(
        matches_csv=args.matches_csv,
        output_dir=args.output_dir,
        target_match_count=args.target_match_count,
        fixed_count=args.fixed_count,
    )
    config.validate()

    LOGGER.info("Loading %s", config.matches_csv)
    matches = select_complete_rounds(
        add_diagnostics(load_matches(config.matches_csv)),
        config.target_match_count,
    )
    ranked = rank_within_round(matches)
    ranks = rank_summary(ranked, config.target_match_count)
    top_k = top_k_summary(ranked, config.target_match_count)
    strategy = strategy_by_round(ranked, config.fixed_count, config.target_match_count)
    thresholds = threshold_summary(ranked)
    summary = build_summary(ranked, strategy, config)

    save_outputs(
        config.output_dir,
        **{
            "v6_confidence_ranked_matches.csv": ranked,
            "v6_confidence_rank_summary.csv": ranks,
            "v6_confidence_top_k_summary.csv": top_k,
            "v6_fixed_double_strategy_by_round.csv": strategy,
            "v6_probability_threshold_summary.csv": thresholds,
        },
    )
    (config.output_dir / "v6_confidence_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
