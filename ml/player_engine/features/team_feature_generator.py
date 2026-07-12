from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def processed_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_team_features(core: pd.DataFrame,
                        momentum: pd.DataFrame,
                        suspension: pd.DataFrame) -> pd.DataFrame:

    if core.empty:
        raise ValueError("player_core_scores.csv not found or empty.")

    core_team = (
        core.groupby("team", as_index=False)
        .agg(
            homeCoreScore=("core_score", "mean"),
            corePlayerCount=("core_score", "count"),
        )
    )

    if not momentum.empty:
        m = (
            momentum.groupby("team", as_index=False)
            .agg(
                homeMomentum=("momentum_score", "mean"),
            )
        )
        core_team = core_team.merge(m, on="team", how="left")
    else:
        core_team["homeMomentum"] = 0.0

    if not suspension.empty:
        s = (
            suspension.groupby("team", as_index=False)
            .agg(
                homeSuspensionLoss=("suspension_impact", "sum"),
                suspendedPlayers=("is_suspended", "sum"),
            )
        )
        core_team = core_team.merge(s, on="team", how="left")
    else:
        core_team["homeSuspensionLoss"] = 0.0
        core_team["suspendedPlayers"] = 0

    return core_team.fillna(0).sort_values("team").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    base = processed_dir()
    p.add_argument("--core", type=Path,
                   default=base / "player_core_scores.csv")
    p.add_argument("--momentum", type=Path,
                   default=base / "player_momentum.csv")
    p.add_argument("--suspension", type=Path,
                   default=base / "player_suspension_features.csv")
    p.add_argument("--output", type=Path,
                   default=base / "team_features.csv")
    args = p.parse_args()

    team = build_team_features(
        read_csv(args.core),
        read_csv(args.momentum),
        read_csv(args.suspension),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    team.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Team Feature Generator")
    print("=" * 60)
    print(team.head(20))
    print()
    print(f"Teams : {len(team)}")
    print(f"Saved : {args.output}")


if __name__ == "__main__":
    main()
