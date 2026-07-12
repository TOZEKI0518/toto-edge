from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    severity: str
    message: str


class PlayerDataValidator:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.matches = self._load("matches.csv")
        self.players = self._load("player_match_history.csv")
        self.substitutions = self._load("substitutions.csv")
        self.goals = self._load("goals.csv")
        self.results: list[CheckResult] = []

    def _load(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.read_csv(path, dtype={"match_card_id": str})

    def add(self, name: str, passed: bool, message: str, severity: str = "ERROR") -> None:
        self.results.append(CheckResult(name, passed, severity, message))

    def validate_required_columns(self) -> None:
        required = {
            "matches": {"match_card_id", "home_team", "away_team", "home_score", "away_score"},
            "players": {"match_card_id", "team", "player_name", "position", "starter", "minutes", "appeared"},
            "substitutions": {"match_card_id", "team", "player_out", "player_in", "minute"},
            "goals": {"match_card_id", "team", "player_name", "minute"},
        }
        frames = {"matches": self.matches, "players": self.players, "substitutions": self.substitutions, "goals": self.goals}
        for name, columns in required.items():
            missing = sorted(columns - set(frames[name].columns))
            self.add(f"required_columns_{name}", not missing, "OK" if not missing else f"Missing: {missing}")

    def validate_match_uniqueness(self) -> None:
        duplicates = int(self.matches.duplicated(["match_card_id"]).sum())
        self.add("unique_matches", duplicates == 0, f"Duplicate match rows: {duplicates}")

    def validate_team_counts(self) -> None:
        bad = []
        for match_id, match in self.matches.groupby("match_card_id"):
            player_teams = set(self.players.loc[self.players["match_card_id"] == match_id, "team"].dropna())
            expected = {str(match.iloc[0]["home_team"]), str(match.iloc[0]["away_team"])}
            if player_teams != expected:
                bad.append(f"{match_id}: expected={expected}, actual={player_teams}")
        self.add("team_alignment", not bad, "All matches aligned" if not bad else "; ".join(bad[:10]))

    def validate_starters(self) -> None:
        bad = []
        starters = self.players[self.players["starter"].astype(str).str.lower().isin(["true", "1"])]
        for (match_id, team), group in starters.groupby(["match_card_id", "team"]):
            if len(group) != 11:
                bad.append(f"{match_id}/{team}={len(group)}")
        self.add("eleven_starters", not bad, "11 starters for every team" if not bad else "Invalid starter counts: " + ", ".join(bad[:10]))

    def validate_player_minutes(self) -> None:
        minutes = pd.to_numeric(self.players["minutes"], errors="coerce")
        invalid = self.players[minutes.isna() | (minutes < 0) | (minutes > 130)]
        self.add("player_minutes_range", invalid.empty, f"Invalid minute rows: {len(invalid)}")

    def validate_substitutions(self) -> None:
        bad = []
        known = set(zip(self.players["match_card_id"], self.players["team"], self.players["player_name"]))
        for row in self.substitutions.itertuples(index=False):
            if row.player_out and (row.match_card_id, row.team, row.player_out) not in known:
                bad.append(f"OUT {row.match_card_id}/{row.team}/{row.player_out}")
            if row.player_in and (row.match_card_id, row.team, row.player_in) not in known:
                bad.append(f"IN {row.match_card_id}/{row.team}/{row.player_in}")
        self.add("substitution_players_exist", not bad, "All substitution players found" if not bad else "; ".join(bad[:10]))

    def validate_goals_vs_score(self) -> None:
        bad = []
        for row in self.matches.itertuples(index=False):
            goals = self.goals[self.goals["match_card_id"] == row.match_card_id]
            home_count = int((goals["team"] == row.home_team).sum())
            away_count = int((goals["team"] == row.away_team).sum())
            if pd.notna(row.home_score) and home_count != int(row.home_score):
                bad.append(f"{row.match_card_id} home score={row.home_score}, goals={home_count}")
            if pd.notna(row.away_score) and away_count != int(row.away_score):
                bad.append(f"{row.match_card_id} away score={row.away_score}, goals={away_count}")
        self.add("goal_count_matches_score", not bad, "Goal events match final scores" if not bad else "; ".join(bad[:10]), severity="WARNING")

    def validate_duplicate_players(self) -> None:
        duplicates = int(self.players.duplicated(["match_card_id", "team", "player_name"]).sum())
        self.add("unique_player_match_rows", duplicates == 0, f"Duplicate player-match rows: {duplicates}")

    def run(self) -> list[CheckResult]:
        self.validate_required_columns()
        self.validate_match_uniqueness()
        self.validate_duplicate_players()
        self.validate_team_counts()
        self.validate_starters()
        self.validate_player_minutes()
        self.validate_substitutions()
        self.validate_goals_vs_score()
        return self.results

    def summary(self) -> dict:
        errors = [r for r in self.results if not r.passed and r.severity == "ERROR"]
        warnings = [r for r in self.results if not r.passed and r.severity == "WARNING"]
        return {
            "passed": len(errors) == 0,
            "checks": len(self.results),
            "failedErrors": len(errors),
            "failedWarnings": len(warnings),
            "matches": len(self.matches),
            "playerRows": len(self.players),
            "substitutions": len(self.substitutions),
            "goals": len(self.goals),
            "results": [asdict(result) for result in self.results],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Project Alpha player data outputs.")
    parser.add_argument("--data-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()

    validator = PlayerDataValidator(args.data_dir)
    validator.run()
    summary = validator.summary()

    print("=" * 64)
    print("Project Alpha - Player Data Validation")
    print("=" * 64)
    for result in validator.results:
        status = "PASS" if result.passed else result.severity
        print(f"[{status:<7}] {result.name:<32} {result.message}")

    report_path = Path(args.data_dir) / "validation_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nValidation Summary")
    print("-" * 40)
    print(f"Matches          : {summary['matches']:,}")
    print(f"Player rows      : {summary['playerRows']:,}")
    print(f"Substitutions    : {summary['substitutions']:,}")
    print(f"Goals            : {summary['goals']:,}")
    print(f"Failed errors    : {summary['failedErrors']}")
    print(f"Failed warnings  : {summary['failedWarnings']}")
    print(f"Saved            : {report_path}")

    failed = summary["failedErrors"] > 0 or (args.strict_warnings and summary["failedWarnings"] > 0)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
