from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup, Tag


DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matches"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


@dataclass(frozen=True)
class ParsedMatch:
    match_card_id: str
    match_date: str
    kickoff_time: str
    stadium: str
    attendance: int | None
    weather: str
    temperature_c: float | None
    humidity_pct: float | None
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    home_first_half_score: int | None
    away_first_half_score: int | None
    home_second_half_score: int | None
    away_second_half_score: int | None
    home_coach: str
    away_coach: str
    regulation_minutes: int
    source_html: str


class JLeagueMatchParser:
    """Parse cached J.League SFMS02 match-detail HTML into tabular records."""

    def __init__(self, regulation_minutes: int = 90) -> None:
        if regulation_minutes <= 0:
            raise ValueError("regulation_minutes must be positive.")
        self.regulation_minutes = regulation_minutes

    @staticmethod
    def clean_text(value: str | None) -> str:
        if value is None:
            return ""
        value = unicodedata.normalize("NFKC", value)
        value = value.replace("\xa0", " ")
        return " ".join(value.split()).strip()

    @staticmethod
    def parse_int(value: str | None) -> int | None:
        text = JLeagueMatchParser.clean_text(value)
        match = re.search(r"-?\d[\d,]*", text)
        if not match:
            return None
        return int(match.group(0).replace(",", ""))

    @staticmethod
    def parse_float(value: str | None) -> float | None:
        text = JLeagueMatchParser.clean_text(value)
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None

    @staticmethod
    def parse_minute(value: str | None) -> int | None:
        text = JLeagueMatchParser.clean_text(value)
        if not text:
            return None
        added = re.search(r"(\d+)\s*\+\s*(\d+)", text)
        if added:
            return int(added.group(1)) + int(added.group(2))
        normal = re.search(r"\d+", text)
        return int(normal.group(0)) if normal else None

    @staticmethod
    def player_key(team: str, player_name: str) -> str:
        normalized = re.sub(r"\s+", "", player_name)
        return f"{team}::{normalized}"

    @staticmethod
    def _table_after_heading(heading: Tag) -> Tag | None:
        node = heading.find_next_sibling()
        while node is not None:
            if isinstance(node, Tag):
                if node.name in {"h3", "h4"}:
                    return None
                table = node if node.name == "table" else node.find("table")
                if table is not None and table.select_one("td.name") is not None:
                    return table
            node = node.find_next_sibling()
        return None

    def _parse_scoreboard(self, soup: BeautifulSoup) -> dict:
        home_el = soup.select_one("#team-name-l")
        away_el = soup.select_one("#team-name-r")
        home_team = self.clean_text(home_el.get_text(" ", strip=True) if home_el else "")
        away_team = self.clean_text(away_el.get_text(" ", strip=True) if away_el else "")

        score_table = home_el.find_parent("table") if home_el else None
        scores = []
        if score_table:
            scores = [self.parse_int(td.get_text(" ", strip=True)) for td in score_table.select("td.score")]

        halves: dict[str, tuple[int | None, int | None]] = {}
        if score_table:
            for dl in score_table.select("td.time dl"):
                label_el = dl.find("dt")
                left_el = dl.select_one("dd.left-area")
                right_el = dl.select_one("dd.right-area")
                label = self.clean_text(label_el.get_text(" ", strip=True) if label_el else "")
                halves[label] = (
                    self.parse_int(left_el.get_text(" ", strip=True) if left_el else ""),
                    self.parse_int(right_el.get_text(" ", strip=True) if right_el else ""),
                )

        first = halves.get("前半", (None, None))
        second = halves.get("後半", (None, None))
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_score": scores[0] if len(scores) > 0 else None,
            "away_score": scores[1] if len(scores) > 1 else None,
            "home_first_half_score": first[0],
            "away_first_half_score": first[1],
            "home_second_half_score": second[0],
            "away_second_half_score": second[1],
        }

    def _parse_match_metadata(self, soup: BeautifulSoup) -> dict:
        labels = {"日付", "キックオフ時刻", "スタジアム", "入場者数", "天候", "気温", "湿度"}
        for table in soup.find_all("table"):
            headers = [self.clean_text(th.get_text(" ", strip=True)) for th in table.find_all("th")]
            if not labels.issubset(set(headers)):
                continue
            body_row = table.find("tbody").find("tr") if table.find("tbody") else None
            values = [self.clean_text(td.get_text(" ", strip=True)) for td in body_row.find_all("td")] if body_row else []
            mapped = dict(zip(headers, values))
            return {
                "match_date": mapped.get("日付", ""),
                "kickoff_time": mapped.get("キックオフ時刻", ""),
                "stadium": mapped.get("スタジアム", ""),
                "attendance": self.parse_int(mapped.get("入場者数")),
                "weather": mapped.get("天候", ""),
                "temperature_c": self.parse_float(mapped.get("気温")),
                "humidity_pct": self.parse_float(mapped.get("湿度")),
            }
        return {
            "match_date": "",
            "kickoff_time": "",
            "stadium": "",
            "attendance": None,
            "weather": "",
            "temperature_c": None,
            "humidity_pct": None,
        }

    def _team_columns(self, soup: BeautifulSoup, home_team: str, away_team: str) -> list[tuple[str, list[Tag]]]:
        return [
            (home_team, list(soup.select("div.two-column-table-box-l"))),
            (away_team, list(soup.select("div.two-column-table-box-r"))),
        ]

    def _parse_roster_section(self, team: str, boxes: list[Tag], section_name: str, starter: bool) -> list[dict]:
        heading = next((h for box in boxes for h in box.find_all("h4", class_="two-column-table-st-base") if self.clean_text(h.get_text(" ", strip=True)) == section_name), None)
        if heading is None:
            return []
        table = self._table_after_heading(heading)
        if table is None:
            return []

        rows: list[dict] = []
        for tr in table.find_all("tr"):
            position_el = tr.select_one("td.position")
            number_el = tr.select_one("td.number")
            name_el = tr.select_one("td.name")
            if name_el is None:
                continue
            name = self.clean_text(name_el.get_text(" ", strip=True))
            if not name:
                continue
            rows.append({
                "team": team,
                "player_name": name,
                "player_key": self.player_key(team, name),
                "position": self.clean_text(position_el.get_text(" ", strip=True) if position_el else ""),
                "shirt_number": self.parse_int(number_el.get_text(" ", strip=True) if number_el else ""),
                "starter": bool(starter),
                "bench": not starter,
            })
        return rows

    def _parse_substitutions(self, team: str, boxes: list[Tag]) -> list[dict]:
        heading = next((h for box in boxes for h in box.find_all("h4", class_="two-column-table-st-base") if self.clean_text(h.get_text(" ", strip=True)) == "交代"), None)
        if heading is None:
            return []
        table = self._table_after_heading(heading)
        if table is None:
            return []

        events: list[dict] = []
        pending_out: dict | None = None
        for tr in table.find_all("tr"):
            change_el = tr.select_one("td.change")
            name_el = tr.select_one("td.name")
            time_el = tr.select_one("td.time")
            if change_el is None or name_el is None:
                continue
            marker = self.clean_text(change_el.get_text(" ", strip=True))
            name = self.clean_text(name_el.get_text(" ", strip=True))
            minute = self.parse_minute(time_el.get_text(" ", strip=True) if time_el else "")
            if marker == "▽":
                pending_out = {"player_out": name, "minute": minute}
            elif marker == "▲":
                if pending_out is None:
                    events.append({"team": team, "player_out": "", "player_in": name, "minute": minute})
                else:
                    events.append({
                        "team": team,
                        "player_out": pending_out["player_out"],
                        "player_in": name,
                        "minute": pending_out["minute"] if pending_out["minute"] is not None else minute,
                    })
                    pending_out = None
        if pending_out is not None:
            events.append({"team": team, "player_out": pending_out["player_out"], "player_in": "", "minute": pending_out["minute"]})
        return events

    def _parse_coach(self, boxes: list[Tag]) -> str:
        heading = next((h for box in boxes for h in box.find_all("h4", class_="two-column-table-st-base") if self.clean_text(h.get_text(" ", strip=True)) == "監督"), None)
        if heading is None:
            return ""
        table = self._table_after_heading(heading)
        name_el = table.select_one("td.name") if table else None
        return self.clean_text(name_el.get_text(" ", strip=True) if name_el else "")

    def _parse_goals(self, soup: BeautifulSoup, home_team: str, away_team: str) -> list[dict]:
        goal_header = next((th for th in soup.find_all("th") if self.clean_text(th.get_text(" ", strip=True)) == "得点"), None)
        if goal_header is None:
            return []
        table = goal_header.find_parent("table")
        main_row = table.find("tr", recursive=False) if table else None
        if main_row is None:
            return []

        goals: list[dict] = []
        for side_class, team in (("left-area", home_team), ("right-area", away_team)):
            side = main_row.find("td", class_=side_class, recursive=False)
            if side is None:
                continue
            for tr in side.select("table tr"):
                cells = [self.clean_text(td.get_text(" ", strip=True)) for td in tr.find_all("td", recursive=False)]
                if len(cells) < 2:
                    continue
                if side_class == "left-area":
                    player, minute_text = cells[0], cells[1]
                else:
                    minute_text, player = cells[0], cells[1]
                minute = self.parse_minute(minute_text)
                if player:
                    goals.append({"team": team, "player_name": player, "minute": minute})
        return goals

    def _build_player_history(self, roster: list[dict], substitutions: list[dict]) -> list[dict]:
        by_key = {row["player_key"]: dict(row) for row in roster}
        for row in by_key.values():
            row.update({"entered_minute": 0 if row["starter"] else None, "left_minute": self.regulation_minutes if row["starter"] else None, "minutes": self.regulation_minutes if row["starter"] else 0, "appeared": bool(row["starter"])})

        for event in substitutions:
            minute = event["minute"]
            if minute is None:
                continue
            out_key = self.player_key(event["team"], event["player_out"]) if event["player_out"] else ""
            in_key = self.player_key(event["team"], event["player_in"]) if event["player_in"] else ""
            if out_key in by_key:
                by_key[out_key]["left_minute"] = minute
                by_key[out_key]["minutes"] = max(0, min(self.regulation_minutes, minute) - int(by_key[out_key]["entered_minute"] or 0))
                by_key[out_key]["appeared"] = True
            if in_key in by_key:
                by_key[in_key]["entered_minute"] = minute
                by_key[in_key]["left_minute"] = self.regulation_minutes
                by_key[in_key]["minutes"] = max(0, self.regulation_minutes - min(self.regulation_minutes, minute))
                by_key[in_key]["appeared"] = True

        return list(by_key.values())

    def parse_file(self, html_path: str | Path, match_card_id: str | None = None) -> dict[str, list[dict]]:
        html_path = Path(html_path)
        if not html_path.exists():
            raise FileNotFoundError(html_path)
        if match_card_id is None:
            match = re.search(r"(\d+)", html_path.stem)
            if not match:
                raise ValueError(f"Cannot infer match_card_id from {html_path.name}")
            match_card_id = match.group(1)

        soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        scoreboard = self._parse_scoreboard(soup)
        metadata = self._parse_match_metadata(soup)
        team_columns = self._team_columns(soup, scoreboard["home_team"], scoreboard["away_team"])

        coaches = {team: self._parse_coach(boxes) for team, boxes in team_columns}
        roster: list[dict] = []
        substitutions: list[dict] = []
        for team, boxes in team_columns:
            roster.extend(self._parse_roster_section(team, boxes, "先発", starter=True))
            roster.extend(self._parse_roster_section(team, boxes, "控え", starter=False))
            substitutions.extend(self._parse_substitutions(team, boxes))

        player_history = self._build_player_history(roster, substitutions)
        goals = self._parse_goals(soup, scoreboard["home_team"], scoreboard["away_team"])
        match_row = ParsedMatch(
            match_card_id=str(match_card_id),
            **metadata,
            **scoreboard,
            home_coach=coaches.get(scoreboard["home_team"], ""),
            away_coach=coaches.get(scoreboard["away_team"], ""),
            regulation_minutes=self.regulation_minutes,
            source_html=str(html_path),
        )

        common = {"match_card_id": str(match_card_id)}
        for collection in (player_history, substitutions, goals):
            for row in collection:
                row.update(common)

        return {
            "matches": [asdict(match_row)],
            "player_match_history": player_history,
            "substitutions": substitutions,
            "goals": goals,
        }


def upsert_csv(path: Path, new_rows: list[dict], keys: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_rows)
    if path.exists() and path.stat().st_size > 0:
        old_df = pd.read_csv(path, dtype={key: str for key in keys})
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df
    if not combined.empty:
        combined = combined.drop_duplicates(subset=keys, keep="last")
        combined = combined.sort_values(keys).reset_index(drop=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def discover_html_files(raw_dir: Path, match_ids: Iterable[str]) -> list[Path]:
    html_dir = raw_dir / "html"
    ids = list(match_ids)
    if ids:
        paths = [html_dir / f"match_{match_id}.html" for match_id in ids]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing cached HTML:\n" + "\n".join(missing))
        return paths
    return sorted(html_dir.glob("match_*.html"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse cached J.League SFMS02 HTML files.")
    parser.add_argument("--match-card-id", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Parse all cached match_*.html files.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--regulation-minutes", type=int, default=90)
    args = parser.parse_args()

    if not args.all and not args.match_card_id:
        parser.error("Specify --match-card-id or --all.")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    html_files = discover_html_files(raw_dir, [] if args.all else args.match_card_id)
    if not html_files:
        raise FileNotFoundError(f"No cached HTML files found under {raw_dir / 'html'}")

    match_parser = JLeagueMatchParser(regulation_minutes=args.regulation_minutes)
    aggregated = {"matches": [], "player_match_history": [], "substitutions": [], "goals": []}

    print("=" * 64)
    print("Project Alpha - J.League Parser")
    print("=" * 64)
    for index, html_path in enumerate(html_files, start=1):
        print(f"[{index}/{len(html_files)}] {html_path.name}")
        parsed = match_parser.parse_file(html_path)
        for name in aggregated:
            aggregated[name].extend(parsed[name])

    outputs = {
        "matches": upsert_csv(output_dir / "matches.csv", aggregated["matches"], ["match_card_id"]),
        "player_match_history": upsert_csv(output_dir / "player_match_history.csv", aggregated["player_match_history"], ["match_card_id", "team", "player_name"]),
        "substitutions": upsert_csv(output_dir / "substitutions.csv", aggregated["substitutions"], ["match_card_id", "team", "player_out", "player_in"]),
        "goals": upsert_csv(output_dir / "goals.csv", aggregated["goals"], ["match_card_id", "team", "player_name", "minute"]),
    }

    summary = {name: len(frame) for name, frame in outputs.items()}
    (output_dir / "parser_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nParser Summary")
    print("-" * 40)
    for name, frame in outputs.items():
        print(f"{name:<24}: {len(frame):,}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
