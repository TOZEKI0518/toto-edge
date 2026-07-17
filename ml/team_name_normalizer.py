from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_ALIAS_MAP: Final[dict[str, str]] = {
    "札幌": "北海道コンサドーレ札幌",
    "仙台": "ベガルタ仙台",
    "秋田": "ブラウブリッツ秋田",
    "山形": "モンテディオ山形",
    "いわき": "いわきFC",
    "水戸": "水戸ホーリーホック",
    "栃木": "栃木SC",
    "群馬": "ザスパ群馬",
    "大宮": "RB大宮アルディージャ",
    "千葉": "ジェフユナイテッド千葉",
    "柏": "柏レイソル",
    "浦和": "浦和レッズ",
    "FC東京": "FC東京",
    "東京V": "東京ヴェルディ",
    "東京Ｖ": "東京ヴェルディ",
    "町田": "FC町田ゼルビア",
    "川崎F": "川崎フロンターレ",
    "川崎Ｆ": "川崎フロンターレ",
    "横浜FM": "横浜F・マリノス",
    "横浜ＦＭ": "横浜F・マリノス",
    "横浜FC": "横浜FC",
    "横浜ＦＣ": "横浜FC",
    "湘南": "湘南ベルマーレ",
    "相模原": "SC相模原",
    "甲府": "ヴァンフォーレ甲府",
    "長野": "AC長野パルセイロ",
    "松本": "松本山雅FC",
    "新潟": "アルビレックス新潟",
    "富山": "カターレ富山",
    "金沢": "ツエーゲン金沢",
    "清水": "清水エスパルス",
    "磐田": "ジュビロ磐田",
    "藤枝": "藤枝MYFC",
    "沼津": "アスルクラロ沼津",
    "名古屋": "名古屋グランパス",
    "岐阜": "FC岐阜",
    "京都": "京都サンガF.C.",
    "G大阪": "ガンバ大阪",
    "Ｇ大阪": "ガンバ大阪",
    "C大阪": "セレッソ大阪",
    "Ｃ大阪": "セレッソ大阪",
    "奈良": "奈良クラブ",
    "神戸": "ヴィッセル神戸",
    "岡山": "ファジアーノ岡山",
    "広島": "サンフレッチェ広島",
    "鳥取": "ガイナーレ鳥取",
    "山口": "レノファ山口FC",
    "讃岐": "カマタマーレ讃岐",
    "徳島": "徳島ヴォルティス",
    "愛媛": "愛媛FC",
    "今治": "FC今治",
    "福岡": "アビスパ福岡",
    "北九州": "ギラヴァンツ北九州",
    "鳥栖": "サガン鳥栖",
    "長崎": "V・ファーレン長崎",
    "熊本": "ロアッソ熊本",
    "大分": "大分トリニータ",
    "宮崎": "テゲバジャーロ宮崎",
    "鹿児島": "鹿児島ユナイテッドFC",
    "琉球": "FC琉球",
    "八戸": "ヴァンラーレ八戸",
    "岩手": "いわてグルージャ盛岡",
    "福島": "福島ユナイテッドFC",
    "YS横浜": "Y.S.C.C.横浜",
    "YSCC横浜": "Y.S.C.C.横浜",
    "鹿島": "鹿島アントラーズ",
}


@dataclass(frozen=True, slots=True)
class NormalizerConfig:
    """Configuration for market team-name normalization."""

    input_csv: Path
    output_csv: Path
    reference_matches_csv: Path
    diagnostics_csv: Path

    def validate(self) -> None:
        for label, path in (
            ("input_csv", self.input_csv),
            ("reference_matches_csv", self.reference_matches_csv),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")


class TeamNameNormalizer:
    """Normalize official toto abbreviations to Project Alpha team names."""

    def __init__(
        self,
        canonical_names: set[str],
        alias_map: dict[str, str] | None = None,
    ) -> None:
        self.canonical_names = canonical_names
        self.alias_map = dict(DEFAULT_ALIAS_MAP)
        if alias_map:
            self.alias_map.update(alias_map)

        self.normalized_canonical = {
            self._normalize_key(name): name
            for name in canonical_names
        }

    def normalize(self, value: str) -> tuple[str, str]:
        """Return canonical name and resolution method."""
        raw = str(value).strip()
        if not raw:
            return "", "missing"

        if raw in self.canonical_names:
            return raw, "exact"

        if raw in self.alias_map:
            candidate = self.alias_map[raw]
            if candidate in self.canonical_names:
                return candidate, "alias"

        key = self._normalize_key(raw)
        if key in self.normalized_canonical:
            return self.normalized_canonical[key], "normalized_exact"

        alias_key_map = {
            self._normalize_key(alias): canonical
            for alias, canonical in self.alias_map.items()
        }
        if key in alias_key_map:
            candidate = alias_key_map[key]
            if candidate in self.canonical_names:
                return candidate, "normalized_alias"

        close = get_close_matches(
            key,
            list(self.normalized_canonical.keys()),
            n=1,
            cutoff=0.72,
        )
        if close:
            return self.normalized_canonical[close[0]], "fuzzy"

        return raw, "unresolved"

    @staticmethod
    def _normalize_key(value: str) -> str:
        text = unicodedata.normalize("NFKC", value)
        text = text.replace("・", "")
        text = text.replace(".", "")
        text = text.replace("．", "")
        text = text.replace("-", "")
        text = text.replace("－", "")
        text = text.replace(" ", "")
        text = text.replace("　", "")
        text = re.sub(r"\s+", "", text)
        return text.casefold()


def load_canonical_names(reference_csv: Path) -> set[str]:
    """Load canonical team names from processed matches.csv."""
    frame = pd.read_csv(
        reference_csv,
        usecols=["home_team", "away_team"],
    )
    names = set(
        frame["home_team"].dropna().astype(str).str.strip()
    )
    names.update(
        frame["away_team"].dropna().astype(str).str.strip()
    )
    if not names:
        raise ValueError("No canonical team names found.")
    return names


def normalize_market_file(config: NormalizerConfig) -> pd.DataFrame:
    """Normalize home/away team names in a market CSV."""
    config.validate()
    market = pd.read_csv(config.input_csv)

    required = {"home_team", "away_team"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(
            "Market CSV missing required columns: "
            + ", ".join(missing)
        )

    canonical = load_canonical_names(config.reference_matches_csv)
    normalizer = TeamNameNormalizer(canonical)

    diagnostics: list[dict[str, str]] = []

    for column in ("home_team", "away_team"):
        normalized_values: list[str] = []
        methods: list[str] = []

        for raw in market[column].fillna("").astype(str):
            normalized, method = normalizer.normalize(raw)
            normalized_values.append(normalized)
            methods.append(method)
            diagnostics.append(
                {
                    "column": column,
                    "original_name": raw,
                    "normalized_name": normalized,
                    "method": method,
                }
            )

        market[column] = normalized_values
        market[f"{column}_normalization_method"] = methods

    unresolved = [
        row
        for row in diagnostics
        if row["method"] in {"missing", "unresolved"}
    ]
    if unresolved:
        LOGGER.warning(
            "Unresolved team-name entries: %d",
            len(unresolved),
        )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    config.diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)

    market.to_csv(
        config.output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(diagnostics).to_csv(
        config.diagnostics_csv,
        index=False,
        encoding="utf-8-sig",
    )

    return market


def main() -> None:
    """Normalize the current official toto market CSV."""
    ml_root = Path(__file__).resolve().parent

    config = NormalizerConfig(
        input_csv=ml_root / "market_data" / "current_toto_market.csv",
        output_csv=(
            ml_root
            / "market_data"
            / "current_toto_market_normalized.csv"
        ),
        reference_matches_csv=(
            ml_root
            / "player_engine"
            / "data"
            / "processed"
            / "matches.csv"
        ),
        diagnostics_csv=(
            ml_root
            / "diagnostics"
            / "market_engine"
            / "team_name_normalization.csv"
        ),
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output = normalize_market_file(config)

    unresolved_count = int(
        (
            output["home_team_normalization_method"].isin(
                ["missing", "unresolved"]
            )
            | output["away_team_normalization_method"].isin(
                ["missing", "unresolved"]
            )
        ).sum()
    )

    print("=" * 84)
    print("Project Alpha Team Name Normalizer")
    print("=" * 84)
    print(f"Rows                  : {len(output)}")
    print(f"Unresolved rows       : {unresolved_count}")
    print(f"Output                : {config.output_csv}")
    print(f"Diagnostics           : {config.diagnostics_csv}")


if __name__ == "__main__":
    main()
