from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://data.j-league.or.jp"
MATCH_DETAIL_PATH = "/SFMS02/"

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_REQUEST_INTERVAL_SECONDS = 2.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0 Safari/537.36 "
    "Toto-Quant-Project-Alpha/1.0"
)


@dataclass(frozen=True)
class CollectedPage:
    """
    取得した1ページの管理情報。
    """

    source_type: str
    source_id: str
    url: str
    fetched_at_utc: str
    status_code: int
    encoding: str
    content_length: int
    sha256: str
    html_path: str
    metadata_path: str
    from_cache: bool


class JLeagueCollector:
    """
    Jリーグ公式データサイトのHTMLを取得し、
    rawデータとしてキャッシュ保存するCollector。

    このクラスではHTML解析を行わない。
    HTML → 選手・交代・カード等への変換は
    後続のparser.pyで実装する。
    """

    def __init__(
        self,
        raw_data_dir: str | Path | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        request_interval_seconds: float = (
            DEFAULT_REQUEST_INTERVAL_SECONDS
        ),
    ) -> None:
        if raw_data_dir is None:
            raw_data_dir = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "raw"
            )

        self.raw_data_dir = Path(raw_data_dir)
        self.timeout_seconds = timeout_seconds
        self.request_interval_seconds = (
            request_interval_seconds
        )

        self.match_html_dir = (
            self.raw_data_dir
            / "matches"
            / "html"
        )

        self.match_metadata_dir = (
            self.raw_data_dir
            / "matches"
            / "metadata"
        )

        self.match_html_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.match_metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._last_request_time: float | None = None

    @staticmethod
    def build_match_url(
        match_card_id: int | str,
    ) -> str:
        match_card_id_text = str(
            match_card_id
        ).strip()

        if not match_card_id_text.isdigit():
            raise ValueError(
                "match_card_id must contain "
                f"digits only: {match_card_id}"
            )

        return (
            f"{BASE_URL}"
            f"{MATCH_DETAIL_PATH}"
            f"?match_card_id={match_card_id_text}"
        )

    def _match_html_path(
        self,
        match_card_id: int | str,
    ) -> Path:
        return (
            self.match_html_dir
            / f"match_{match_card_id}.html"
        )

    def _match_metadata_path(
        self,
        match_card_id: int | str,
    ) -> Path:
        return (
            self.match_metadata_dir
            / f"match_{match_card_id}.json"
        )

    def _wait_if_needed(self) -> None:
        """
        公式サイトへ短時間に連続アクセスしないための待機。
        """

        if self._last_request_time is None:
            return

        elapsed = (
            time.monotonic()
            - self._last_request_time
        )

        remaining = (
            self.request_interval_seconds
            - elapsed
        )

        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _detect_encoding(
        response_headers,
    ) -> str:
        content_type = response_headers.get(
            "Content-Type",
            "",
        )

        lower_content_type = (
            content_type.lower()
        )

        if "charset=" in lower_content_type:
            charset = (
                lower_content_type
                .split("charset=", 1)[1]
                .split(";", 1)[0]
                .strip()
                .strip('"')
                .strip("'")
            )

            if charset:
                return charset

        # JリーグData Siteは通常UTF-8。
        return "utf-8"

    @staticmethod
    def _calculate_sha256(
        content: bytes,
    ) -> str:
        return hashlib.sha256(
            content
        ).hexdigest()

    def _read_cached_result(
        self,
        match_card_id: int | str,
    ) -> CollectedPage | None:
        html_path = self._match_html_path(
            match_card_id
        )

        metadata_path = (
            self._match_metadata_path(
                match_card_id
            )
        )

        if (
            not html_path.exists()
            or not metadata_path.exists()
        ):
            return None

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        metadata["from_cache"] = True

        return CollectedPage(
            **metadata
        )

    def collect_match(
        self,
        match_card_id: int | str,
        force: bool = False,
    ) -> CollectedPage:
        """
        試合詳細ページを1件取得する。

        force=False:
            キャッシュが存在すれば再取得しない。

        force=True:
            キャッシュがあっても再取得する。
        """

        match_card_id_text = str(
            match_card_id
        ).strip()

        if not force:
            cached = self._read_cached_result(
                match_card_id_text
            )

            if cached is not None:
                print(
                    "[CACHE] "
                    f"match_card_id="
                    f"{match_card_id_text}"
                )

                return cached

        url = self.build_match_url(
            match_card_id_text
        )

        self._wait_if_needed()

        request = Request(
            url=url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": (
                    "text/html,"
                    "application/xhtml+xml,"
                    "application/xml;q=0.9,"
                    "*/*;q=0.8"
                ),
                "Accept-Language": (
                    "ja,en-US;q=0.8,en;q=0.6"
                ),
                "Cache-Control": "no-cache",
            },
            method="GET",
        )

        print(
            "[GET] "
            f"match_card_id="
            f"{match_card_id_text}"
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                self._last_request_time = (
                    time.monotonic()
                )

                status_code = int(
                    response.status
                )

                encoding = (
                    self._detect_encoding(
                        response.headers
                    )
                )

                content = response.read()

        except HTTPError as error:
            raise RuntimeError(
                "J.League request failed: "
                f"HTTP {error.code} "
                f"for {url}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                "J.League connection failed: "
                f"{error.reason}"
            ) from error

        except TimeoutError as error:
            raise RuntimeError(
                "J.League request timed out: "
                f"{url}"
            ) from error

        if status_code != 200:
            raise RuntimeError(
                "Unexpected HTTP status: "
                f"{status_code}"
            )

        if not content:
            raise RuntimeError(
                "Downloaded HTML is empty."
            )

        html_path = self._match_html_path(
            match_card_id_text
        )

        metadata_path = (
            self._match_metadata_path(
                match_card_id_text
            )
        )

        html_path.write_bytes(content)

        fetched_at_utc = (
            datetime.now(timezone.utc)
            .isoformat()
        )

        metadata = CollectedPage(
            source_type="match_detail",
            source_id=match_card_id_text,
            url=url,
            fetched_at_utc=fetched_at_utc,
            status_code=status_code,
            encoding=encoding,
            content_length=len(content),
            sha256=self._calculate_sha256(
                content
            ),
            html_path=str(html_path),
            metadata_path=str(
                metadata_path
            ),
            from_cache=False,
        )

        metadata_dict = asdict(metadata)

        # キャッシュ保存時は常にFalse。
        # 読み込み時だけTrueへ変更する。
        metadata_path.write_text(
            json.dumps(
                metadata_dict,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "[SAVED] "
            f"{html_path}"
        )

        return metadata

    def collect_matches(
        self,
        match_card_ids: Iterable[
            int | str
        ],
        force: bool = False,
    ) -> list[CollectedPage]:
        results: list[CollectedPage] = []

        normalized_ids = []

        for match_card_id in match_card_ids:
            match_card_id_text = str(
                match_card_id
            ).strip()

            if not match_card_id_text:
                continue

            normalized_ids.append(
                match_card_id_text
            )

        if not normalized_ids:
            raise ValueError(
                "No match_card_ids were supplied."
            )

        total = len(normalized_ids)

        for index, match_card_id in enumerate(
            normalized_ids,
            start=1,
        ):
            print(
                f"\n[{index}/{total}] "
                f"Collecting match "
                f"{match_card_id}"
            )

            result = self.collect_match(
                match_card_id=match_card_id,
                force=force,
            )

            results.append(result)

        return results


def parse_match_ids(
    values: list[str],
) -> list[str]:
    """
    次の両方を受け付ける。

    --match-card-id 24973

    --match-card-ids 24973 25229 26054

    カンマ区切りも使用可能。
    """

    match_ids: list[str] = []

    for value in values:
        pieces = value.split(",")

        for piece in pieces:
            match_id = piece.strip()

            if not match_id:
                continue

            if not match_id.isdigit():
                raise ValueError(
                    "Invalid match_card_id: "
                    f"{match_id}"
                )

            match_ids.append(match_id)

    # 順番を維持したまま重複削除
    return list(
        dict.fromkeys(match_ids)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw match-detail HTML "
            "from the official J.League "
            "Data Site."
        )
    )

    parser.add_argument(
        "--match-card-id",
        action="append",
        default=[],
        help=(
            "Single match_card_id. "
            "This option may be repeated."
        ),
    )

    parser.add_argument(
        "--match-card-ids",
        nargs="*",
        default=[],
        help=(
            "Multiple match_card_ids "
            "separated by spaces or commas."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Download again even when "
            "cached files already exist."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=(
            DEFAULT_TIMEOUT_SECONDS
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=(
            DEFAULT_REQUEST_INTERVAL_SECONDS
        ),
        help=(
            "Minimum seconds between "
            "requests."
        ),
    )

    parser.add_argument(
        "--raw-data-dir",
        default=None,
        help=(
            "Optional raw-data directory."
        ),
    )

    args = parser.parse_args()

    supplied_values = (
        args.match_card_id
        + args.match_card_ids
    )

    match_ids = parse_match_ids(
        supplied_values
    )

    if not match_ids:
        parser.error(
            "Specify --match-card-id "
            "or --match-card-ids."
        )

    collector = JLeagueCollector(
        raw_data_dir=args.raw_data_dir,
        timeout_seconds=args.timeout,
        request_interval_seconds=(
            args.interval
        ),
    )

    print("=" * 64)
    print("Project Alpha - J.League Collector")
    print("=" * 64)

    print(
        f"Matches       : "
        f"{len(match_ids)}"
    )

    print(
        f"Force         : "
        f"{args.force}"
    )

    print(
        f"Interval      : "
        f"{args.interval:.1f} sec"
    )

    print(
        f"Raw Data Dir  : "
        f"{collector.raw_data_dir}"
    )

    results = collector.collect_matches(
        match_card_ids=match_ids,
        force=args.force,
    )

    print()
    print("=" * 64)
    print("Collector Summary")
    print("=" * 64)

    downloaded = sum(
        not result.from_cache
        for result in results
    )

    cached = sum(
        result.from_cache
        for result in results
    )

    print(
        f"Downloaded    : "
        f"{downloaded}"
    )

    print(
        f"From Cache    : "
        f"{cached}"
    )

    print(
        f"Total         : "
        f"{len(results)}"
    )

    print()

    for result in results:
        source = (
            "CACHE"
            if result.from_cache
            else "DOWNLOAD"
        )

        print(
            f"{result.source_id:<8} "
            f"{source:<10} "
            f"{result.content_length:>9,} bytes"
        )


if __name__ == "__main__":
    main()