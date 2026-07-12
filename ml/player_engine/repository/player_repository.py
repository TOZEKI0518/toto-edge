from pathlib import Path

import pandas as pd


class PlayerRepository:
    """
    Project Alpha
    Player Repository

    Responsible for loading/saving all player related datasets.

    data/
        players.csv
        matches.csv
        cards.csv
    """

    def __init__(self, data_dir=None):

        if data_dir is None:
            data_dir = (
                Path(__file__).resolve().parents[1]
                / "data"
            )

        self.data_dir = Path(data_dir)

        self.players_file = self.data_dir / "players.csv"
        self.matches_file = self.data_dir / "matches.csv"
        self.cards_file = self.data_dir / "cards.csv"

        self.data_dir.mkdir(parents=True, exist_ok=True)

    ######################################################
    # Players
    ######################################################

    def load_players(self):

        if not self.players_file.exists():
            return pd.DataFrame()

        return pd.read_csv(self.players_file)

    def save_players(self, df):

        df.to_csv(
            self.players_file,
            index=False,
            encoding="utf-8-sig"
        )

    ######################################################
    # Matches
    ######################################################

    def load_matches(self):

        if not self.matches_file.exists():
            return pd.DataFrame()

        return pd.read_csv(self.matches_file)

    def save_matches(self, df):

        df.to_csv(
            self.matches_file,
            index=False,
            encoding="utf-8-sig"
        )

    ######################################################
    # Cards
    ######################################################

    def load_cards(self):

        if not self.cards_file.exists():
            return pd.DataFrame()

        return pd.read_csv(self.cards_file)

    def save_cards(self, df):

        df.to_csv(
            self.cards_file,
            index=False,
            encoding="utf-8-sig"
        )

    ######################################################
    # Summary
    ######################################################

    def summary(self):

        print("=" * 60)
        print("Player Repository")
        print("=" * 60)

        print("Players :", len(self.load_players()))
        print("Matches :", len(self.load_matches()))
        print("Cards   :", len(self.load_cards()))