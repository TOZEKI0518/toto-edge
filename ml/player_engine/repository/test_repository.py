import pandas as pd

from player_repository import PlayerRepository


repo = PlayerRepository()

players = pd.DataFrame([
    {
        "player_id": 1,
        "player_name": "Test Player",
        "team": "Project Alpha FC",
        "position": "FW"
    }
])

repo.save_players(players)

loaded = repo.load_players()

print(loaded)

repo.summary()