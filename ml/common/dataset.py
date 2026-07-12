from pathlib import Path
import pandas as pd

DATA_PATH = Path("toto-training-dataset.csv")

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["result"]).copy()
    df = df.sort_values("roundNo").reset_index(drop=True)
    return df