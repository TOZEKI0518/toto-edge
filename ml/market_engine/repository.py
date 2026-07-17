from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from collections.abc import Sequence
import pandas as pd
from market_engine.models import MarketMatch
class MarketRepository:
    def __init__(self,historical_csv:Path,current_csv:Path)->None:self.historical_csv=historical_csv;self.current_csv=current_csv
    @staticmethod
    def frame(matches:Sequence[MarketMatch])->pd.DataFrame:return pd.DataFrame([asdict(x) for x in matches])
    def save_current(self,matches:Sequence[MarketMatch])->pd.DataFrame:
        f=self.frame(matches);self.current_csv.parent.mkdir(parents=True,exist_ok=True);f.to_csv(self.current_csv,index=False,encoding='utf-8-sig');return f
    def append_history(self,matches:Sequence[MarketMatch])->pd.DataFrame:
        inc=self.frame(matches);self.historical_csv.parent.mkdir(parents=True,exist_ok=True)
        old=pd.read_csv(self.historical_csv) if self.historical_csv.exists() else pd.DataFrame()
        out=pd.concat([old,inc],ignore_index=True).drop_duplicates(['round_id','toto_match_no','collected_at'],keep='last').sort_values(['round_id','toto_match_no','collected_at'])
        out.to_csv(self.historical_csv,index=False,encoding='utf-8-sig');return out
