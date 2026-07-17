from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class MarketMatch:
    round_id:int; toto_match_no:int; home_team:str; away_team:str
    market_prob_home:float; market_prob_draw:float; market_prob_away:float
    match_day:str=""; collected_at:str=""; source_url:str=""

@dataclass(frozen=True, slots=True)
class MarketEngineConfig:
    output_dir:Path; historical_csv:Path; current_market_csv:Path
    current_features_csv:Path; diagnostics_dir:Path; round_id:int|None=None
    timeout_seconds:float=30.0; retries:int=3; retry_wait_seconds:float=2.0
    verify_ssl:bool=True; save_html:bool=True
    def validate(self)->None:
        if self.round_id is not None and self.round_id<=0: raise ValueError('round_id must be positive')
        if self.timeout_seconds<=0 or self.retries<=0: raise ValueError('invalid retry settings')
