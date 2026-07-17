from __future__ import annotations
import math
from collections.abc import Sequence
import numpy as np,pandas as pd
from market_engine.models import MarketMatch
class MarketFeatureBuilder:
    def build_all(self,matches:Sequence[MarketMatch])->pd.DataFrame:
        rows=[]
        for m in matches:
            p=np.array([m.market_prob_home,m.market_prob_draw,m.market_prob_away],float);p/=p.sum();order=np.argsort(-p);r=np.empty_like(order);r[order]=np.arange(1,4);ent=-float(np.sum(np.clip(p,1e-15,1)*np.log(np.clip(p,1e-15,1))))/math.log(3)
            rows.append({'round_id':m.round_id,'toto_match_no':m.toto_match_no,'home_team':m.home_team,'away_team':m.away_team,'market_prob_home':p[0],'market_prob_draw':p[1],'market_prob_away':p[2],'market_favorite':('H','D','A')[order[0]],'market_favorite_probability':p[order[0]],'market_second_probability':p[order[1]],'market_probability_margin':p[order[0]]-p[order[1]],'market_entropy':ent,'market_concentration':1-ent,'market_home_rank':int(r[0]),'market_draw_rank':int(r[1]),'market_away_rank':int(r[2]),'market_home_bias':p[0]-1/3,'market_draw_bias':p[1]-1/3,'market_away_bias':p[2]-1/3})
        return pd.DataFrame(rows)
