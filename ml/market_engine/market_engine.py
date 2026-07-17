from __future__ import annotations
import argparse,json,logging,sys
from pathlib import Path
if __package__ in (None,''):
    root=Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
from market_engine.collector import TotoMarketCollector
from market_engine.feature_builder import MarketFeatureBuilder
from market_engine.models import MarketEngineConfig
from market_engine.parser import TotoMarketParser
from market_engine.repository import MarketRepository

def config_default()->MarketEngineConfig:
    ml=Path(__file__).resolve().parents[1];out=ml/'market_data';diag=ml/'diagnostics'/'market_engine'
    return MarketEngineConfig(out,out/'historical_market.csv',out/'current_toto_market.csv',out/'current_market_features.csv',diag)
def run(c:MarketEngineConfig)->dict[str,object]:
    c.validate();c.output_dir.mkdir(parents=True,exist_ok=True);c.diagnostics_dir.mkdir(parents=True,exist_ok=True)
    col=TotoMarketCollector(c);rid=c.round_id or col.discover_round_id();html,url=col.fetch_round(rid);matches=TotoMarketParser().parse(html,rid,url)
    repo=MarketRepository(c.historical_csv,c.current_market_csv);cur=repo.save_current(matches);hist=repo.append_history(matches);feat=MarketFeatureBuilder().build_all(matches);feat.to_csv(c.current_features_csv,index=False,encoding='utf-8-sig')
    cur.to_csv(c.output_dir/f'toto_market_round_{rid}.csv',index=False,encoding='utf-8-sig')
    if c.save_html:(c.output_dir/f'toto_market_round_{rid}.html').write_text(html,encoding='utf-8')
    probs=['market_prob_home','market_prob_draw','market_prob_away'];s=cur[probs].sum(axis=1);summary={'round_id':rid,'rows':len(cur),'feature_columns':len(feat.columns),'missing_team_names':int((cur.home_team.fillna('').eq('')|cur.away_team.fillna('').eq('')).sum()),'duplicate_match_numbers':int(cur.toto_match_no.duplicated().sum()),'missing_probability_values':int(cur[probs].isna().sum().sum()),'probability_sum_min':float(s.min()),'probability_sum_max':float(s.max()),'historical_rows':len(hist),'current_market_csv':str(c.current_market_csv),'current_features_csv':str(c.current_features_csv),'historical_csv':str(c.historical_csv),'source_url':url}
    (c.diagnostics_dir/f'market_engine_round_{rid}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');return summary
def main()->None:
    d=config_default();p=argparse.ArgumentParser();p.add_argument('--round-id',type=int);p.add_argument('--output-dir',type=Path,default=d.output_dir);p.add_argument('--timeout',type=float,default=30);p.add_argument('--retries',type=int,default=3);p.add_argument('--no-save-html',action='store_true');p.add_argument('--no-verify-ssl',action='store_true');p.add_argument('--verbose',action='store_true');a=p.parse_args();logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,format='%(asctime)s | %(levelname)s | %(name)s | %(message)s');diag=Path(__file__).resolve().parents[1]/'diagnostics'/'market_engine';c=MarketEngineConfig(a.output_dir,a.output_dir/'historical_market.csv',a.output_dir/'current_toto_market.csv',a.output_dir/'current_market_features.csv',diag,a.round_id,a.timeout,a.retries,2.0,not a.no_verify_ssl,not a.no_save_html);s=run(c)
    print('='*88);print('Project Alpha Market Engine');print('='*88)
    for k in ('round_id','rows','feature_columns','missing_team_names','duplicate_match_numbers','missing_probability_values','probability_sum_min','probability_sum_max','historical_rows','current_market_csv','current_features_csv','historical_csv'):print(f'{k:28}: {s[k]}')
if __name__=='__main__':main()
