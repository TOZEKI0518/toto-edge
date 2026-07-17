from __future__ import annotations
import logging,re,time
from urllib.parse import urlencode
import requests
from bs4 import BeautifulSoup
from market_engine.models import MarketEngineConfig
LOGGER=logging.getLogger(__name__)
BASE='https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501ForwardVotetotoSP.form'
DISCOVERY='https://store.toto-dream.com/dcs/subos/screen/ps01/spsl000/PGSPSL00001InitTotoMulti.form'
class TotoMarketCollector:
    def __init__(self,config:MarketEngineConfig)->None:
        config.validate(); self.config=config; self.session=requests.Session(); self.session.headers.update({'User-Agent':'Mozilla/5.0','Accept-Language':'ja'})
    @staticmethod
    def build_url(round_id:int)->str:
        return BASE+'?'+urlencode({'commodityId':'01','fromId':'SSIN026','gameAssortment':'9','holdCntId':str(round_id)})
    def fetch(self,url:str)->str:
        last=None
        for attempt in range(1,self.config.retries+1):
            try:
                r=self.session.get(url,timeout=self.config.timeout_seconds,verify=self.config.verify_ssl); r.raise_for_status(); r.encoding=r.apparent_encoding or r.encoding
                if not r.text.strip(): raise ValueError('empty response')
                LOGGER.info('Fetched %s (%d bytes)',url,len(r.content)); return r.text
            except Exception as exc:
                last=exc
                if attempt<self.config.retries: time.sleep(self.config.retry_wait_seconds)
        raise RuntimeError('official page fetch failed') from last
    def discover_round_id(self)->int:
        html=self.fetch(DISCOVERY); ids={int(x) for x in re.findall(r'holdCntId(?:=|%3D|&amp;holdCntId=)([0-9]{3,5})',html)}
        text=BeautifulSoup(html,'html.parser').get_text(' ',strip=True); ids.update(int(x) for x in re.findall(r'(?:第\s*)?([0-9]{3,5})\s*回',text))
        if not ids: raise ValueError('round discovery failed; use --round-id')
        return max(ids)
    def fetch_round(self,round_id:int)->tuple[str,str]:
        url=self.build_url(round_id); return self.fetch(url),url
