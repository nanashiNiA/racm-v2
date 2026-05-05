"""V5: ETH cache build (SIMPLE - Binance API only, like SOL)
============================================================
Tardis book は file too large → SOL と同じ方式で API のみ
"""
import sys,os,json,urllib.request,time,pickle
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V5: ETH cache (Binance API)',flush=True)
print('='*70,flush=True)

CACHE_DIR='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'

def fetch_klines(symbol, interval='1h', start_ms=None, end_ms=None):
    url_base='https://fapi.binance.com/fapi/v1/klines'
    all_data=[]
    current=start_ms
    pages=0
    while current<end_ms and pages<200:
        url=f'{url_base}?symbol={symbol}&interval={interval}&startTime={current}&limit=1500'
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'V5'})
            with urllib.request.urlopen(req,timeout=15) as r:
                data=json.loads(r.read())
            if not data:break
            all_data.extend(data)
            current=int(data[-1][0])+1
            pages+=1
            time.sleep(0.15)
            if len(data)<1500:break
        except:break
    return all_data

def fetch_funding(symbol):
    url_base='https://fapi.binance.com/fapi/v1/fundingRate'
    all_data=[]
    end_time=int(pd.Timestamp.now().timestamp()*1000)
    current=int(pd.Timestamp('2020-01-01').timestamp()*1000)
    pages=0
    while current<end_time and pages<30:
        url=f'{url_base}?symbol={symbol}&startTime={current}&limit=1000'
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'V5'})
            with urllib.request.urlopen(req,timeout=15) as r:
                data=json.loads(r.read())
            if not data:break
            all_data.extend(data)
            current=int(data[-1]['fundingTime'])+1
            pages+=1
            time.sleep(0.15)
            if len(data)<1000:break
        except:break
    df=pd.DataFrame(all_data)
    df['fundingTime']=pd.to_datetime(df['fundingTime'],unit='ms',utc=True)
    df['fundingRate']=df['fundingRate'].astype(float)
    return df.set_index('fundingTime').sort_index().drop_duplicates()

print('\nFetching ETH funding...',flush=True)
funding=fetch_funding('ETHUSDT')
print(f'  {len(funding)} records, mean: {funding["fundingRate"].mean()*100:.5f}%/8H = {funding["fundingRate"].mean()*3*365*100:+.1f}%/yr',flush=True)

for year in range(2021,2026):
    cache_file=os.path.join(CACHE_DIR,f'ETHUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(cache_file):
        print(f'  {year} cache exists',flush=True)
        continue

    print(f'\nFetching ETH klines {year}...',flush=True)
    start_ms=int(pd.Timestamp(f'{year}-01-01').timestamp()*1000)
    end_ms=int(pd.Timestamp(f'{year}-12-31 23:59:59').timestamp()*1000)
    klines=fetch_klines('ETHUSDT', '1h', start_ms, end_ms)
    if not klines:continue

    df=pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms',utc=True)
    for c in ['open','high','low','close','volume']:
        df[c]=df[c].astype(float)
    df['mid']=(df['open']+df['close'])/2
    df['spread_pct']=0.01
    df['imbalance_0']=0.0
    df['depth_imbalance']=0.0
    df['total_depth']=df['volume']
    df['bid_pressure']=0.5
    df['ask_pressure']=0.5

    ohlc_df=df[['timestamp','open','high','low','close','mid','spread_pct',
                'imbalance_0','depth_imbalance','total_depth','bid_pressure','ask_pressure','volume']]

    funding_year=funding[(funding.index.year==year)].copy()
    funding_year['timestamp']=funding_year.index
    ticker_year=funding_year[['timestamp','fundingRate']].rename(columns={'fundingRate':'funding_rate'})

    print(f'  Saving {len(ohlc_df)} bars',flush=True)
    with open(cache_file,'wb') as fp:
        pickle.dump((ohlc_df, ticker_year, None), fp)

print('\nDone.',flush=True)
