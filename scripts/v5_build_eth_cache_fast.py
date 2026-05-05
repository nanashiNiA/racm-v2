"""V5: ETH cache build - FAST version
======================================
書 file 1日 100MB+ を全行読むと遅い → 各日 1000行サンプル
精度低下するが時間短縮 (1日数秒で済む)
"""
import sys,os,json,urllib.request,time,pickle,gzip
from datetime import date,timedelta
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V5 FAST: ETH cache build',flush=True)
print('='*70,flush=True)

CACHE_DIR='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
ETH_BOOK_DIR='Y:/tardis_data/perpetual/ETHUSDT/binance-futures/book_snapshot_25'

# Reuse funding from earlier fetch (still valid)
def fetch_funding(symbol='ETHUSDT'):
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
funding=fetch_funding()
print(f'  Got {len(funding)} records',flush=True)

# FAST loader: skip rows for speed
def load_book_fast(file_path):
    """Load only every Nth row for speed"""
    try:
        cols=['timestamp','asks[0].price','asks[0].amount','bids[0].price','bids[0].amount']
        # Sample every 100 rows (=~1 per second from 100/s data)
        df=pd.read_csv(file_path,compression='gzip',usecols=cols,skiprows=lambda i:i%100!=0 and i!=0)
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
        df['mid']=(df['asks[0].price']+df['bids[0].price'])/2
        df['spread_pct']=(df['asks[0].price']-df['bids[0].price'])/(df['mid']+1e-10)*100
        bv=df['bids[0].amount']
        av=df['asks[0].amount']
        df['imbalance_0']=(bv-av)/(bv+av+1e-10)
        df['depth_imbalance']=df['imbalance_0']  # approximate
        df['total_depth']=bv+av
        df['bid_pressure']=bv/(df['total_depth']+1e-10)
        df['ask_pressure']=av/(df['total_depth']+1e-10)
        return df[['timestamp','mid','spread_pct','imbalance_0','depth_imbalance',
                   'total_depth','bid_pressure','ask_pressure']]
    except Exception as e:
        return None

def process_year(year):
    print(f'\n  Processing year {year}...',flush=True)
    cache_file=os.path.join(CACHE_DIR,f'ETHUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(cache_file):
        print(f'    Cache exists',flush=True)
        return True

    all_ohlc=[]
    days_done=0
    t_start=time.time()
    for month in range(1,13):
        month_dir=os.path.join(ETH_BOOK_DIR,str(year),f'{month:02d}')
        if not os.path.exists(month_dir):continue
        files=sorted([f for f in os.listdir(month_dir) if f.endswith('.csv.gz')])
        for f in files:
            file_path=os.path.join(month_dir,f)
            df=load_book_fast(file_path)
            if df is None or len(df)==0:continue
            df=df.set_index('timestamp').sort_index()
            ohlc=df['mid'].resample('1H').agg(open='first',high='max',low='min',close='last')
            features=df.resample('1H').agg({
                'spread_pct':'mean','imbalance_0':'mean','depth_imbalance':'mean',
                'total_depth':'mean','bid_pressure':'mean','ask_pressure':'mean'
            })
            ohlc_df=pd.concat([ohlc,features],axis=1).dropna(subset=['open','close'])
            all_ohlc.append(ohlc_df.reset_index())
            days_done+=1
            if days_done%30==0:
                elapsed=time.time()-t_start
                print(f'    {days_done} days ({elapsed:.0f}s, {days_done/elapsed:.1f}/s)',flush=True)

    if not all_ohlc:
        print(f'    No data for {year}',flush=True)
        return False

    ohlc_all=pd.concat(all_ohlc,ignore_index=True).sort_values('timestamp')
    ohlc_all=ohlc_all.drop_duplicates('timestamp').reset_index(drop=True)
    ticker_year=funding.copy()
    ticker_year['timestamp']=ticker_year.index
    ticker_year=ticker_year[['timestamp','fundingRate']].rename(columns={'fundingRate':'funding_rate'})

    print(f'    Saving {len(ohlc_all)} bars to cache',flush=True)
    with open(cache_file,'wb') as fp:
        pickle.dump((ohlc_all, ticker_year, None), fp)
    print(f'    OK',flush=True)
    return True

for year in range(2021,2026):
    process_year(year)

print('\nDone.',flush=True)
