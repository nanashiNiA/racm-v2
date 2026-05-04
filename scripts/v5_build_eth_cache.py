"""V5: Build ETH cache for swin_v25 multi-asset
================================================
ETH データ整備:
1. Tardis book_snapshot_25 ETHUSDT (2021-2025) → 板特徴量
2. Binance API で ETH funding rate
3. Binance API で ETH liquidations
4. swin_v25 互換 pkl 形式で cache 保存
"""
import sys,os,json,urllib.request,time,pickle,gzip
from datetime import date,timedelta
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V5: ETH cache build for swin_v25',flush=True)
print('='*70,flush=True)

CACHE_DIR='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
ETH_BOOK_DIR='Y:/tardis_data/perpetual/ETHUSDT/binance-futures/book_snapshot_25'

# ============================================================
# 1. ETH funding from Binance API
# ============================================================
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
        except Exception as e:
            print(f'  ERROR {symbol} page {pages}: {e}',flush=True)
            break
    df=pd.DataFrame(all_data)
    if len(df):
        df['fundingTime']=pd.to_datetime(df['fundingTime'],unit='ms',utc=True)
        df['fundingRate']=df['fundingRate'].astype(float)
        df=df.set_index('fundingTime').sort_index().drop_duplicates()
    return df

print('\nFetching ETH funding...',flush=True)
funding=fetch_funding('ETHUSDT')
print(f'  Got {len(funding)} funding records ({funding.index.min()} → {funding.index.max()})',flush=True)
print(f'  Mean: {funding["fundingRate"].mean()*100:.5f}%/8H = {funding["fundingRate"].mean()*3*365*100:+.1f}%/yr',flush=True)

# ============================================================
# 2. ETH liquidations from Binance API
# ============================================================
def fetch_liquidations(symbol='ETHUSDT', days_back=365*5):
    """Note: Binance API only provides recent liquidations, no full history"""
    url=f'https://fapi.binance.com/fapi/v1/allForceOrders?symbol={symbol}&limit=1000'
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'V5'})
        with urllib.request.urlopen(req,timeout=15) as r:
            data=json.loads(r.read())
        df=pd.DataFrame(data)
        if len(df)>0:
            df['time']=pd.to_datetime(df['time'],unit='ms',utc=True)
            df['price']=df['price'].astype(float)
            df['origQty']=df['origQty'].astype(float)
            df['liq_usd']=df['price']*df['origQty']
            df['side']=df['side']
        return df.set_index('time').sort_index() if len(df)>0 else pd.DataFrame()
    except Exception as e:
        print(f'  ERROR: {e}',flush=True)
        return pd.DataFrame()

print('\nFetching ETH liquidations (limited)...',flush=True)
liq=fetch_liquidations('ETHUSDT')
print(f'  Got {len(liq)} liquidation records (recent only)',flush=True)

# ============================================================
# 3. ETH book_snapshot_25 → 1H aggregated OHLC + features
# ============================================================
def load_book_snapshot_day(file_path, levels=5):
    try:
        cols=['timestamp']
        for i in range(levels):
            cols.extend([f'asks[{i}].price',f'asks[{i}].amount',
                        f'bids[{i}].price',f'bids[{i}].amount'])
        df=pd.read_csv(file_path,compression='gzip',usecols=cols)
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
        df['mid']=(df['asks[0].price']+df['bids[0].price'])/2
        df['spread_pct']=(df['asks[0].price']-df['bids[0].price'])/(df['mid']+1e-10)*100
        # Imbalance
        bid_vol_0=df['bids[0].amount']
        ask_vol_0=df['asks[0].amount']
        df['imbalance_0']=(bid_vol_0-ask_vol_0)/(bid_vol_0+ask_vol_0+1e-10)
        # Depth
        bid_depth=sum(df[f'bids[{i}].amount'] for i in range(levels))
        ask_depth=sum(df[f'asks[{i}].amount'] for i in range(levels))
        df['depth_imbalance']=(bid_depth-ask_depth)/(bid_depth+ask_depth+1e-10)
        df['total_depth']=bid_depth+ask_depth
        df['bid_pressure']=bid_vol_0/(df['total_depth']+1e-10)
        df['ask_pressure']=ask_vol_0/(df['total_depth']+1e-10)
        return df[['timestamp','mid','spread_pct','imbalance_0','depth_imbalance',
                   'total_depth','bid_pressure','ask_pressure']]
    except Exception as e:
        return None

def process_year(year):
    print(f'\n  Processing year {year}...',flush=True)
    cache_file=os.path.join(CACHE_DIR,f'ETHUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(cache_file):
        print(f'    Cache exists: {cache_file}',flush=True)
        return True

    all_ohlc=[]
    days_done=0
    for month in range(1,13):
        month_dir=os.path.join(ETH_BOOK_DIR,str(year),f'{month:02d}')
        if not os.path.exists(month_dir):continue
        files=sorted([f for f in os.listdir(month_dir) if f.endswith('.csv.gz')])
        for f in files:
            file_path=os.path.join(month_dir,f)
            df=load_book_snapshot_day(file_path)
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
            if days_done%30==0:print(f'    {days_done} days',flush=True)

    if not all_ohlc:
        print(f'    No data for {year}',flush=True)
        return False

    ohlc_all=pd.concat(all_ohlc,ignore_index=True).sort_values('timestamp')
    ohlc_all=ohlc_all.drop_duplicates('timestamp').reset_index(drop=True)

    # Map funding to 1H
    ticker_year=funding.copy()
    ticker_year['timestamp']=ticker_year.index
    ticker_year=ticker_year[['timestamp','fundingRate']].rename(columns={'fundingRate':'funding_rate'})

    # Liquidations: dummy (not enough historical)
    liq_year=None

    # Save in same format as BTC cache
    print(f'    Saving {len(ohlc_all)} bars to {cache_file}',flush=True)
    with open(cache_file,'wb') as fp:
        pickle.dump((ohlc_all, ticker_year, liq_year), fp)
    print(f'    OK',flush=True)
    return True

# Process each year
for year in range(2021,2026):
    process_year(year)

print('\nDone.',flush=True)
