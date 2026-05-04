"""Build Binance Futures 1H cache - FAST VERSION
Sample every 3rd day to reduce processing time by 3x.
Interpolate missing hours for continuous coverage.
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from datetime import date,timedelta

DATA_DIR='Y:/tardis_data/binance-futures/perpetual'
CACHE_DIR='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
SYMBOL='BTCUSDT'

def load_book_fast(file_path):
    """Load only top-of-book (level 0) for speed."""
    try:
        cols=['timestamp','asks[0].price','asks[0].amount','bids[0].price','bids[0].amount']
        df=pd.read_csv(file_path,compression='gzip',usecols=cols)
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
        df['mid_price']=(df['asks[0].price']+df['bids[0].price'])/2
        df['spread_pct']=(df['asks[0].price']-df['bids[0].price'])/(df['mid_price']+1e-12)*100
        bv=df['bids[0].amount'];av=df['asks[0].amount']
        df['imbalance_0']=(bv-av)/(bv+av+1e-10)
        df['bid_pressure']=bv/(bv+av+1e-10)
        df['ask_pressure']=av/(bv+av+1e-10)
        return df[['timestamp','mid_price','spread_pct','imbalance_0','bid_pressure','ask_pressure']]
    except Exception as ex:
        return None

def load_liq_day(file_path):
    try:
        df=pd.read_csv(file_path,compression='gzip',usecols=['timestamp','side','price','amount'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
        df['liq_usd']=df['price']*df['amount']
        return df
    except:
        return None

def process_year(year, sample_every=1):
    cache_file=os.path.join(CACHE_DIR,f'BINANCE_{SYMBOL}_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(cache_file):
        print(f'  {year}: cache exists, skipping',flush=True)
        return True

    print(f'  {year}: processing (sample every {sample_every} days)...',flush=True)
    all_ohlc=[];all_liq=[]
    start=date(year,1,1);end=date(year,12,31)
    current=start;days_ok=0;day_count=0

    while current<=end:
        day_count+=1
        y,m=current.year,current.month
        ds=current.strftime('%Y-%m-%d')
        folder=os.path.join(DATA_DIR,str(y),f'{m:02d}')

        # Process every Nth day for speed
        if day_count%sample_every==0 or day_count==1:
            book_file=os.path.join(folder,f'binance-futures_perpetual_book_snapshot_25_{ds}_{SYMBOL}.csv.gz')
            if os.path.exists(book_file):
                df=load_book_fast(book_file)
                if df is not None and len(df)>0:
                    df=df.set_index('timestamp').sort_index()
                    ohlc=df['mid_price'].resample('1H').agg(open='first',high='max',low='min',close='last')
                    features=df.resample('1H').agg({
                        'spread_pct':'mean','imbalance_0':'mean',
                        'bid_pressure':'mean','ask_pressure':'mean'
                    })
                    ohlc_df=pd.concat([ohlc,features],axis=1).dropna(subset=['open','close'])
                    all_ohlc.append(ohlc_df.reset_index())
                    days_ok+=1

            liq_file=os.path.join(folder,f'binance-futures_perpetual_liquidations_{ds}_{SYMBOL}.csv.gz')
            if os.path.exists(liq_file):
                df=load_liq_day(liq_file)
                if df is not None and len(df)>0:
                    df=df.set_index('timestamp').sort_index()
                    longs=df[df['side']=='sell']
                    shorts=df[df['side']=='buy']
                    long_agg=longs.resample('1H').agg({'liq_usd':'sum'}).rename(columns={'liq_usd':'long_liq_usd'})
                    short_agg=shorts.resample('1H').agg({'liq_usd':'sum'}).rename(columns={'liq_usd':'short_liq_usd'})
                    liq_count=df.resample('1H').size().to_frame('liq_count')
                    liq_df=pd.concat([long_agg,short_agg,liq_count],axis=1).fillna(0)
                    total_liq=liq_df['long_liq_usd']+liq_df['short_liq_usd']
                    liq_df['liq_imbalance']=np.where(total_liq>0,
                        (liq_df['short_liq_usd']-liq_df['long_liq_usd'])/total_liq,0)
                    all_liq.append(liq_df.reset_index())

        if days_ok>0 and days_ok%50==0:
            print(f'    {days_ok} days processed...',flush=True)
        current+=timedelta(days=1)

    if not all_ohlc:
        print(f'  {year}: NO DATA',flush=True)
        return False

    ohlc_all=pd.concat(all_ohlc,ignore_index=True).sort_values('timestamp')
    # Fill gaps by forward-filling (for sampled days)
    ohlc_all=ohlc_all.drop_duplicates('timestamp').set_index('timestamp')
    full_idx=pd.date_range(ohlc_all.index[0],ohlc_all.index[-1],freq='1H')
    ohlc_all=ohlc_all.reindex(full_idx).ffill().bfill()
    ohlc_all=ohlc_all.reset_index().rename(columns={'index':'timestamp'})

    liq_all=None
    if all_liq:
        liq_all=pd.concat(all_liq,ignore_index=True).sort_values('timestamp')
        liq_all=liq_all.drop_duplicates('timestamp').set_index('timestamp')
        liq_all=liq_all.reindex(full_idx).fillna(0)
        liq_all=liq_all.reset_index().rename(columns={'index':'timestamp'})

    print(f'  {year}: {len(ohlc_all)} bars from {days_ok} days',flush=True)

    with open(cache_file,'wb') as f:
        pickle.dump((ohlc_all,None,liq_all),f)
    print(f'  {year}: saved',flush=True)
    return True

print('='*70,flush=True)
print('  BUILD BINANCE FUTURES 1H CACHE (FAST)',flush=True)
print('='*70,flush=True)

# Process every day but only read level 0 (not all 25 levels)
for year in range(2021,2026):
    process_year(year, sample_every=1)

print('\nDone.',flush=True)
