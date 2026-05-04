"""Build 1H cached features from Binance Futures orderbook data
Same structure as Bybit cache in prediction_model_project/src/data_cache/
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from datetime import date,timedelta

DATA_DIR='Y:/tardis_data/binance-futures/perpetual'
CACHE_DIR='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
SYMBOL='BTCUSDT'
FREQ='1H'

def load_book_snapshot_day(file_path, levels=5):
    try:
        cols=['timestamp']
        for i in range(levels):
            cols.extend([f'asks[{i}].price',f'asks[{i}].amount',
                        f'bids[{i}].price',f'bids[{i}].amount'])
        df=pd.read_csv(file_path,compression='gzip',usecols=cols)
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
        df['best_ask']=df['asks[0].price'];df['best_bid']=df['bids[0].price']
        df['mid_price']=(df['best_ask']+df['best_bid'])/2
        df['spread_pct']=(df['best_ask']-df['best_bid'])/(df['mid_price']+1e-12)*100
        df['bid_vol_0']=df['bids[0].amount'];df['ask_vol_0']=df['asks[0].amount']
        df['imbalance_0']=(df['bid_vol_0']-df['ask_vol_0'])/(df['bid_vol_0']+df['ask_vol_0']+1e-10)
        bid_depth=sum(df[f'bids[{i}].amount'] for i in range(levels))
        ask_depth=sum(df[f'asks[{i}].amount'] for i in range(levels))
        df['depth_imbalance']=(bid_depth-ask_depth)/(bid_depth+ask_depth+1e-10)
        df['total_depth']=bid_depth+ask_depth
        df['bid_pressure']=df['bid_vol_0']/(df['total_depth']+1e-10)
        df['ask_pressure']=df['ask_vol_0']/(df['total_depth']+1e-10)
        return df[['timestamp','mid_price','best_bid','best_ask','spread_pct',
                   'imbalance_0','depth_imbalance','total_depth','bid_pressure','ask_pressure']]
    except Exception as ex:
        return None

def load_ticker_day(file_path):
    try:
        df=pd.read_csv(file_path,compression='gzip',usecols=[
            'timestamp','funding_rate','open_interest','last_price','mark_price'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
        return df
    except:
        return None

def load_liq_day(file_path):
    try:
        df=pd.read_csv(file_path,compression='gzip',usecols=['timestamp','side','price','amount'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
        df['liq_usd']=df['price']*df['amount']
        return df
    except:
        return None

def process_year(year):
    cache_file=os.path.join(CACHE_DIR,f'BINANCE_{SYMBOL}_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(cache_file):
        print(f'  {year}: cache exists, skipping',flush=True)
        return True

    print(f'  {year}: processing...',flush=True)
    all_ohlc=[];all_ticker=[];all_liq=[]
    start=date(year,1,1);end=date(year,12,31)
    current=start;days_ok=0;days_fail=0

    while current<=end:
        y,m=current.year,current.month
        ds=current.strftime('%Y-%m-%d')
        folder=os.path.join(DATA_DIR,str(y),f'{m:02d}')

        # Book snapshot
        book_file=os.path.join(folder,f'binance-futures_perpetual_book_snapshot_25_{ds}_{SYMBOL}.csv.gz')
        if os.path.exists(book_file):
            df=load_book_snapshot_day(book_file)
            if df is not None and len(df)>0:
                df=df.set_index('timestamp').sort_index()
                ohlc=df['mid_price'].resample(FREQ).agg(open='first',high='max',low='min',close='last')
                features=df.resample(FREQ).agg({
                    'spread_pct':'mean','imbalance_0':'mean','depth_imbalance':'mean',
                    'total_depth':'mean','bid_pressure':'mean','ask_pressure':'mean'
                })
                ohlc_df=pd.concat([ohlc,features],axis=1).dropna(subset=['open','close'])
                all_ohlc.append(ohlc_df.reset_index())
                days_ok+=1
            else:
                days_fail+=1
        else:
            days_fail+=1

        # Ticker
        ticker_file=os.path.join(folder,f'binance-futures_perpetual_derivative_ticker_{ds}_{SYMBOL}.csv.gz')
        if os.path.exists(ticker_file):
            df=load_ticker_day(ticker_file)
            if df is not None and len(df)>0:
                df=df.set_index('timestamp').sort_index()
                resampled=df.resample(FREQ).agg({
                    'funding_rate':'last','open_interest':'last',
                    'last_price':'last','mark_price':'last'
                }).dropna(how='all')
                all_ticker.append(resampled.reset_index())

        # Liquidations
        liq_file=os.path.join(folder,f'binance-futures_perpetual_liquidations_{ds}_{SYMBOL}.csv.gz')
        if os.path.exists(liq_file):
            df=load_liq_day(liq_file)
            if df is not None and len(df)>0:
                df=df.set_index('timestamp').sort_index()
                longs=df[df['side']=='sell']
                shorts=df[df['side']=='buy']
                long_agg=longs.resample(FREQ).agg({'liq_usd':'sum'}).rename(columns={'liq_usd':'long_liq_usd'})
                short_agg=shorts.resample(FREQ).agg({'liq_usd':'sum'}).rename(columns={'liq_usd':'short_liq_usd'})
                liq_count=df.resample(FREQ).size().to_frame('liq_count')
                liq_df=pd.concat([long_agg,short_agg,liq_count],axis=1).fillna(0)
                total_liq=liq_df['long_liq_usd']+liq_df['short_liq_usd']
                liq_df['liq_imbalance']=np.where(total_liq>0,
                    (liq_df['short_liq_usd']-liq_df['long_liq_usd'])/total_liq,0)
                all_liq.append(liq_df.reset_index())

        if days_ok%30==0 and days_ok>0:
            print(f'    {days_ok} days processed...',flush=True)
        current+=timedelta(days=1)

    if not all_ohlc:
        print(f'  {year}: NO DATA',flush=True)
        return False

    ohlc_all=pd.concat(all_ohlc,ignore_index=True)
    ticker_all=pd.concat(all_ticker,ignore_index=True) if all_ticker else None
    liq_all=pd.concat(all_liq,ignore_index=True) if all_liq else None

    print(f'  {year}: {len(ohlc_all)} bars, {days_ok} days OK, {days_fail} days missing',flush=True)

    with open(cache_file,'wb') as f:
        pickle.dump((ohlc_all,ticker_all,liq_all),f)
    print(f'  {year}: saved to {cache_file}',flush=True)
    return True

print('='*70,flush=True)
print('  BUILD BINANCE FUTURES 1H CACHE',flush=True)
print('='*70,flush=True)

for year in range(2021,2026):
    process_year(year)

print('\nDone.',flush=True)
