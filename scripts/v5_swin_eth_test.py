"""V5: swin_v25 を ETH に適用 (cache 完成後実行)
==================================================
ETH cache (build_eth_cache.py で生成) を読み込み、
swin_v25 と同等の ML パイプラインで 2024 年 OOS テスト
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V5: swin_v25 ETH 2024 OOS test',flush=True)
print('='*70,flush=True)

# Check ETH cache exists
cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
eth_files=[f for f in os.listdir(cache_dir) if f.startswith('ETHUSDT_')]
print(f'  ETH cache files: {eth_files}',flush=True)

if not eth_files:
    print('  ERROR: No ETH cache, run v5_build_eth_cache.py first',flush=True)
    sys.exit(1)

# Use swin_v25 directly with ETH symbol
# But swin_v25 hardcodes Bybit path - need to override
from src import swin_v25

# Monkey-patch the data loader to load ETH cache instead
ORIG_LOAD=swin_v25.load_all_data_chunked
def custom_load(data_dir, start_date, end_date, symbol="BTCUSDT", freq='30min'):
    """Override to load from local ETH cache"""
    print(f'    Custom loader: ETHUSDT cache from {start_date} to {end_date}',flush=True)
    all_ohlc=[];all_ticker=[];all_liq=[]
    for year in range(start_date.year, end_date.year+1):
        cache_file=os.path.join(cache_dir,f'ETHUSDT_{year}0101_{year}1231_1H.pkl')
        if os.path.exists(cache_file):
            with open(cache_file,'rb') as f:
                ohlc,ticker,liq=pickle.load(f)
            if ohlc is not None:all_ohlc.append(ohlc)
            if ticker is not None:all_ticker.append(ticker)
            if liq is not None:all_liq.append(liq)

    if not all_ohlc:return None,None,None
    ohlc_df=pd.concat(all_ohlc,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    ticker_df=pd.concat(all_ticker,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp') if all_ticker else None
    liq_df=pd.concat(all_liq,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp') if all_liq else None

    # Add 'volume' if missing (swin_v25 expects it)
    if 'volume' not in ohlc_df.columns:
        ohlc_df['volume']=1.0  # dummy

    print(f'    Loaded: ohlc={len(ohlc_df)}, ticker={len(ticker_df) if ticker_df is not None else 0}, liq={len(liq_df) if liq_df is not None else 0}',flush=True)
    return ohlc_df, ticker_df, liq_df

swin_v25.load_all_data_chunked = custom_load

# Run for ETH on 2024
print('\nRunning swin_v25 on ETH 2024...',flush=True)
result = swin_v25.run_single_config('IGNORED','1H','2024',symbol='ETHUSDT',use_v25_engine=False)
if result:
    print(f'\n  ETH Results:')
    print(f'    Net Return: {result["net_return"]:+.1f}%',flush=True)
    print(f'    Trades: {result["trades"]}, WR: {result["win_rate"]:.1f}%',flush=True)
    print(f'    MaxDD: {result["max_dd"]:.1f}%',flush=True)
    print(f'    B&H: {result["bnh_return"]:+.1f}%',flush=True)
    print(f'    vs B&H: {result["vs_bnh"]:+.1f}%',flush=True)

print('\nDone.',flush=True)
