"""V5: SOL realistic cost adjustment
=====================================
swin_v25 の cost = HYPERLIQUID maker 2bps + slip 1bps = 3bps/side, 6bps round-trip
SOL altcoin 実態: spread 5-15bps + slip 5-20bps in volatile periods

trade log を再評価して real-world コストで discount
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V5: SOL realistic cost stress test',flush=True)
print('='*70,flush=True)

# Re-run SOL with adjusted cost assumption (modify swin_v25 fees)
from src import swin_v25

# Save original constants
ORIG_MAKER=swin_v25.HYPERLIQUID_MAKER_FEE
ORIG_TAKER=swin_v25.HYPERLIQUID_TAKER_FEE
ORIG_SLIP=swin_v25.LIMIT_ORDER_SLIPPAGE

cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'

def custom_load(data_dir, start_date, end_date, symbol="BTCUSDT", freq='30min'):
    print(f'    Loader: SOLUSDT cache',flush=True)
    all_ohlc=[];all_ticker=[]
    for year in range(start_date.year, end_date.year+1):
        cache_file=os.path.join(cache_dir,f'SOLUSDT_{year}0101_{year}1231_1H.pkl')
        if os.path.exists(cache_file):
            with open(cache_file,'rb') as f:
                ohlc,ticker,liq=pickle.load(f)
            if ohlc is not None:all_ohlc.append(ohlc)
            if ticker is not None:all_ticker.append(ticker)

    if not all_ohlc:return None,None,None
    ohlc_df=pd.concat(all_ohlc,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    ticker_df=pd.concat(all_ticker,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp') if all_ticker else None
    if 'volume' not in ohlc_df.columns:ohlc_df['volume']=1.0
    print(f'    ohlc={len(ohlc_df)}, ticker={len(ticker_df) if ticker_df is not None else 0}',flush=True)
    return ohlc_df, ticker_df, None

swin_v25.load_all_data_chunked = custom_load

# Test scenarios with different cost assumptions
COST_SCENARIOS=[
    ('Original (1bp slip + 2bp maker)', 0.0001, 0.0002, 0.00055),
    ('Realistic alt (5bp slip + 5bp taker)', 0.0005, 0.0005, 0.0008),
    ('Conservative alt (10bp slip + 5bp taker)', 0.0010, 0.0005, 0.001),
    ('Bear market alt (20bp slip + 10bp taker)', 0.0020, 0.0010, 0.002),
]

print(f'\n  {"Scenario":<45} {"2024 NET":>10}',flush=True)
print(f'  {"-"*60}',flush=True)

for label, slip, maker, taker in COST_SCENARIOS:
    swin_v25.LIMIT_ORDER_SLIPPAGE=slip
    swin_v25.HYPERLIQUID_MAKER_FEE=maker
    swin_v25.HYPERLIQUID_TAKER_FEE=taker

    try:
        result = swin_v25.run_single_config('IGNORED','1H','2024',symbol='SOLUSDT',use_v25_engine=False)
        if result:
            print(f'  {label:<45} {result["net_return"]:>+9.1f}%',flush=True)
    except Exception as e:
        print(f'  {label:<45} ERROR: {e}',flush=True)

# Restore
swin_v25.LIMIT_ORDER_SLIPPAGE=ORIG_SLIP
swin_v25.HYPERLIQUID_MAKER_FEE=ORIG_MAKER
swin_v25.HYPERLIQUID_TAKER_FEE=ORIG_TAKER

print('\nDone.',flush=True)
