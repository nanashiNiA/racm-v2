"""V5: swin_v25 SOL test (using cached data)
==============================================
SOL は book features dummy なので price/funding/OI ベースの alpha のみ
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V5: swin_v25 SOL test',flush=True)
print('='*70,flush=True)

cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'

from src import swin_v25

ASSET='SOLUSDT'

def custom_load(data_dir, start_date, end_date, symbol="BTCUSDT", freq='30min'):
    print(f'    Custom loader: {ASSET} cache from {start_date} to {end_date}',flush=True)
    all_ohlc=[];all_ticker=[]
    for year in range(start_date.year, end_date.year+1):
        cache_file=os.path.join(cache_dir,f'{ASSET}_{year}0101_{year}1231_1H.pkl')
        if os.path.exists(cache_file):
            with open(cache_file,'rb') as f:
                ohlc,ticker,liq=pickle.load(f)
            if ohlc is not None:all_ohlc.append(ohlc)
            if ticker is not None:all_ticker.append(ticker)

    if not all_ohlc:return None,None,None
    ohlc_df=pd.concat(all_ohlc,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    ticker_df=pd.concat(all_ticker,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp') if all_ticker else None

    if 'volume' not in ohlc_df.columns:ohlc_df['volume']=1.0
    print(f'    Loaded: ohlc={len(ohlc_df)}, ticker={len(ticker_df) if ticker_df is not None else 0}',flush=True)
    return ohlc_df, ticker_df, None

swin_v25.load_all_data_chunked = custom_load

# Test on multiple years
results={}
for period in ['2022','2024','2025']:
    print(f'\n{"="*70}')
    print(f'  swin_v25 {ASSET} OOS test: {period}')
    print(f'{"="*70}')
    try:
        result = swin_v25.run_single_config('IGNORED','1H',period,symbol=ASSET,use_v25_engine=False)
        if result:
            results[period]=result
            print(f'\n  Net: {result["net_return"]:+.1f}%, B&H: {result["bnh_return"]:+.1f}%, vs B&H: {result["vs_bnh"]:+.1f}%')
            print(f'  Trades: {result["trades"]}, WR: {result["win_rate"]:.1f}%, MDD: {result["max_dd"]:.1f}%')
    except Exception as e:
        print(f'  ERROR: {e}',flush=True)
        import traceback;traceback.print_exc()

# Summary
print(f'\n{"="*70}')
print(f'  SOL swin_v25 SUMMARY')
print(f'{"="*70}')
for p,r in results.items():
    print(f'  {p}: Net {r["net_return"]:+.1f}%, B&H {r["bnh_return"]:+.1f}%')

if len(results)==3:
    cagr_factor=1.0
    for r in results.values():cagr_factor *= (1+r['net_return']/100)
    cagr=cagr_factor**(1/len(results))-1
    print(f'\n  3-year compound: {(cagr_factor-1)*100:+.1f}%')
    print(f'  CAGR: {cagr*100:+.1f}%/yr')

print('\nDone.',flush=True)
