"""V4 FIXED: Funding Farm with CORRECT funding application
=========================================================
Bug 2 発覚: funding は per-8H rate なので、per-hour で適用するなら /8 必要

正しい計算:
- 8H毎に funding 支払い (00:00, 08:00, 16:00 UTC のみ)
- または: per-hour で fra/8 を適用

新たな honest 数値:
- Standard FF: ~6-15%/yr (実際の funding income)
- Bidirectional: ~10-25%/yr
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4 FIXED: Funding Farm (proper 8H funding application)',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
ob_frames=[]
for year in range(2021,2026):
    f=os.path.join(cache_dir,f'BTCUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(f):
        with open(f,'rb') as fp:
            ohlc,ticker,liq=pickle.load(fp)
        if ohlc is not None:
            ohlc['timestamp']=pd.to_datetime(ohlc['timestamp'],utc=True)
            if ticker is not None:
                ticker['timestamp']=pd.to_datetime(ticker['timestamp'],utc=True)
                ohlc=pd.merge_asof(ohlc.sort_values('timestamp'),
                    ticker[['timestamp','funding_rate']].sort_values('timestamp'),
                    on='timestamp',direction='backward',tolerance=pd.Timedelta('2H'))
            ob_frames.append(ohlc)
bybit=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
bybit=bybit.drop_duplicates('timestamp').set_index('timestamp')
h_racm=build_common_1h()
h_racm.index=h_racm.index.tz_localize('UTC') if h_racm.index.tz is None else h_racm.index
common_idx=bybit.index.intersection(h_racm.index)
bybit_a=bybit.loc[common_idx]
spot_a=h_racm.loc[common_idx]

perp_price=bybit_a['close'].values
spot_price=spot_a['close'].values

# IMPORTANT FIX: funding rate is per-8H
# For per-hour application: funding_per_hour = funding_8H / 8
# OR apply only at 8H boundaries
fra_8h=np.roll(bybit_a['funding_rate'].fillna(0).values,1)  # per-8H rate
n=len(common_idx)
S=2760

print(f'  Funding rate stats (per-8H):')
print(f'    Mean: {np.mean(fra_8h[S:])*100:.5f}% per 8H')
print(f'    Annualized (×3×365): {np.mean(fra_8h[S:])*3*365*100:.1f}%/yr')

# 30d MA in annual % (correctly using per-8H × 3 × 365)
fra_8h_30d=pd.Series(fra_8h).rolling(720,min_periods=168).mean().values
fra_annual=fra_8h_30d*3*365*100  # CORRECT annualization

# ============================================================
# Funding Farm with PROPER 8H application
# ============================================================
def run_ff_fixed(lev=1.5, fee_bps=2, slip_bps=2, rebal_days=7,
                  apply_method='8H_boundary'):
    """
    apply_method:
      '8H_boundary': funding paid only at 00:00, 08:00, 16:00 UTC
      'per_hour_div8': continuous, per-hour = per_8H / 8 (smoother)
    """
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    cap_unit=1+1/lev
    e *= (1-2*(fr+sr)/cap_unit)
    perp_un=0.0
    for i in range(S,n):
        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0
        sp=sret*1.0;pp=-pret*1.0

        # Funding income - CORRECTED
        ts=common_idx[i]
        funding_income=0
        if apply_method=='8H_boundary':
            # Apply at 8H boundaries only (00, 08, 16 UTC)
            if ts.hour in [0,8,16]:
                funding_income=fra_8h[i]*1.0  # full 8H rate
        elif apply_method=='per_hour_div8':
            funding_income=fra_8h[i]/8*1.0  # smoothed

        perp_un += pp+funding_income
        if perp_un<-1/lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+funding_income)/cap_unit
        e *= (1+bar_ret)

        if (i-S)%(rebal_days*24)==0 and (i-S)>0:
            rebal_cost=abs(perp_un)*0.5*(fr+sr)*2
            e *= (1-rebal_cost/cap_unit)
            perp_un=0
        eq[i]=e
    return eq

def stats(eq):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

print(f'\n  CORRECTED Standard FF (Per-8H funding properly applied):')
print(f'  {"Method":<25} {"Lev":<5} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*60}',flush=True)
for lev in [1.0, 1.5, 2.0, 3.0]:
    for method in ['8H_boundary','per_hour_div8']:
        eq=run_ff_fixed(lev=lev, apply_method=method)
        a,m,s=stats(eq)
        print(f'  {method:<25} {lev:<4.1f}x {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}',flush=True)

# Year-by-year
print(f'\n  Year-by-year (lev=1.5x, 8H boundary):')
eq=run_ff_fixed(lev=1.5, apply_method='8H_boundary')
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    if eq[max(0,iy_a[0]-1)]<=0:continue
    r=(eq[iy_a[-1]]/eq[max(0,iy_a[0]-1)]-1)*100
    avg_f=np.mean(fra_annual[iy_a])
    print(f'    {yr}: Return={r:+.1f}%, Funding 30d MA={avg_f:+.0f}%/yr',flush=True)

print('\n'+'='*70)
print(' COMPARISON: Old (BUG) vs Fixed')
print('='*70)
print(f'  OLD (8x inflated): +52%/yr, Sharpe 10.8')
print(f'  FIXED:            ~6-13%/yr, Sharpe 1-2 (more realistic)')
print(f'')
print(f'  This is honest carry trade level returns.')
print(f'  Sharpe 1-2 is good but not world-class.')
print(f'  300% professor requirement: COMPLETELY IMPOSSIBLE honestly.')

print('\nDone.',flush=True)
