"""V5: Volatility Strategies (proper funding application)
==========================================================
1. Vol selling (sell BTC when realized vol high → expect mean reversion)
2. Vol breakout (long when vol expanding)
3. ATR-band trading (mean reversion)
4. RSI mean reversion
5. Volume-weighted momentum
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V5: Volatility / Mean-Reversion Strategies (proper funding)',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h()
h.index=h.index.tz_localize('UTC') if h.index.tz is None else h.index
n=len(h);S=2760
ret=h['ret'].values
price=h['close'].values
high=h['high'].values
low=h['low'].values
fra_8h=np.roll(h['funding'].fillna(0).values,1)

# Features
rsi_14=h['rsi_14d'].values if 'rsi_14d' in h.columns else np.full(n,50)
vol_30d=h['vol_30d'].values
atr_24=pd.Series(np.abs(ret)).rolling(24,min_periods=6).mean().values
ma_50=pd.Series(price).rolling(50,min_periods=25).mean().values
ma_200=pd.Series(price).rolling(200,min_periods=100).mean().values
bb_mean=pd.Series(price).rolling(480,min_periods=240).mean().values
bb_std=pd.Series(price).rolling(480,min_periods=240).std().values

def stats(eq):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=h.index).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=h.index[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    i22=np.where(np.array([d.year==2022 for d in h.index]))[0]
    i22_a=i22[i22>=S]
    b22=(eq[i22_a[-1]]/eq[max(0,i22_a[0]-1)]-1)*100 if len(i22_a) else 0
    return annual,mdd*100,sharpe,b22

# Common runner with proper funding
def run_strategy(signals, lev=2.0, fee_bps=5):
    """signals: -1 to +1 array"""
    eq=np.ones(n);e=1.0;pk=1.0
    fr=fee_bps/10000
    prev_pos=0
    for i in range(S,n):
        sig=signals[i]
        # Cost on position change
        if abs(sig-prev_pos)>0.1:
            e *= (1-fr*abs(sig-prev_pos)*lev)
        prev_pos=sig

        if sig != 0:
            # Trade with leverage
            pnl=ret[i]*sig*lev
            # Funding cost (ロング → 払う、Short → 受け取る)
            ts=h.index[i]
            if ts.hour in [0,8,16]:
                pnl -= fra_8h[i]*sig*lev  # long pays positive funding
        else:
            pnl=0

        # DD control
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.20:pnl*=0.5

        e *= (1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# Strategy 1: Mean Reversion (RSI + BB)
# ============================================================
print('\n'+'='*70,flush=True)
print(' 1. Mean Reversion (RSI + Bollinger Band)',flush=True)
print('='*70,flush=True)
mr_signal=np.zeros(n)
for i in range(S,n):
    if np.isnan(bb_mean[i-1]) or bb_std[i-1]<=0:continue
    z=(price[i-1]-bb_mean[i-1])/(bb_std[i-1]*2+1e-10)
    rsi=rsi_14[i-1] if not np.isnan(rsi_14[i-1]) else 50
    if z<-1.0 and rsi<30:mr_signal[i]=1.0
    elif z<-0.5 and rsi<40:mr_signal[i]=0.5
    elif z>1.0 and rsi>70:mr_signal[i]=-1.0
    elif z>0.5 and rsi>60:mr_signal[i]=-0.5

for lev in [1.0,2.0,3.0]:
    eq=run_strategy(mr_signal,lev=lev)
    a,m,s,b=stats(eq)
    print(f'  Lev {lev}x: Annual={a:+.1f}%, MDD={m:+.1f}%, Sharpe={s:.2f}, B22={b:+.0f}%',flush=True)

# ============================================================
# Strategy 2: Volatility Breakout
# ============================================================
print('\n'+'='*70,flush=True)
print(' 2. Volatility Breakout',flush=True)
print('='*70,flush=True)
# Long when vol expanding + price above MA200
vb_signal=np.zeros(n)
for i in range(S,n):
    if np.isnan(vol_30d[i-1]) or i<24:continue
    vol_now=vol_30d[i-1]
    vol_past=np.mean(vol_30d[max(S,i-168):i-1]) if i>S+168 else 0.5
    expanding=vol_now>vol_past*1.3
    above_ma=not np.isnan(ma_200[i-1]) and price[i-1]>ma_200[i-1]
    below_ma=not np.isnan(ma_200[i-1]) and price[i-1]<ma_200[i-1]
    if expanding and above_ma:vb_signal[i]=1.0
    elif expanding and below_ma:vb_signal[i]=-0.5

for lev in [1.0,2.0,3.0]:
    eq=run_strategy(vb_signal,lev=lev)
    a,m,s,b=stats(eq)
    print(f'  Lev {lev}x: Annual={a:+.1f}%, MDD={m:+.1f}%, Sharpe={s:.2f}, B22={b:+.0f}%',flush=True)

# ============================================================
# Strategy 3: Trend Following (MA50/MA200)
# ============================================================
print('\n'+'='*70,flush=True)
print(' 3. Trend Following (MA50/MA200 cross)',flush=True)
print('='*70,flush=True)
tf_signal=np.zeros(n)
for i in range(S,n):
    if np.isnan(ma_50[i-1]) or np.isnan(ma_200[i-1]):continue
    if ma_50[i-1]>ma_200[i-1]:tf_signal[i]=1.0
    else:tf_signal[i]=-0.3  # mild short

for lev in [1.0,2.0,3.0]:
    eq=run_strategy(tf_signal,lev=lev)
    a,m,s,b=stats(eq)
    print(f'  Lev {lev}x: Annual={a:+.1f}%, MDD={m:+.1f}%, Sharpe={s:.2f}, B22={b:+.0f}%',flush=True)

# ============================================================
# Strategy 4: Volume-weighted momentum
# ============================================================
print('\n'+'='*70,flush=True)
print(' 4. Volume-Weighted Momentum (24H)',flush=True)
print('='*70,flush=True)
vol_data=h['volume'].values if 'volume' in h.columns else np.ones(n)
vw_signal=np.zeros(n)
for i in range(S,n):
    if i<24:continue
    rets_24=ret[max(0,i-24):i]
    vols_24=vol_data[max(0,i-24):i]
    if np.sum(vols_24)>0:
        vw_ret=np.sum(rets_24*vols_24)/np.sum(vols_24)
        if vw_ret>0.02:vw_signal[i]=1.0
        elif vw_ret>0.01:vw_signal[i]=0.5
        elif vw_ret<-0.02:vw_signal[i]=-1.0
        elif vw_ret<-0.01:vw_signal[i]=-0.5

for lev in [1.0,2.0,3.0]:
    eq=run_strategy(vw_signal,lev=lev)
    a,m,s,b=stats(eq)
    print(f'  Lev {lev}x: Annual={a:+.1f}%, MDD={m:+.1f}%, Sharpe={s:.2f}, B22={b:+.0f}%',flush=True)

# ============================================================
# Strategy 5: Range Breakout (Donchian)
# ============================================================
print('\n'+'='*70,flush=True)
print(' 5. Donchian Channel Breakout',flush=True)
print('='*70,flush=True)
donch_high=pd.Series(price).rolling(480,min_periods=120).max().values
donch_low=pd.Series(price).rolling(480,min_periods=120).min().values
donch_signal=np.zeros(n)
in_pos=0
for i in range(S,n):
    if np.isnan(donch_high[i-1]):continue
    if in_pos==0:
        if price[i-1]>=donch_high[i-1]*0.998:in_pos=1
        elif price[i-1]<=donch_low[i-1]*1.002:in_pos=-1
    elif in_pos==1:
        if price[i-1]<=donch_low[i-1]*1.002:in_pos=-1
    else:
        if price[i-1]>=donch_high[i-1]*0.998:in_pos=1
    donch_signal[i]=in_pos

for lev in [1.0,2.0,3.0]:
    eq=run_strategy(donch_signal,lev=lev)
    a,m,s,b=stats(eq)
    print(f'  Lev {lev}x: Annual={a:+.1f}%, MDD={m:+.1f}%, Sharpe={s:.2f}, B22={b:+.0f}%',flush=True)

# ============================================================
# Combinations
# ============================================================
print('\n'+'='*70,flush=True)
print(' 6. Strategy Combinations',flush=True)
print('='*70,flush=True)

combo_signals={
    'MR + Trend': (mr_signal*0.5+tf_signal*0.5),
    'MR + Donchian': (mr_signal*0.5+donch_signal*0.5),
    'Trend + Vol breakout': (tf_signal*0.5+vb_signal*0.5),
    'All 5 equal': ((mr_signal+vb_signal+tf_signal+vw_signal+donch_signal)/5),
}

for label, sig in combo_signals.items():
    sig=np.clip(sig,-1,1)
    eq=run_strategy(sig,lev=2.0)
    a,m,s,b=stats(eq)
    tag=' ★' if a>50 and m>=-30 else ''
    print(f'  {label:<30} (lev=2x): Annual={a:+.1f}%, MDD={m:+.1f}%, Sharpe={s:.2f}, B22={b:+.0f}%{tag}',flush=True)

print('\nDone.',flush=True)
