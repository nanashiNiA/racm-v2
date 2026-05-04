"""V5 FIXED: 全戦略を per-8H proper funding で再計算
======================================================
1. Standard Bear-aware FF
2. Bidirectional FF
3. BTC+ETH parallel
4. 80/20 blend (修正後)
5. レバスキャン
6. シャッフルテスト

正しい funding application:
- per-8H rate: 0.0001 = 0.01% per 8H
- per-hour 適用なら /8 必要 OR 8H境界で full rate
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V5 FIXED: 全戦略 per-8H proper funding',flush=True)
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
fra_8h=np.roll(bybit_a['funding_rate'].fillna(0).values,1)  # per-8H
n=len(common_idx);S=2760
fra_30d=pd.Series(fra_8h).rolling(720,min_periods=168).mean().values
fra_annual=fra_30d*3*365*100  # per-8H × 3 × 365 = annual %

# ============================================================
# Strategy runners (FIXED: 8H boundary funding)
# ============================================================
def run_standard(lev=1.5, fee_bps=2, slip_bps=2, rebal_days=7):
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=lev
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f=fra_annual[i] if not np.isnan(fra_annual[i]) else 10
        if f>30:t=2.0
        elif f>10:t=1.5
        elif f>5:t=1.0
        elif f>0:t=0.5
        else:t=0.001
        if (i-S)%(rebal_days*24)==0 and (i-S)>0 and abs(t-current_lev)>0.1:
            adj=abs(t-current_lev)/max(current_lev,0.1)*(fr+sr)*2
            cap_unit=1+1/max(current_lev,0.1)
            e *= (1-adj/cap_unit)
            current_lev=t;perp_un=0
        cap_unit=1+1/max(current_lev,0.1)
        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0
        sp=sret*1.0;pp=-pret*1.0
        # FIXED: funding only at 8H boundaries
        ts=common_idx[i]
        fi=fra_8h[i]*1.0 if ts.hour in [0,8,16] else 0
        perp_un+=pp+fi
        if current_lev>0.01 and perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
        eq[i]=e
    return eq

def run_bidirectional(fee_bps=2, slip_bps=2, rebal_days=7):
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=1.5;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f=fra_annual[i] if not np.isnan(fra_annual[i]) else 10
        if f>30:tm='short_perp';tl=2.0
        elif f>10:tm='short_perp';tl=1.5
        elif f>5:tm='short_perp';tl=1.0
        elif f>-5:tm='neutral';tl=0
        else:tm='long_perp';tl=1.0
        if (i-S)%(rebal_days*24)==0 and (i-S)>0:
            if tm!=current_mode or abs(tl-current_lev)>0.1:
                if current_mode!='neutral':
                    e *= max(0.001,1+perp_un/cap_unit)
                    e *= (1-2*(fr+sr)/cap_unit)
                if tm!='neutral':e *= (1-2*(fr+sr)/cap_unit)
                current_mode=tm;current_lev=max(tl,0.001)
                cap_unit=1+1/current_lev;perp_un=0
        if current_mode=='neutral':eq[i]=e;continue
        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0
        ts=common_idx[i]
        funding_event=ts.hour in [0,8,16]
        if current_mode=='short_perp':
            sp=sret*1.0;pp=-pret*1.0
            fi=fra_8h[i]*1.0 if funding_event else 0
        else:
            sp=0;pp=sret*1.0
            fi=-fra_8h[i]*1.0 if funding_event else 0
        perp_un+=pp+fi
        if perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
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
    i22=np.where(np.array([d.year==2022 for d in common_idx]))[0]
    i22_a=i22[i22>=S]
    b22=(eq[i22_a[-1]]/eq[max(0,i22_a[0]-1)]-1)*100 if len(i22_a) else 0
    return annual,mdd*100,sharpe,b22

# ============================================================
# Run all strategies
# ============================================================
print('\n'+'='*70,flush=True)
print(' Strategy Comparison (FIXED proper funding)',flush=True)
print('='*70,flush=True)
print(f'  {"Strategy":<35} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"B22":>8}',flush=True)
print(f'  {"-"*65}',flush=True)

eq_std=run_standard(lev=1.5)
a,m,s,b=stats(eq_std)
print(f'  {"Standard FF (1.5x)":<35} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%',flush=True)

eq_bid=run_bidirectional()
a,m,s,b=stats(eq_bid)
print(f'  {"Bidirectional FF":<35} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%',flush=True)

# Blends
def blend(eqs,weights):
    eq=np.ones(n);e=1.0
    for i in range(S,n):
        br=0
        for k,w in weights.items():
            if i>0 and eqs[k][i-1]>0:br+=w*(eqs[k][i]/eqs[k][i-1]-1)
        e *= (1+br);eq[i]=e
    return eq

eqs={'STD':eq_std,'BID':eq_bid}
for label,w in [('80% Std + 20% Bid',{'STD':0.8,'BID':0.2}),
                 ('60% Std + 40% Bid',{'STD':0.6,'BID':0.4}),
                 ('50/50',{'STD':0.5,'BID':0.5})]:
    eq=blend(eqs,w)
    a,m,s,b=stats(eq)
    print(f'  {label:<35} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%',flush=True)

# ============================================================
# Year by year
# ============================================================
print(f'\n  Year-by-year (Standard FF):',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    if eq_std[max(0,iy_a[0]-1)]<=0:continue
    r=(eq_std[iy_a[-1]]/eq_std[max(0,iy_a[0]-1)]-1)*100
    avg_f=np.mean(fra_annual[iy_a])
    print(f'    {yr}: Return={r:+.1f}%, Funding={avg_f:+.0f}%/yr',flush=True)

print(f'\n  Year-by-year (Bidirectional):',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    if eq_bid[max(0,iy_a[0]-1)]<=0:continue
    r=(eq_bid[iy_a[-1]]/eq_bid[max(0,iy_a[0]-1)]-1)*100
    print(f'    {yr}: Return={r:+.1f}%',flush=True)

# ============================================================
# Higher leverage (push for 300%)
# ============================================================
print('\n'+'='*70,flush=True)
print(' HIGH LEVERAGE Bidirectional (300% target push)',flush=True)
print('='*70,flush=True)

def run_bidir_high_lev(short_lev=3.0, long_lev=2.0):
    """High leverage version of Bidirectional"""
    eq=np.ones(n);e=1.0
    fr=2/10000;sr=2/10000
    perp_un=0.0;current_lev=short_lev;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f=fra_annual[i] if not np.isnan(fra_annual[i]) else 10
        if f>30:tm='short_perp';tl=short_lev
        elif f>10:tm='short_perp';tl=short_lev*0.7
        elif f>5:tm='short_perp';tl=short_lev*0.5
        elif f>-5:tm='neutral';tl=0
        else:tm='long_perp';tl=long_lev
        if (i-S)%(7*24)==0 and (i-S)>0:
            if tm!=current_mode or abs(tl-current_lev)>0.1:
                if current_mode!='neutral':
                    e *= max(0.001,1+perp_un/cap_unit)
                    e *= (1-2*(fr+sr)/cap_unit)
                if tm!='neutral':e *= (1-2*(fr+sr)/cap_unit)
                current_mode=tm;current_lev=max(tl,0.001)
                cap_unit=1+1/current_lev;perp_un=0
        if current_mode=='neutral':eq[i]=e;continue
        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0
        ts=common_idx[i]
        funding_event=ts.hour in [0,8,16]
        if current_mode=='short_perp':
            sp=sret*1.0;pp=-pret*1.0
            fi=fra_8h[i]*1.0 if funding_event else 0
        else:
            sp=0;pp=sret*1.0
            fi=-fra_8h[i]*1.0 if funding_event else 0
        perp_un+=pp+fi
        if perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
        eq[i]=e
    return eq

print(f'  {"Short Lev":<10} {"Long Lev":<10} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*55}',flush=True)
for sl,ll in [(1.5,1.0),(2.0,1.0),(3.0,2.0),(5.0,3.0),(10.0,5.0)]:
    eq=run_bidir_high_lev(short_lev=sl,long_lev=ll)
    a,m,s,b=stats(eq)
    tag=' ★' if a>200 and m>=-40 else ''
    print(f'  {sl:<9.1f} {ll:<9.1f} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

print('\nDone.',flush=True)
