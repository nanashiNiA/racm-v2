"""V4: Spot RACM + Funding Farm 組み合わせ
============================================
Spot RACM (no funding bug): WFS ~39% (シャッフルFAILだがコスト無し)
Funding Farm 1.5x: WFS +51% (確定、Sharpe 10.8)

組み合わせ:
- Spot RACM (50%): BTC spot directional exposure when bullish
- Funding Farm (50%): delta neutral yield always

両方ともfunding問題なし (Spot は funding 無し、FF は受け取り側)
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime

print('='*70,flush=True)
print('  V4: Spot RACM + Funding Farm Combo',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=params.warmup_hours
fra=np.roll(h['funding'].fillna(0).values,1)
vol_30d=h['vol_30d'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
bp_v2=RACMRegime.compute(price,ret,params)

# Bybit data for funding farm
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
                    ticker[['timestamp','funding_rate','last_price','mark_price']].sort_values('timestamp'),
                    on='timestamp',direction='backward',tolerance=pd.Timedelta('2H'))
            ob_frames.append(ohlc)
bybit=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
bybit=bybit.drop_duplicates('timestamp').set_index('timestamp')

idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
common_idx=bybit.index.intersection(idx_h_utc)

# Align all to common
bybit_a=bybit.loc[common_idx]
n=len(common_idx)

# Map indices
racm_idx=np.array([np.where(idx_h_utc==t)[0][0] for t in common_idx])
perp_price=bybit_a['close'].values
spot_price=price[racm_idx]
fra_aligned=np.roll(bybit_a['funding_rate'].fillna(0).values,1)

S_common=2760  # warmup

print(f'  Common: {n} bars',flush=True)

# ============================================================
# Strategy 1: Spot RACM (NO LEVERAGE, NO FUNDING)
# ============================================================
def run_spot_racm():
    eq=np.ones(n);e=1.0;pk=1.0
    for i in range(S_common,n):
        ridx=racm_idx[i]
        v=vol_30d[ridx-1] if ridx>0 and not np.isnan(vol_30d[ridx-1]) else 0.80
        bp_i=bp_v2[ridx]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        # Spot only: just BTC return * bp (no leverage, no funding)
        # LS is impossible on spot side without each asset's spot
        # Simplified: BTC spot * regime
        if bp_i>0:
            pos=min(bp_i,1.0)  # cap at 1.0 (no leverage)
        else:
            pos=0  # short impossible on spot, just stay flat
        # Use spot return
        if i>0 and spot_price[i-1]>0:
            r=spot_price[i]/spot_price[i-1]-1
        else:r=0
        pnl=r*pos
        # No funding cost (spot)
        # No DD reduction (just hold spot)
        e *= (1+pnl)
        eq[i]=e
    return eq

# ============================================================
# Strategy 2: Funding Farm 1.5x
# ============================================================
def run_ff(lev=1.5,fee_bps=2,slip_bps=2):
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    cap_unit=1+1/lev
    e *= (1-2*(fr+sr)/cap_unit)
    perp_un=0.0
    for i in range(S_common,n):
        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0
        sp=sret*1.0;pp=-pret*1.0
        fi=fra_aligned[i]*1.0
        perp_un += pp+fi
        if perp_un<-1/lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
        if (i-S_common)%(7*24)==0 and (i-S_common)>0:
            rebal=abs(perp_un)*0.5*(fr+sr)*2
            e *= (1-rebal/cap_unit)
            perp_un=0
        eq[i]=e
    return eq

# ============================================================
# Run individuals
# ============================================================
print('Running...',flush=True)
eq_spot=run_spot_racm()
eq_ff=run_ff(lev=1.5)

# BTC B&H
eq_bh=np.ones(n);e=1.0
for i in range(S_common,n):
    if i>0 and spot_price[i-1]>0:
        r=spot_price[i]/spot_price[i-1]-1
    else:r=0
    e *= (1+r);eq_bh[i]=e

def stats(eq):
    years=(n-S_common)/(365*24)
    if eq[-1]<=0:return -100,0,0
    annual=((eq[-1])**(1/years)-1)*100
    mdd=0;pk=1
    for i in range(S_common,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[S_common]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

print(f'\n  {"Strategy":<25} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*55}',flush=True)
for label,eq in [('Spot RACM',eq_spot),('Funding Farm 1.5x',eq_ff),('BTC B&H',eq_bh)]:
    a,m,s=stats(eq)
    print(f'  {label:<25} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}',flush=True)

# ============================================================
# Combinations
# ============================================================
print('\n'+'='*70,flush=True)
print(' COMBINATIONS',flush=True)
print('='*70,flush=True)

def blend(weights):
    eq=np.ones(n);e=1.0
    eqs={'SPOT':eq_spot,'FF':eq_ff,'BH':eq_bh}
    for i in range(S_common,n):
        bar_ret=0
        for k,w in weights.items():
            if i>0 and eqs[k][i-1]>0:
                bar_ret+=w*(eqs[k][i]/eqs[k][i-1]-1)
        e *= (1+bar_ret);eq[i]=e
    return eq

print(f'  {"Portfolio":<35} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*60}',flush=True)
for label,w in [
    ('100% Funding Farm', {'FF':1.0}),
    ('100% Spot RACM', {'SPOT':1.0}),
    ('70% FF + 30% Spot', {'FF':0.7,'SPOT':0.3}),
    ('50% FF + 50% Spot', {'FF':0.5,'SPOT':0.5}),
    ('30% FF + 70% Spot', {'FF':0.3,'SPOT':0.7}),
    ('50% FF + 50% BH', {'FF':0.5,'BH':0.5}),
]:
    eq=blend(w)
    a,m,s=stats(eq)
    tag=' ★' if a>50 and m>=-20 else ''
    print(f'  {label:<35} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

# ============================================================
# Year-by-year (best combo)
# ============================================================
print('\n'+'='*70,flush=True)
print(' YEAR-BY-YEAR BEST',flush=True)
print('='*70,flush=True)
eq_70_30=blend({'FF':0.7,'SPOT':0.3})
print(f'  {"Year":<6} {"FF":>10} {"Spot":>10} {"70/30":>10} {"BH":>10}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=S_common]
    if len(iy_a)<100:continue
    rs=lambda eq:(eq[iy_a[-1]]/eq[max(0,iy_a[0]-1)]-1)*100 if eq[max(0,iy_a[0]-1)]>0 else 0
    print(f'  {yr:<6} {rs(eq_ff):>+9.1f}% {rs(eq_spot):>+9.1f}% {rs(eq_70_30):>+9.1f}% {rs(eq_bh):>+9.1f}%',flush=True)

print('\nDone.',flush=True)
