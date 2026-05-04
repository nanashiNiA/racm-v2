"""V5: Ultimate Ensemble - Multiple weak alpha → strong combined
================================================================
Funding Farm + Donchian + Vol + ML + leverage を統合
全部 honest 数値で、組み合わせで 50%+/yr を狙う
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V5: Ultimate Ensemble (honest 50%+/yr 狙い)',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h()
h.index=h.index.tz_localize('UTC') if h.index.tz is None else h.index

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
common_idx=bybit.index.intersection(h.index)
bybit_a=bybit.loc[common_idx]
spot_a=h.loc[common_idx]
perp_price=bybit_a['close'].values
spot_price=spot_a['close'].values
fra_8h=np.roll(bybit_a['funding_rate'].fillna(0).values,1)
ret=np.diff(spot_price,prepend=spot_price[0])/spot_price
n=len(common_idx);S=2760
fra_30d=pd.Series(fra_8h).rolling(720,min_periods=168).mean().values
fra_annual=fra_30d*3*365*100

# Features
ma_50=pd.Series(spot_price).rolling(50,min_periods=25).mean().values
ma_200=pd.Series(spot_price).rolling(200,min_periods=100).mean().values
donch_high=pd.Series(spot_price).rolling(480,min_periods=120).max().values
donch_low=pd.Series(spot_price).rolling(480,min_periods=120).min().values

# ============================================================
# Component 1: Standard Funding Farm (delta neutral)
# ============================================================
def run_ff(lev=1.5):
    eq=np.ones(n);e=1.0
    fr=2/10000;sr=2/10000
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
        if (i-S)%(7*24)==0 and (i-S)>0 and abs(t-current_lev)>0.1:
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

# ============================================================
# Component 2: Donchian Breakout (directional)
# ============================================================
def run_donchian(lev=2.0,fee_bps=5):
    eq=np.ones(n);e=1.0;pk=1.0
    fr=fee_bps/10000
    in_pos=0
    for i in range(S,n):
        if np.isnan(donch_high[i-1]):eq[i]=e;continue
        new_pos=in_pos
        if in_pos==0:
            if spot_price[i-1]>=donch_high[i-1]*0.998:new_pos=1
            elif spot_price[i-1]<=donch_low[i-1]*1.002:new_pos=-1
        elif in_pos==1:
            if spot_price[i-1]<=donch_low[i-1]*1.002:new_pos=-1
        else:
            if spot_price[i-1]>=donch_high[i-1]*0.998:new_pos=1
        if new_pos!=in_pos:
            e *= (1-fr*abs(new_pos-in_pos)*lev)
            in_pos=new_pos
        pnl=ret[i]*in_pos*lev
        ts=common_idx[i]
        if ts.hour in [0,8,16] and in_pos!=0:
            pnl -= fra_8h[i]*in_pos*lev  # long pays positive funding
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.20:pnl*=0.5
        e *= (1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# Component 3: Trend Following (MA crossover)
# ============================================================
def run_trend(lev=1.5,fee_bps=5):
    eq=np.ones(n);e=1.0;pk=1.0
    fr=fee_bps/10000
    prev_pos=0
    for i in range(S,n):
        if np.isnan(ma_50[i-1]) or np.isnan(ma_200[i-1]):eq[i]=e;continue
        pos=1 if ma_50[i-1]>ma_200[i-1] else -0.3
        if abs(pos-prev_pos)>0.1:
            e *= (1-fr*abs(pos-prev_pos)*lev)
        prev_pos=pos
        pnl=ret[i]*pos*lev
        ts=common_idx[i]
        if ts.hour in [0,8,16]:
            pnl -= fra_8h[i]*pos*lev
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.20:pnl*=0.5
        e *= (1+pnl);eq[i]=e;pk=max(pk,e)
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

# Run components
print('Running components...',flush=True)
eq_ff=run_ff(lev=1.5)
eq_dc=run_donchian(lev=2.0)
eq_tr=run_trend(lev=1.5)

print(f'\n  {"Component":<25} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"B22":>8}',flush=True)
print(f'  {"-"*55}',flush=True)
for label,eq in [('Funding Farm 1.5x',eq_ff),('Donchian 2x',eq_dc),('Trend 1.5x',eq_tr)]:
    a,m,s,b=stats(eq)
    print(f'  {label:<25} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%',flush=True)

# Correlations
print(f'\n  Correlations:',flush=True)
def daily_ret(eq):
    return pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
d_ff=daily_ret(eq_ff);d_dc=daily_ret(eq_dc);d_tr=daily_ret(eq_tr)
common=d_ff.index.intersection(d_dc.index).intersection(d_tr.index)
corrs=pd.DataFrame({'FF':d_ff.loc[common],'DC':d_dc.loc[common],'TR':d_tr.loc[common]}).corr()
print(corrs.to_string(),flush=True)

# Blends
print(f'\n  ENSEMBLES:',flush=True)
def blend(eqs,weights):
    eq=np.ones(n);e=1.0
    for i in range(S,n):
        br=0
        for k,w in weights.items():
            if i>0 and eqs[k][i-1]>0:br+=w*(eqs[k][i]/eqs[k][i-1]-1)
        e *= (1+br);eq[i]=e
    return eq

eqs={'FF':eq_ff,'DC':eq_dc,'TR':eq_tr}

print(f'  {"Portfolio":<40} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"B22":>8}',flush=True)
print(f'  {"-"*68}',flush=True)
portfolios=[
    ('100% FF',{'FF':1.0}),
    ('100% DC',{'DC':1.0}),
    ('100% TR',{'TR':1.0}),
    ('50% FF + 50% DC',{'FF':0.5,'DC':0.5}),
    ('33/33/33',{'FF':0.33,'DC':0.33,'TR':0.34}),
    ('60% FF + 40% DC',{'FF':0.6,'DC':0.4}),
    ('40% FF + 60% DC',{'FF':0.4,'DC':0.6}),
    ('70% FF + 15% DC + 15% TR',{'FF':0.7,'DC':0.15,'TR':0.15}),
]
for label,w in portfolios:
    eq=blend(eqs,w)
    a,m,s,b=stats(eq)
    tag=' ★' if a>15 and m>=-15 and s>1 else ''
    print(f'  {label:<40} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%{tag}',flush=True)

print('\nDone.',flush=True)
