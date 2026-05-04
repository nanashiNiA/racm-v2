"""V4: Combined Strategies (Standard FF + Bidirectional + swin_v25)
=====================================================================
3戦略のcorrelation を確認し、ブレンドで Sharpe / Annual を最大化
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V4: Combined Strategies (FF, Bidir, swin)',flush=True)
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
df=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
df=df.drop_duplicates('timestamp').set_index('timestamp')
n=len(df);S=2760
fra=df['funding_rate'].fillna(0).shift(1).values
spot_price=df['close'].values
fra_ma_30d=pd.Series(fra).rolling(720,min_periods=168).mean().values
fra_annual=fra_ma_30d*365*24*100

def run_standard_ff():
    eq=np.ones(n);e=1.0
    fr=2/10000;sr=2/10000
    perp_un=0.0;current_lev=1.5
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f30=fra_annual[i] if not np.isnan(fra_annual[i]) else 50
        if f30>100:target=2.0
        elif f30>50:target=1.5
        elif f30>20:target=1.0
        elif f30>0:target=0.5
        else:target=0.001
        if (i-S)%(7*24)==0 and (i-S)>0 and abs(target-current_lev)>0.1:
            adj=abs(target-current_lev)/max(current_lev,0.1)*(fr+sr)*2
            cap_unit=1+1/max(current_lev,0.1)
            e *= (1-adj/cap_unit)
            current_lev=target
            perp_un=0
        cap_unit=1+1/max(current_lev,0.1)
        if i>0 and spot_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0
        sp=sret*1.0;pp=-sret*1.0;fi=fra[i]*1.0
        perp_un += pp+fi
        if current_lev>0.01 and perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
        eq[i]=e
    return eq

def run_bidirectional_ff():
    eq=np.ones(n);e=1.0
    fr=2/10000;sr=2/10000
    perp_un=0.0;current_lev=1.5;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f30=fra_annual[i] if not np.isnan(fra_annual[i]) else 50
        if f30>50:target_mode='short_perp';target_lev=1.5
        elif f30>20:target_mode='short_perp';target_lev=1.0
        elif f30>0:target_mode='neutral';target_lev=0
        elif f30>-10:target_mode='neutral';target_lev=0
        else:target_mode='long_perp';target_lev=1.0
        if (i-S)%(7*24)==0 and (i-S)>0:
            if target_mode!=current_mode or abs(target_lev-current_lev)>0.1:
                if current_mode!='neutral':
                    e *= max(0.001,1+perp_un/cap_unit)
                    e *= (1-2*(fr+sr)/cap_unit)
                if target_mode!='neutral':
                    e *= (1-2*(fr+sr)/cap_unit)
                current_mode=target_mode
                current_lev=max(target_lev,0.001)
                cap_unit=1+1/current_lev
                perp_un=0
        if current_mode=='neutral':eq[i]=e;continue
        if i>0 and spot_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0
        if current_mode=='short_perp':
            sp=sret*1.0;pp=-sret*1.0;fi=fra[i]*1.0
        else:
            sp=0;pp=sret*1.0;fi=-fra[i]*1.0
        perp_un += pp+fi
        if perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
        eq[i]=e
    return eq

# Simulated swin_v25-like (proxy: longer trades with HGBR-like accuracy ~57%)
# We don't have full swin_v25 multi-year results, so simulate a 57% WR / 1.5R:R strategy
def run_swing_proxy():
    eq=np.ones(n);e=1.0
    np.random.seed(42)
    # ~225 trades per year × 5 years = 1125 trades
    # Random entry every ~40 bars
    trade_bar = S
    in_pos = False
    while trade_bar < n - 100:
        trade_bar += np.random.randint(20, 80)
        if trade_bar >= n: break
        # 57% chance of win, 1.5R reward, 1R risk
        # at 4x lev, 0.25 size
        # expected per trade: 0.57*1.5 - 0.43*1 = 0.43R
        # R = 1.5% (sl distance)
        # Each trade: gross return = +1.5%*0.25*4 = +1.5% if win, -1%*0.25*4 = -1% if loss
        is_win = np.random.random() < 0.57
        r = 0.015 if is_win else -0.010
        e *= (1 + r)
    eq[:S] = 1.0
    # Spread the trades evenly
    n_active = n - S
    growth_per_bar = (e - 1) / n_active
    for i in range(S, n):
        eq[i] = 1.0 + growth_per_bar * (i - S)
    return eq

print('Computing strategies...',flush=True)
eq_std = run_standard_ff()
eq_bidir = run_bidirectional_ff()
eq_swing = run_swing_proxy()

def stats(eq):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=df.index).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=df.index[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

a_std,m_std,s_std=stats(eq_std)
a_bid,m_bid,s_bid=stats(eq_bidir)
a_swi,m_swi,s_swi=stats(eq_swing)

print(f'\n  {"Strategy":<25} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*55}',flush=True)
print(f'  {"Standard FF":<25} {a_std:>+9.1f}% {m_std:>+7.1f}% {s_std:>7.2f}',flush=True)
print(f'  {"Bidirectional FF":<25} {a_bid:>+9.1f}% {m_bid:>+7.1f}% {s_bid:>7.2f}',flush=True)
print(f'  {"Swing proxy (sim)":<25} {a_swi:>+9.1f}% {m_swi:>+7.1f}% {s_swi:>7.2f}',flush=True)

# Correlations (daily)
print('\n'+'='*70,flush=True)
print(' Daily Return Correlations',flush=True)
print('='*70,flush=True)
def daily_ret(eq):
    return pd.Series(eq,index=df.index).resample('1D').last().pct_change().dropna()
d_std=daily_ret(eq_std)
d_bid=daily_ret(eq_bidir)
d_swi=daily_ret(eq_swing)
common=d_std.index.intersection(d_bid.index).intersection(d_swi.index)
corrs=pd.DataFrame({'STD':d_std.loc[common],'BID':d_bid.loc[common],'SWI':d_swi.loc[common]}).corr()
print(corrs.to_string(),flush=True)

# Blended portfolios
print('\n'+'='*70,flush=True)
print(' BLENDED PORTFOLIOS',flush=True)
print('='*70,flush=True)
def blend(weights, eqs):
    eq_p=np.ones(n);e=1.0
    for i in range(S,n):
        bar_ret=0
        for k,w in weights.items():
            if i>0 and eqs[k][i-1]>0:
                bar_ret+=w*(eqs[k][i]/eqs[k][i-1]-1)
        e *= (1+bar_ret);eq_p[i]=e
    return eq_p

eqs={'STD':eq_std,'BID':eq_bidir,'SWI':eq_swing}

print(f'  {"Portfolio":<40} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*65}',flush=True)
portfolios=[
    ('100% STD',{'STD':1.0}),
    ('100% BID',{'BID':1.0}),
    ('100% SWI',{'SWI':1.0}),
    ('50% STD + 50% BID',{'STD':0.5,'BID':0.5}),
    ('33/33/33',{'STD':0.33,'BID':0.33,'SWI':0.34}),
    ('40% STD + 30% BID + 30% SWI',{'STD':0.4,'BID':0.3,'SWI':0.3}),
    ('60% BID + 40% SWI',{'BID':0.6,'SWI':0.4}),
    ('70% BID + 30% SWI',{'BID':0.7,'SWI':0.3}),
]
for label,w in portfolios:
    eq=blend(w,eqs)
    a,m,s=stats(eq)
    tag=' ★' if a>100 and m>=-15 and s>3 else ''
    print(f'  {label:<40} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

print('\nDone.',flush=True)
