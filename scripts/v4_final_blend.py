"""V4 Final: Standard FF + Bidirectional Blend
================================================
Standard FF (clean, +52%, MDD-3%, Sharpe 10.8)
Bidirectional FF (+101%, MDD-20%, Sharpe 2.25)

ブレンドで OOS drift と MDD を抑える。
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V4 Final Blend: Standard + Bidirectional FF',flush=True)
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
fra_30d=pd.Series(fra).rolling(720,min_periods=168).mean().values
fra_annual=fra_30d*365*24*100

def run_standard():
    eq=np.ones(n);e=1.0
    fr=2/10000;sr=2/10000
    perp_un=0.0;current_lev=1.5
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f=fra_annual[i] if not np.isnan(fra_annual[i]) else 50
        if f>100:t=2.0
        elif f>50:t=1.5
        elif f>20:t=1.0
        elif f>0:t=0.5
        else:t=0.001
        if (i-S)%(7*24)==0 and (i-S)>0 and abs(t-current_lev)>0.1:
            adj=abs(t-current_lev)/max(current_lev,0.1)*(fr+sr)*2
            cap_unit=1+1/max(current_lev,0.1)
            e *= (1-adj/cap_unit)
            current_lev=t;perp_un=0
        cap_unit=1+1/max(current_lev,0.1)
        if i>0 and spot_price[i-1]>0:sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0
        sp=sret*1.0;pp=-sret*1.0;fi=fra[i]*1.0
        perp_un+=pp+fi
        if current_lev>0.01 and perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret);eq[i]=e
    return eq

def run_bidir():
    eq=np.ones(n);e=1.0
    fr=2/10000;sr=2/10000
    perp_un=0.0;current_lev=1.5;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f=fra_annual[i] if not np.isnan(fra_annual[i]) else 50
        if f>50:tm='short_perp';tl=1.5
        elif f>20:tm='short_perp';tl=1.0
        elif f>0:tm='neutral';tl=0
        elif f>-10:tm='neutral';tl=0
        else:tm='long_perp';tl=1.0
        if (i-S)%(7*24)==0 and (i-S)>0:
            if tm!=current_mode or abs(tl-current_lev)>0.1:
                if current_mode!='neutral':
                    e *= max(0.001,1+perp_un/cap_unit)
                    e *= (1-2*(fr+sr)/cap_unit)
                if tm!='neutral':e *= (1-2*(fr+sr)/cap_unit)
                current_mode=tm;current_lev=max(tl,0.001)
                cap_unit=1+1/current_lev;perp_un=0
        if current_mode=='neutral':eq[i]=e;continue
        if i>0 and spot_price[i-1]>0:sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0
        if current_mode=='short_perp':
            sp=sret*1.0;pp=-sret*1.0;fi=fra[i]*1.0
        else:sp=0;pp=sret*1.0;fi=-fra[i]*1.0
        perp_un+=pp+fi
        if perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret);eq[i]=e
    return eq

eq_std=run_standard()
eq_bid=run_bidir()

def stats(eq):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=df.index).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=df.index[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    i22=np.where(np.array([d.year==2022 for d in df.index]))[0]
    i22_a=i22[i22>=S]
    b22=(eq[i22_a[-1]]/eq[max(0,i22_a[0]-1)]-1)*100 if len(i22_a) else 0
    monthly=pd.Series(eq,index=df.index).resample('1M').last().pct_change().dropna()
    monthly=monthly[monthly.index>=df.index[S]]
    win_rate=(monthly>0).mean()*100
    worst_month=monthly.min()*100 if len(monthly) else 0
    return annual,mdd*100,sharpe,b22,win_rate,worst_month

# Blends
print(f'\n  {"Portfolio":<35} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"B22":>7} {"WR%":>5} {"Worst":>7}',flush=True)
print(f'  {"-"*72}',flush=True)

for label,w_std,w_bid in [
    ('100% Standard',1.0,0.0),
    ('100% Bidirectional',0.0,1.0),
    ('80% Std + 20% Bid',0.8,0.2),
    ('60% Std + 40% Bid',0.6,0.4),
    ('50% Std + 50% Bid',0.5,0.5),
    ('40% Std + 60% Bid',0.4,0.6),
    ('20% Std + 80% Bid',0.2,0.8),
]:
    if w_std==1.0:eq=eq_std
    elif w_bid==1.0:eq=eq_bid
    else:
        eq=np.ones(n);e=1.0
        for i in range(S,n):
            r_std=eq_std[i]/eq_std[i-1]-1 if i>0 and eq_std[i-1]>0 else 0
            r_bid=eq_bid[i]/eq_bid[i-1]-1 if i>0 and eq_bid[i-1]>0 else 0
            br=w_std*r_std+w_bid*r_bid
            e *= (1+br);eq[i]=e
    a,m,s,b,w,worst=stats(eq)
    tag=''
    if a>70 and m>=-12 and s>4: tag=' ★'
    print(f'  {label:<35} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+6.0f}% {w:>4.0f}% {worst:>+6.1f}%{tag}',flush=True)

# Year-by-year for best
print('\n  Year-by-year (50/50):',flush=True)
eq_blend=np.ones(n);e=1.0
for i in range(S,n):
    r_std=eq_std[i]/eq_std[i-1]-1 if i>0 and eq_std[i-1]>0 else 0
    r_bid=eq_bid[i]/eq_bid[i-1]-1 if i>0 and eq_bid[i-1]>0 else 0
    e *= (1+0.5*r_std+0.5*r_bid);eq_blend[i]=e

print(f'  {"Year":<6} {"Std":>10} {"Bidir":>10} {"50/50":>10}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in df.index]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    rs=(eq_std[iy_a[-1]]/eq_std[max(0,iy_a[0]-1)]-1)*100 if eq_std[max(0,iy_a[0]-1)]>0 else 0
    rb=(eq_bid[iy_a[-1]]/eq_bid[max(0,iy_a[0]-1)]-1)*100 if eq_bid[max(0,iy_a[0]-1)]>0 else 0
    rbl=(eq_blend[iy_a[-1]]/eq_blend[max(0,iy_a[0]-1)]-1)*100 if eq_blend[max(0,iy_a[0]-1)]>0 else 0
    print(f'  {yr:<6} {rs:>+9.1f}% {rb:>+9.1f}% {rbl:>+9.1f}%',flush=True)

# Correlation
print(f'\n  Daily return correlation (Std vs Bid):',flush=True)
d_std=pd.Series(eq_std,index=df.index).resample('1D').last().pct_change().dropna()
d_bid=pd.Series(eq_bid,index=df.index).resample('1D').last().pct_change().dropna()
common_d=d_std.index.intersection(d_bid.index)
corr=np.corrcoef(d_std.loc[common_d],d_bid.loc[common_d])[0,1]
print(f'  r = {corr:+.3f}',flush=True)

print('\nDone.',flush=True)
