"""V4: Bidirectional FF + 追加最適化
====================================
Bidirectional FF (+101%/yr) を更に改善する案:
1. レバレッジスキャン (現在 short=1.5x, long=1.0x が最適か)
2. 閾値スキャン (-10%/yr が最適か)
3. Stop loss追加でMDD削減 (-20% → -10%)
4. 各年ごとの分解詳細
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V4: Bidirectional FF 最適化',flush=True)
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

def run_bidir(short_lev=1.5, long_lev=1.0, neg_threshold=-10,
              fee_bps=2, slip_bps=2, rebal_days=7,
              use_stoploss=False, sl_pct=0.10):
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=short_lev;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    pk=1.0
    for i in range(S,n):
        f30=fra_annual[i] if not np.isnan(fra_annual[i]) else 50

        if f30>50:target_mode='short_perp';target_lev=short_lev
        elif f30>20:target_mode='short_perp';target_lev=short_lev*0.67
        elif f30>0:target_mode='neutral';target_lev=0
        elif f30>neg_threshold:target_mode='neutral';target_lev=0
        else:target_mode='long_perp';target_lev=long_lev

        if (i-S)%(rebal_days*24)==0 and (i-S)>0:
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

        if current_mode=='neutral':
            eq[i]=e
            pk=max(pk,e)
            continue

        if i>0 and spot_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0

        if current_mode=='short_perp':
            sp=sret*1.0;pp=-sret*1.0;fi=fra[i]*1.0
        else:  # long_perp
            sp=0;pp=sret*1.0;fi=-fra[i]*1.0

        perp_un += pp+fi

        # Stop loss check
        if use_stoploss:
            dd=(e-pk)/pk if pk>0 else 0
            if dd<-sl_pct and current_mode=='long_perp':
                # Exit long position on stop
                e *= max(0.001,1+perp_un/cap_unit)
                e *= (1-2*(fr+sr)/cap_unit)
                current_mode='neutral'
                current_lev=0.001
                cap_unit=1+1/current_lev
                perp_un=0
                eq[i]=e
                continue

        if perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue

        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
        eq[i]=e
        pk=max(pk,e)

    return eq

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
    return annual,mdd*100,sharpe,b22

# ============================================================
# Leverage scan
# ============================================================
print('\n'+'='*70,flush=True)
print(' LEVERAGE SCAN',flush=True)
print('='*70,flush=True)
print(f'  {"short_lev":<10} {"long_lev":<10} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"B22":>8}',flush=True)
print(f'  {"-"*60}',flush=True)
for sl in [1.0, 1.5, 2.0]:
    for ll in [0.5, 1.0, 1.5]:
        eq=run_bidir(short_lev=sl, long_lev=ll)
        a,m,s,b=stats(eq)
        tag=' ★' if a>100 and m>=-25 else ''
        print(f'  {sl:<10.1f} {ll:<10.1f} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%{tag}',flush=True)

# ============================================================
# Threshold scan
# ============================================================
print('\n'+'='*70,flush=True)
print(' NEGATIVE THRESHOLD SCAN',flush=True)
print('='*70,flush=True)
print(f'  {"threshold":<12} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"B22":>8}',flush=True)
print(f'  {"-"*50}',flush=True)
for th in [0, -5, -10, -20, -30]:
    eq=run_bidir(neg_threshold=th)
    a,m,s,b=stats(eq)
    tag=' default' if th==-10 else ''
    print(f'  {th:<12} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%{tag}',flush=True)

# ============================================================
# Stop loss for inverse FF
# ============================================================
print('\n'+'='*70,flush=True)
print(' STOP LOSS for Inverse FF',flush=True)
print('='*70,flush=True)
print(f'  {"SL%":<10} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"B22":>8}',flush=True)
print(f'  {"-"*50}',flush=True)
for sl in [None, 0.05, 0.10, 0.15, 0.20]:
    if sl is None:
        eq=run_bidir(use_stoploss=False)
        label='no SL'
    else:
        eq=run_bidir(use_stoploss=True, sl_pct=sl)
        label=f'SL {sl*100:.0f}%'
    a,m,s,b=stats(eq)
    tag=' ★' if m>=-15 and a>80 else ''
    print(f'  {label:<10} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {b:>+7.0f}%{tag}',flush=True)

# ============================================================
# Year-by-year detail
# ============================================================
print('\n'+'='*70,flush=True)
print(' YEAR-BY-YEAR DETAIL (Bidirectional FF best)',flush=True)
print('='*70,flush=True)
eq=run_bidir(short_lev=1.5,long_lev=1.0,neg_threshold=-10,use_stoploss=True,sl_pct=0.10)
print(f'  {"Year":<6} {"Return":>10} {"Avg Funding":>12} {"Days neg fund":>15}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in df.index]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    if eq[max(0,iy_a[0]-1)]<=0:continue
    r=(eq[iy_a[-1]]/eq[max(0,iy_a[0]-1)]-1)*100
    avg_f=np.mean(fra_annual[iy_a])
    days_neg=np.sum(fra_annual[iy_a]<0)/24
    print(f'  {yr:<6} {r:>+9.1f}% {avg_f:>+10.0f}% {days_neg:>13.0f}',flush=True)

print('\nDone.',flush=True)
