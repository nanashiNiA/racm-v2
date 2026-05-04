"""V4 Bidirectional FF: 厳密検証
=================================
+101% annual はあまりに良すぎる。徹底検証:
1. シャッフル検証 (funding rate timing)
2. リーク検証 (lag shifts)
3. OOS分割
4. 月別詳細
5. 教授要件チェック
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V4 Bidirectional FF 徹底検証',flush=True)
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

def run_bidir(fra_use=None, fee_bps=2, slip_bps=2, rebal_days=7):
    if fra_use is None:fra_use=fra
    fra_30d=pd.Series(fra_use).rolling(720,min_periods=168).mean().values
    fra_ann=fra_30d*365*24*100

    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=1.5;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)

    for i in range(S,n):
        f30=fra_ann[i] if not np.isnan(fra_ann[i]) else 50

        if f30>50:target_mode='short_perp';target_lev=1.5
        elif f30>20:target_mode='short_perp';target_lev=1.0
        elif f30>0:target_mode='neutral';target_lev=0
        elif f30>-10:target_mode='neutral';target_lev=0
        else:target_mode='long_perp';target_lev=1.0

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
            eq[i]=e;continue

        if i>0 and spot_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0

        if current_mode=='short_perp':
            sp=sret*1.0;pp=-sret*1.0;fi=fra_use[i]*1.0
        else:  # long_perp
            sp=0;pp=sret*1.0;fi=-fra_use[i]*1.0

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

# Real
eq_real=run_bidir()
ann_real,mdd_real,sh_real,b22_real=stats(eq_real)
print(f'\n  Real: Annual={ann_real:+.1f}%, MDD={mdd_real:+.1f}%, Sharpe={sh_real:.2f}, Bear22={b22_real:+.1f}%',flush=True)

# 1. Shuffle test
print('\n'+'='*70,flush=True)
print(' 1. SHUFFLE TEST (200x)',flush=True)
print('='*70,flush=True)

shuf_anns=[];shuf_mdds=[];shuf_b22s=[]
for seed in range(200):
    np.random.seed(seed+888888)
    fra_sh=fra.copy()
    np.random.shuffle(fra_sh[S:])
    eq_s=run_bidir(fra_use=fra_sh)
    a,m,_,b=stats(eq_s)
    shuf_anns.append(a)
    shuf_mdds.append(m)
    shuf_b22s.append(b)

p_ann=np.mean([a>=ann_real for a in shuf_anns])
p_b22=np.mean([b>=b22_real for b in shuf_b22s])
print(f'  Annual: Real={ann_real:+.1f}%, Shuffle={np.mean(shuf_anns):+.1f}%+/-{np.std(shuf_anns):.1f}%, p={p_ann:.3f} {"PASS" if p_ann<0.05 else "FAIL"}',flush=True)
print(f'  Bear22: Real={b22_real:+.1f}%, Shuffle={np.mean(shuf_b22s):+.1f}%+/-{np.std(shuf_b22s):.1f}%, p={p_b22:.3f} {"PASS" if p_b22<0.05 else "FAIL"}',flush=True)

# 2. Leak check
print('\n'+'='*70,flush=True)
print(' 2. LEAK CHECK (lag shifts)',flush=True)
print('='*70,flush=True)
print(f'  {"Shift":<8} {"Annual":>10} {"MDD":>8}',flush=True)
for shift in [-2,-1,0,1,2,4,8]:
    fra_s=np.roll(fra,shift)
    eq=run_bidir(fra_use=fra_s)
    a,m,_,_=stats(eq)
    marker=' base' if shift==0 else ''
    print(f'  {shift:+d}h {a:>+9.1f}% {m:>+7.1f}%{marker}',flush=True)

# 3. OOS split
print('\n'+'='*70,flush=True)
print(' 3. OOS SPLIT',flush=True)
print('='*70,flush=True)
train_end=np.where(np.array([d<pd.Timestamp('2024-01-01',tz='UTC') for d in df.index]))[0][-1]
eq=run_bidir()
train_eq=eq[S:train_end+1]
train_years=(train_end-S)/(365*24)
train_ann=((train_eq[-1]/train_eq[0])**(1/train_years)-1)*100 if train_eq[0]>0 else 0
test_eq=eq[train_end+1:]/eq[train_end] if eq[train_end]>0 else eq[train_end+1:]
test_years=(n-train_end-1)/(365*24)
test_ann=((test_eq[-1])**(1/test_years)-1)*100 if test_eq[-1]>0 else 0
print(f'  Train (2021-23): {train_ann:.1f}%/yr',flush=True)
print(f'  Test  (2024-25): {test_ann:.1f}%/yr',flush=True)
diff=abs(train_ann-test_ann)
print(f'  Consistent: {"YES" if diff<50 else f"DRIFT ({diff:.0f}%)"}',flush=True)

# 4. Monthly detail
print('\n'+'='*70,flush=True)
print(' 4. MONTHLY DETAIL',flush=True)
print('='*70,flush=True)
eq_s=pd.Series(eq,index=df.index)
monthly=eq_s.resample('1M').last().pct_change().dropna()
monthly=monthly[monthly.index>=df.index[S]]
print(f'  Total months: {len(monthly)}',flush=True)
print(f'  Mean monthly: {monthly.mean()*100:+.2f}%',flush=True)
print(f'  Std monthly: {monthly.std()*100:.2f}%',flush=True)
print(f'  Win rate: {(monthly>0).mean()*100:.1f}%',flush=True)
print(f'  Worst month: {monthly.min()*100:+.2f}% ({monthly.idxmin().strftime("%Y-%m")})',flush=True)
print(f'  Best month: {monthly.max()*100:+.2f}% ({monthly.idxmax().strftime("%Y-%m")})',flush=True)

neg_months=monthly[monthly<0]
print(f'\n  Negative months ({len(neg_months)}):',flush=True)
for ts,r in neg_months.items():
    print(f'    {ts.strftime("%Y-%m")}: {r*100:+.2f}%',flush=True)

# 5. 教授要件
print('\n'+'='*70,flush=True)
print(' 5. 教授要件チェック',flush=True)
print('='*70,flush=True)
print(f'  WFS (annual): {ann_real:.0f}%  要件 ≥300%  {"✓" if ann_real>=300 else "✗"}',flush=True)
print(f'  MDD:          {mdd_real:+.1f}%  要件 ≤-30%  {"✓" if mdd_real>=-30 else "✗"}',flush=True)
print(f'  Bear 2022:    {b22_real:+.1f}%  要件 >0%    {"✓" if b22_real>0 else "✗"}',flush=True)
print(f'  Sharpe:       {sh_real:.2f}',flush=True)

print('\nDone.',flush=True)
