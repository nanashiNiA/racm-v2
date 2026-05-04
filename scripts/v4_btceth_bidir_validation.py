"""V4: BTC+ETH Bidirectional FF 厳密検証
==========================================
1. シャッフル (200x): each asset独立
2. リーク
3. OOS分割 (Train 20-23, Test 24-25)
4. 月別詳細
5. 教授要件
"""
import sys,os,json,urllib.request,time
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import load_6assets_1h
from datetime import datetime,timezone

print('='*70,flush=True)
print('  BTC+ETH Bidirectional FF: 徹底検証',flush=True)
print('='*70,flush=True)

ASSETS=['BTC','ETH']
def fetch(symbol):
    url_base='https://fapi.binance.com/fapi/v1/fundingRate'
    all_data=[]
    current=int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp()*1000)
    for _ in range(20):
        url=f'{url_base}?symbol={symbol}&startTime={current}&limit=1000'
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'V4'})
            with urllib.request.urlopen(req,timeout=15) as r:
                data=json.loads(r.read())
            if not data:break
            all_data.extend(data)
            current=int(data[-1]['fundingTime'])+1
            time.sleep(0.15)
            if len(data)<1000:break
        except:break
    df=pd.DataFrame(all_data)
    df['fundingTime']=pd.to_datetime(df['fundingTime'],unit='ms',utc=True)
    df['fundingRate']=df['fundingRate'].astype(float)
    return df.set_index('fundingTime').sort_index().drop_duplicates()

print('Loading...',flush=True)
funding={a:fetch(f'{a}USDT') for a in ASSETS}
common_start=max(df.index.min() for df in funding.values())
common_end=min(df.index.max() for df in funding.values())

a1=load_6assets_1h()
common_idx=list(a1.values())[0].index
for df in a1.values():common_idx=common_idx.intersection(df.index)
common_idx=common_idx.tz_localize('UTC') if common_idx.tz is None else common_idx
common_idx=common_idx[(common_idx>=common_start)&(common_idx<=common_end)]

asset_rets={a:a1[a].assign(idx=lambda d:d.index.tz_localize('UTC') if d.index.tz is None else d.index).set_index('idx').loc[common_idx,'return'].fillna(0).values
            if a in a1 else None for a in ASSETS}
asset_funding={a:funding[a]['fundingRate'].reindex(common_idx,method='ffill').fillna(0).values for a in ASSETS}
n=len(common_idx);S=720
funding_30d_pct={a:pd.Series(asset_funding[a]).rolling(720,min_periods=168).mean().values*365*24*100
                  for a in ASSETS}

def single_asset_bidir(asset, fra_use=None):
    eq=np.ones(n);e=1.0
    fr=2/10000;sr=2/10000
    perp_un=0.0;current_lev=1.5;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    rets=asset_rets[asset]
    frates=fra_use if fra_use is not None else asset_funding[asset]
    f30=pd.Series(frates).rolling(720,min_periods=168).mean().values*365*24*100
    for i in range(S,n):
        f=f30[i] if not np.isnan(f30[i]) else 50
        if f>50:target_mode='short_perp';target_lev=1.5
        elif f>20:target_mode='short_perp';target_lev=1.0
        elif f>0:target_mode='neutral';target_lev=0
        elif f>-10:target_mode='neutral';target_lev=0
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
        sret=rets[i]
        if current_mode=='short_perp':
            sp=sret*1.0;pp=-sret*1.0;fi=frates[i]*1.0
        else:
            sp=0;pp=sret*1.0;fi=-frates[i]*1.0
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

def equal_weight_portfolio(eqs):
    eq=np.ones(n);e=1.0
    n_a=len(eqs)
    for i in range(S,n):
        bar_ret=0
        for a,a_eq in eqs.items():
            if i>0 and a_eq[i-1]>0:
                bar_ret += (a_eq[i]/a_eq[i-1]-1) / n_a
        e *= (1+bar_ret);eq[i]=e
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

# Real
eqs={a:single_asset_bidir(a) for a in ASSETS}
eq_real=equal_weight_portfolio(eqs)
ann_r,mdd_r,sh_r,b22_r=stats(eq_real)
print(f'\n  Real: Annual={ann_r:+.1f}%, MDD={mdd_r:+.1f}%, Sharpe={sh_r:.2f}, B22={b22_r:+.1f}%',flush=True)

# 1. Shuffle
print('\n'+'='*70,flush=True)
print(' 1. SHUFFLE TEST (200x, both assets)',flush=True)
print('='*70,flush=True)

shuf_anns=[];shuf_b22s=[]
for seed in range(200):
    np.random.seed(seed+11111)
    eqs_s={}
    for a in ASSETS:
        f_s=asset_funding[a].copy()
        np.random.shuffle(f_s[S:])
        eqs_s[a]=single_asset_bidir(a, fra_use=f_s)
    eq_s=equal_weight_portfolio(eqs_s)
    a_s,_,_,b_s=stats(eq_s)
    shuf_anns.append(a_s);shuf_b22s.append(b_s)

p_a=np.mean([a>=ann_r for a in shuf_anns])
p_b=np.mean([b>=b22_r for b in shuf_b22s])
print(f'  Annual: Real={ann_r:+.1f}%, Shuffle={np.mean(shuf_anns):+.1f}%+/-{np.std(shuf_anns):.1f}%, p={p_a:.3f} {"PASS" if p_a<0.05 else "FAIL"}',flush=True)
print(f'  Bear22: Real={b22_r:+.1f}%, Shuffle={np.mean(shuf_b22s):+.1f}%+/-{np.std(shuf_b22s):.1f}%, p={p_b:.3f} {"PASS" if p_b<0.05 else "FAIL"}',flush=True)

# 2. Leak check
print('\n'+'='*70,flush=True)
print(' 2. LEAK CHECK',flush=True)
print('='*70,flush=True)
print(f'  {"Shift":<8} {"Annual":>10}',flush=True)
for shift in [-2,-1,0,1,2,4,8]:
    eqs_l={a:single_asset_bidir(a, fra_use=np.roll(asset_funding[a],shift)) for a in ASSETS}
    eq_l=equal_weight_portfolio(eqs_l)
    a,_,_,_=stats(eq_l)
    marker=' base' if shift==0 else ''
    print(f'  {shift:+d}h {a:>+9.1f}%{marker}',flush=True)

# 3. OOS split
print('\n'+'='*70,flush=True)
print(' 3. OOS SPLIT',flush=True)
print('='*70,flush=True)
train_end=np.where(np.array([d<pd.Timestamp('2024-01-01',tz='UTC') for d in common_idx]))[0][-1]
train_eq=eq_real[S:train_end+1]
years_t=(train_end-S)/(365*24)
ann_t=((train_eq[-1]/train_eq[0])**(1/years_t)-1)*100 if train_eq[0]>0 else 0
test_eq=eq_real[train_end+1:]/eq_real[train_end] if eq_real[train_end]>0 else eq_real[train_end+1:]
years_test=(n-train_end-1)/(365*24)
ann_test=((test_eq[-1])**(1/years_test)-1)*100 if test_eq[-1]>0 else 0
print(f'  Train (2020-23): {ann_t:.1f}%/yr',flush=True)
print(f'  Test  (2024-25): {ann_test:.1f}%/yr',flush=True)
print(f'  Consistent: {"YES" if abs(ann_t-ann_test)<60 else f"DRIFT ({abs(ann_t-ann_test):.0f}%)"}',flush=True)

# 4. Monthly
print('\n'+'='*70,flush=True)
print(' 4. MONTHLY DETAIL',flush=True)
print('='*70,flush=True)
eq_s=pd.Series(eq_real,index=common_idx)
monthly=eq_s.resample('1M').last().pct_change().dropna()
monthly=monthly[monthly.index>=common_idx[S]]
print(f'  Total: {len(monthly)} months',flush=True)
print(f'  Mean: {monthly.mean()*100:+.2f}%, Std: {monthly.std()*100:.2f}%',flush=True)
print(f'  Win rate: {(monthly>0).mean()*100:.1f}%',flush=True)
print(f'  Worst: {monthly.min()*100:+.2f}% ({monthly.idxmin().strftime("%Y-%m")})',flush=True)
print(f'  Best: {monthly.max()*100:+.2f}% ({monthly.idxmax().strftime("%Y-%m")})',flush=True)

# 5. 教授要件
print('\n'+'='*70,flush=True)
print(' 5. 教授要件',flush=True)
print('='*70,flush=True)
print(f'  WFS:    {ann_r:.0f}%  要件 ≥300%  {"✓" if ann_r>=300 else "✗"}',flush=True)
print(f'  MDD:    {mdd_r:+.1f}%  要件 ≤-30%  {"✓" if mdd_r>=-30 else "✗"}',flush=True)
print(f'  Bear22: {b22_r:+.1f}%  要件 >0%    {"✓" if b22_r>0 else "✗"}',flush=True)
print(f'  Sharpe: {sh_r:.2f}',flush=True)

print('\nDone.',flush=True)
