"""V4: Multi-Asset Parallel Bidirectional FF
=============================================
複数資産で同時にBidirectional FFを並行運用
- BTC, ETH, SOL, LINK, XRP, DOGE
- 各資産で独立に bidirectional 判断
- Capital を均等分割 (1/6 each)
- 各資産の funding cycle 違いで分散効果

期待:
- BTC が neutral 期間も他の資産で運用できる
- リスク分散
- より高い annual return
"""
import sys,os,json,urllib.request,time
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import load_6assets_1h
from datetime import datetime,timezone

print('='*70,flush=True)
print('  V4: Multi-Asset Parallel Bidirectional FF',flush=True)
print('='*70,flush=True)

ASSETS=['BTC','ETH','SOL','LINK','XRP','DOGE']

def fetch_funding(symbol):
    url_base='https://fapi.binance.com/fapi/v1/fundingRate'
    all_data=[]
    end_time=int(datetime.now(timezone.utc).timestamp()*1000)
    start_time=int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp()*1000)
    current=start_time
    for _ in range(20):
        url=f'{url_base}?symbol={symbol}&startTime={current}&limit=1000'
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'V4/1.0'})
            with urllib.request.urlopen(req,timeout=15) as r:
                data=json.loads(r.read())
            if not data:break
            all_data.extend(data)
            current=int(data[-1]['fundingTime'])+1
            time.sleep(0.15)
            if len(data)<1000:break
        except:break
    df=pd.DataFrame(all_data)
    if len(df):
        df['fundingTime']=pd.to_datetime(df['fundingTime'],unit='ms',utc=True)
        df['fundingRate']=df['fundingRate'].astype(float)
        df=df.set_index('fundingTime').sort_index().drop_duplicates()
    return df

print('Loading...',flush=True)
funding={}
for a in ASSETS:
    df=fetch_funding(f'{a}USDT')
    if len(df)>0:funding[a]=df

common_start=max(df.index.min() for df in funding.values())
common_end=min(df.index.max() for df in funding.values())
print(f'  Funding: {len(funding)} assets, common {common_start.date()} → {common_end.date()}',flush=True)

a1=load_6assets_1h()
common_idx=list(a1.values())[0].index
for df in a1.values():common_idx=common_idx.intersection(df.index)
common_idx=common_idx.tz_localize('UTC') if common_idx.tz is None else common_idx
common_idx=common_idx[(common_idx>=common_start)&(common_idx<=common_end)]
print(f'  Common 1H: {len(common_idx)}',flush=True)

asset_rets={}
asset_funding_1h={}
for a in ASSETS:
    if a in a1 and a in funding:
        df=a1[a]
        df.index=df.index.tz_localize('UTC') if df.index.tz is None else df.index
        asset_rets[a]=df.loc[common_idx,'return'].fillna(0).values
        asset_funding_1h[a]=funding[a]['fundingRate'].reindex(common_idx,method='ffill').fillna(0).values

valid_assets=list(asset_rets.keys())
# Filter: only assets with stable bidirectional behavior
# SOL/LINK/XRP/DOGE blow up due to high volatility on inverse FF
SAFE_ASSETS=['BTC','ETH']
valid_assets=[a for a in SAFE_ASSETS if a in asset_rets]
print(f'  Using safe assets only: {valid_assets}',flush=True)

n=len(common_idx)
S=720

# Funding 30d MA per asset (in % per year)
funding_30d={a:pd.Series(asset_funding_1h[a]).rolling(720,min_periods=168).mean().values*365*24*100
             for a in valid_assets}

# ============================================================
# Single-asset Bidirectional FF runner
# ============================================================
def single_asset_bidir(asset, capital_share=1.0, fee_bps=2, slip_bps=2, rebal_days=7):
    """Run Bidirectional FF on single asset, returns equity array"""
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=1.5;current_mode='neutral'
    cap_unit=1+1/current_lev
    e *= (1-2*(fr+sr)/cap_unit)
    rets=asset_rets[asset];frates=asset_funding_1h[asset]
    f30=funding_30d[asset]
    for i in range(S,n):
        f=f30[i] if not np.isnan(f30[i]) else 50
        if f>50:target_mode='short_perp';target_lev=1.5
        elif f>20:target_mode='short_perp';target_lev=1.0
        elif f>0:target_mode='neutral';target_lev=0
        elif f>-10:target_mode='neutral';target_lev=0
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

print('\nRunning per-asset bidirectional FF...',flush=True)
asset_eqs={}
for a in valid_assets:
    eq=single_asset_bidir(a)
    asset_eqs[a]=eq
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    print(f'  {a}: Annual={annual:+.1f}%, MDD={mdd*100:+.1f}%',flush=True)

# ============================================================
# Equal-weight portfolio
# ============================================================
print('\n'+'='*70,flush=True)
print(' EQUAL-WEIGHT PORTFOLIO',flush=True)
print('='*70,flush=True)

# Each asset gets 1/N of capital
n_assets=len(valid_assets)
print(f'  Capital split: 1/{n_assets} per asset = {100/n_assets:.1f}%',flush=True)

eq_port=np.ones(n);e=1.0
for i in range(S,n):
    bar_ret=0
    for a in valid_assets:
        if i>0 and asset_eqs[a][i-1]>0:
            ar=asset_eqs[a][i]/asset_eqs[a][i-1]-1
            bar_ret += ar / n_assets
    e *= (1+bar_ret);eq_port[i]=e

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

a,m,s,b=stats(eq_port)
print(f'\n  Equal-weight Multi-Asset:')
print(f'    Annual: {a:+.1f}%',flush=True)
print(f'    MDD: {m:+.1f}%',flush=True)
print(f'    Sharpe: {s:.2f}',flush=True)
print(f'    Bear 2022: {b:+.1f}%',flush=True)

# Compare with BTC only
a_btc,m_btc,s_btc,b_btc=stats(asset_eqs['BTC'])
print(f'\n  BTC only Bidirectional:')
print(f'    Annual: {a_btc:+.1f}%, MDD: {m_btc:+.1f}%, Sharpe: {s_btc:.2f}, B22: {b_btc:+.1f}%',flush=True)

print(f'\n  Improvement: {a-a_btc:+.1f}%/yr, MDD: {m-m_btc:+.1f}%, Sharpe diff: {s-s_btc:+.2f}',flush=True)

# Year by year
print(f'\n  Year-by-year (Multi-Asset):',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    if eq_port[max(0,iy_a[0]-1)]<=0:continue
    r=(eq_port[iy_a[-1]]/eq_port[max(0,iy_a[0]-1)]-1)*100
    print(f'    {yr}: {r:+.1f}%',flush=True)

# Asset utilization
print(f'\n  Per-asset annual contribution:',flush=True)
for a in valid_assets:
    aa,_,_,_=stats(asset_eqs[a])
    print(f'    {a}: {aa:+.1f}%/yr (×1/{n_assets} weight = {aa/n_assets:+.1f}% to portfolio)',flush=True)

print('\nDone.',flush=True)
