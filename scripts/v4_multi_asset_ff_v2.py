"""V4 Multi-Asset Funding Farm v2 (proper pagination)
======================================================
Binance API で複数資産の funding を full history 取得 (適切なページネーション)
"""
import sys,os,json,urllib.request,time
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import load_6assets_1h
from datetime import datetime,timezone

print('='*70,flush=True)
print('  V4: Multi-Asset Funding Farm v2',flush=True)
print('='*70,flush=True)

ASSETS=['BTC','ETH','SOL','LINK','XRP','DOGE']

def fetch_funding_history_full(symbol):
    """Use startTime+endTime properly for full history"""
    url_base='https://fapi.binance.com/fapi/v1/fundingRate'
    all_data=[]
    # Start from 2020 to now
    end_time=int(datetime.now(timezone.utc).timestamp()*1000)
    start_time=int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp()*1000)

    current_start = start_time
    pages = 0
    max_pages = 30  # 30K records max

    while current_start < end_time and pages < max_pages:
        url=f'{url_base}?symbol={symbol}&startTime={current_start}&limit=1000'
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'V4/1.0'})
            with urllib.request.urlopen(req,timeout=15) as r:
                data=json.loads(r.read())
            if not data: break
            all_data.extend(data)
            # Move start_time to last record + 1
            current_start = int(data[-1]['fundingTime']) + 1
            pages += 1
            time.sleep(0.15)
            if len(data)<1000: break  # no more data
        except Exception as e:
            print(f'  ERROR {symbol} page {pages}: {e}',flush=True)
            break
    df=pd.DataFrame(all_data)
    if len(df):
        df['fundingTime']=pd.to_datetime(df['fundingTime'],unit='ms',utc=True)
        df['fundingRate']=df['fundingRate'].astype(float)
        df=df.set_index('fundingTime').sort_index().drop_duplicates()
    return df, pages

print('Fetching full funding history for all assets...',flush=True)
funding_data={}
for a in ASSETS:
    df, pages = fetch_funding_history_full(f'{a}USDT')
    if len(df)>0:
        funding_data[a]=df
        annual_avg=df['fundingRate'].mean()*3*365*100
        print(f'  {a}USDT: {len(df)} records ({pages} pages), {df.index.min().date()} → {df.index.max().date()}, avg annual: {annual_avg:+.1f}%',flush=True)

# Find common range across all assets
common_start=max(df.index.min() for df in funding_data.values())
common_end=min(df.index.max() for df in funding_data.values())
print(f'\nCommon range: {common_start.date()} → {common_end.date()}',flush=True)

# Annual funding stats per asset (in common range)
print(f'\nAnnual funding (common range):',flush=True)
common_funding={}
for a, df in funding_data.items():
    df_c=df[(df.index>=common_start)&(df.index<=common_end)]
    if len(df_c)>0:
        common_funding[a]=df_c
        annual=df_c['fundingRate'].mean()*3*365*100
        pos_pct=(df_c['fundingRate']>0).mean()*100
        print(f'  {a}: {annual:+.1f}%/yr (positive {pos_pct:.0f}% of time, n={len(df_c)})',flush=True)

# ============================================================
# Multi-asset FF: rotate to highest funding asset each 8H
# ============================================================
print('\n'+'='*70,flush=True)
print(' MULTI-ASSET FF: ROTATION STRATEGY',flush=True)
print('='*70,flush=True)
print('  Strategy: every 8H, pick the asset with highest funding rate',flush=True)
print('  Run delta-neutral FF on that asset only',flush=True)

# Load asset returns (1H)
print('\nLoading 1H returns...',flush=True)
a1=load_6assets_1h()
common_idx=list(a1.values())[0].index
for df in a1.values():common_idx=common_idx.intersection(df.index)
common_idx=common_idx.tz_localize('UTC') if common_idx.tz is None else common_idx
# Restrict to funding data range
common_idx=common_idx[(common_idx>=common_start)&(common_idx<=common_end)]
print(f'  Common 1H bars: {len(common_idx)}',flush=True)

# Build per-asset returns
asset_rets={}
for a in ASSETS:
    if a not in a1:continue
    df=a1[a]
    df.index=df.index.tz_localize('UTC') if df.index.tz is None else df.index
    asset_rets[a]=df.loc[common_idx,'return'].fillna(0).values

# Build per-asset funding (forward-fill to 1H)
asset_funding_1h={}
for a in ASSETS:
    if a not in common_funding:continue
    df=common_funding[a]
    df_1h=df['fundingRate'].reindex(common_idx,method='ffill').fillna(0)
    asset_funding_1h[a]=df_1h.values

valid_assets=[a for a in ASSETS if a in asset_rets and a in asset_funding_1h]
print(f'  Valid assets: {valid_assets}',flush=True)

# ============================================================
# Strategy: Rotate to highest-funding asset each 8H
# ============================================================
def run_rotation_ff(lev=1.5, fee_bps=5, slip_bps=2):
    """Each 8H, rotate to asset with highest 30d funding average"""
    n=len(common_idx)
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000

    # 30d funding MA per asset
    funding_30d={}
    for a in valid_assets:
        funding_30d[a]=pd.Series(asset_funding_1h[a]).rolling(720,min_periods=168).mean().values

    capital_unit = 1+1/lev
    current_asset = None
    perp_unrealized = 0.0
    rotations = 0
    e *= (1 - 2*(fr+sr)/capital_unit)  # initial setup

    for i in range(720, n):
        # Decide which asset to be in
        if (i-720) % 8 == 0:  # every 8H
            best_asset = None
            best_funding = -999
            for a in valid_assets:
                f = funding_30d[a][i] if not np.isnan(funding_30d[a][i]) else 0
                if f > best_funding:
                    best_funding = f
                    best_asset = a

            # Negative funding → exit
            if best_funding <= 0:
                if current_asset is not None:
                    # Exit position
                    e *= max(0.001, 1 + perp_unrealized/capital_unit)
                    e *= (1 - 2*(fr+sr)/capital_unit)
                    perp_unrealized = 0
                    current_asset = None
                eq[i] = e
                continue

            # Switch asset if needed
            if best_asset != current_asset:
                if current_asset is not None:
                    # Close old position
                    e *= max(0.001, 1 + perp_unrealized/capital_unit)
                    perp_unrealized = 0
                # Open new position
                e *= (1 - 2*(fr+sr)/capital_unit)
                current_asset = best_asset
                rotations += 1

        # Apply funding farm P&L for current asset
        if current_asset is not None:
            r = asset_rets[current_asset][i]
            spot_pnl = r * 1.0
            perp_pnl = -r * 1.0  # short perp, opposite of long spot
            funding_income = asset_funding_1h[current_asset][i] * 1.0
            perp_unrealized += perp_pnl + funding_income
            # Liquidation check
            if perp_unrealized < -1/lev * 0.5:
                e *= max(0.001, 1+perp_unrealized/capital_unit)
                perp_unrealized = 0
                e *= (1 - 2*(fr+sr)/capital_unit)
                continue
            bar_return = (spot_pnl + perp_pnl + funding_income) / capital_unit
            e *= (1+bar_return)

        eq[i] = e

    return eq, rotations

def run_btc_only(lev=1.5, fee_bps=5, slip_bps=2):
    """Compare: BTC only FF"""
    n=len(common_idx)
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    capital_unit=1+1/lev
    e *= (1 - 2*(fr+sr)/capital_unit)
    perp_un=0.0
    for i in range(720, n):
        r = asset_rets['BTC'][i]
        sp=r*1.0;pp=-r*1.0
        fi=asset_funding_1h['BTC'][i]*1.0
        perp_un += pp+fi
        if perp_un<-1/lev*0.5:
            e *= max(0.001,1+perp_un/capital_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/capital_unit)
            continue
        bar_ret=(sp+pp+fi)/capital_unit
        e *= (1+bar_ret)
        eq[i]=e
    return eq

def stats(eq):
    n=len(common_idx)
    valid=eq[eq>0]
    if len(valid)<10:return -100,0,0
    years=(n-720)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(720,n):
        if eq[i]<=0:continue
        pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[720]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

eq_rot, rot = run_rotation_ff(lev=1.5, fee_bps=5, slip_bps=2)
eq_btc = run_btc_only(lev=1.5, fee_bps=5, slip_bps=2)

print(f'\n  {"Strategy":<35} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*60}',flush=True)
for label,eq in [('BTC only FF (baseline)',eq_btc),(f'Rotation FF ({rot} rotations)',eq_rot)]:
    a,m,s=stats(eq)
    tag=' ★' if a>50 else ''
    print(f'  {label:<35} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

# Year by year
print(f'\n  Year-by-year:',flush=True)
print(f'  {"Year":<6} {"BTC only":>12} {"Rotation":>12} {"Diff":>10}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=720]
    if len(iy_a)<100:continue
    if eq_btc[iy_a[0]]<=0 or eq_rot[iy_a[0]]<=0:continue
    rb=(eq_btc[iy_a[-1]]/eq_btc[max(0,iy_a[0]-1)]-1)*100
    rr=(eq_rot[iy_a[-1]]/eq_rot[max(0,iy_a[0]-1)]-1)*100
    print(f'  {yr:<6} {rb:>+11.1f}% {rr:>+11.1f}% {rr-rb:>+9.1f}%',flush=True)

# Asset utilization in rotation
print(f'\n  Asset utilization (rotation strategy):',flush=True)
asset_chosen={a:0 for a in valid_assets}
funding_30d_all={a:pd.Series(asset_funding_1h[a]).rolling(720,min_periods=168).mean().values for a in valid_assets}
none_count=0
for i in range(720,len(common_idx),8):  # every 8H
    best_a=None;best_f=-999
    for a in valid_assets:
        f=funding_30d_all[a][i] if not np.isnan(funding_30d_all[a][i]) else -999
        if f>best_f:best_f=f;best_a=a
    if best_f<=0:none_count+=1
    else:asset_chosen[best_a]+=1
total=sum(asset_chosen.values())+none_count
for a,c in sorted(asset_chosen.items(),key=lambda x:-x[1]):
    print(f'    {a}: {c} times ({c/total*100:.1f}%)',flush=True)
print(f'    NONE (negative funding): {none_count} ({none_count/total*100:.1f}%)',flush=True)

print('\nDone.',flush=True)
