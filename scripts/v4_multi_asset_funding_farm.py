"""V4 Multi-Asset Funding Farm
================================
複数資産で並行 funding farm:
- BTC, ETH, SOL, LINK, XRP, DOGE
- 各資産の funding rate を別個に獲得
- Basis variation の分散効果

データ取得:
- Binance Futures API で historical funding rate (各資産)
- 各資産の spot price は v2core.data_loader から
"""
import sys,os,json,urllib.request,time
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import load_6assets_8h,load_6assets_1h
from datetime import datetime,timezone

print('='*70,flush=True)
print('  V4: Multi-Asset Funding Farm',flush=True)
print('='*70,flush=True)

# ============================================================
# Fetch funding rates for multiple assets
# ============================================================
ASSETS=['BTC','ETH','SOL','LINK','XRP','DOGE']
print('Fetching historical funding rates...',flush=True)

def fetch_funding_history(symbol):
    """Fetch all historical funding rates for symbol"""
    url_base='https://fapi.binance.com/fapi/v1/fundingRate'
    all_data=[]
    end_time=int(datetime.now(timezone.utc).timestamp()*1000)
    for _ in range(20):  # max 20 pages × 1000 = 20K records
        url=f'{url_base}?symbol={symbol}&limit=1000&endTime={end_time}'
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'V4/1.0'})
            with urllib.request.urlopen(req,timeout=15) as r:
                data=json.loads(r.read())
            if not data: break
            all_data.extend(data)
            end_time=int(data[0]['fundingTime'])-1
            time.sleep(0.1)
        except Exception as e:
            print(f'  ERROR fetching {symbol}: {e}',flush=True)
            break
    df=pd.DataFrame(all_data)
    if len(df):
        df['fundingTime']=pd.to_datetime(df['fundingTime'],unit='ms',utc=True)
        df['fundingRate']=df['fundingRate'].astype(float)
        df=df.set_index('fundingTime').sort_index().drop_duplicates()
    return df

funding_data={}
for a in ASSETS:
    print(f'  {a}USDT...',end='',flush=True)
    df=fetch_funding_history(f'{a}USDT')
    if len(df)>0:
        funding_data[a]=df
        annual=df['fundingRate'].mean()*3*365*100
        print(f' {len(df)} records, mean annual funding: {annual:+.1f}%',flush=True)
    else:
        print(' FAILED',flush=True)

# ============================================================
# Get common time range
# ============================================================
if not funding_data:
    print('No funding data, exit',flush=True)
    sys.exit(1)

# Use intersection of all asset time ranges
common_start=max(df.index.min() for df in funding_data.values())
common_end=min(df.index.max() for df in funding_data.values())
print(f'\n  Common range: {common_start} → {common_end}',flush=True)

# Annual funding by asset
print(f'\n  Annual funding rate by asset (mean):',flush=True)
for a in ASSETS:
    if a not in funding_data:continue
    df=funding_data[a]
    df_common=df[(df.index>=common_start)&(df.index<=common_end)]
    annual=df_common['fundingRate'].mean()*3*365*100
    pos_pct=(df_common['fundingRate']>0).mean()*100
    print(f'    {a}USDT: {annual:+.1f}%/yr (positive {pos_pct:.0f}% of time, n={len(df_common)})',flush=True)

# ============================================================
# Multi-Asset Funding Farm Simulation
# ============================================================
print('\n'+'='*70,flush=True)
print(' MULTI-ASSET FUNDING FARM SIMULATION',flush=True)
print('='*70,flush=True)

# Load 1H asset returns for spot side
print('Loading 1H asset returns...',flush=True)
a1=load_6assets_1h()
common_idx=list(a1.values())[0].index
for df in a1.values():common_idx=common_idx.intersection(df.index)
common_idx=common_idx.tz_localize('UTC') if common_idx.tz is None else common_idx

# Restrict to funding data range
common_idx=common_idx[(common_idx>=common_start)&(common_idx<=common_end)]
print(f'  Common 1H bars: {len(common_idx)}',flush=True)

# Build spot returns matrix
asset_rets={}
for a in ASSETS:
    if a not in a1:continue
    df=a1[a]
    df.index=df.index.tz_localize('UTC') if df.index.tz is None else df.index
    asset_rets[a]=df.loc[common_idx,'return'].fillna(0).values

# Build funding rate matrix (interpolate to 1H)
asset_funding={}
for a in ASSETS:
    if a not in funding_data:continue
    df=funding_data[a]
    # Reindex to 1H, forward fill (funding paid every 8H)
    df_1h=df['fundingRate'].reindex(common_idx,method='ffill').fillna(0)
    # Convert 8H rate to per-bar (per 1H), since funding is applied at the 8H boundary
    # But we'll apply it at each 8H mark as a discrete payment
    asset_funding[a]=df_1h.values

print(f'  Assets with both data: {[a for a in ASSETS if a in asset_rets and a in asset_funding]}',flush=True)

# ============================================================
# Run multi-asset funding farm
# ============================================================
def run_multi_asset_ff(assets, equal_weight=True, lev=1.5,
                        spot_fee=5, perp_fee=0, slip=2):
    """
    Run funding farm on multiple assets simultaneously.
    Total capital split equally across assets.
    Each asset: Spot Long + Perp Short delta neutral.
    """
    n=len(common_idx)
    eq=np.ones(n);e=1.0
    fee_rate=(spot_fee+perp_fee)/10000
    slip_rate=slip/10000

    n_assets=len(assets)
    capital_per_asset=1.0/n_assets

    # Per-asset: capital_per_asset = (1 + 1/lev) × position_per_asset
    # → position_per_asset = capital_per_asset / (1 + 1/lev)
    pos_per_asset=capital_per_asset/(1+1/lev)

    # Initial setup: 2 trades per asset
    e *= (1 - 2*(fee_rate+slip_rate)*n_assets)

    # Track perp unrealized per asset
    perp_unrealized={a:0 for a in assets}

    # Funding paid at 00:00, 08:00, 16:00 UTC (Binance convention)
    funding_hours=[0,8,16]

    for i in range(720, n):  # warmup 30 days
        # Per-asset P&L
        total_bar_return=0
        for a in assets:
            if a not in asset_rets or a not in asset_funding:continue
            r=asset_rets[a][i]
            # Spot long + Perp short = delta neutral (assuming spot ≈ perp)
            spot_pnl=r*pos_per_asset
            perp_pnl=-r*pos_per_asset

            # Funding income at 8H boundaries
            ts=common_idx[i]
            funding_income=0
            if ts.hour in funding_hours:
                # Apply 8H funding rate
                funding_income=asset_funding[a][i]*pos_per_asset

            perp_unrealized[a]+=perp_pnl+funding_income
            total_bar_return+=spot_pnl+perp_pnl+funding_income

        e *= (1+total_bar_return)
        eq[i]=e

    return eq

# Run with various asset combinations
print(f'\n  Multi-asset combinations (lev=1.5x):',flush=True)
print(f'  {"Assets":<25} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*55}',flush=True)

valid_assets=[a for a in ASSETS if a in asset_rets and a in asset_funding]
combinations=[
    (['BTC'], 'BTC only'),
    (['BTC','ETH'], 'BTC+ETH'),
    (['BTC','ETH','SOL'], 'BTC+ETH+SOL'),
    (['BTC','ETH','SOL','LINK'], '4-asset'),
    (valid_assets, f'All {len(valid_assets)} assets'),
]

def stats_eq(eq):
    n_active=np.sum(eq>0)
    if n_active==0:return -100,0,0
    valid=eq[eq>0]
    if len(valid)<10:return -100,0,0
    years=len(valid)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(720,len(eq)):
        if eq[i]<=0:continue
        pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[720]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

for assets,label in combinations:
    if not all(a in asset_rets for a in assets):continue
    eq=run_multi_asset_ff(assets,lev=1.5)
    a,m,s=stats_eq(eq)
    tag=' ★' if a>50 and m>=-10 else ''
    print(f'  {label:<25} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

# ============================================================
# Year-by-year multi-asset
# ============================================================
print('\n  Year-by-year (BTC+ETH+SOL+LINK, lev=1.5x):',flush=True)
eq_4=run_multi_asset_ff(['BTC','ETH','SOL','LINK'],lev=1.5)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=720]
    if len(iy_a)<100:continue
    if eq_4[iy_a[0]]<=0 or eq_4[iy_a[-1]]<=0:continue
    r=(eq_4[iy_a[-1]]/eq_4[max(0,iy_a[0]-1)]-1)*100
    print(f'    {yr}: {r:+.1f}%',flush=True)

# ============================================================
# Compare BTC only vs Multi-asset
# ============================================================
print('\n'+'='*70,flush=True)
print(' BTC only vs Multi-asset',flush=True)
print('='*70,flush=True)
eq_btc=run_multi_asset_ff(['BTC'],lev=1.5)
eq_multi=run_multi_asset_ff(valid_assets,lev=1.5)
a_b,m_b,s_b=stats_eq(eq_btc)
a_m,m_m,s_m=stats_eq(eq_multi)
print(f'  BTC only:    {a_b:+.1f}% annual, MDD {m_b:+.1f}%, Sharpe {s_b:.2f}',flush=True)
print(f'  Multi-asset: {a_m:+.1f}% annual, MDD {m_m:+.1f}%, Sharpe {s_m:.2f}',flush=True)
print(f'  Difference: {a_m-a_b:+.1f}%/year',flush=True)

print('\nDone.',flush=True)
