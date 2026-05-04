"""V4: Funding Spread Arbitrage
================================
複数資産の funding rate 差を捕獲する arbitrage 戦略

ロジック:
- 高funding資産で perp short (受け取る)
- 低funding資産で perp long (払う、または受け取る)
- ベータ調整: 同等の market exposure
- Net delta = 0 if same beta
- Net P&L = funding spread (高 - 低)

例:
- BTC funding +50%/yr, ETH funding +20%/yr
- → Short BTC perp $1, Long ETH perp $1*β (β=ETH beta to BTC)
- → Capture 30%/yr funding difference
"""
import sys,os,json,urllib.request,time
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import load_6assets_1h
from datetime import datetime,timezone

print('='*70,flush=True)
print('  V4: Funding Spread Arbitrage',flush=True)
print('='*70,flush=True)

# Re-fetch funding
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

print('Fetching funding...',flush=True)
funding={}
for a in ASSETS:
    print(f'  {a}USDT...',end='',flush=True)
    df=fetch_funding(f'{a}USDT')
    if len(df)>0:
        funding[a]=df
        print(f' {len(df)} records',flush=True)

# Common range
common_start=max(df.index.min() for df in funding.values())
common_end=min(df.index.max() for df in funding.values())
print(f'\n  Common: {common_start.date()} → {common_end.date()}',flush=True)

# Load 1H returns
print('Loading returns...',flush=True)
a1=load_6assets_1h()
common_idx=list(a1.values())[0].index
for df in a1.values():common_idx=common_idx.intersection(df.index)
common_idx=common_idx.tz_localize('UTC') if common_idx.tz is None else common_idx
common_idx=common_idx[(common_idx>=common_start)&(common_idx<=common_end)]
print(f'  Common 1H: {len(common_idx)}',flush=True)

# Get rets and funding aligned
asset_rets={}
asset_funding_1h={}
for a in ASSETS:
    if a in a1 and a in funding:
        df=a1[a]
        df.index=df.index.tz_localize('UTC') if df.index.tz is None else df.index
        asset_rets[a]=df.loc[common_idx,'return'].fillna(0).values
        # Funding interpolated to 1H
        asset_funding_1h[a]=funding[a]['fundingRate'].reindex(common_idx,method='ffill').fillna(0).values

valid_assets=list(asset_rets.keys())
print(f'  Valid: {valid_assets}',flush=True)

# ============================================================
# Compute beta of each asset to BTC
# ============================================================
print('\n'+'='*70,flush=True)
print(' BETAS to BTC',flush=True)
print('='*70,flush=True)
btc_ret=asset_rets['BTC']
betas={}
for a in valid_assets:
    if a == 'BTC':betas[a]=1.0;continue
    cov=np.cov(asset_rets[a], btc_ret)[0,1]
    var=np.var(btc_ret)
    beta=cov/(var+1e-10)
    betas[a]=beta
    print(f'  {a} beta to BTC: {beta:.3f}',flush=True)

# ============================================================
# Funding spread strategy
# ============================================================
print('\n'+'='*70,flush=True)
print(' FUNDING SPREAD ARBITRAGE',flush=True)
print('='*70,flush=True)

n=len(common_idx)
S=720  # 30 day warmup

# 30d funding average per asset (re-aligned)
funding_30d={a:pd.Series(asset_funding_1h[a]).rolling(720,min_periods=168).mean().values
             for a in valid_assets}

def run_spread_arb(rebal_hours=24, fee_bps=2):
    """
    At each rebal:
    - Find highest funding asset → short
    - Find lowest funding asset → long (beta-adjusted)
    - Hold until next rebal
    """
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000
    short_a=None;long_a=None
    spread_count=0
    rebal_count=0

    for i in range(S, n):
        if (i-S) % rebal_hours == 0:
            # Rank assets by funding
            ranks=[]
            for a in valid_assets:
                f=funding_30d[a][i] if not np.isnan(funding_30d[a][i]) else 0
                ranks.append((f,a))
            ranks.sort(reverse=True)
            new_short=ranks[0][1] if ranks[0][0]>0 else None
            new_long=ranks[-1][1] if ranks[-1][0]<ranks[0][0] else None
            spread=ranks[0][0]-ranks[-1][0]

            # Only enter if spread is large enough
            if spread > 30:  # 30%/yr spread minimum
                if new_short != short_a or new_long != long_a:
                    # Rebalance cost: 4 trades (close 2 + open 2)
                    n_changes=int(new_short!=short_a)+int(new_long!=long_a)
                    e *= (1 - n_changes*2*fr)  # 2 trades per change
                    short_a=new_short;long_a=new_long
                    rebal_count+=1
                spread_count+=1
            else:
                # No spread, exit
                if short_a is not None or long_a is not None:
                    e *= (1 - 2*fr)
                    short_a=None;long_a=None

        # Apply per-bar P&L
        if short_a and long_a:
            # Short return = -short_ret
            short_pnl = -asset_rets[short_a][i]
            # Long return adjusted by beta to match short exposure
            beta_ratio = betas[short_a] / betas[long_a] if betas[long_a] > 0.1 else 1.0
            beta_ratio = np.clip(beta_ratio, 0.5, 2.0)  # cap
            long_pnl = asset_rets[long_a][i] * beta_ratio
            # Funding income (lag-1 already)
            short_funding = asset_funding_1h[short_a][i]  # short receives positive
            long_funding = -asset_funding_1h[long_a][i] * beta_ratio  # long pays
            bar_ret = short_pnl + long_pnl + short_funding + long_funding
            e *= (1 + bar_ret * 0.5)  # half size to limit risk

        eq[i] = e

    return eq, spread_count, rebal_count

def stats(eq):
    n_eq=len(common_idx)
    years=(n_eq-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n_eq):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

print(f'\n  {"Rebal Period":<15} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"Spread bars":>12}',flush=True)
print(f'  {"-"*60}',flush=True)
for rebal in [8, 24, 168, 720]:
    eq, sp, rb = run_spread_arb(rebal_hours=rebal)
    a,m,s = stats(eq)
    label=f'{rebal}h'
    if rebal==24:label+=' (daily)'
    elif rebal==168:label+=' (weekly)'
    elif rebal==720:label+=' (monthly)'
    print(f'  {label:<15} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {sp:>11}',flush=True)

# Funding rate distribution analysis
print('\n'+'='*70,flush=True)
print(' SPREAD MAGNITUDE ANALYSIS',flush=True)
print('='*70,flush=True)
spreads=[]
for i in range(S,n,24):
    fundings=[funding_30d[a][i] for a in valid_assets if not np.isnan(funding_30d[a][i])]
    if len(fundings)>=2:
        spreads.append(max(fundings)-min(fundings))
spreads=np.array(spreads)
if len(spreads)>0:
    print(f'  Daily max spread (annualized %):',flush=True)
    print(f'    Mean: {np.mean(spreads):.1f}%/yr',flush=True)
    print(f'    Median: {np.median(spreads):.1f}%/yr',flush=True)
    print(f'    p25: {np.percentile(spreads,25):.1f}%/yr',flush=True)
    print(f'    p75: {np.percentile(spreads,75):.1f}%/yr',flush=True)
    print(f'    p95: {np.percentile(spreads,95):.1f}%/yr',flush=True)

print('\nDone.',flush=True)
