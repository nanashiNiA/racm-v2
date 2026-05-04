"""V4: Funding Farm with Dynamic Leverage
==========================================
Idea: funding rate が高い時にレバを上げる、低い時に下げる

ロジック:
- 過去30日 funding 平均が高 (>0.02%/8H = 年22%超) → lev=2.0
- 中 (0.005-0.02%/8H) → lev=1.5
- 低 (<0.005%/8H = 年5%未満) → lev=1.0
- 負 → exit (lev=0, just spot only)

期待効果:
- Bear 2022 (低funding) で過剰レバを避ける → MDD改善
- Bull 2021/2024 (高funding) でフルレバ → リターン増
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4: Funding Farm with Dynamic Leverage',flush=True)
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
bybit=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
bybit=bybit.drop_duplicates('timestamp').set_index('timestamp')

h_racm=build_common_1h()
h_racm.index=h_racm.index.tz_localize('UTC') if h_racm.index.tz is None else h_racm.index
common_idx=bybit.index.intersection(h_racm.index)
bybit_a=bybit.loc[common_idx]
spot_a=h_racm.loc[common_idx]

perp_price=bybit_a['close'].values
spot_price=spot_a['close'].values
fra=np.roll(bybit_a['funding_rate'].fillna(0).values,1)

n=len(common_idx)
S=2760

# 30日 rolling funding average for dynamic decision
fra_ma_30d=pd.Series(fra).rolling(720,min_periods=168).mean().values

# Annual funding rate equivalent
fra_annual_pct=fra_ma_30d*365*24*100  # convert per-hour to per-year

print(f'  funding 30d MA distribution:',flush=True)
fra_active=fra_annual_pct[S:]
for pct in [10,25,50,75,90]:
    print(f'    p{pct}: {np.percentile(fra_active,pct):+.0f}%/yr',flush=True)

# ============================================================
# Dynamic FF
# ============================================================
def run_ff_dynamic(use_dynamic=True, fee_bps=2, slip_bps=2, rebal_days=7):
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0
    bars_per_rebal=rebal_days*24
    current_lev=1.5  # initial
    last_lev=1.5
    rebalances=0

    for i in range(S,n):
        # Decide leverage
        if use_dynamic:
            f30 = fra_annual_pct[i] if not np.isnan(fra_annual_pct[i]) else 50
            if f30 > 100:    # very high funding (>100%/yr) → max leverage
                target_lev = 2.0
            elif f30 > 50:   # high → lev 1.5
                target_lev = 1.5
            elif f30 > 20:   # moderate → lev 1.0
                target_lev = 1.0
            elif f30 > 0:    # low positive → lev 0.5
                target_lev = 0.5
            else:            # negative → exit perp, hold spot only
                target_lev = 0.001  # near zero
        else:
            target_lev = 1.5

        # Rebalance leverage at funding intervals
        if (i-S)%bars_per_rebal==0 and (i-S)>0 and abs(target_lev-current_lev)>0.1:
            # Adjust position size cost
            adj_cost = abs(target_lev-current_lev)/current_lev * (fr+sr) * 2
            cap_unit = 1+1/current_lev
            e *= (1 - adj_cost/cap_unit)
            current_lev = target_lev
            perp_un = 0  # reset on rebal
            rebalances += 1

        cap_unit = 1+1/current_lev

        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0

        sp=sret*1.0
        pp=-pret*1.0
        fi=fra[i]*1.0

        perp_un += pp+fi

        if perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue

        bar_ret=(sp+pp+fi)/cap_unit
        e *= (1+bar_ret)
        eq[i]=e

    return eq, rebalances

def stats(eq):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

eq_static, _ = run_ff_dynamic(use_dynamic=False)
eq_dynamic, n_rebal = run_ff_dynamic(use_dynamic=True)

print(f'\n  {"Strategy":<30} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*55}',flush=True)
for label,eq in [('Static FF 1.5x',eq_static),(f'Dynamic FF (rebal {n_rebal})',eq_dynamic)]:
    a,m,s=stats(eq)
    print(f'  {label:<30} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}',flush=True)

# Year by year
print(f'\n  Year-by-year:',flush=True)
print(f'  {"Year":<6} {"Static":>10} {"Dynamic":>10} {"Diff":>10} {"Avg Funding":>12}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    rs=(eq_static[iy_a[-1]]/eq_static[max(0,iy_a[0]-1)]-1)*100
    rd=(eq_dynamic[iy_a[-1]]/eq_dynamic[max(0,iy_a[0]-1)]-1)*100
    avgf=np.mean(fra_annual_pct[iy_a])
    print(f'  {yr:<6} {rs:>+9.1f}% {rd:>+9.1f}% {rd-rs:>+9.1f}% {avgf:>+10.0f}%',flush=True)

# ============================================================
# Multiple dynamic strategies
# ============================================================
print('\n'+'='*70,flush=True)
print(' Dynamic Strategy Variations',flush=True)
print('='*70,flush=True)

def run_ff_variant(thresholds, lev_levels, fee_bps=2, slip_bps=2, rebal_days=7):
    """thresholds: [low, mid, high] in %/yr; lev_levels: [vlow, low, mid, high]"""
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=lev_levels[2]
    bars_per_rebal=rebal_days*24
    for i in range(S,n):
        f30=fra_annual_pct[i] if not np.isnan(fra_annual_pct[i]) else 50
        if f30>thresholds[2]:target_lev=lev_levels[3]
        elif f30>thresholds[1]:target_lev=lev_levels[2]
        elif f30>thresholds[0]:target_lev=lev_levels[1]
        else:target_lev=lev_levels[0]

        if (i-S)%bars_per_rebal==0 and (i-S)>0 and abs(target_lev-current_lev)>0.1:
            adj=abs(target_lev-current_lev)/max(current_lev,0.1)*(fr+sr)*2
            cap=1+1/max(current_lev,0.1)
            e *= (1-adj/cap)
            current_lev=target_lev
            perp_un=0

        cap=1+1/max(current_lev,0.1)
        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0
        sp=sret*1.0;pp=-pret*1.0;fi=fra[i]*1.0
        perp_un += pp+fi
        if current_lev>0.01 and perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap)
            perp_un=0
            e *= (1-2*(fr+sr)/cap)
            continue
        bar_ret=(sp+pp+fi)/cap
        e *= (1+bar_ret)
        eq[i]=e
    return eq

variants=[
    ('Aggressive (10/30/100, 0.001/1/2/3)', [10,30,100], [0.001,1.0,2.0,3.0]),
    ('Moderate (20/50/150, 0.5/1/1.5/2.5)', [20,50,150], [0.5,1.0,1.5,2.5]),
    ('Conservative (30/60/120, 1/1/1.5/2)', [30,60,120], [1.0,1.0,1.5,2.0]),
    ('Bear-aware (0/40/80, 0/1/1.5/2)', [0,40,80], [0.001,1.0,1.5,2.0]),
    ('Static 1.0x', [0,0,0], [1.0,1.0,1.0,1.0]),
    ('Static 1.5x (baseline)', [0,0,0], [1.5,1.5,1.5,1.5]),
    ('Static 2.0x', [0,0,0], [2.0,2.0,2.0,2.0]),
]

print(f'  {"Variant":<45} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*72}',flush=True)
for label,t,l in variants:
    eq=run_ff_variant(t,l)
    a,m,s=stats(eq)
    tag=' ★' if a>52 and m>=-7 else ''
    print(f'  {label:<45} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

print('\nDone.',flush=True)
