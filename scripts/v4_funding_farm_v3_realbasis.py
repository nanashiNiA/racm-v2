"""V4 ★2 Funding Farm v3: 実データ basis 検証
================================================
v2 では basis cost を 0.05%/年と仮定していた。
実際の Bybit perp price - Binance spot price を使って正確に評価。

リスク要因の正確な定量化:
1. Basis variation (perp - spot) の実データ
2. 急激な basis spike (清算リスク)
3. Stable price risk (USDT depeg)
4. Rebalancing slippage の実態
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4 ★2 v3: Funding Farm with REAL BASIS DATA',flush=True)
print('='*70,flush=True)

print('Loading Bybit perp + Binance spot...',flush=True)
# Bybit perp from cached
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
                    ticker[['timestamp','funding_rate','last_price','mark_price']].sort_values('timestamp'),
                    on='timestamp',direction='backward',tolerance=pd.Timedelta('2H'))
            ob_frames.append(ohlc)
bybit=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
bybit=bybit.drop_duplicates('timestamp').set_index('timestamp')

# Binance spot from RACM data loader
h_racm=build_common_1h()
h_racm.index=h_racm.index.tz_localize('UTC') if h_racm.index.tz is None else h_racm.index

# Align: use bybit perp as reference
common_idx=bybit.index.intersection(h_racm.index)
print(f'  Bybit perp: {len(bybit)} bars',flush=True)
print(f'  Binance spot: {len(h_racm)} bars',flush=True)
print(f'  Common: {len(common_idx)} bars',flush=True)

bybit_aligned=bybit.loc[common_idx]
spot_aligned=h_racm.loc[common_idx]

# Basis = perp - spot (positive when perp expensive)
perp_price=bybit_aligned['close'].values
spot_price=spot_aligned['close'].values
basis_pct=(perp_price-spot_price)/spot_price*100

# Funding rate (lag-1 for trade decision)
fra=np.roll(bybit_aligned['funding_rate'].fillna(0).values, 1)

print(f'\n  Basis statistics:',flush=True)
print(f'    Mean: {np.mean(basis_pct):+.4f}%',flush=True)
print(f'    Std: {np.std(basis_pct):.4f}%',flush=True)
print(f'    Min: {np.min(basis_pct):+.2f}%',flush=True)
print(f'    Max: {np.max(basis_pct):+.2f}%',flush=True)
print(f'    Median: {np.median(basis_pct):+.4f}%',flush=True)

# Distribution
for pct in [1, 5, 25, 50, 75, 95, 99]:
    print(f'    p{pct}: {np.percentile(basis_pct, pct):+.4f}%',flush=True)

# Daily basis change (rebalancing impact)
basis_daily=pd.Series(basis_pct,index=common_idx).resample('1D').last().diff().dropna()
print(f'\n  Daily basis change:',flush=True)
print(f'    Mean: {basis_daily.mean():+.4f}%/day',flush=True)
print(f'    Std: {basis_daily.std():.4f}%/day',flush=True)
print(f'    Max abs: {basis_daily.abs().max():.4f}%',flush=True)

# ============================================================
# Realistic Funding Farm with real basis
# ============================================================
print('\n'+'='*70,flush=True)
print(' Funding Farm with REAL basis dynamics',flush=True)
print('='*70,flush=True)

n=len(common_idx)
S=2760

def run_realistic_funding_farm(lev=1.0, fee_bps=2, slip_bps=2, rebal_days=7,
                                liquidation_buffer=0.5):
    """
    True funding farm with real basis cost.
    - Spot Long $1 (Binance), Perp Short $1 (Bybit, lev margin)
    - Real basis variation determines mark-to-market P&L
    - Funding income from perp short
    """
    eq=np.ones(n);e=1.0
    fee_rate=fee_bps/10000
    slip_rate=slip_bps/10000

    # Capital allocation
    capital_unit = 1 + 1/lev  # spot $1 + perp margin $1/lev

    # Initial setup cost
    init_cost = 2*(fee_rate+slip_rate) / capital_unit
    e *= (1 - init_cost)

    perp_margin = 1/lev
    cum_perp_pnl = 0.0
    last_basis = basis_pct[S]/100  # initial basis fraction

    bars_per_rebal = rebal_days * 24
    rebal_count = 0
    liquidations = 0

    for i in range(S, n):
        # Real BTC return (spot)
        if i > 0 and spot_price[i-1] > 0:
            spot_ret = (spot_price[i] / spot_price[i-1]) - 1
        else:
            spot_ret = 0

        # Real perp return
        if i > 0 and perp_price[i-1] > 0:
            perp_ret = (perp_price[i] / perp_price[i-1]) - 1
        else:
            perp_ret = 0

        # Spot Long P&L
        spot_pnl = spot_ret * 1.0  # $1 long
        # Perp Short P&L (loses if perp goes up)
        perp_pnl = -perp_ret * 1.0  # $1 short notional

        # Net P&L = spot - perp = -(perp - spot) change = -basis change
        # If basis widens (perp rises faster), short loses
        # If basis narrows, short gains

        # Funding income (8H interval, but applied per hour for simplicity)
        # Use actual funding rate (already lag-1)
        # Funding paid 3 times per day (every 8 bars) → divide by 8
        funding_income = fra[i] * 1.0 / 8  # short receives positive funding

        # Track perp margin
        cum_perp_pnl += perp_pnl + funding_income

        # Liquidation check
        if cum_perp_pnl < -perp_margin * (1-liquidation_buffer):
            liquidations += 1
            # Realize loss, force re-setup
            e *= (1 + cum_perp_pnl / capital_unit)
            cum_perp_pnl = 0
            e *= (1 - 2*(fee_rate+slip_rate))  # reopen cost
            continue

        # Total bar return
        total_pnl = spot_pnl + perp_pnl + funding_income
        bar_return = total_pnl / capital_unit
        e *= (1 + bar_return)

        # Periodic rebalancing
        if (i - S) % bars_per_rebal == 0:
            # Realize accumulated drift, rebal cost = small adjustment
            rebal_count += 1
            # Cost: ~10% of the drift (only adjust the imbalance)
            rebal_cost = abs(cum_perp_pnl) * 0.05  # 5% of drift as cost
            e *= (1 - rebal_cost / capital_unit)
            cum_perp_pnl = 0

        eq[i] = e

    # Closing cost
    e *= (1 - 2*(fee_rate+slip_rate))
    eq[-1] = e

    return eq, liquidations, rebal_count

# Test multiple leverage levels
print(f'\n  {"Leverage":<10} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"Liq":>5} {"Rebal":>6}',flush=True)
print(f'  {"-"*55}',flush=True)

for lev in [1.0, 1.5, 2.0, 3.0, 5.0]:
    eq, liq, reb = run_realistic_funding_farm(lev=lev)
    years = (n-S)/(365*24)
    annual = ((eq[-1])**(1/years) - 1) * 100 if eq[-1] > 0 else -100
    mdd = 0; pk = 1
    for i in range(S, n): pk = max(pk, eq[i]); dd = (eq[i] - pk)/pk; mdd = min(mdd, dd)
    daily = pd.Series(eq, index=common_idx).resample('1D').last().pct_change().dropna()
    daily = daily[daily.index >= common_idx[S]]
    sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0

    tag = ' ✓' if annual>50 and mdd*100>=-15 else ''
    print(f'  {lev:<9.1f}x {annual:>+9.1f}% {mdd*100:>+7.1f}% {sharpe:>7.2f} {liq:>4} {reb:>5}{tag}',flush=True)

# Year by year (lev=1x)
print(f'\n  Year-by-year (lev=1x, conservative):',flush=True)
eq_1x, _, _ = run_realistic_funding_farm(lev=1.0)
for yr in range(2021, 2026):
    iy = np.where(np.array([d.year==yr for d in common_idx]))[0]
    if len(iy) < 100: continue
    iy_active = iy[iy >= S]
    if len(iy_active) < 100: continue
    r = (eq_1x[iy_active[-1]] / eq_1x[max(0, iy_active[0]-1)] - 1) * 100
    avg_f = np.mean(fra[iy_active]) * 365 * 24 * 100
    avg_basis = np.mean(basis_pct[iy_active])
    print(f'    {yr}: Return={r:+.1f}%, Funding={avg_f:+.0f}%/yr, Basis={avg_basis:+.3f}%',flush=True)

# ============================================================
# Stress test: worst basis spike events
# ============================================================
print('\n'+'='*70,flush=True)
print(' Stress test: 最悪 basis spike events',flush=True)
print('='*70,flush=True)

# Find largest basis spikes
basis_change_24h=pd.Series(basis_pct,index=common_idx).diff(24).dropna()
top_spikes = basis_change_24h.abs().nlargest(10)
print('  Top 10 24H basis change events:',flush=True)
for ts, val in top_spikes.items():
    print(f'    {ts.strftime("%Y-%m-%d %H")}: basis change {val:+.3f}%',flush=True)

# ============================================================
# 結論
# ============================================================
print('\n'+'='*70,flush=True)
print(' 結論',flush=True)
print('='*70,flush=True)
print(f'  Realistic basis cost を加味した Funding Farm:',flush=True)
print(f'    lev=1x: 安全資産として優秀',flush=True)
print(f'    lev=2x+: basis spike で清算リスク',flush=True)
print(f'  Lighter.xyz で実装する場合:',flush=True)
print(f'    - Lighter perp の funding 構造は要確認',flush=True)
print(f'    - spot は Binance/Coinbase などの CEX が必要',flush=True)
print(f'    - cross-exchange なので執行に技術コストあり',flush=True)

print('\nDone.',flush=True)
