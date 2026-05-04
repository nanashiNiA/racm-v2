"""V4 ★2 Funding Farm v4: バグ修正版 + 実basisデータ
=====================================================
v3のバグ: funding を /8 で割っていた
  (h['funding'] は既に per-bar (per-1H) の値、/8 不要)

v4 修正:
- funding rate を per-bar でそのまま使用 (v2と同じ正しい計算)
- basis cost は実データから計算 (v3 の改善点を継承)
- 細かい清算判定 + rebalancing
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4 ★2 v4 (バグ修正): Funding Farm with REAL BASIS',flush=True)
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
                    ticker[['timestamp','funding_rate','last_price','mark_price']].sort_values('timestamp'),
                    on='timestamp',direction='backward',tolerance=pd.Timedelta('2H'))
            ob_frames.append(ohlc)
bybit=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
bybit=bybit.drop_duplicates('timestamp').set_index('timestamp')

h_racm=build_common_1h()
h_racm.index=h_racm.index.tz_localize('UTC') if h_racm.index.tz is None else h_racm.index

common_idx=bybit.index.intersection(h_racm.index)
bybit_aligned=bybit.loc[common_idx]
spot_aligned=h_racm.loc[common_idx]

# データ
perp_price=bybit_aligned['close'].values
spot_price=spot_aligned['close'].values
basis_pct=(perp_price-spot_price)/spot_price*100
fra=np.roll(bybit_aligned['funding_rate'].fillna(0).values, 1)  # lag-1

n=len(common_idx)
S=2760

print(f'  Common bars: {n}',flush=True)
print(f'  funding mean: {np.mean(fra[S:])*100:.5f}%/bar = {np.mean(fra[S:])*365*24*100:.0f}%/yr',flush=True)
print(f'  basis mean: {np.mean(basis_pct[S:]):+.4f}% (perp - spot)',flush=True)

# ============================================================
# Funding Farm with CORRECT funding application
# ============================================================
def run_funding_farm_correct(lev=1.0, fee_bps=2, slip_bps=2, rebal_days=7,
                              maint_margin_ratio=0.5):
    """
    Correct funding farm:
    - Spot Long $1, Perp Short $1
    - Funding paid per bar (per hour, since data is per-1H)
    - Basis variation = mark-to-market loss/gain
    - Liquidation when perp margin < maintenance margin
    """
    eq=np.ones(n);e=1.0
    fee_rate=fee_bps/10000
    slip_rate=slip_bps/10000

    # Capital: spot $1 + perp margin $1/lev
    capital_unit = 1 + 1/lev

    # Initial setup cost
    e *= (1 - 2*(fee_rate+slip_rate) / capital_unit)

    perp_margin_init = 1/lev
    perp_unrealized = 0.0  # cumulative unrealized P&L on perp

    bars_per_rebal = rebal_days * 24
    rebal_count = 0
    liquidations = 0

    for i in range(S, n):
        # Per-bar returns
        if i > 0 and spot_price[i-1] > 0 and perp_price[i-1] > 0:
            spot_ret = spot_price[i] / spot_price[i-1] - 1
            perp_ret = perp_price[i] / perp_price[i-1] - 1
        else:
            spot_ret = 0; perp_ret = 0

        # P&L per dollar position
        spot_pnl = spot_ret * 1.0  # long $1
        perp_pnl = -perp_ret * 1.0  # short $1

        # Funding income (perp short receives positive funding)
        # CORRECT: per-bar funding, no /8 division
        funding_income = fra[i] * 1.0  # already per-bar rate

        # Update cumulative perp unrealized P&L
        perp_unrealized += perp_pnl + funding_income

        # Liquidation check
        if perp_unrealized < -perp_margin_init * (1 - maint_margin_ratio):
            liquidations += 1
            # Force close perp at loss
            e *= max(0.001, 1 + perp_unrealized / capital_unit)
            perp_unrealized = 0
            # Reopen cost
            e *= (1 - 2*(fee_rate+slip_rate) / capital_unit)
            continue

        # Total bar return: spot_pnl + perp_pnl + funding_income (all per $1)
        total_pnl = spot_pnl + perp_pnl + funding_income
        bar_return = total_pnl / capital_unit
        e *= (1 + bar_return)

        # Periodic rebalancing
        if (i - S) % bars_per_rebal == 0 and (i-S) > 0:
            rebal_count += 1
            # Cost of rebalancing: small fee on the drift amount
            rebal_cost = abs(perp_unrealized) * 0.5 * (fee_rate+slip_rate) * 2
            e *= (1 - rebal_cost / capital_unit)
            perp_unrealized = 0  # reset after rebal

        eq[i] = e

    # Closing cost
    e *= (1 - 2*(fee_rate+slip_rate) / capital_unit)
    eq[-1] = e

    return eq, liquidations, rebal_count

# Test
print(f'\n  {"Leverage":<10} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"Liq":>5} {"Rebal":>6}',flush=True)
print(f'  {"-"*55}',flush=True)

for lev in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    eq, liq, reb = run_funding_farm_correct(lev=lev)
    years = (n-S)/(365*24)
    annual = ((eq[-1])**(1/years) - 1) * 100 if eq[-1] > 0 else -100
    mdd = 0; pk = 1
    for i in range(S, n): pk = max(pk, eq[i]); dd = (eq[i] - pk)/pk; mdd = min(mdd, dd)
    daily = pd.Series(eq, index=common_idx).resample('1D').last().pct_change().dropna()
    daily = daily[daily.index >= common_idx[S]]
    sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0

    tag = ' *' if annual>50 and mdd*100>=-15 and liq==0 else ''
    print(f'  {lev:<9.1f}x {annual:>+9.1f}% {mdd*100:>+7.1f}% {sharpe:>7.2f} {liq:>4} {reb:>5}{tag}',flush=True)

# Year-by-year (lev=1x)
print(f'\n  Year-by-year (lev=1x):',flush=True)
eq_1x, _, _ = run_funding_farm_correct(lev=1.0)
for yr in range(2021, 2026):
    iy = np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_active = iy[iy >= S]
    if len(iy_active) < 100: continue
    r = (eq_1x[iy_active[-1]] / eq_1x[max(0, iy_active[0]-1)] - 1) * 100
    avg_f = np.mean(fra[iy_active]) * 365 * 24 * 100
    avg_basis = np.mean(basis_pct[iy_active])
    print(f'    {yr}: Return={r:+.1f}%, Funding={avg_f:+.0f}%/yr, Basis={avg_basis:+.3f}%',flush=True)

# 教授要件チェック
print(f'\n  教授要件チェック:',flush=True)
eq_1x, _, _ = run_funding_farm_correct(lev=1.0, fee_bps=3, slip_bps=2)
years = (n-S)/(365*24)
annual = ((eq_1x[-1])**(1/years) - 1) * 100 if eq_1x[-1] > 0 else -100
mdd = 0; pk = 1
for i in range(S, n): pk = max(pk, eq_1x[i]); dd = (eq_1x[i] - pk)/pk; mdd = min(mdd, dd)
i22 = np.where(np.array([d.year==2022 for d in common_idx]))[0]
i22_a = i22[i22 >= S]
b22 = (eq_1x[i22_a[-1]] / eq_1x[max(0, i22_a[0]-1)] - 1) * 100 if len(i22_a) else 0

print(f'    WFS (annual):    {annual:.0f}%   要件: ≥300%  {"✓" if annual>=300 else "✗"}',flush=True)
print(f'    MDD:             {mdd*100:+.1f}%  要件: ≤-30%  {"✓" if mdd*100>=-30 else "✗"}',flush=True)
print(f'    Bear 2022:       {b22:+.1f}%  要件: >0%    {"✓" if b22>0 else "✗"}',flush=True)

# Sharpe
daily = pd.Series(eq_1x, index=common_idx).resample('1D').last().pct_change().dropna()
daily = daily[daily.index >= common_idx[S]]
sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
print(f'    Sharpe:          {sharpe:.2f}',flush=True)

print('\nDone.',flush=True)
