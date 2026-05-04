"""V4 ★2: Funding Rate Yield Farm
====================================
構造: BTC spot Long + BTC perp Short = delta neutral
収益: perp funding rate (年100% 期待値) を全額獲得
リスク: spot/perp basis 変動、ステーブル価格リスク

検証:
1. funding rate を「払う側」(現在のRACM) vs「稼ぐ側」(funding farm) の比較
2. delta neutral の実現可能性 (basis 変動の影響)
3. 年別 funding 獲得率
4. レバレッジによる増幅 (cross margin)
5. 実際のexecution cost (taker fee, slippage)
"""
import sys,os
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4 ★2: FUNDING RATE YIELD FARM',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
fra=np.roll(h['funding'].fillna(0).values,1)  # lag-1 (previous funding rate)
S=2760

# Funding rate stats
print(f'\nFunding rate statistics:',flush=True)
print(f'  Mean per bar: {np.mean(fra[S:])*100:.5f}%',flush=True)
print(f'  Annualized (1x): {np.mean(fra[S:])*365*24*100:.1f}%',flush=True)
print(f'  Positive bars: {np.sum(fra[S:]>0)/(nn-S)*100:.1f}%',flush=True)
print(f'  Negative bars: {np.sum(fra[S:]<0)/(nn-S)*100:.1f}%',flush=True)

# ============================================================
# STRATEGY: Spot Long + Perp Short (delta neutral funding capture)
# ============================================================
print('\n'+'='*70,flush=True)
print(' STRATEGY 1: Pure Funding Farm (delta neutral)',flush=True)
print('='*70,flush=True)
print('  Long spot, Short perp same size → delta = 0',flush=True)
print('  P&L = funding income - basis variation - execution cost',flush=True)

def run_funding_farm(lev=1.0, fee_bps=2, slippage_bps=2, basis_drag=True):
    """
    Long spot at 1x, short perp at lev×spot_amount
    Each 8H: receive funding from short side
    Total capital = spot_amount + perp_margin (cross margin)

    Effective leverage on capital:
      spot uses 1x of capital
      perp short uses 1/lev of spot value (if cross margin)
      net deposit = 1 + 1/lev
      effective return on capital = (lev * funding) / (1 + 1/lev)
    """
    eq=np.ones(nn);e=1.0
    fee_rate=fee_bps/10000
    slippage_rate=slippage_bps/10000
    # Initial setup cost (open positions): 2x fees (spot + perp)
    e *= (1 - 2*(fee_rate+slippage_rate))

    # Hourly funding payments occur every 8H (3 per day)
    # Receive funding when short and funding > 0
    # Pay funding when short and funding < 0
    for i in range(S,nn):
        # Spot side: fully exposed to BTC ret (long)
        spot_pnl = ret[i] * 1.0
        # Perp short side: opposite of BTC ret * leverage
        perp_pnl = -ret[i] * lev
        # Net delta: 1 - lev (we want close to 0, so lev=1 is delta neutral)
        # Funding received: lev * funding_rate (short receives positive funding)
        funding_pnl = fra[i] * lev  # short earns positive funding, pays negative
        # Basis drag: spot/perp price difference fluctuation
        # Approximate: 0.05% annual basis cost (conservative)
        basis_cost = 0.0005 / (365*24) if basis_drag else 0

        total_pnl = spot_pnl + perp_pnl + funding_pnl - basis_cost
        # Capital = 1 (spot) + 1/lev (perp margin) for cross margin
        # Effective return on total capital
        capital = 1 + 1/lev
        e *= (1 + total_pnl / capital)
        eq[i] = e

    # Closing cost
    e *= (1 - 2*(fee_rate+slippage_rate))
    eq[-1] = e
    return eq

# Test different leverage levels
print(f'\n  {"Leverage":<12} {"Total Return":>15} {"Annualized":>12} {"Sharpe":>8} {"MDD":>8}',flush=True)
print(f'  {"-"*60}',flush=True)

results_ff = {}
for lev in [1.0, 2.0, 3.0, 5.0, 10.0]:
    eq = run_funding_farm(lev=lev, fee_bps=2, slippage_bps=2)
    total = (eq[-1] - 1) * 100
    years = (nn - S) / (365 * 24)
    annualized = ((eq[-1])**(1/years) - 1) * 100

    # Sharpe
    eq_s = pd.Series(eq, index=idx_h)
    daily = eq_s.resample('1D').last().pct_change().dropna()
    daily = daily[daily.index >= idx_h[S]]
    sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0

    # MDD
    mdd = 0; pk = 1
    for i in range(S, nn): pk = max(pk, eq[i]); dd = (eq[i] - pk)/pk; mdd = min(mdd, dd)

    results_ff[lev] = (total, annualized, sharpe, mdd)
    print(f'  {lev:<11.1f}x {total:>+13.1f}% {annualized:>+11.1f}% {sharpe:>7.2f} {mdd*100:>+7.1f}%',flush=True)

# ============================================================
# YEAR BY YEAR
# ============================================================
print(f'\n  Year-by-year (lev=3x):',flush=True)
eq_3x = run_funding_farm(lev=3.0)
for yr in range(2021, 2026):
    iy = np.where(np.array([d.year == yr for d in idx_h]))[0]
    if len(iy) < 100: continue
    r = (eq_3x[iy[-1]] / eq_3x[max(0, iy[0]-1)] - 1) * 100
    avg_f = np.mean(fra[iy]) * 365 * 24 * 100
    print(f'    {yr}: Return={r:+.1f}%, Avg Funding (1x)={avg_f:+.1f}%/yr',flush=True)

# ============================================================
# SENSITIVITY: Cost analysis
# ============================================================
print('\n'+'='*70,flush=True)
print(' SENSITIVITY: Cost Impact',flush=True)
print('='*70,flush=True)
print(f'  {"Fee+Slip (bps)":<18} {"Annualized":>12}',flush=True)
print(f'  {"-"*32}',flush=True)
for cost in [0, 1, 2, 5, 10, 20]:
    eq = run_funding_farm(lev=3.0, fee_bps=cost, slippage_bps=cost)
    years = (nn - S) / (365 * 24)
    annual = ((eq[-1])**(1/years) - 1) * 100
    print(f'  {cost*2:<17} {annual:>+11.1f}%',flush=True)

# ============================================================
# REBALANCING: How often do we need to rebalance?
# ============================================================
print('\n'+'='*70,flush=True)
print(' REBALANCING: Delta Drift Analysis',flush=True)
print('='*70,flush=True)
print('  spot vs perp price drift creates delta exposure over time',flush=True)
print('  Perp ret ≈ spot ret + funding adjustment (over 8H window)',flush=True)
# Compute cumulative drift
# In practice: spot price - mark price difference accumulates
# Approximate: each 8H, drift = funding * exposure
funding_8h = pd.Series(fra).resample('8H').sum() if hasattr(idx_h, 'resample') else None
print(f'  Average 8H funding: {np.mean(fra[S:])*8*100:.4f}%',flush=True)
print(f'  Drift after 1 day (no rebal): ~{np.mean(fra[S:])*24*100:.3f}% delta exposure',flush=True)
print(f'  Drift after 1 week: ~{np.mean(fra[S:])*24*7*100:.2f}% delta exposure',flush=True)
print(f'  → Rebalance every 1-7 days to maintain delta neutral',flush=True)

# ============================================================
# COMPARISON vs BTC B&H
# ============================================================
print('\n'+'='*70,flush=True)
print(' COMPARISON: Funding Farm vs BTC B&H',flush=True)
print('='*70,flush=True)
eq_bh = np.ones(nn); e = 1.0
for i in range(S, nn): e *= (1 + ret[i]); eq_bh[i] = e
years = (nn - S) / (365 * 24)
bh_return = (eq_bh[-1] - 1) * 100
bh_annual = ((eq_bh[-1])**(1/years) - 1) * 100
bh_mdd = 0; pk = 1
for i in range(S, nn): pk = max(pk, eq_bh[i]); dd = (eq_bh[i] - pk)/pk; bh_mdd = min(bh_mdd, dd)

print(f'  {"Strategy":<25} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*52}',flush=True)
print(f'  {"BTC B&H":<25} {bh_annual:>+9.1f}% {bh_mdd*100:>+7.1f}% {"":>7}',flush=True)
for lev, (total, annual, sharpe, mdd) in results_ff.items():
    print(f'  {f"FF lev={lev}x":<25} {annual:>+9.1f}% {mdd*100:>+7.1f}% {sharpe:>7.2f}',flush=True)

# ============================================================
# REALISTIC SCENARIO: Lighter.xyz (0 fees)
# ============================================================
print('\n'+'='*70,flush=True)
print(' REALISTIC: Lighter.xyz (0 fees)',flush=True)
print('='*70,flush=True)
# Lighter has 0 fees but we still have:
# - Spot fee (probably small, 1-5 bps on Binance/Coinbase)
# - Slippage on size (1-3 bps)
# - Basis variation
print(f'  Assumption: Lighter perp 0 fee, Spot side 5bps, slippage 3bps',flush=True)
for lev in [1.0, 2.0, 3.0, 5.0]:
    # Asymmetric: spot side has fees, perp side (Lighter) doesn't
    eq = run_funding_farm(lev=lev, fee_bps=2.5, slippage_bps=1.5)  # average
    annual = ((eq[-1])**(1/years) - 1) * 100
    print(f'  Lev={lev}x: Annual={annual:+.1f}%',flush=True)

print('\nDone.',flush=True)
