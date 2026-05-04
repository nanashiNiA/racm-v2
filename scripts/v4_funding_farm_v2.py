"""V4 ★2: Funding Farm - 修正版
==================================
前回バグ: lev で perp 名目を変えていた → 実際は delta neutralでない

正しい構造:
- Spot Long $1 (margin $1)
- Perp Short $1 (same notional, margin $1/lev)
- Net delta = 0 (lev に関わらず)
- Total capital = $1 (spot) + $1/lev (perp margin)
- Funding income per bar = $1 * funding_rate
- Return on capital = $1 * funding_rate / (1 + 1/lev)

つまり lev は資本効率を高めるだけ、ポジションサイズは変えない。
リスク: perp 側の清算 (証拠金 < 維持証拠金)
"""
import sys,os
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4 ★2 修正版: Funding Farm (正しいdelta neutral)',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
fra=np.roll(h['funding'].fillna(0).values,1)
S=2760

def run_funding_farm(lev=1.0, fee_bps=2, slippage_bps=2,
                     liquidation_buffer=0.5, rebal_days=7):
    """
    True delta-neutral funding farm.

    lev: leverage on PERP MARGIN only
    Position: Spot $1 long + Perp $1 short
    Margin: Spot $1 + Perp $1/lev
    Total capital: $1 + $1/lev

    Per bar:
      net_delta_pnl = ret*$1 + (-ret*$1) = 0
      funding_pnl = fra*$1 (positive when fra>0, short receives)

    Liquidation risk: if BTC pumps too fast, perp short loses > margin
      Liquidation when: (price increase) * $1 > $1/lev * liquidation_buffer
      → price increase > liquidation_buffer/lev

    Rebalancing: spot vs perp positions drift due to funding payments
      → need periodic rebalancing
    """
    eq=np.ones(nn);e=1.0
    fee_rate=fee_bps/10000
    slip_rate=slippage_bps/10000

    # Initial setup: 2 trades (spot + perp)
    # Capital = 1 + 1/lev
    capital_unit = 1 + 1/lev
    initial_cost = 2*(fee_rate+slip_rate) / capital_unit
    e *= (1 - initial_cost)

    # Track perp margin level (for liquidation check)
    perp_margin = 1/lev  # initial margin
    spot_position = 1.0   # constant
    cum_perp_pnl = 0.0

    bars_per_rebal = rebal_days * 24
    rebal_count = 0
    liquidations = 0

    for i in range(S,nn):
        # Funding income (perp short receives when funding>0, pays when <0)
        funding_income = fra[i] * 1.0  # $1 perp short notional

        # Track perp margin: starts at 1/lev, increases with funding income, decreases with adverse moves
        # Perp short P&L: -ret[i] * 1.0 (per dollar notional)
        perp_unrealized = -ret[i] * 1.0
        cum_perp_pnl += perp_unrealized + funding_income

        # Liquidation check: if cumulative perp loss > initial margin * (1-buffer)
        # We use buffer=0.5 (50% maintenance margin)
        if cum_perp_pnl < -perp_margin * (1-liquidation_buffer):
            liquidations += 1
            # Force close perp position with loss
            e *= (1 + cum_perp_pnl / capital_unit)  # realize loss
            cum_perp_pnl = 0
            # Close cost
            e *= (1 - 2*(fee_rate+slip_rate))
            # Resize positions (back to delta neutral with smaller capital)
            # Simplified: stop trading until next rebal
            continue

        # Net P&L per bar (delta neutral, only funding matters)
        # Both spot and perp track BTC, so spot_pnl + perp_pnl = 0
        # Net = funding_income / capital
        bar_return = funding_income / capital_unit
        e *= (1 + bar_return)

        # Periodic rebalancing
        if (i - S) % bars_per_rebal == 0:
            # Rebalance cost: 2 trades to restore delta neutral
            rebal_count += 1
            e *= (1 - 2*(fee_rate+slip_rate)*0.1)  # smaller adjustment
            cum_perp_pnl = 0  # reset tracking after rebal

        eq[i] = e

    # Closing cost
    e *= (1 - 2*(fee_rate+slip_rate))
    eq[-1] = e

    return eq, liquidations, rebal_count

# Test multiple leverage levels
print(f'\n  {"Leverage":<10} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"Liq":>6} {"Rebal":>7}',flush=True)
print(f'  {"-"*54}',flush=True)

results = {}
for lev in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
    eq, liq, reb = run_funding_farm(lev=lev, fee_bps=2, slippage_bps=2)
    years = (nn - S) / (365 * 24)
    annual = ((eq[-1])**(1/years) - 1) * 100 if eq[-1] > 0 else -100

    # Sharpe
    eq_s = pd.Series(eq, index=idx_h)
    daily = eq_s.resample('1D').last().pct_change().dropna()
    daily = daily[daily.index >= idx_h[S]]
    sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0

    # MDD
    mdd = 0; pk = 1
    for i in range(S, nn): pk = max(pk, eq[i]); dd = (eq[i] - pk)/pk; mdd = min(mdd, dd)

    results[lev] = (annual, mdd*100, sharpe, liq)
    print(f'  {lev:<9.1f}x {annual:>+9.1f}% {mdd*100:>+7.1f}% {sharpe:>7.2f} {liq:>5} {reb:>6}',flush=True)

# ============================================================
# Year-by-year (lev=5x)
# ============================================================
print('\n  Year-by-year (lev=5x):',flush=True)
eq_5x, _, _ = run_funding_farm(lev=5.0)
for yr in range(2021, 2026):
    iy = np.where(np.array([d.year == yr for d in idx_h]))[0]
    if len(iy) < 100: continue
    r = (eq_5x[iy[-1]] / eq_5x[max(0, iy[0]-1)] - 1) * 100
    avg_f = np.mean(fra[iy]) * 365 * 24 * 100
    print(f'    {yr}: Return={r:+.1f}%, Avg Funding={avg_f:+.1f}%/yr',flush=True)

# ============================================================
# REALISTIC SCENARIOS
# ============================================================
print('\n'+'='*70,flush=True)
print(' REALISTIC SCENARIOS',flush=True)
print('='*70,flush=True)

scenarios = [
    ('Conservative (lev=2x, fees=5bps)', 2.0, 5, 5),
    ('Moderate (lev=3x, fees=3bps)', 3.0, 3, 3),
    ('Lighter perp 0 fee (lev=5x, spot=5bps)', 5.0, 2.5, 2.5),
    ('Aggressive Lighter (lev=10x, low fee)', 10.0, 2, 2),
]

print(f'  {"Scenario":<45} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*72}',flush=True)
for label, lev, fee, slip in scenarios:
    eq, liq, _ = run_funding_farm(lev=lev, fee_bps=fee, slippage_bps=slip)
    years = (nn - S) / (365 * 24)
    annual = ((eq[-1])**(1/years) - 1) * 100 if eq[-1] > 0 else -100
    mdd = 0; pk = 1
    for i in range(S, nn): pk = max(pk, eq[i]); dd = (eq[i] - pk)/pk; mdd = min(mdd, dd)
    eq_s = pd.Series(eq, index=idx_h)
    daily = eq_s.resample('1D').last().pct_change().dropna()
    daily = daily[daily.index >= idx_h[S]]
    sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    print(f'  {label:<45} {annual:>+9.1f}% {mdd*100:>+7.1f}% {sharpe:>7.2f}',flush=True)

# ============================================================
# 教授要件チェック
# ============================================================
print('\n'+'='*70,flush=True)
print(' 教授要件チェック (Funding Farm lev=3x)',flush=True)
print('='*70,flush=True)

eq, liq, reb = run_funding_farm(lev=3.0, fee_bps=3, slippage_bps=3)
years = (nn - S) / (365 * 24)
annual = ((eq[-1])**(1/years) - 1) * 100 if eq[-1] > 0 else -100
mdd = 0; pk = 1
for i in range(S, nn): pk = max(pk, eq[i]); dd = (eq[i] - pk)/pk; mdd = min(mdd, dd)
eq_s = pd.Series(eq, index=idx_h)
daily = eq_s.resample('1D').last().pct_change().dropna()
daily = daily[daily.index >= idx_h[S]]
sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0

# Bear 2022
i22 = np.where(np.array([d.year == 2022 for d in idx_h]))[0]
b22 = (eq[i22[-1]] / eq[max(0, i22[0]-1)] - 1) * 100 if len(i22) > 100 else 0

print(f'  WFS (annual):    {annual:.0f}%   要件: ≥300%  → {"✓" if annual>=300 else "✗"}',flush=True)
print(f'  MDD:             {mdd*100:+.1f}%  要件: ≤-30%  → {"✓" if mdd*100>=-30 else "✗"}',flush=True)
print(f'  Bear 2022:       {b22:+.0f}%   要件: >0%    → {"✓" if b22>0 else "✗"}',flush=True)
print(f'  Sharpe:          {sharpe:.2f}',flush=True)
print(f'  取引/日:         ~{2/7:.2f} (週次rebal x 2取引)',flush=True)
print(f'  → 取引頻度要件 (≥3/日) は別途追加トレードで対応必要',flush=True)
print(f'  Liquidations:    {liq}',flush=True)

print('\nDone.',flush=True)
