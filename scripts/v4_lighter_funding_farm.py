"""V4 ★5 Lighter Native Funding Farm
======================================
Lighter.xyz 仕様 (推定):
- Perp 取引手数料: 0% (retail)
- スリッページ: ~1bps (DEX なので CEX より大きい想定)
- Funding rate: Binance/Bybit と類似と仮定 (arbitrage で連動)

Spot 側:
- Binance/Coinbase など CEX
- 手数料: 5-10bps (taker)
- スリッページ: 2-3bps

検証ケース:
1. Lighter (perp 0 fee) vs Binance (perp 2-5 bps fee)
2. Cross-exchange execution の追加コスト
3. Lighter 流動性想定 (50% slippage premium)
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4 ★5: Lighter.xyz Native Funding Farm',flush=True)
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

perp_price=bybit_aligned['close'].values
spot_price=spot_aligned['close'].values
fra=np.roll(bybit_aligned['funding_rate'].fillna(0).values, 1)

n=len(common_idx)
S=2760

# ============================================================
# Funding Farm with venue-specific cost
# ============================================================
def run_ff(lev=1.5, spot_fee=5, spot_slip=3, perp_fee=0, perp_slip=1,
           rebal_days=7, maint_buffer=0.5):
    """
    Asymmetric venue costs:
    - spot_fee/slip: CEX side (Binance/Coinbase)
    - perp_fee/slip: Lighter or competitor side
    Both in bps
    """
    eq=np.ones(n);e=1.0
    spot_fee_r=spot_fee/10000;spot_slip_r=spot_slip/10000
    perp_fee_r=perp_fee/10000;perp_slip_r=perp_slip/10000

    capital_unit = 1 + 1/lev
    # Initial setup: open spot + open perp
    init_cost = (spot_fee_r + spot_slip_r + perp_fee_r + perp_slip_r) / capital_unit
    e *= (1 - init_cost)

    perp_unrealized = 0.0
    bars_per_rebal = rebal_days * 24
    liquidations = 0

    for i in range(S, n):
        if i > 0 and spot_price[i-1] > 0 and perp_price[i-1] > 0:
            spot_ret = spot_price[i] / spot_price[i-1] - 1
            perp_ret = perp_price[i] / perp_price[i-1] - 1
        else:
            spot_ret = 0; perp_ret = 0

        spot_pnl = spot_ret * 1.0
        perp_pnl = -perp_ret * 1.0
        funding_income = fra[i] * 1.0

        perp_unrealized += perp_pnl + funding_income

        if perp_unrealized < -1/lev * (1 - maint_buffer):
            liquidations += 1
            e *= max(0.001, 1 + perp_unrealized / capital_unit)
            perp_unrealized = 0
            # Reopen perp only (spot still safe)
            e *= (1 - (perp_fee_r + perp_slip_r) / capital_unit)
            continue

        bar_return = (spot_pnl + perp_pnl + funding_income) / capital_unit
        e *= (1 + bar_return)

        if (i - S) % bars_per_rebal == 0 and (i - S) > 0:
            # Rebalance: small adjustment on both sides
            rebal_cost = abs(perp_unrealized) * 0.5 * (spot_fee_r + spot_slip_r + perp_fee_r + perp_slip_r)
            e *= (1 - rebal_cost / capital_unit)
            perp_unrealized = 0

        eq[i] = e

    # Closing
    e *= (1 - (spot_fee_r + spot_slip_r + perp_fee_r + perp_slip_r) / capital_unit)
    eq[-1] = e
    return eq, liquidations

def stats(eq):
    years = (n - S) / (365 * 24)
    annual = ((eq[-1])**(1/years) - 1) * 100 if eq[-1] > 0 else -100
    mdd = 0; pk = 1
    for i in range(S, n): pk = max(pk, eq[i]); dd = (eq[i] - pk)/pk; mdd = min(mdd, dd)
    daily = pd.Series(eq, index=common_idx).resample('1D').last().pct_change().dropna()
    daily = daily[daily.index >= common_idx[S]]
    sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    return annual, mdd*100, sharpe

# ============================================================
# Compare venues
# ============================================================
print('\n  Venue comparison (lev=1.5x):',flush=True)
print(f'  {"Venue Setup":<45} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"Liq":>5}',flush=True)
print(f'  {"-"*78}',flush=True)

scenarios = [
    ('Bybit/Binance perp (5bps fee, 3bps slip)', 5, 3, 5, 3),
    ('Bybit perp (2bps maker, 2bps slip)', 5, 3, 2, 2),
    ('Lighter perp (0 fee, 1bps slip)', 5, 3, 0, 1),
    ('Lighter perp (0 fee, 3bps slip - illiq)', 5, 3, 0, 3),
    ('Lighter perp (0 fee, 5bps slip - thin)', 5, 3, 0, 5),
    ('All zero (theoretical max)', 0, 0, 0, 0),
]

for label, sf, ss, pf, ps in scenarios:
    eq, liq = run_ff(lev=1.5, spot_fee=sf, spot_slip=ss, perp_fee=pf, perp_slip=ps)
    a, m, s = stats(eq)
    tag = ' ★' if 'Lighter' in label and a > 50 else ''
    print(f'  {label:<45} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {liq:>4}{tag}',flush=True)

# ============================================================
# Lighter-specific: leverage scan
# ============================================================
print('\n'+'='*70,flush=True)
print(' Lighter Funding Farm: Leverage Scan',flush=True)
print('='*70,flush=True)
print(f'  {"Leverage":<10} {"Annual":>10} {"MDD":>8} {"Sharpe":>8} {"Liq":>5}',flush=True)
print(f'  {"-"*48}',flush=True)

for lev in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]:
    eq, liq = run_ff(lev=lev, spot_fee=5, spot_slip=3, perp_fee=0, perp_slip=1)
    a, m, s = stats(eq)
    tag = ' ✓' if a > 50 and m >= -15 and liq == 0 else ''
    print(f'  {lev:<9.1f}x {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f} {liq:>4}{tag}',flush=True)

# ============================================================
# Year-by-year (Lighter, lev=1.5x)
# ============================================================
print('\n  Year-by-year (Lighter lev=1.5x):',flush=True)
eq_l, _ = run_ff(lev=1.5, spot_fee=5, spot_slip=3, perp_fee=0, perp_slip=1)
eq_b, _ = run_ff(lev=1.5, spot_fee=5, spot_slip=3, perp_fee=2, perp_slip=2)

print(f'  {"Year":<6} {"Lighter":>12} {"Bybit":>12} {"Diff":>10}',flush=True)
for yr in range(2021, 2026):
    iy = np.where(np.array([d.year == yr for d in common_idx]))[0]
    iy_a = iy[iy >= S]
    if len(iy_a) < 100: continue
    rl = (eq_l[iy_a[-1]] / eq_l[max(0, iy_a[0]-1)] - 1) * 100
    rb = (eq_b[iy_a[-1]] / eq_b[max(0, iy_a[0]-1)] - 1) * 100
    print(f'  {yr:<6} {rl:>+11.1f}% {rb:>+11.1f}% {rl-rb:>+9.2f}%',flush=True)

# ============================================================
# Cost breakdown
# ============================================================
print('\n'+'='*70,flush=True)
print(' Lighter benefit analysis',flush=True)
print('='*70,flush=True)

# How much does Lighter save?
years = (n - S) / (365 * 24)
print(f'  Annual rebalances (weekly): ~{52} per year',flush=True)
print(f'  Each rebalance: 2 trades (perp adjust + spot adjust)',flush=True)
print(f'  Bybit perp 2bps × 2 trades × 52 = {2*2*52} bps/year = 2.08% drag',flush=True)
print(f'  Lighter perp 0bps × 2 × 52 = 0% (theoretical)',flush=True)
print(f'  → Lighter saves ~2% per year on rebalancing alone',flush=True)
print(f'  + plus liquidation/restart cost savings',flush=True)
print(f'  + plus initial setup cost ~5bps savings',flush=True)

# ============================================================
# 結論
# ============================================================
print('\n'+'='*70,flush=True)
print(' 結論: Lighter Native Funding Farm',flush=True)
print('='*70,flush=True)

eq_lighter, _ = run_ff(lev=1.5, spot_fee=5, spot_slip=3, perp_fee=0, perp_slip=1)
eq_bybit, _ = run_ff(lev=1.5, spot_fee=5, spot_slip=3, perp_fee=2, perp_slip=2)
a_l, m_l, s_l = stats(eq_lighter)
a_b, m_b, s_b = stats(eq_bybit)

print(f'  Lighter (lev=1.5x): {a_l:+.1f}% annual, MDD {m_l:+.1f}%, Sharpe {s_l:.2f}',flush=True)
print(f'  Bybit  (lev=1.5x): {a_b:+.1f}% annual, MDD {m_b:+.1f}%, Sharpe {s_b:.2f}',flush=True)
print(f'  Lighter advantage: +{a_l-a_b:.2f}%/year',flush=True)
print(f'',flush=True)
print(f'  注意: Lighter funding rate を Bybit の proxy で計算',flush=True)
print(f'    実際の Lighter funding は arbitrage で似るはずだが要検証',flush=True)
print(f'    0xArchive から実データ取得すれば確定値が出る',flush=True)

print('\nDone.',flush=True)
