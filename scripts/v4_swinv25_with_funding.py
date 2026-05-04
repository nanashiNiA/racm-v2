"""V4 swin_v25 本体を呼び出して funding cost を後から加算
=========================================================
swin_v25 のオリジナル backtest を実行 (もう走らせる)
完了後、trade_log を解析して funding cost を正確に計算
purging 効果も確認
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from datetime import datetime

print('='*70,flush=True)
print('  V4: swin_v25 本体実行 + funding コスト後付け検証',flush=True)
print('='*70,flush=True)

# Run swin_v25 with smaller window for faster validation
print('\nRunning swin_v25 on 2024 1H (smaller WF for speed)...',flush=True)
print('  Modified: train=1200 bars, test=300 bars, step=300 (faster)',flush=True)

# Monkey-patch walk_forward_backtest to be faster
from src import swin_v25
orig_wf = swin_v25.walk_forward_backtest

def fast_wf(df, feature_cols, train_window=1200, test_window=300, step=300, **kwargs):
    return orig_wf(df, feature_cols, train_window=train_window,
                    test_window=test_window, step=step, **kwargs)
swin_v25.walk_forward_backtest = fast_wf

# Run
result = swin_v25.run_single_config(
    'Y:/tardis_data/bybit/perpetual', '1H', '2024',
    use_v25_engine=False
)

if result is None:
    print('FAILED to run swin_v25',flush=True)
    sys.exit(1)

trades = result['trade_log']
equity_curve = result['equity_curve']
print(f'\n  Original swin_v25 result:',flush=True)
print(f'    Net Return: {result["net_return"]:+.1f}%',flush=True)
print(f'    Trades: {result["trades"]}, WR: {result["win_rate"]:.1f}%',flush=True)
print(f'    MaxDD: {result["max_dd"]:.1f}%',flush=True)
print(f'    B&H: {result["bnh_return"]:+.1f}%',flush=True)

if not trades:
    print('No trades, exit.',flush=True)
    sys.exit(0)

# ============================================================
# Add funding cost to each trade
# ============================================================
print('\n'+'='*70,flush=True)
print(' funding コスト追加検証',flush=True)
print('='*70,flush=True)

# Load funding rate from cache
cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
with open(f'{cache_dir}/BTCUSDT_20240101_20241231_1H.pkl','rb') as f:
    ohlc, ticker, liq = pickle.load(f)

ticker['timestamp']=pd.to_datetime(ticker['timestamp'],utc=True)
ticker=ticker.set_index('timestamp').sort_index()

# Convert trade times
total_funding_cost = 0
total_funding_count = 0
funding_per_trade = []

for t in trades:
    entry_time = pd.Timestamp(t['entry_time'])
    exit_time = pd.Timestamp(t['exit_time'])
    if entry_time.tz is None:
        entry_time = entry_time.tz_localize('UTC')
    if exit_time.tz is None:
        exit_time = exit_time.tz_localize('UTC')

    # Get funding rates during the trade period
    mask = (ticker.index >= entry_time) & (ticker.index <= exit_time)
    fra_window = ticker.loc[mask, 'funding_rate'].fillna(0).values

    # 1H bars: funding payment every 8H, but applied per bar
    # Total funding for the trade period
    if t['side'] == 'LONG':
        # Long pays positive funding
        funding_cost = np.sum(fra_window) * t['leverage'] * t['position_size']
    else:
        # Short receives positive funding
        funding_cost = -np.sum(fra_window) * t['leverage'] * t['position_size']

    total_funding_cost += funding_cost
    total_funding_count += 1
    funding_per_trade.append(funding_cost)

    t['funding_cost'] = funding_cost
    t['net_return_with_funding'] = t['net_return'] - funding_cost

# Compute equity with funding
equity_with_funding = 1.0
equity_curve_funded = [1.0]
for t in trades:
    equity_with_funding *= (1 + t['net_return_with_funding'])
    equity_curve_funded.append(equity_with_funding)

# Stats
total_return_funded = (equity_with_funding - 1) * 100
win_rate_funded = sum(1 for t in trades if t['net_return_with_funding'] > 0) / len(trades) * 100
peak = np.maximum.accumulate(equity_curve_funded)
max_dd_funded = abs(((np.array(equity_curve_funded) - peak) / peak).min()) * 100

print(f'\n  Original (no funding cost):  Return={result["net_return"]:+.1f}%, WR={result["win_rate"]:.1f}%, MDD={result["max_dd"]:.1f}%',flush=True)
print(f'  WITH funding cost (correct): Return={total_return_funded:+.1f}%, WR={win_rate_funded:.1f}%, MDD={max_dd_funded:.1f}%',flush=True)
print(f'',flush=True)
print(f'  Total funding cost: {total_funding_cost*100:+.2f}% of capital',flush=True)
print(f'  Avg funding per trade: {np.mean(funding_per_trade)*100:+.4f}%',flush=True)
print(f'  Long trades funding (paid): {sum(t["funding_cost"] for t in trades if t["side"]=="LONG")*100:+.2f}%',flush=True)
print(f'  Short trades funding (received): {sum(t["funding_cost"] for t in trades if t["side"]=="SHORT")*100:+.2f}%',flush=True)

# Long vs Short distribution
long_trades = [t for t in trades if t['side']=='LONG']
short_trades = [t for t in trades if t['side']=='SHORT']
print(f'\n  Long trades: {len(long_trades)}, avg net (no fund): {np.mean([t["net_return"] for t in long_trades])*100:+.3f}%',flush=True)
print(f'  Short trades: {len(short_trades)}, avg net (no fund): {np.mean([t["net_return"] for t in short_trades])*100:+.3f}%',flush=True)

# ============================================================
# Hold time analysis
# ============================================================
print('\n'+'='*70,flush=True)
print(' Hold time analysis',flush=True)
print('='*70,flush=True)

hold_times = [t['holding_bars'] for t in trades]
print(f'  Avg hold: {np.mean(hold_times):.1f} bars ({np.mean(hold_times):.1f}H)',flush=True)
print(f'  Median: {np.median(hold_times):.0f} bars',flush=True)
print(f'  Max: {max(hold_times)} bars ({max(hold_times)/24:.1f}d)',flush=True)
print(f'  Funding payments per trade: {np.mean(hold_times)/8:.2f} (every 8H)',flush=True)

# ============================================================
# Exit reason analysis
# ============================================================
print('\n'+'='*70,flush=True)
print(' Exit reasons',flush=True)
print('='*70,flush=True)
exit_counts = {}
for t in trades:
    er = t['exit_reason']
    if er not in exit_counts: exit_counts[er] = []
    exit_counts[er].append(t['net_return_with_funding'])
for er, rets in exit_counts.items():
    print(f'  {er:<10}: {len(rets):>4} trades, avg ret (with fund): {np.mean(rets)*100:+.3f}%',flush=True)

# ============================================================
# 結論
# ============================================================
print('\n'+'='*70,flush=True)
print(' 結論: swin_v25 with proper funding cost',flush=True)
print('='*70,flush=True)

if total_return_funded > 30:
    print(f'  ✓ 年+30% 以上 → 戦略は funding を加味しても利益あり',flush=True)
elif total_return_funded > 0:
    print(f'  △ 年+0-30% → 戦略は弱い利益、改善余地あり',flush=True)
else:
    print(f'  ✗ funding 加算後マイナス → 戦略は実用不可',flush=True)
print(f'  Net (with funding): {total_return_funded:+.1f}%',flush=True)
print(f'  vs B&H 2024: {result["bnh_return"]:+.1f}%',flush=True)
print(f'  vs Funding Farm 1x (年+64%): {"勝ち" if total_return_funded > 64 else "負け"}',flush=True)

print('\nDone.',flush=True)
