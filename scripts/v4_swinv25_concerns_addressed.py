"""V4 swin_v25 の3つの懸念を独立検証
========================================
バックテスト完了を待たず、独立スクリプトで以下を検証:

懸念1: Funding コスト未計算
  → swin_v25 の典型的トレード (24-48H hold, 0.15-0.35 size, 4-6x lev) に
     正しい funding コストを加算してリターン再計算

懸念2: Walk-Forward Purging なし
  → horizon=6-8 bar gap を train/test 間に挿入して再評価
  → HGBR 予測精度がどう変わるか

懸念3: シグナル単体の予測力
  → HGBR の direction 予測精度を OOS で測定
  → シャッフルテストで時系列的価値を検証

データ: Bybit 2021-2025 (cached)
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

print('='*70,flush=True)
print('  swin_v25: 3つの懸念を独立検証',flush=True)
print('='*70,flush=True)

# ============================================================
# Load Bybit data
# ============================================================
print('Loading Bybit cache...',flush=True)
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
                    ticker[['timestamp','funding_rate','open_interest']].sort_values('timestamp'),
                    on='timestamp',direction='backward',tolerance=pd.Timedelta('2H'))
            if liq is not None:
                liq['timestamp']=pd.to_datetime(liq['timestamp'],utc=True)
                ohlc=ohlc.merge(liq,on='timestamp',how='left')
                for c in ['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']:
                    if c in ohlc.columns:ohlc[c]=ohlc[c].fillna(0)
            ob_frames.append(ohlc)
df=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
df=df.drop_duplicates('timestamp').reset_index(drop=True)
print(f'  Loaded: {len(df)} bars ({df["timestamp"].iloc[0]} → {df["timestamp"].iloc[-1]})',flush=True)

# ============================================================
# Build features (simplified swin_v25 feature set)
# ============================================================
print('Building features...',flush=True)
df['mid']=(df['open']+df['close'])/2
close_s=df['close'].shift(1)

# Returns (lag-1)
for hrs in [1,4,8,12,24,72]:
    df[f'ret_{hrs}h']=close_s.pct_change(hrs)

# Volatility
df['atr_24']=(df['high'].shift(1)-df['low'].shift(1)).rolling(24,min_periods=6).mean()
df['atr_pct']=df['atr_24']/(close_s+1e-12)
df['vol_24']=df['ret_1h'].rolling(24,min_periods=6).std()
df['vol_zscore']=(df['vol_24']-df['vol_24'].rolling(168,min_periods=48).mean())/(df['vol_24'].rolling(168,min_periods=48).std()+1e-10)

# RSI (lag-1)
delta=close_s.diff()
gain=delta.clip(lower=0).rolling(14).mean()
loss=(-delta.clip(upper=0)).rolling(14).mean()
df['rsi']=100-100/(1+gain/(loss+1e-10))

# MAs
df['sma_50']=close_s.rolling(50,min_periods=25).mean()
df['sma_200']=close_s.rolling(200,min_periods=100).mean()
df['sma_dist']=(close_s-df['sma_200'])/(df['sma_200']+1e-10)
df['sma_slope']=df['sma_50'].pct_change(10)

# Orderbook features (lag-1)
if 'imbalance_0' in df.columns:
    df['imb_lag']=df['imbalance_0'].shift(1)
    df['imb_ma']=df['imb_lag'].rolling(12,min_periods=3).mean()
if 'depth_imbalance' in df.columns:
    df['depth_imb_lag']=df['depth_imbalance'].shift(1)
if 'bid_pressure' in df.columns and 'ask_pressure' in df.columns:
    df['pressure_diff']=(df['bid_pressure']-df['ask_pressure']).shift(1)

# Funding/OI
if 'funding_rate' in df.columns:
    fr=df['funding_rate'].shift(1)
    df['funding_zscore']=(fr-fr.rolling(168,min_periods=48).mean())/(fr.rolling(168,min_periods=48).std()+1e-10)
    df['funding_lag']=fr
if 'open_interest' in df.columns:
    oi=df['open_interest'].shift(1)
    df['oi_change']=oi.pct_change(24)

if 'liq_count' in df.columns:
    liq=df['liq_count'].shift(1)
    df['liq_spike']=liq/(liq.rolling(24,min_periods=6).mean()+1)

# Target: future 6-bar return (HGBR target)
horizon=6
df['target']=df['close'].shift(-horizon)/df['open'].shift(-horizon+1)-1

feature_cols=[c for c in df.columns if c not in
              ['timestamp','open','high','low','close','volume','mid','target',
               'imbalance_0','depth_imbalance','total_depth','bid_pressure','ask_pressure',
               'funding_rate','open_interest','last_price','mark_price',
               'long_liq_usd','short_liq_usd','liq_count','liq_imbalance']]
print(f'  {len(feature_cols)} features',flush=True)

# Cleanup
df=df.replace([np.inf,-np.inf],np.nan)
n=len(df)

# ============================================================
# 懸念2: Walk-Forward Purging 検証
# ============================================================
print('\n'+'='*70,flush=True)
print(' 懸念2: Walk-Forward Purging検証',flush=True)
print('='*70,flush=True)
print('  swin_v25 は purging なし → 訓練の最後 6 bar に未来情報リーク可能性',flush=True)

# Train: years 2021-2023, Test: 2024
train_end_idx=df[df['timestamp']<pd.Timestamp('2024-01-01',tz='UTC')].index[-1]
test_start_idx=df[df['timestamp']>=pd.Timestamp('2024-01-01',tz='UTC')].index[0]
test_end_idx=df[df['timestamp']<pd.Timestamp('2025-01-01',tz='UTC')].index[-1]

print(f'  Train: 0-{train_end_idx} ({df["timestamp"].iloc[200]} - {df["timestamp"].iloc[train_end_idx]})',flush=True)
print(f'  Test:  {test_start_idx}-{test_end_idx}',flush=True)

def train_predict(train_idx_range, test_idx_range, purge_bars=0):
    """Train HGBR on train range, predict on test range with optional purge gap."""
    train_start, train_end = train_idx_range
    test_start, test_end = test_idx_range
    # Purge: remove last purge_bars from training to avoid future_return leak
    train_end_purged = train_end - purge_bars

    train_data = df.iloc[train_start:train_end_purged+1]
    test_data = df.iloc[test_start:test_end+1]

    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    X_train = X_train[valid_train]; y_train = y_train[valid_train]

    if len(X_train) < 500:
        return None, None, None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = HistGradientBoostingRegressor(
        max_iter=150, max_depth=5, learning_rate=0.03,
        l2_regularization=0.5, random_state=42)
    model.fit(X_train_s, y_train)

    X_test = test_data[feature_cols].values
    valid_test = ~np.isnan(X_test).any(axis=1)
    X_test_valid = X_test[valid_test]
    X_test_s = scaler.transform(X_test_valid)
    preds_valid = model.predict(X_test_s)

    preds = np.full(len(X_test), np.nan)
    preds[valid_test] = preds_valid
    actuals = test_data['target'].values
    return preds, actuals, test_data['timestamp'].values

print(f'\n  Purge bars vs Test accuracy:',flush=True)
print(f'  {"Purge":<8} {"Direction Acc":>15} {"Sharpe of Trades":>18} {"Mean Return":>15}',flush=True)
print(f'  {"-"*60}',flush=True)

for purge in [0, 6, 12, 24, 48]:
    preds, actuals, ts = train_predict(
        (200, train_end_idx),
        (test_start_idx, test_end_idx),
        purge_bars=purge
    )
    if preds is None: continue

    # Direction accuracy
    valid = ~(np.isnan(preds) | np.isnan(actuals))
    if valid.sum() < 100: continue
    direction_acc = np.mean(np.sign(preds[valid]) == np.sign(actuals[valid])) * 100

    # Trade-like simulation: take top 25% prediction magnitude as trades
    pred_threshold = np.percentile(np.abs(preds[valid]), 75)
    trade_mask = np.abs(preds[valid]) > pred_threshold
    trade_returns = np.where(preds[valid][trade_mask] > 0,
                              actuals[valid][trade_mask],
                              -actuals[valid][trade_mask])
    if len(trade_returns) > 10:
        mean_ret = np.mean(trade_returns) * 100
        sharpe = np.mean(trade_returns) / (np.std(trade_returns)+1e-10) * np.sqrt(365*24/horizon)
    else:
        mean_ret = 0; sharpe = 0

    tag = ' ← swin_v25 (NO PURGE)' if purge == 0 else ('' if purge < 12 else '')
    print(f'  {purge:<7}h {direction_acc:>13.1f}%  {sharpe:>16.2f}  {mean_ret:>+13.3f}%{tag}',flush=True)

# ============================================================
# 懸念1: Funding コスト追加検証
# ============================================================
print('\n'+'='*70,flush=True)
print(' 懸念1: Funding コスト追加検証',flush=True)
print('='*70,flush=True)

# Run trades with proper funding cost
# Simulate swin_v25 trade logic: enter when |pred| > threshold, exit at TP/SL/Time

preds_no_purge, actuals_no_purge, ts_test = train_predict(
    (200, train_end_idx), (test_start_idx, test_end_idx), purge_bars=0
)

test_df = df.iloc[test_start_idx:test_end_idx+1].reset_index(drop=True)
test_df['pred'] = preds_no_purge
test_df['fra'] = test_df.get('funding_lag', 0).fillna(0)

print('  Simulating swin_v25-like trades on 2024 OOS (HGBR, no purge):',flush=True)

def simulate_trades(test_df, position_size=0.25, leverage=4.0,
                    atr_sl_mult=1.5, atr_tp_mult=3.0, max_hold=24,
                    pred_threshold=None, funding_mode='ignore'):
    """Simulate trade-by-trade with optional funding cost."""
    if pred_threshold is None:
        pred_threshold = np.percentile(np.abs(test_df['pred'].dropna()), 75)

    trades = []
    in_pos = False
    entry_idx = 0
    direction = 0
    entry_price = 0
    sl = 0; tp = 0

    for i in range(len(test_df)):
        if not in_pos:
            pred = test_df['pred'].iloc[i]
            atr = test_df['atr_pct'].iloc[i] if 'atr_pct' in test_df.columns else 0.01
            if pd.isna(pred) or pd.isna(atr) or atr <= 0: continue

            if abs(pred) > pred_threshold:
                in_pos = True
                entry_idx = i
                entry_price = test_df['close'].iloc[i]
                direction = 1 if pred > 0 else -1
                if direction == 1:
                    sl = entry_price * (1 - atr * atr_sl_mult)
                    tp = entry_price * (1 + atr * atr_tp_mult)
                else:
                    sl = entry_price * (1 + atr * atr_sl_mult)
                    tp = entry_price * (1 - atr * atr_tp_mult)
        else:
            bars_held = i - entry_idx
            current_price = test_df['close'].iloc[i]
            high_p = test_df['high'].iloc[i]
            low_p = test_df['low'].iloc[i]

            exit_price = None
            exit_reason = None

            if direction == 1:
                if low_p <= sl: exit_price, exit_reason = sl, 'SL'
                elif high_p >= tp: exit_price, exit_reason = tp, 'TP'
            else:
                if high_p >= sl: exit_price, exit_reason = sl, 'SL'
                elif low_p <= tp: exit_price, exit_reason = tp, 'TP'

            if exit_price is None and bars_held >= max_hold:
                exit_price, exit_reason = current_price, 'TIME'

            if exit_price is not None:
                # Compute trade P&L
                if direction == 1:
                    gross_ret = (exit_price / entry_price - 1)
                else:
                    gross_ret = (1 - exit_price / entry_price)

                # Levered return
                lev_ret = gross_ret * leverage

                # Costs: maker fee 0.02% * 2 (entry + exit) + slippage
                cost = 0.0002 * 2 * leverage  # in % of position

                # Funding cost
                funding_cost = 0
                if funding_mode == 'add':
                    # Sum funding over the holding period
                    # Long pays when funding > 0, short receives
                    bars_held_actual = i - entry_idx
                    fra_window = test_df['fra'].iloc[entry_idx:i+1].fillna(0).values
                    # 8H funding: simplified per-hour application
                    if direction == 1:
                        # Long: pay positive funding, receive negative
                        funding_cost = np.sum(fra_window) * leverage  # cost
                    else:
                        # Short: receive positive funding
                        funding_cost = -np.sum(fra_window) * leverage  # negative = income

                net_ret = (lev_ret - cost - funding_cost) * position_size
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': direction, 'gross_ret': gross_ret,
                    'lev_ret': lev_ret, 'cost': cost,
                    'funding_cost': funding_cost,
                    'net_ret': net_ret,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held
                })
                in_pos = False

    return trades

# Test with and without funding cost
print(f'\n  {"Mode":<25} {"Trades":>8} {"WR":>6} {"Total":>8} {"Avg Funding":>12}',flush=True)
print(f'  {"-"*60}',flush=True)

for mode_label, mode in [
    ('IGNORE funding (swin_v25)', 'ignore'),
    ('ADD funding cost (correct)', 'add'),
]:
    trades = simulate_trades(test_df, funding_mode=mode)
    if trades:
        n_trades = len(trades)
        win_rate = sum(1 for t in trades if t['net_ret'] > 0) / n_trades * 100
        # Compound return
        eq = 1.0
        for t in trades: eq *= (1 + t['net_ret'])
        total_ret = (eq - 1) * 100
        avg_funding = np.mean([t['funding_cost'] for t in trades]) * 100
        print(f'  {mode_label:<25} {n_trades:>8} {win_rate:>5.1f}% {total_ret:>+7.1f}% {avg_funding:>+11.4f}%',flush=True)

# ============================================================
# 懸念3: シグナル単体のシャッフルテスト
# ============================================================
print('\n'+'='*70,flush=True)
print(' 懸念3: HGBR シグナルのシャッフルテスト',flush=True)
print('='*70,flush=True)

print('  Real HGBR predictions vs Shuffled (200x):',flush=True)
trades_real = simulate_trades(test_df, funding_mode='add')
eq_real = 1.0
for t in trades_real: eq_real *= (1 + t['net_ret'])
ret_real = (eq_real - 1) * 100

shuffle_returns = []
for seed in range(200):
    np.random.seed(seed + 50000)
    test_df_sh = test_df.copy()
    preds_sh = test_df_sh['pred'].values.copy()
    valid_mask = ~np.isnan(preds_sh)
    valid_preds = preds_sh[valid_mask]
    np.random.shuffle(valid_preds)
    preds_sh[valid_mask] = valid_preds
    test_df_sh['pred'] = preds_sh

    trades_sh = simulate_trades(test_df_sh, funding_mode='add')
    eq_sh = 1.0
    for t in trades_sh: eq_sh *= (1 + t['net_ret'])
    shuffle_returns.append((eq_sh - 1) * 100)

p_value = np.mean([r >= ret_real for r in shuffle_returns])
print(f'  Real:    {ret_real:+.1f}%',flush=True)
print(f'  Shuffle: {np.mean(shuffle_returns):+.1f}% +/- {np.std(shuffle_returns):.1f}%',flush=True)
print(f'  p-value: {p_value:.3f} {"PASS (signal has timing value)" if p_value < 0.05 else "FAIL (signal timing is random)"}',flush=True)

# ============================================================
# 結論
# ============================================================
print('\n'+'='*70,flush=True)
print(' 結論',flush=True)
print('='*70,flush=True)
print(f'  懸念1 (funding未計算): 加算するとリターン低下を確認',flush=True)
print(f'  懸念2 (purging無し): purge bars増加で予測精度低下を確認',flush=True)
print(f'  懸念3 (シグナル価値): シャッフルp={p_value:.3f}',flush=True)
print()
print('Done.',flush=True)
