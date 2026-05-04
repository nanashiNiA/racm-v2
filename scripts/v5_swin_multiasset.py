"""V5: swin_v25 Multi-Asset Extension
=======================================
swin_v25 (BTC 2024 +154% NET, 2022 +52% NET) は funding バグの影響なし
→ 真のalphaの可能性が高い

Multi-Asset 拡張で 300%目標を狙う:
- BTC + ETH + SOL の 3資産
- 各資産で並行 swin_v25 trading
- Capital 1/3 each
- 各 +85%/yr の同時実行 → 合計 +255%/yr 期待
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from datetime import datetime

print('='*70,flush=True)
print('  V5: swin_v25 Multi-Asset (300% honest 目標)',flush=True)
print('='*70,flush=True)

# Tardis bybit perpetual で BTC のみ → ETH/SOL は別途データ必要
# まず BTC のみで multi-year 検証完了確認
# 次に 同じ ML model を別資産に適用可能か判定

print('\nNote: Tardis Bybit data は BTC のみ',flush=True)
print('ETH/SOL の swin_v25 適用には別途 data 必要',flush=True)
print('代替案:',flush=True)
print('  1. Binance perp 1H bars + funding (我々がもっているもの)',flush=True)
print('  2. v2core load_6assets_1h() で 6資産1H ロード',flush=True)

# Load 6 assets data
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
from v2core.data_loader import load_6assets_1h

print('\nLoading 6 assets 1H...',flush=True)
a1=load_6assets_1h()
print(f'  Available assets: {list(a1.keys())}',flush=True)
for a,df in a1.items():
    print(f'    {a}: {len(df)} bars, {df.index.min()} → {df.index.max()}',flush=True)

# ============================================================
# Build features (swin_v25 style, simplified)
# ============================================================
def build_features(df, lookback_bars=24):
    """Simplified swin_v25 features"""
    feat=df.copy()
    close_s=feat['close'].shift(1)
    feat['return_1']=close_s.pct_change(1)
    feat['return_4']=close_s.pct_change(4)
    feat['return_24']=close_s.pct_change(24)

    # SMA
    feat['sma_50']=close_s.rolling(50,min_periods=25).mean()
    feat['sma_200']=close_s.rolling(200,min_periods=100).mean()
    feat['sma_dist']=(close_s-feat['sma_200'])/(feat['sma_200']+1e-10)

    # ATR
    high_s=feat['high'].shift(1) if 'high' in feat else close_s*1.005
    low_s=feat['low'].shift(1) if 'low' in feat else close_s*0.995
    tr=np.maximum(high_s-low_s, np.maximum((high_s-close_s.shift(1)).abs(), (low_s-close_s.shift(1)).abs()))
    feat['atr_14']=pd.Series(tr).rolling(14,min_periods=7).mean().values
    feat['atr_pct']=feat['atr_14']/(close_s+1e-10)

    # RSI
    delta=close_s.diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    feat['rsi']=100-100/(1+gain/(loss+1e-10))

    # Vol
    feat['vol_24']=feat['return_1'].rolling(24,min_periods=12).std()

    # MA cross
    ema8=close_s.ewm(span=8).mean()
    ema24=close_s.ewm(span=24).mean()
    feat['ema_diff']=(ema8-ema24)/(close_s+1e-10)

    # Target: future 6-bar return
    feat['target']=feat['close'].shift(-6)/feat['open'].shift(-5)-1
    return feat

# Test on BTC, ETH, SOL
TEST_ASSETS=['BTC','ETH','SOL']
features={}
for a in TEST_ASSETS:
    if a not in a1:continue
    feat=build_features(a1[a])
    features[a]=feat
    print(f'  {a}: features built ({len(feat)} bars)',flush=True)

# ============================================================
# Walk-forward ML for each asset
# ============================================================
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print('Need sklearn',flush=True)
    sys.exit(1)

feature_cols=['return_1','return_4','return_24','sma_dist','atr_pct','rsi','vol_24','ema_diff']

def train_predict_2024(asset, feat_df):
    """Train on 2021-2023, predict on 2024"""
    feat_df.index=feat_df.index.tz_localize('UTC') if feat_df.index.tz is None else feat_df.index
    train_mask=(feat_df.index<pd.Timestamp('2024-01-01',tz='UTC'))&(feat_df.index>=pd.Timestamp('2021-01-01',tz='UTC'))
    test_mask=(feat_df.index>=pd.Timestamp('2024-01-01',tz='UTC'))&(feat_df.index<pd.Timestamp('2025-01-01',tz='UTC'))

    train=feat_df[train_mask]
    test=feat_df[test_mask]

    X_train=train[feature_cols].values
    y_train=train['target'].values
    valid=~(np.isnan(X_train).any(axis=1)|np.isnan(y_train))
    X_train,y_train=X_train[valid],y_train[valid]

    if len(X_train)<500:return None

    scaler=StandardScaler()
    X_tr_s=scaler.fit_transform(X_train)
    model=HistGradientBoostingRegressor(max_iter=150,max_depth=5,learning_rate=0.03,
                                         l2_regularization=0.5,random_state=42)
    model.fit(X_tr_s,y_train)

    X_test=test[feature_cols].values
    valid_t=~np.isnan(X_test).any(axis=1)
    X_test_s=scaler.transform(X_test[valid_t])
    preds=np.full(len(X_test),np.nan)
    preds[valid_t]=model.predict(X_test_s)
    return test, preds

# Trade simulation per asset
def simulate(test_df, preds, lev=4.0, pos_size=0.30, atr_sl=1.5, atr_tp=3.0,
             max_hold=20, fee_bps=2, slip_bps=2):
    test_df=test_df.copy()
    test_df['pred']=preds
    threshold=np.percentile(np.abs(test_df['pred'].dropna()),75)

    in_pos=False;entry_idx=0;entry_price=0;direction=0;sl=0;tp=0
    eq_list=[1.0];e=1.0;trades=0;wins=0
    for i in range(len(test_df)):
        if not in_pos:
            pred=test_df['pred'].iloc[i]
            atr=test_df['atr_pct'].iloc[i] if 'atr_pct' in test_df.columns else 0.01
            if pd.isna(pred) or pd.isna(atr) or atr<=0:eq_list.append(e);continue
            if abs(pred)>threshold:
                in_pos=True
                entry_idx=i
                entry_price=test_df['close'].iloc[i]
                direction=1 if pred>0 else -1
                if direction==1:sl=entry_price*(1-atr*atr_sl);tp=entry_price*(1+atr*atr_tp)
                else:sl=entry_price*(1+atr*atr_sl);tp=entry_price*(1-atr*atr_tp)
        else:
            bars=i-entry_idx
            cur_h=test_df['high'].iloc[i] if 'high' in test_df.columns else test_df['close'].iloc[i]
            cur_l=test_df['low'].iloc[i] if 'low' in test_df.columns else test_df['close'].iloc[i]
            cur_c=test_df['close'].iloc[i]
            exit_p=None
            if direction==1:
                if cur_l<=sl:exit_p=sl
                elif cur_h>=tp:exit_p=tp
            else:
                if cur_h>=sl:exit_p=sl
                elif cur_l<=tp:exit_p=tp
            if exit_p is None and bars>=max_hold:exit_p=cur_c
            if exit_p is not None:
                if direction==1:gross=exit_p/entry_price-1
                else:gross=1-exit_p/entry_price
                cost=2*(fee_bps+slip_bps)/10000*lev
                net=(gross*lev-cost)*pos_size
                e *= (1+net)
                trades+=1
                if net>0:wins+=1
                in_pos=False
        eq_list.append(e)
    return e-1, trades, wins

# Run for each asset
print('\n'+'='*70,flush=True)
print(' Per-Asset Results (2024 OOS, train 2021-2023)',flush=True)
print('='*70,flush=True)
print(f'  {"Asset":<6} {"Net Return":>12} {"Trades":>8} {"WR":>6}',flush=True)
print(f'  {"-"*36}',flush=True)

asset_results={}
for a in TEST_ASSETS:
    if a not in features:continue
    result=train_predict_2024(a, features[a])
    if result is None:continue
    test_df, preds = result
    net_ret, trades, wins = simulate(test_df, preds)
    wr=wins/max(1,trades)*100
    asset_results[a]={'ret':net_ret*100,'trades':trades,'wr':wr}
    print(f'  {a:<6} {net_ret*100:>+11.1f}% {trades:>8} {wr:>5.0f}%',flush=True)

# Multi-asset portfolio (equal weight)
if len(asset_results)>=2:
    avg=np.mean([r['ret'] for r in asset_results.values()])
    sum_ret=sum(r['ret'] for r in asset_results.values())
    print(f'\n  Multi-Asset (equal weight):',flush=True)
    print(f'    Average per-asset: {avg:+.1f}%',flush=True)
    print(f'    Sum (independent): {sum_ret:+.1f}%',flush=True)
    print(f'    Equal-weight portfolio: {avg:+.1f}% (each asset gets 1/{len(asset_results)} capital)',flush=True)

print('\nDone.',flush=True)
