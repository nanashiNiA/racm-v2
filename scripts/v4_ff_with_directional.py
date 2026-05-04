"""V4: Funding Farm + Small Directional Bias
==============================================
FF (delta neutral) に小さな方向性バイアスを追加。
- 強気時: spot を perp より少し多く保有 → 上昇で利益
- 弱気時: perp を spot より少し多く保有 → 下落で利益
- bias size: 5-15% (大きすぎるとFFの安全性失う)

検証:
1. bias size の最適化
2. directional signal の選択 (MA crossover, RSI, momentum)
3. FF baseline との比較
4. シャッフル検証
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4: Funding Farm + Directional Bias',flush=True)
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
fra_ma_30d=pd.Series(fra).rolling(720,min_periods=168).mean().values
fra_annual_pct=fra_ma_30d*365*24*100
n=len(common_idx);S=2760

# Directional signals (lag-1)
ma_50=pd.Series(spot_price).rolling(50,min_periods=25).mean().shift(1).values
ma_200=pd.Series(spot_price).rolling(200,min_periods=100).mean().shift(1).values
rsi_period=14
delta=pd.Series(spot_price).diff()
gain=delta.clip(lower=0).rolling(rsi_period).mean().shift(1)
loss=(-delta.clip(upper=0)).rolling(rsi_period).mean().shift(1)
rsi=(100-100/(1+gain/(loss+1e-10))).fillna(50).values

def get_directional_signal(i, signal_type='ma_cross'):
    """Returns -1 (short bias), 0 (neutral), +1 (long bias)"""
    if signal_type == 'ma_cross':
        if not np.isnan(ma_50[i]) and not np.isnan(ma_200[i]):
            return 1 if ma_50[i] > ma_200[i] else -1
        return 0
    elif signal_type == 'rsi':
        r = rsi[i]
        if r < 30: return 1  # oversold → long
        elif r > 70: return -1  # overbought → short
        return 0
    elif signal_type == 'momentum':
        if i > 168 and spot_price[i-168] > 0:
            mom = spot_price[i-1] / spot_price[i-168] - 1
            if mom > 0.05: return 1
            elif mom < -0.05: return -1
        return 0
    return 0

def get_target_lev(i):
    """Bear-aware leverage"""
    f30 = fra_annual_pct[i] if not np.isnan(fra_annual_pct[i]) else 50
    if f30 > 100: return 2.0
    elif f30 > 50: return 1.5
    elif f30 > 20: return 1.0
    elif f30 > 0: return 0.5
    return 0.001

def run_ff_with_bias(bias_size=0.0, signal_type='ma_cross',
                      fee_bps=2, slip_bps=2, rebal_days=7):
    """
    FF with directional bias:
    - bias_size: 0 = pure delta neutral, 0.1 = ±10% bias
    - When signal=+1 (bullish): spot_size = 1 + bias, perp_size = 1
      → net long bias by `bias` units
    - When signal=-1 (bearish): spot_size = 1, perp_size = 1 + bias
      → net short bias
    """
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=1.5
    bars_per_rebal=rebal_days*24

    for i in range(S,n):
        target_lev = get_target_lev(i)
        signal = get_directional_signal(i, signal_type) if bias_size > 0 else 0

        if (i-S)%bars_per_rebal==0 and (i-S)>0 and abs(target_lev-current_lev)>0.1:
            adj=abs(target_lev-current_lev)/max(current_lev,0.1)*(fr+sr)*2
            cap=1+1/max(current_lev,0.1)
            e *= (1-adj/cap)
            current_lev=target_lev
            perp_un=0

        cap=1+1/max(current_lev,0.1)

        # Position sizing with bias
        if signal == 1:
            spot_size = 1 + bias_size
            perp_size = 1
        elif signal == -1:
            spot_size = 1
            perp_size = 1 + bias_size
        else:
            spot_size = 1
            perp_size = 1

        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
            pret=perp_price[i]/perp_price[i-1]-1
        else:sret=0;pret=0

        sp = sret * spot_size
        pp = -pret * perp_size
        fi = fra[i] * perp_size

        perp_un += pp + fi

        if current_lev > 0.01 and perp_un < -1/current_lev * 0.5:
            e *= max(0.001, 1 + perp_un/cap)
            perp_un = 0
            e *= (1 - 2*(fr+sr)/cap)
            continue

        bar_ret = (sp + pp + fi) / cap
        e *= (1 + bar_ret)
        eq[i] = e

    return eq

def stats(eq):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

# ============================================================
# Test bias sizes
# ============================================================
print(f'\n  {"Strategy":<35} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*60}',flush=True)

for bias in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    for sig in ['ma_cross']:
        eq=run_ff_with_bias(bias_size=bias, signal_type=sig)
        a,m,s=stats(eq)
        label=f'FF + bias {bias:.2f} ({sig})'
        tag=' ★' if a>54 and m>=-10 else ''
        print(f'  {label:<35} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

# Compare signal types at best bias
print(f'\n  Signal type comparison (bias=0.10):',flush=True)
for sig in ['ma_cross','rsi','momentum']:
    eq=run_ff_with_bias(bias_size=0.10, signal_type=sig)
    a,m,s=stats(eq)
    print(f'  bias 0.10 ({sig:<10}): Annual={a:+.1f}%, MDD={m:+.1f}%, Sharpe={s:.2f}',flush=True)

# ============================================================
# Year-by-year for best
# ============================================================
print('\n  Year-by-year:',flush=True)
eq_pure=run_ff_with_bias(bias_size=0.0)
eq_bias=run_ff_with_bias(bias_size=0.10, signal_type='ma_cross')

print(f'  {"Year":<6} {"Pure FF":>12} {"FF+10%bias":>12} {"Diff":>10}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    rp=(eq_pure[iy_a[-1]]/eq_pure[max(0,iy_a[0]-1)]-1)*100
    rb=(eq_bias[iy_a[-1]]/eq_bias[max(0,iy_a[0]-1)]-1)*100
    print(f'  {yr:<6} {rp:>+11.1f}% {rb:>+11.1f}% {rb-rp:>+9.1f}%',flush=True)

# ============================================================
# Shuffle test on best biased
# ============================================================
print('\n'+'='*70,flush=True)
print(' SHUFFLE TEST (bias signal timing)',flush=True)
print('='*70,flush=True)

eq_real=run_ff_with_bias(bias_size=0.10, signal_type='ma_cross')
ann_real=stats(eq_real)[0]
print(f'  Real (bias 0.10 ma_cross): {ann_real:+.1f}%',flush=True)

print('  Shuffling MA values (200x)...',flush=True)
shuffle_anns=[]
ma_50_orig=ma_50.copy();ma_200_orig=ma_200.copy()
for seed in range(200):
    np.random.seed(seed+888888)
    ma50_s=ma_50.copy();ma200_s=ma_200.copy()
    np.random.shuffle(ma50_s[S:]);np.random.shuffle(ma200_s[S:])
    ma_50[:]=ma50_s;ma_200[:]=ma200_s
    eq_s=run_ff_with_bias(bias_size=0.10, signal_type='ma_cross')
    ma_50[:]=ma_50_orig;ma_200[:]=ma_200_orig
    a,_,_=stats(eq_s)
    shuffle_anns.append(a)

p=np.mean([a>=ann_real for a in shuffle_anns])
print(f'  Real: {ann_real:+.1f}%, Shuffle: {np.mean(shuffle_anns):+.1f}%+/-{np.std(shuffle_anns):.1f}%',flush=True)
print(f'  p-value: {p:.3f} {"PASS (bias has alpha)" if p<0.05 else "FAIL"}',flush=True)

print('\nDone.',flush=True)
