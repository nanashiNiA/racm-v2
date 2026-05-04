"""V4 ★3 Multi-Strategy Capital Allocator
==========================================
構成:
- Funding Farm 1.5x (確定: 年+52%, MDD-6%, Sharpe 10.8) → 50%
- Simple Swing (RSI/Funding contrarian + ATR stop) → 30%
- Cash/USDT lending (年5%想定) → 20%

検証:
- 個別 vs ブレンド比較
- 月次リバランス
- 各戦略の相関
- ストレス期間 (2022 Bear) でのパフォーマンス
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h

print('='*70,flush=True)
print('  V4 ★3 Multi-Strategy Capital Allocator',flush=True)
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
high_price=spot_aligned['high'].values
low_price=spot_aligned['low'].values
fra=np.roll(bybit_aligned['funding_rate'].fillna(0).values, 1)
ret=np.diff(spot_price,prepend=spot_price[0])/spot_price

n=len(common_idx)
S=2760

# RSI
delta=pd.Series(spot_price).diff()
gain=delta.clip(lower=0).rolling(14).mean()
loss=(-delta.clip(upper=0)).rolling(14).mean()
rsi=(100-100/(1+gain/(loss+1e-10))).shift(1).fillna(50).values

# ATR
tr=(pd.Series(high_price)-pd.Series(low_price)).rolling(24,min_periods=6).mean()
atr_pct=(tr/pd.Series(spot_price).shift(1)).fillna(0.01).values

print(f'  Data: {n} bars, S={S}',flush=True)

# ============================================================
# Strategy 1: Funding Farm 1.5x (確定済み)
# ============================================================
def run_funding_farm(lev=1.5, fee_bps=2, slip_bps=2, rebal_days=7):
    eq=np.ones(n);e=1.0
    fee_rate=fee_bps/10000;slip_rate=slip_bps/10000
    capital_unit=1+1/lev
    e *= (1 - 2*(fee_rate+slip_rate)/capital_unit)
    perp_unrealized=0.0
    bars_per_rebal=rebal_days*24
    for i in range(S, n):
        if i>0 and spot_price[i-1]>0 and perp_price[i-1]>0:
            spot_ret=spot_price[i]/spot_price[i-1]-1
            perp_ret=perp_price[i]/perp_price[i-1]-1
        else:
            spot_ret=0;perp_ret=0
        spot_pnl=spot_ret*1.0
        perp_pnl=-perp_ret*1.0
        funding_income=fra[i]*1.0
        perp_unrealized += perp_pnl + funding_income
        if perp_unrealized < -1/lev * 0.5:
            e *= max(0.001, 1 + perp_unrealized/capital_unit)
            perp_unrealized=0
            e *= (1 - 2*(fee_rate+slip_rate)/capital_unit)
            continue
        bar_return=(spot_pnl+perp_pnl+funding_income)/capital_unit
        e *= (1+bar_return)
        if (i-S) % bars_per_rebal == 0 and (i-S)>0:
            rebal_cost=abs(perp_unrealized)*0.5*(fee_rate+slip_rate)*2
            e *= (1 - rebal_cost/capital_unit)
            perp_unrealized=0
        eq[i]=e
    return eq

# ============================================================
# Strategy 2: Simple Swing (RSI/Funding contrarian, ATR stop)
# Funding コストを正しく加算
# ============================================================
def run_simple_swing(position_size=0.5, lev=2.0, atr_sl=1.5, atr_tp=3.0,
                      max_hold=24, fee_bps=2, slip_bps=2):
    """
    Long when oversold (RSI<30) + funding negative (shorts paying)
    Short when overbought (RSI>70) + funding positive (longs paying)
    ATR stops, max hold 24H
    """
    eq=np.ones(n);e=1.0
    fee_rate=fee_bps/10000;slip_rate=slip_rate=slip_bps/10000

    in_pos=False
    entry_idx=0;entry_price=0;direction=0;sl=0;tp=0
    trades_count=0
    wins=0

    for i in range(S, n):
        if not in_pos:
            r=rsi[i];f=fra[i];a=atr_pct[i]
            if a<=0:continue
            entry_signal=0
            # Long: oversold + negative funding
            if r<30 and f<-0.0001:
                entry_signal=1
            # Short: overbought + positive funding
            elif r>70 and f>0.0001:
                entry_signal=-1
            if entry_signal!=0:
                in_pos=True
                entry_idx=i
                entry_price=spot_price[i]
                direction=entry_signal
                if direction==1:
                    sl=entry_price*(1-a*atr_sl);tp=entry_price*(1+a*atr_tp)
                else:
                    sl=entry_price*(1+a*atr_sl);tp=entry_price*(1-a*atr_tp)
        else:
            bars_held=i-entry_idx
            cur_h=high_price[i];cur_l=low_price[i];cur_c=spot_price[i]
            exit_price=None
            if direction==1:
                if cur_l<=sl:exit_price=sl
                elif cur_h>=tp:exit_price=tp
            else:
                if cur_h>=sl:exit_price=sl
                elif cur_l<=tp:exit_price=tp
            if exit_price is None and bars_held>=max_hold:
                exit_price=cur_c

            if exit_price is not None:
                # Compute trade return
                if direction==1:
                    gross=(exit_price/entry_price-1)
                else:
                    gross=(1-exit_price/entry_price)
                lev_ret=gross*lev

                # Cost: maker fees + slippage
                cost=2*(fee_rate+slip_rate)*lev

                # Funding cost during hold
                fra_window=fra[entry_idx:i+1]
                if direction==1:
                    funding_cost=np.sum(fra_window)*lev  # long pays positive
                else:
                    funding_cost=-np.sum(fra_window)*lev  # short receives positive

                net_ret=(lev_ret - cost - funding_cost)*position_size
                e *= (1+net_ret)
                trades_count+=1
                if net_ret>0:wins+=1
                in_pos=False

        eq[i]=e

    return eq, trades_count, wins

# ============================================================
# Run individual strategies
# ============================================================
print('\n'+'='*70,flush=True)
print(' INDIVIDUAL STRATEGIES',flush=True)
print('='*70,flush=True)

eq_ff = run_funding_farm(lev=1.5)
eq_sw, n_trades, n_wins = run_simple_swing(position_size=0.5, lev=2.0)
# Cash: simple 5% annual yield
cash_rate_per_bar = 0.05/(365*24)
eq_cash = np.array([(1+cash_rate_per_bar)**(i-S) if i>=S else 1.0 for i in range(n)])
# BTC B&H
eq_bh = np.ones(n);e=1.0
for i in range(S, n):e *= (1+ret[i]);eq_bh[i]=e

def stats(eq, label):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S, n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=common_idx[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

print(f'  {"Strategy":<30} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*60}',flush=True)
for label,eq in [('BTC B&H',eq_bh),('Funding Farm 1.5x',eq_ff),
                  (f'Simple Swing (n={n_trades},wr={n_wins/max(1,n_trades)*100:.0f}%)',eq_sw),
                  ('Cash 5% yield',eq_cash)]:
    a,m,s=stats(eq,label)
    print(f'  {label:<30} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}',flush=True)

# ============================================================
# Multi-Strategy Portfolio
# ============================================================
print('\n'+'='*70,flush=True)
print(' MULTI-STRATEGY PORTFOLIOS',flush=True)
print('='*70,flush=True)

def blend(weights_dict, eqs):
    """weights_dict: {name: weight}, eqs: {name: equity_array}"""
    eq_p = np.ones(n);e=1.0
    rebal_bars = 30*24  # monthly rebal
    last_rebal = S
    weights = weights_dict.copy()
    for i in range(S, n):
        # Compute bar return for each strategy
        bar_returns = {}
        for name, eq in eqs.items():
            if i>0 and eq[i-1]>0:
                bar_returns[name]=eq[i]/eq[i-1]-1
            else:
                bar_returns[name]=0
        # Weighted return
        port_ret = sum(weights[name]*bar_returns[name] for name in weights)
        e *= (1+port_ret)
        eq_p[i]=e
    return eq_p

eqs = {
    'FF': eq_ff,
    'SW': eq_sw,
    'CASH': eq_cash,
    'BH': eq_bh,
}

print(f'  {"Portfolio":<40} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*68}',flush=True)

portfolios = [
    ('60% FF + 40% Cash (defensive)', {'FF':0.6,'CASH':0.4}),
    ('70% FF + 30% Swing', {'FF':0.7,'SW':0.3}),
    ('50% FF + 30% Swing + 20% Cash', {'FF':0.5,'SW':0.3,'CASH':0.2}),
    ('40% FF + 30% Swing + 30% B&H', {'FF':0.4,'SW':0.3,'BH':0.3}),
    ('100% FF (1.5x)', {'FF':1.0}),
    ('100% B&H', {'BH':1.0}),
]

for label, w in portfolios:
    eq_p = blend(w, eqs)
    a,m,s = stats(eq_p, label)
    tag=' ✓' if a>50 and m>=-15 else ''
    print(f'  {label:<40} {a:>+9.1f}% {m:>+7.1f}% {s:>7.2f}{tag}',flush=True)

# ============================================================
# Year-by-year (best portfolio)
# ============================================================
print('\n'+'='*70,flush=True)
print(' YEAR-BY-YEAR (50% FF + 30% Swing + 20% Cash)',flush=True)
print('='*70,flush=True)

eq_best = blend({'FF':0.5,'SW':0.3,'CASH':0.2}, eqs)
print(f'  {"Year":<6} {"Best Port":>12} {"FF only":>12} {"BH":>12}',flush=True)
for yr in range(2021,2026):
    iy = np.where(np.array([d.year==yr for d in common_idx]))[0]
    iy_a = iy[iy>=S]
    if len(iy_a)<100: continue
    rb = (eq_best[iy_a[-1]]/eq_best[max(0,iy_a[0]-1)]-1)*100
    rf = (eq_ff[iy_a[-1]]/eq_ff[max(0,iy_a[0]-1)]-1)*100
    rh = (eq_bh[iy_a[-1]]/eq_bh[max(0,iy_a[0]-1)]-1)*100
    print(f'  {yr:<6} {rb:>+11.1f}% {rf:>+11.1f}% {rh:>+11.1f}%',flush=True)

# ============================================================
# Strategy correlations
# ============================================================
print('\n'+'='*70,flush=True)
print(' STRATEGY CORRELATIONS (daily returns)',flush=True)
print('='*70,flush=True)

daily_rets = {}
for name, eq in eqs.items():
    daily = pd.Series(eq,index=common_idx).resample('1D').last().pct_change().dropna()
    daily = daily[daily.index>=common_idx[S]]
    daily_rets[name] = daily

corr_df = pd.DataFrame(daily_rets).corr()
print(corr_df.to_string(),flush=True)

print('\nDone.',flush=True)
