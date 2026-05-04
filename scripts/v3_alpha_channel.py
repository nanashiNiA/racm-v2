"""V3+ Alpha Channel: 平常時に毎バー小さく稼ぐ
==================================================
RACMのベースポジションに重ねる短期アルファチャンネル。
手数料0 (Lighter.xyz) を活かして高頻度で小さな利益を積む。

テスト:
A) 1H Mean Reversion: RSI/BB過熱→逆張りoverlay (±0.3x)
B) 1H LS: 8H→1Hリバランスで追加alpha
C) Funding micro-tilt: funding方向に微調整
D) 板インバランス短期: OBI方向にmicro-tilt
E) Volatility compression breakout: 圧縮からのブレイクアウト

各チャンネルは小サイズ (0.1-0.5x) でRACMに干渉しない。
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3+ ALPHA CHANNEL: 平常時アルファ探索',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=params.warmup_hours
fra=np.roll(h['funding'].fillna(0).values,1)
vol_30d=h['vol_30d'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
bp_v2=RACMRegime.compute(price,ret,params)

# 1H assets for LS
a1=load_6assets_1h();c1=list(a1.values())[0].index
for df in a1.values():c1=c1.intersection(df.index)
c1=c1.intersection(idx_h)
ar_al={}
for a in a1:
    arr=np.zeros(nn);rvals=a1[a].loc[c1,'return'].values
    for i,ts in enumerate(c1):
        j=np.searchsorted(idx_h,ts)
        if j<nn and i<len(rvals):arr[j]=rvals[i]
    ar_al[a]=arr

# Orderbook
cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
ob_frames=[]
for year in range(2021,2026):
    f=os.path.join(cache_dir,f'BTCUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(f):
        with open(f,'rb') as fp:
            ohlc,ticker,liq=pickle.load(fp)
        if ohlc is not None:
            ohlc['timestamp']=pd.to_datetime(ohlc['timestamp'],utc=True)
            if liq is not None:
                liq['timestamp']=pd.to_datetime(liq['timestamp'],utc=True)
                ohlc=ohlc.merge(liq,on='timestamp',how='left')
                for c in ['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']:
                    if c in ohlc.columns: ohlc[c]=ohlc[c].fillna(0)
            ob_frames.append(ohlc)
ob_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ob_all=ob_all.drop_duplicates('timestamp').set_index('timestamp')
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
ob_safe=ob_all.reindex(idx_h_utc, method='ffill')
spread_raw=ob_safe['spread_pct'].fillna(0).values
total_liq_raw=(ob_safe['long_liq_usd'].fillna(0).values+ob_safe['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in ob_safe.columns else np.zeros(nn))
obi_raw=ob_safe['imbalance_0'].fillna(0).values if 'imbalance_0' in ob_safe.columns else np.zeros(nn)

sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety=RACMMicrostructure.safety_score(sz,lz)

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
print('Data ready.\n',flush=True)

# ============================================================
# Pre-compute alpha channel signals
# ============================================================
print('Computing alpha channels...',flush=True)

# A) Mean Reversion (RSI + BB)
rsi_14=h['rsi_14d'].values if 'rsi_14d' in h.columns else np.full(nn,50)
bb_mean=pd.Series(price).rolling(480,min_periods=240).mean().values
bb_std=pd.Series(price).rolling(480,min_periods=240).std().values

mr_signal=np.zeros(nn)  # -0.3 to +0.3
for i in range(S,nn):
    if np.isnan(bb_mean[i-1]) or bb_std[i-1]<=0:continue
    z=(price[i-1]-bb_mean[i-1])/(bb_std[i-1]*2+1e-10)
    rsi=rsi_14[i-1] if not np.isnan(rsi_14[i-1]) else 50
    if z<-1.0 and rsi<30: mr_signal[i]=0.3
    elif z<-0.5 and rsi<40: mr_signal[i]=0.15
    elif z>1.0 and rsi>70: mr_signal[i]=-0.3
    elif z>0.5 and rsi>60: mr_signal[i]=-0.15

# B) 1H LS momentum (higher frequency rebalancing)
print('  B) 1H LS...',flush=True)
asset_names=list(ar_al.keys())
na=len(asset_names)
# 1H momentum scores (30d lookback)
ls_1h_pnl=np.zeros(nn)
mom_1h={a: np.roll(pd.Series(ar_al[a]).rolling(720,min_periods=240).sum().values,1) for a in asset_names}
vol_1h={a: np.roll(pd.Series(np.abs(ar_al[a])).rolling(168,min_periods=48).mean().values,1) for a in asset_names}
for i in range(S,nn):
    scores=[]
    for a in asset_names:
        m=mom_1h[a][i];v=vol_1h[a][i]
        if not np.isnan(m) and v>1e-10:
            scores.append((m/v, a, ar_al[a][i]))
    if len(scores)<3:continue
    scores.sort(key=lambda x:x[0], reverse=True)
    # Long top, short bottom (equal weight)
    ls_1h_pnl[i]=(scores[0][2]-scores[-1][2])/(2*na)

# Excess over 8H LS (the "additional" alpha from 1H)
ls_1h_excess=ls_1h_pnl-lp1h  # 1H LS minus 8H LS already in V3

# C) Funding micro-tilt
funding_raw=h['funding'].fillna(0).values
funding_ma=pd.Series(funding_raw).rolling(168,min_periods=24).mean().values
funding_tilt=np.zeros(nn)
for i in range(S,nn):
    fv=funding_ma[i-1] if not np.isnan(funding_ma[i-1]) else 0
    # Opposite of funding: short when funding positive (earn carry)
    if fv>0.0002: funding_tilt[i]=-0.15
    elif fv>0.0001: funding_tilt[i]=-0.08
    elif fv<-0.0002: funding_tilt[i]=0.15
    elif fv<-0.0001: funding_tilt[i]=0.08

# D) OBI micro-tilt
obi_ma=np.roll(pd.Series(obi_raw).rolling(24,min_periods=6).mean().values,1)
obi_signal=np.zeros(nn)
for i in range(S,nn):
    ob=obi_ma[i] if not np.isnan(obi_ma[i]) else 0
    if ob>0.15: obi_signal[i]=0.2  # strong bid → micro long
    elif ob>0.05: obi_signal[i]=0.1
    elif ob<-0.15: obi_signal[i]=-0.2  # strong ask → micro short
    elif ob<-0.05: obi_signal[i]=-0.1

# E) Volatility compression breakout
atr_24=pd.Series(np.abs(ret)).rolling(24,min_periods=6).mean().values
atr_168=pd.Series(np.abs(ret)).rolling(168,min_periods=48).mean().values
vol_ratio=np.zeros(nn)
for i in range(S,nn):
    if atr_168[i-1]>1e-8:
        vol_ratio[i]=atr_24[i-1]/atr_168[i-1]

donch_high=pd.Series(price).rolling(480,min_periods=120).max().values
donch_low=pd.Series(price).rolling(480,min_periods=120).min().values
breakout_signal=np.zeros(nn)
for i in range(S,nn):
    if vol_ratio[i]<0.6:  # compression
        if price[i-1]>=donch_high[i-1]*0.995: breakout_signal[i]=0.3  # breakout up
        elif price[i-1]<=donch_low[i-1]*1.005: breakout_signal[i]=-0.3  # breakout down

print('Channels computed.',flush=True)

# ============================================================
# WF folds
# ============================================================
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

# ============================================================
# Backtest: V3 base + alpha channels
# ============================================================
def run_v3_plus(alpha_pnl_arr=None, alpha_overlay_arr=None, label=''):
    """
    alpha_pnl_arr: additional PnL per bar (e.g., 1H LS excess)
    alpha_overlay_arr: position overlay (e.g., +0.3 = add 0.3x long on BTC)
    """
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp_v2[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        lev=1.5+(3.0-1.5)*safety[i]

        # Base RACM PnL
        pnl_base=(dw*ret[i]*bp_i+lw*lp1h[i])*lev

        # Alpha channel: additional PnL
        pnl_alpha=0
        if alpha_pnl_arr is not None:
            pnl_alpha+=alpha_pnl_arr[i]*lev  # scaled by leverage
        if alpha_overlay_arr is not None:
            pnl_alpha+=ret[i]*alpha_overlay_arr[i]  # overlay = direct BTC position

        pnl=pnl_base+pnl_alpha+fra[i]*abs(lev)

        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

def eval_fi(eq):
    """Fold-independent evaluation"""
    fold_rets=[]
    for fold_idx in folds:
        if len(fold_idx)<10:continue
        r=(eq[fold_idx[-1]]/eq[max(0,fold_idx[0]-1)]-1)*100
        fold_rets.append(r)
    wfs=np.mean(fold_rets)*12 if fold_rets else 0
    win=sum(1 for r in fold_rets if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else 0
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return wfs,win,len(fold_rets),b22,mdd*100

# ============================================================
# RESULTS
# ============================================================
print('\n'+' RESULTS '.center(70,'='),flush=True)
print(f'  {"Model":<40} {"WFS":>5} {"Win":>5} {"B22":>6} {"MDD":>6}',flush=True)
print(f'  {"-"*64}',flush=True)

# Base V3
eq_base=run_v3_plus()
wfs_base,win_base,nf,b22_base,mdd_base=eval_fi(eq_base)
print(f'  {"V3 Base (no alpha)":<40} {wfs_base:>4.0f}% {win_base:>2}/{nf} {b22_base:>+5.0f}% {mdd_base:>+5.1f}%',flush=True)

tests=[
    ('A) +Mean Reversion (±0.3x)', None, mr_signal),
    ('B) +1H LS excess', ls_1h_excess, None),
    ('C) +Funding micro-tilt', None, funding_tilt),
    ('D) +OBI micro-tilt', None, obi_signal),
    ('E) +Vol compression breakout', None, breakout_signal),
    # Combinations
    ('A+C (MR + Funding)', None, mr_signal+funding_tilt),
    ('A+D (MR + OBI)', None, mr_signal+obi_signal),
    ('A+E (MR + Breakout)', None, mr_signal+breakout_signal),
    ('B+C (1H LS + Funding)', ls_1h_excess, funding_tilt),
    ('A+C+E (MR+Fund+Break)', None, mr_signal+funding_tilt+breakout_signal),
    ('ALL (A+B+C+D+E)', ls_1h_excess, mr_signal+funding_tilt+obi_signal+breakout_signal),
]

for label,alpha_pnl,alpha_overlay in tests:
    eq=run_v3_plus(alpha_pnl, alpha_overlay)
    wfs,win,nf,b22,mdd=eval_fi(eq)
    tag=''
    if wfs>wfs_base+5 and b22>=b22_base-5 and mdd>=-30: tag=' ***'
    elif wfs>wfs_base: tag=' +'
    print(f'  {label:<40} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# Standalone alpha channel performance (no RACM)
# ============================================================
print('\n'+' STANDALONE ALPHA CHANNELS '.center(70,'='),flush=True)
print('  (RACMなし、各チャンネル単独でBTCを取引した場合)',flush=True)

def run_standalone(signal, label):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        pnl=ret[i]*signal[i]
        # No cost (Lighter 0bps)
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

for label,sig in [
    ('A) Mean Reversion', mr_signal),
    ('C) Funding tilt', funding_tilt),
    ('D) OBI tilt', obi_signal),
    ('E) Breakout', breakout_signal),
]:
    eq=run_standalone(sig, label)
    wfs,win,nf,b22,mdd=eval_fi(eq)
    print(f'  {label:<30} WFS={wfs:>4.0f}%, B22={b22:>+5.0f}%, MDD={mdd:>+5.1f}%',flush=True)

# ============================================================
# Shuffle test on best alpha channel
# ============================================================
print('\n'+' SHUFFLE TEST (best channel, 200x) '.center(70,'='),flush=True)

# Find best channel
best_label=None;best_wfs=wfs_base;best_alpha_pnl=None;best_alpha_overlay=None
for label,alpha_pnl,alpha_overlay in tests:
    eq=run_v3_plus(alpha_pnl, alpha_overlay)
    wfs,_,_,_,_=eval_fi(eq)
    if wfs>best_wfs:
        best_wfs=wfs;best_label=label;best_alpha_pnl=alpha_pnl;best_alpha_overlay=alpha_overlay

if best_label:
    print(f'  Best: {best_label} (WFS={best_wfs:.0f}%)',flush=True)
    eq_real=run_v3_plus(best_alpha_pnl, best_alpha_overlay)
    wfs_real,_,_,_,_=eval_fi(eq_real)
    wfs_sh=[]
    for seed in range(200):
        np.random.seed(seed+100000)
        if best_alpha_overlay is not None:
            ov=best_alpha_overlay.copy()
            np.random.shuffle(ov[S:])
            eq_s=run_v3_plus(best_alpha_pnl, ov)
        elif best_alpha_pnl is not None:
            ap=best_alpha_pnl.copy()
            np.random.shuffle(ap[S:])
            eq_s=run_v3_plus(ap, None)
        else:
            eq_s=eq_real
        w,_,_,_,_=eval_fi(eq_s)
        wfs_sh.append(w)
    p=np.mean([w>=wfs_real for w in wfs_sh])
    print(f'  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p:.3f} {"PASS" if p<0.05 else "FAIL"}',flush=True)
else:
    print('  No channel improved over V3 base.',flush=True)

print('\nDone.',flush=True)
