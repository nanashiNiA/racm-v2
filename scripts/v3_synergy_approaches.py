"""V3 Synergy: 短期予測の「別の使い方」を探索
===============================================
従来の失敗: 短期予測の方向をposition overlayに使用 → 全FAIL

新しいアプローチ:
1. Vol予測 → 事前レバ調整（方向ではなく振幅を予測）
2. Regime遷移の先読み（DC>=1.0になる直前を検知）
3. LS リバランスの質の改善（「良いタイミング」で入替）
4. 独立並行戦略の資本配分（RACM 80% + swin 20%）
5. 1H内の最良約定タイミング（VWAP的アプローチ）
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3 SYNERGY: 短期予測の新しい活用法',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=params.warmup_hours
fra=np.roll(h['funding'].fillna(0).values,1)
vol_30d=h['vol_30d'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
bp_v2=RACMRegime.compute(price,ret,params)

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
                    if c in ohlc.columns:ohlc[c]=ohlc[c].fillna(0)
            ob_frames.append(ohlc)
ob_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ob_all=ob_all.drop_duplicates('timestamp').set_index('timestamp')
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
ob_safe=ob_all.reindex(idx_h_utc,method='ffill')
spread_raw=ob_safe['spread_pct'].fillna(0).values
total_liq_raw=(ob_safe['long_liq_usd'].fillna(0).values+ob_safe['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in ob_safe.columns else np.zeros(nn))
sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety=RACMMicrostructure.safety_score(sz,lz)

dc_arr=np.zeros(nn)
for i in range(S,nn):
    dl=ds=0
    if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:dl+=1
    if not np.isnan(ma20[i-1]) and price[i-1]<ma20[i-1]:ds+=1
    if not np.isnan(skew_a[i-1]) and skew_a[i-1]<-0.5:dl+=1;ds+=1
    if dd_a[i-1]<-0.12:dl+=1;ds+=1
    dc_arr[i]=dl*0.3+ds*0.7

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

print('Data ready.\n',flush=True)

# ============================================================
# APPROACH 1: Volatility Prediction → Pre-emptive Lev Cut
# ============================================================
print('='*70,flush=True)
print(' APPROACH 1: Vol予測 → 事前レバ調整',flush=True)
print('='*70,flush=True)
print('  次の1-4Hの|return|を予測し、大きい振幅が来る前にレバ縮小',flush=True)

# "Vol predictor": past features → future |return|
# Use simple features: recent vol trend, spread, liq
# Target: |ret[i+1:i+4]|.max() > threshold → "high vol coming"
future_vol=np.zeros(nn)
for i in range(S,nn-4):
    future_vol[i]=np.max(np.abs(ret[i+1:i+5]))  # next 4 bars max |return|

# Feature: can we predict this from past data?
# Use: vol_30d trend, spread_z, liq_z, recent ret magnitude
vol_pred_score=np.zeros(nn)
recent_vol=pd.Series(np.abs(ret)).rolling(6,min_periods=3).mean().values  # last 6H vol
vol_trend=np.zeros(nn)
for i in range(S,nn):
    if i>168:
        v_now=np.mean(np.abs(ret[max(S,i-24):i]))
        v_past=np.mean(np.abs(ret[max(S,i-168):i-24]))
        vol_trend[i]=v_now/(v_past+1e-10)

# Simple rule: high recent vol + high spread + high liq → vol likely to continue
for i in range(S,nn):
    score=0
    if vol_trend[i]>1.5: score+=1  # vol accelerating
    if sz[i]>1.0: score+=1  # spread widening
    if lz[i]>1.5: score+=1  # liquidations happening
    if np.abs(ret[i-1])>0.02: score+=1  # big recent move
    vol_pred_score[i]=score

# Evaluate: does vol_pred_score predict future_vol?
print('  Vol prediction accuracy:',flush=True)
for thresh in [0,1,2,3]:
    mask=(vol_pred_score[S:nn-4]>=thresh)
    if mask.sum()>0:
        avg_fvol_high=np.mean(future_vol[S:nn-4][mask])
        avg_fvol_low=np.mean(future_vol[S:nn-4][~mask])
        ratio=avg_fvol_high/(avg_fvol_low+1e-10)
        print(f'    score>={thresh}: avg future |ret|={avg_fvol_high*100:.3f}% vs {avg_fvol_low*100:.3f}% (ratio={ratio:.2f})',flush=True)

# Use as safety modifier: high vol_pred → reduce leverage
def run_vol_pred(cut_thresh=2, cut_mult=0.7):
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
        # Vol prediction cut
        if vol_pred_score[i]>=cut_thresh:
            lev*=cut_mult
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# APPROACH 2: Regime Transition Early Warning
# ============================================================
print('\n'+'='*70,flush=True)
print(' APPROACH 2: Regime遷移の先読み',flush=True)
print('='*70,flush=True)
print('  DCが0.5→1.5に上がる「前」を検知できれば損失回避',flush=True)

# Pre-danger signal: 1-8H前にdanger上昇を予測
# Features: MA distance shrinking, skew deteriorating, DD deepening
pre_danger=np.zeros(nn)
ma110_dist=np.zeros(nn)
for i in range(S,nn):
    if not np.isnan(ma110[i-1]) and ma110[i-1]>0:
        ma110_dist[i]=(price[i-1]-ma110[i-1])/ma110[i-1]

# Signal: price approaching MA110 from above + skew deteriorating
for i in range(S,nn):
    score=0
    # Price close to MA110 and falling
    if 0<ma110_dist[i]<0.03 and ret[i-1]<0: score+=1  # within 3% above MA, falling
    if 0<ma110_dist[i]<0.01: score+=1  # very close to MA
    # Skew deteriorating
    if not np.isnan(skew_a[i-1]) and skew_a[i-1]<-0.3: score+=1
    # DD deepening
    if dd_a[i-1]<-0.08: score+=1
    # Vol increasing
    if vol_trend[i]>1.3: score+=1
    pre_danger[i]=score

# How early does pre_danger fire before actual dc>=1.5?
# Find transitions: dc goes from <1.0 to >=1.5
transitions=[]
for i in range(S+1,nn):
    if dc_arr[i]>=1.5 and dc_arr[i-1]<1.0:
        transitions.append(i)

print(f'  Regime transitions found: {len(transitions)}',flush=True)
if transitions:
    lead_times=[]
    for t in transitions:
        # How many bars before transition does pre_danger>=2?
        for lead in range(1,25):
            if t-lead>=S and pre_danger[t-lead]>=2:
                lead_times.append(lead);break
        else:
            lead_times.append(0)
    detected=sum(1 for l in lead_times if l>0)
    print(f'  Detected {detected}/{len(transitions)} transitions with >=2H lead',flush=True)
    if lead_times:
        print(f'  Average lead time: {np.mean([l for l in lead_times if l>0]):.1f}H',flush=True)

# Use pre-danger as early lev reduction
def run_early_warning(thresh=3, cut=0.5):
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
        # Early warning cut (before V2 regime reacts)
        if pre_danger[i]>=thresh and dc_arr[i]<1.5:  # pre-danger but V2 hasn't reacted yet
            lev*=cut
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# APPROACH 3: LS Rebalance Quality (skip bad timing)
# ============================================================
print('\n'+'='*70,flush=True)
print(' APPROACH 3: LS リバランス品質向上',flush=True)
print('='*70,flush=True)
print('  高ボラ/スプレッド拡大時のリバランスをスキップ',flush=True)

# Skip LS rebalancing when conditions are bad
# Bad conditions: high spread, high vol, big recent move
# This means: hold previous LS position longer when market is chaotic

# Simulate: use 8H LS but delay rebalancing when spread is high
ls_quality_pnl=np.zeros(nn)
asset_names=list(ar8.keys())
prev_long='';prev_short=''
for lb_days in [60,90]:
    lb=lb_days*3;w=0.5/len([60,90])
    am={n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in ar8.items()}
    av={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in ar8.items()}
    for i in range(200,len(c8)):
        ms=[(am[n][i]/(av[n][i]+1e-10),n) for n in ar8 if not np.isnan(am[n][i])]
        if len(ms)<3:continue
        ms.sort(key=lambda x:x[0],reverse=True)
        new_long=ms[0][1];new_short=ms[-1][1]
        # Check if we should rebalance (skip if spread high)
        ts8=c8[i]
        j=np.searchsorted(idx_h_utc,ts8.tz_localize('UTC') if ts8.tzinfo is None else ts8)
        skip=False
        if j<nn and sz[j]>1.5:skip=True  # high spread → skip rebalance
        if not skip:
            prev_long=new_long;prev_short=new_short
        # Use current (possibly held) positions
        if prev_long and prev_short:
            lp_this=(ar8[prev_long][i]-ar8[prev_short][i])/(2*len(ar8))
        else:
            lp_this=(ar8.get(new_long,np.zeros(len(c8)))[i]-ar8.get(new_short,np.zeros(len(c8)))[i])/(2*len(ar8))
        # Map to 1H
        ts_end=c8[i+1] if i+1<len(c8) else ts8+pd.Timedelta(hours=8)
        idxs=np.where((idx_h>=ts8)&(idx_h<ts_end))[0]
        for ii in idxs:
            ls_quality_pnl[ii]+=lp_this*w/max(1,len(idxs))

# ============================================================
# APPROACH 4: Dual Strategy Capital Allocation
# ============================================================
print('\n'+'='*70,flush=True)
print(' APPROACH 4: 独立並行戦略 (RACM + Swing)',flush=True)
print('='*70,flush=True)
print('  RACMとswin_v25を別々に動かし、equity curveをブレンド',flush=True)

# Simulate swin_v25-like: trade every signal with ATR stop
# Simplified: buy when RSI<30 + OBI bullish, sell when RSI>70
# With ATR-based TP/SL
rsi_14=h['rsi_14d'].values if 'rsi_14d' in h.columns else np.full(nn,50)
obi_raw=ob_safe['imbalance_0'].fillna(0).values if 'imbalance_0' in ob_safe.columns else np.zeros(nn)
atr_raw=pd.Series(np.abs(ret)).rolling(24,min_periods=6).mean().values

swing_pnl=np.zeros(nn)
in_trade=False;entry_price=0;direction=0;bars_held=0;stop_loss=0;take_profit=0
for i in range(S,nn):
    if in_trade:
        bars_held+=1
        if direction==1:
            if price[i]<=stop_loss or price[i]>=take_profit or bars_held>=48:
                swing_pnl[i]=(price[i]/entry_price-1)*0.15  # 15% of portfolio
                in_trade=False
        else:
            if price[i]>=stop_loss or price[i]<=take_profit or bars_held>=48:
                swing_pnl[i]=(1-price[i]/entry_price)*0.15
                in_trade=False
    else:
        rsi=rsi_14[i-1] if not np.isnan(rsi_14[i-1]) else 50
        obi=obi_raw[i-1] if not np.isnan(obi_raw[i-1]) else 0
        atr=atr_raw[i-1] if not np.isnan(atr_raw[i-1]) else 0.01
        # Entry conditions
        if rsi<30 and obi>0.05 and not in_trade:
            in_trade=True;direction=1;entry_price=price[i]
            stop_loss=entry_price*(1-atr*1.5);take_profit=entry_price*(1+atr*3)
            bars_held=0
        elif rsi>70 and obi<-0.05 and not in_trade:
            in_trade=True;direction=-1;entry_price=price[i]
            stop_loss=entry_price*(1+atr*1.5);take_profit=entry_price*(1-atr*3)
            bars_held=0

# ============================================================
# RESULTS
# ============================================================
def eval_full(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else 0
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return wfs,win,len(wf),b22,o25,mdd*100

# Base V3
def run_base():
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
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# V3 + swing parallel
def run_with_swing(swing_weight=0.15):
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
        pnl_racm=(dw*ret[i]*bp_i+lw*lp1h[i])*lev+fra[i]*abs(lev)
        pnl=(1-swing_weight)*pnl_racm+swing_weight*swing_pnl[i] if swing_pnl[i]!=0 else pnl_racm
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# V3 + quality LS
def run_quality_ls():
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
        # Use quality-filtered LS instead of standard
        pnl=(dw*ret[i]*bp_i+lw*ls_quality_pnl[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

print('\n'+' ALL APPROACHES COMPARISON '.center(70,'='),flush=True)
hdr=f'  {"Model":<45} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6}'
print(hdr,flush=True);print('  '+'-'*75,flush=True)

eq_base=run_base()
tests=[
    ('V3 Base (reference)', eq_base),
    ('1a) Vol pred cut (thresh=2, cut=0.7)', run_vol_pred(2, 0.7)),
    ('1b) Vol pred cut (thresh=3, cut=0.5)', run_vol_pred(3, 0.5)),
    ('2a) Early warning (thresh=3, cut=0.5)', run_early_warning(3, 0.5)),
    ('2b) Early warning (thresh=4, cut=0.3)', run_early_warning(4, 0.3)),
    ('3) LS Quality (skip high-spread rebal)', run_quality_ls()),
    ('4a) +Swing 15%', run_with_swing(0.15)),
    ('4b) +Swing 25%', run_with_swing(0.25)),
]

for label,eq in tests:
    wfs,win,nf,b22,o25,mdd=eval_full(eq)
    tag=' <--' if 'reference' in label else (' ***' if wfs>375 and b22>100 else '')
    print(f'  {label:<45} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# Shuffle test on best improvement
print('\n  Shuffle test on best approaches (200x):',flush=True)
# Vol pred shuffle
eq_vp=run_vol_pred(2,0.7)
wfs_vp,_,_,b22_vp,_,_=eval_full(eq_vp)
wfs_sh=[]
for seed in range(200):
    np.random.seed(seed+500000)
    vps=vol_pred_score.copy()
    np.random.shuffle(vps[S:])
    # Temporarily replace
    vol_pred_score_orig=vol_pred_score.copy()
    vol_pred_score[:]=vps
    eq_s=run_vol_pred(2,0.7)
    vol_pred_score[:]=vol_pred_score_orig
    w,_,_,_,_,_=eval_full(eq_s)
    wfs_sh.append(w)
p=np.mean([w>=wfs_vp for w in wfs_sh])
print(f'    Vol Pred: WFS={wfs_vp:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p:.3f} {"PASS" if p<0.05 else "FAIL"}',flush=True)

# Early warning shuffle
eq_ew=run_early_warning(3,0.5)
wfs_ew,_,_,b22_ew,_,_=eval_full(eq_ew)
wfs_sh2=[]
for seed in range(200):
    np.random.seed(seed+600000)
    pds=pre_danger.copy()
    np.random.shuffle(pds[S:])
    pre_danger_orig=pre_danger.copy()
    pre_danger[:]=pds
    eq_s=run_early_warning(3,0.5)
    pre_danger[:]=pre_danger_orig
    w,_,_,_,_,_=eval_full(eq_s)
    wfs_sh2.append(w)
p2=np.mean([w>=wfs_ew for w in wfs_sh2])
print(f'    Early Warning: WFS={wfs_ew:.0f}%, Shuffle={np.mean(wfs_sh2):.0f}%±{np.std(wfs_sh2):.0f}%, p={p2:.3f} {"PASS" if p2<0.05 else "FAIL"}',flush=True)

print('\nDone.',flush=True)
