"""V3 + swin_v25 ML Overlay
============================
swin_v25のHGBR予測 + Confidence Scoreを
V3の動的レバに追加統合。

テスト:
1. HGBR予測をWFで生成（Train→Test、リークなし）
2. ML予測 → レバ調整シグナルとして統合
3. V3 (MS only) vs V3+ML vs ML only の比較
4. 板特徴量のML特有の「良さ」: imbalance_change, pressure_diff, depth_ratio
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_HGBR=True
except:
    HAS_HGBR=False
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3 + swin_v25 ML OVERLAY',flush=True)
print('='*70,flush=True)

# ============================================================
# Load all data
# ============================================================
print('Loading RACM base...',flush=True)
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

# Orderbook
print('Loading orderbook...',flush=True)
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
                    if c in ohlc.columns: ohlc[c]=ohlc[c].fillna(0)
            ob_frames.append(ohlc)

ob_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ob_all=ob_all.drop_duplicates('timestamp').set_index('timestamp')
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
ob_safe=ob_all.reindex(idx_h_utc, method='ffill')

print(f'  Orderbook: {len(ob_safe)} bars, columns: {list(ob_safe.columns)}',flush=True)

# MS signals (existing V3)
spread_raw=ob_safe['spread_pct'].fillna(0).values
total_liq_raw=(ob_safe['long_liq_usd'].fillna(0).values+ob_safe['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in ob_safe.columns else np.zeros(nn))
sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety_ms=RACMMicrostructure.safety_score(sz,lz)

# ============================================================
# Build ML features (swin_v25 style, all lag-1)
# ============================================================
print('Building ML features...',flush=True)

feat=pd.DataFrame(index=range(nn))
# Price features (all lag-1 via shift on price)
p=pd.Series(price)
feat['ret_1h']=p.pct_change(1).shift(1).values
feat['ret_4h']=p.pct_change(4).shift(1).values
feat['ret_8h']=p.pct_change(8).shift(1).values
feat['ret_24h']=p.pct_change(24).shift(1).values
feat['ret_7d']=p.pct_change(168).shift(1).values

# Volatility
tr=pd.Series(np.abs(ret))
feat['atr_pct']=(tr.rolling(24,min_periods=6).mean()/p.shift(1)).shift(1).values
feat['vol_z']=((feat['atr_pct']-pd.Series(feat['atr_pct']).rolling(168,min_periods=48).mean())/
               (pd.Series(feat['atr_pct']).rolling(168,min_periods=48).std()+1e-10)).values

# Trend
ema8=p.ewm(span=8).mean().shift(1)
ema24=p.ewm(span=24).mean().shift(1)
ema72=p.ewm(span=72).mean().shift(1)
feat['ema_diff_8_24']=((ema8-ema24)/(p.shift(1)+1e-10)).values
feat['ema_diff_24_72']=((ema24-ema72)/(p.shift(1)+1e-10)).values

# RSI
delta=p.diff().shift(1)
gain=delta.clip(lower=0).rolling(14).mean()
loss=(-delta.clip(upper=0)).rolling(14).mean()
feat['rsi']=(100-100/(1+gain/(loss+1e-10))).values

# ADX (simplified)
feat['adx']=pd.Series(h['adx'].values).shift(1).values if 'adx' in h.columns else 0

# Orderbook features (the "良さ" from swin_v25)
if 'imbalance_0' in ob_safe.columns:
    imb=ob_safe['imbalance_0'].fillna(0).values
    feat['imb_0']=np.roll(imb,1)  # lag-1
    feat['imb_ma12']=np.roll(pd.Series(imb).rolling(12,min_periods=3).mean().values,1)
    feat['imb_change4']=np.roll(pd.Series(imb).diff(4).values,1)

if 'depth_imbalance' in ob_safe.columns:
    depth=ob_safe['depth_imbalance'].fillna(0).values
    feat['depth_imb']=np.roll(depth,1)
    feat['depth_imb_ma']=np.roll(pd.Series(depth).rolling(12,min_periods=3).mean().values,1)

if 'bid_pressure' in ob_safe.columns and 'ask_pressure' in ob_safe.columns:
    bp_ob=ob_safe['bid_pressure'].fillna(0).values
    ap_ob=ob_safe['ask_pressure'].fillna(0).values
    feat['pressure_diff']=np.roll(bp_ob-ap_ob,1)
    feat['pressure_ma']=np.roll(pd.Series(bp_ob-ap_ob).rolling(12,min_periods=3).mean().values,1)

feat['spread_z']=sz  # already lag-1
feat['liq_z']=lz  # already lag-1

if 'funding_rate' in ob_safe.columns:
    fr_ob=ob_safe['funding_rate'].fillna(0).values
    feat['funding_z']=np.roll((pd.Series(fr_ob)-pd.Series(fr_ob).rolling(168,min_periods=48).mean())/
                              (pd.Series(fr_ob).rolling(168,min_periods=48).std()+1e-10),1).astype(float)

if 'open_interest' in ob_safe.columns:
    oi_ob=ob_safe['open_interest'].fillna(method='ffill').fillna(0).values
    feat['oi_change_1h']=np.roll(pd.Series(oi_ob).pct_change(1).values,1)
    feat['oi_change_24h']=np.roll(pd.Series(oi_ob).pct_change(24).values,1)

# Target: next 8-bar return (for ML training)
feat['target']=pd.Series(price).pct_change(8).shift(-8).values  # FUTURE - only for training

# Clean
feat=feat.replace([np.inf,-np.inf],np.nan)
feature_cols=[c for c in feat.columns if c!='target']
print(f'  {len(feature_cols)} features: {feature_cols[:10]}...',flush=True)

# ============================================================
# Walk-Forward HGBR (Train 6M → Predict next 1M)
# ============================================================
print('\nWalk-Forward ML training...',flush=True)
ml_pred=np.zeros(nn)  # ML prediction for each bar
ml_conf=np.zeros(nn)  # prediction confidence (abs percentile)

if not HAS_HGBR:
    print('  ERROR: HistGradientBoostingRegressor not available!',flush=True)
    sys.exit(1)

# WF windows: train 6 months, predict next 1 month
wf_train_months=6
wf_start=pd.Timestamp('2021-07-01')  # first test after 6M train
wf_step=pd.DateOffset(months=1)
cur=wf_start
n_windows=0

while cur<idx_h[-1]:
    train_end=cur
    train_start=cur-pd.DateOffset(months=wf_train_months)
    test_end=cur+wf_step

    train_mask=np.array([(d>=train_start and d<train_end) for d in idx_h])
    test_mask=np.array([(d>=train_end and d<test_end) for d in idx_h])

    train_idx=np.where(train_mask)[0]
    test_idx=np.where(test_mask)[0]

    if len(train_idx)<500 or len(test_idx)<50:
        cur+=wf_step;continue

    # Train data
    X_train=feat.loc[train_idx, feature_cols].values
    y_train=feat.loc[train_idx, 'target'].values

    # Remove NaN rows
    valid=~(np.isnan(X_train).any(axis=1)|np.isnan(y_train))
    X_train=X_train[valid];y_train=y_train[valid]

    if len(X_train)<200:
        cur+=wf_step;continue

    # Scale
    scaler=StandardScaler()
    X_train_s=scaler.fit_transform(X_train)

    # Fit HGBR
    model=HistGradientBoostingRegressor(
        max_iter=150, max_depth=5, learning_rate=0.03,
        l2_regularization=0.5, random_state=42)
    model.fit(X_train_s, y_train)

    # Predict on test
    X_test=feat.loc[test_idx, feature_cols].values
    valid_test=~np.isnan(X_test).any(axis=1)
    if valid_test.sum()>0:
        X_test_s=scaler.transform(X_test[valid_test])
        preds=model.predict(X_test_s)
        j=0
        for i,v in zip(test_idx,valid_test):
            if v:
                ml_pred[i]=preds[j];j+=1

    n_windows+=1
    cur+=wf_step

print(f'  {n_windows} WF windows trained',flush=True)
print(f'  ML predictions coverage: {np.sum(ml_pred!=0)}/{nn} ({np.sum(ml_pred!=0)/nn*100:.1f}%)',flush=True)

# ML confidence: rolling percentile of |prediction|
ml_abs=np.abs(ml_pred)
ml_conf_arr=np.zeros(nn)
for i in range(S,nn):
    if ml_pred[i]==0:continue
    past=ml_abs[max(S,i-720):i]
    past=past[past>0]
    if len(past)>50:
        ml_conf_arr[i]=np.sum(past<=ml_abs[i])/len(past)

# ============================================================
# WF Folds
# ============================================================
folds=[];cur_f=pd.Timestamp('2021-01-01')
while cur_f+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur_f+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur_f+=pd.DateOffset(months=1)

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]

def run_strategy(safety_arr, lev_min, lev_max, dd_levels):
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
        lev=lev_min+(lev_max-lev_min)*safety_arr[i]
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_levels):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

def eval_strat(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return wfs,win,len(wf),b22,o25,mdd*100

# ============================================================
# Build ML-enhanced safety scores
# ============================================================
# ML as directional tilt: positive prediction → boost, negative → cut
safety_ml_only=np.ones(nn)*0.5
for i in range(S,nn):
    if ml_pred[i]==0:safety_ml_only[i]=0.5;continue
    # Scale prediction to safety adjustment
    pred_z=ml_pred[i]*100  # scale up (predictions are small decimals)
    s=1.0
    if pred_z>0.5: s+=0.2  # bullish → boost
    elif pred_z>0.2: s+=0.1
    elif pred_z<-0.5: s-=0.2  # bearish → cut
    elif pred_z<-0.2: s-=0.1
    safety_ml_only[i]=float(np.clip(s,0.3,1.5))

# Combined: MS + ML
safety_ms_ml=np.ones(nn)*0.5
for i in range(S,nn):
    s_ms=safety_ms[i]
    s_ml=safety_ml_only[i]
    # Weighted average: 60% MS (proven) + 40% ML (new)
    safety_ms_ml[i]=float(np.clip(0.6*s_ms+0.4*s_ml, 0.0, 1.5))

# ML confidence-gated: only use ML when confidence is high
safety_ms_ml_conf=np.ones(nn)*0.5
for i in range(S,nn):
    s=safety_ms[i]
    if ml_conf_arr[i]>0.7 and ml_pred[i]!=0:  # high confidence ML
        pred_z=ml_pred[i]*100
        if pred_z>0.3: s=min(s+0.15,1.5)
        elif pred_z<-0.3: s=max(s-0.15,0.0)
    safety_ms_ml_conf[i]=float(np.clip(s,0.0,1.5))

# ============================================================
# RESULTS
# ============================================================
print('\n'+' RESULTS (lev 1.5-3.0x, dd_default) '.center(70,'='),flush=True)

tests=[
    ('V2 fixed 3.0x', np.ones(nn)*1.0, 3.0, 3.0),
    ('V3 MS only (baseline)', safety_ms, 1.5, 3.0),
    ('ML only', safety_ml_only, 1.5, 3.0),
    ('MS + ML (60/40 blend)', safety_ms_ml, 1.5, 3.0),
    ('MS + ML conf-gated', safety_ms_ml_conf, 1.5, 3.0),
]

print(f'  {"Model":<30} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6}',flush=True)
print(f'  {"-"*60}',flush=True)

for label,s_arr,lmin,lmax in tests:
    eq=run_strategy(s_arr,lmin,lmax,dd_default)
    wfs,win,nf,b22,o25,mdd=eval_strat(eq)
    tag=''
    if 'baseline' in label: tag=' <--'
    elif wfs>370 and b22>96: tag=' ***'
    print(f'  {label:<30} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# Year-by-year
print(f'\n  Year-by-year (V3 MS vs V3 MS+ML):',flush=True)
eq_ms=run_strategy(safety_ms,1.5,3.0,dd_default)
eq_msml=run_strategy(safety_ms_ml,1.5,3.0,dd_default)
eq_msml_c=run_strategy(safety_ms_ml_conf,1.5,3.0,dd_default)

print(f'  {"Year":<6} {"V3 MS":>10} {"MS+ML":>10} {"MS+ML conf":>10}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:continue
    r1=(eq_ms[iy[-1]]/eq_ms[max(0,iy[0]-1)]-1)*100
    r2=(eq_msml[iy[-1]]/eq_msml[max(0,iy[0]-1)]-1)*100
    r3=(eq_msml_c[iy[-1]]/eq_msml_c[max(0,iy[0]-1)]-1)*100
    def fmt(v):
        if abs(v)>10000:return f'{v/1000:>+7.0f}K%'
        return f'{v:>+9.1f}%'
    print(f'  {yr:<6} {fmt(r1):>10} {fmt(r2):>10} {fmt(r3):>10}',flush=True)

# Shuffle test on ML signal
print(f'\n  ML Shuffle Test (200x):',flush=True)
eq_real=run_strategy(safety_ms_ml_conf,1.5,3.0,dd_default)
wfs_real,_,_,b22_real,_,_=eval_strat(eq_real)
wfs_sh=[];b22_sh=[]
for seed in range(200):
    np.random.seed(seed+99000)
    ss=safety_ms_ml_conf.copy()
    active=ss[S:].copy()
    np.random.shuffle(active)
    ss[S:]=active
    eq_s=run_strategy(ss,1.5,3.0,dd_default)
    w,_,_,b,_,_=eval_strat(eq_s)
    wfs_sh.append(w);b22_sh.append(b)

p_w=np.mean([w>=wfs_real for w in wfs_sh])
p_b=np.mean([b>=b22_real for b in b22_sh])
print(f'  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p_w:.3f} {"PASS" if p_w<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_sh):+.0f}%±{np.std(b22_sh):.0f}%, p={p_b:.3f} {"PASS" if p_b<0.05 else "FAIL"}',flush=True)

print('\nDone.',flush=True)
