"""V3 Integration Verification
================================
racm_core.pyに統合したRACMMicrostructureクラスが
スクリプト版と同一の結果を出すか検証。
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3 INTEGRATION VERIFICATION',flush=True)
print('='*70,flush=True)

# Load data
print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams()
S=params.warmup_hours;fra=np.roll(h['funding'].fillna(0).values,1)
vol_30d=h['vol_30d'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)

# V2 regime (using existing build_bp)
bp_v2=RACMRegime.compute(price, ret, params)

# Load orderbook (ffill = leak-safe)
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

print('Computing V3 features via racm_core...',flush=True)

# Use RACMMicrostructure class
sz=RACMMicrostructure.spread_zscore(spread_raw, params.ms_z_window, params.ms_z_min_periods)
lz=RACMMicrostructure.liq_zscore(total_liq_raw, params.ms_z_window, params.ms_z_min_periods)
safety=RACMMicrostructure.safety_score(sz, lz,
    sp_thresh_high=params.ms_sp_thresh_high, sp_thresh_med=params.ms_sp_thresh_med,
    liq_thresh_high=params.ms_liq_thresh_high, liq_thresh_med=params.ms_liq_thresh_med,
    boost_sp_thresh=params.ms_boost_sp, boost_liq_thresh=params.ms_boost_liq)

print(f'  spread_z range: [{np.percentile(sz[S:],5):.2f}, {np.percentile(sz[S:],95):.2f}]',flush=True)
print(f'  liq_z range: [{np.percentile(lz[S:],5):.2f}, {np.percentile(lz[S:],95):.2f}]',flush=True)
print(f'  safety range: [{np.percentile(safety[S:],5):.2f}, {np.percentile(safety[S:],95):.2f}]',flush=True)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

# Run V3 backtest
def run_v3(bp, safety_arr, params):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        pnl_base=dw*ret[i]*bp_i+lw*lp1h[i]

        lev=RACMMicrostructure.dynamic_leverage(safety_arr[i], params.ms_lev_min, params.ms_lev_max)
        pnl=pnl_base*lev+fra[i]*abs(lev)

        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(params.dd_levels):
            if dd<level:pnl*=mult;break

        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# Run V2 baseline
def run_v2(bp):
    eq=np.ones(nn);e=1.0;pk=1.0
    lev=3.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(params.dd_levels):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

eq_v2=run_v2(bp_v2)
eq_v3=run_v3(bp_v2, safety, params)

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

# Results
print('\n'+' RESULTS '.center(70,'='),flush=True)
print(f'  {"Model":<25} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6}',flush=True)
print(f'  {"-"*55}',flush=True)

wfs2,win2,nf2,b22_2,o25_2,mdd2=eval_strat(eq_v2)
wfs3,win3,nf3,b22_3,o25_3,mdd3=eval_strat(eq_v3)

print(f'  {"V2 (fixed 3.0x)":<25} {wfs2:>4.0f}% {win2:>2}/{nf2} {b22_2:>+5.0f}% {o25_2:>+5.0f}% {mdd2:>+5.1f}%',flush=True)
print(f'  {"V3-A (MS DynLev)":<25} {wfs3:>4.0f}% {win3:>2}/{nf3} {b22_3:>+5.0f}% {o25_3:>+5.0f}% {mdd3:>+5.1f}%',flush=True)

# Expected values from previous runs
exp_wfs=588;exp_b22=117
print(f'\n  Expected: WFS~{exp_wfs}%, B22~+{exp_b22}%',flush=True)
print(f'  Got:      WFS={wfs3:.0f}%, B22={b22_3:+.0f}%',flush=True)
wfs_ok=abs(wfs3-exp_wfs)<20
b22_ok=abs(b22_3-exp_b22)<15
print(f'  Match: WFS {"✓" if wfs_ok else "✗"}, B22 {"✓" if b22_ok else "✗"}',flush=True)

if wfs_ok and b22_ok:
    print(f'\n  ✓ INTEGRATION VERIFIED — racm_core.RACMMicrostructure is correct',flush=True)
else:
    print(f'\n  ⚠️ Values differ from expected — check implementation',flush=True)

print('\nDone.',flush=True)
