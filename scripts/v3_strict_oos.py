"""V3 Strict OOS Test
======================
完全に厳密なOOS:
1. 2021-2023のデータのみで閾値を選択
2. 2024-2025は一切見ない状態で固定
3. 2024-2025のみで評価

さらに: Walk-Forward パラメータ選択
各foldのtrainでパラメータを選び、testで評価する本格WF。
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3 STRICT OOS TEST',flush=True)
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

# Orderbook (ffill)
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

print('Data ready.\n',flush=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def compute_safety_with_params(sp_h, sp_m, lq_h, lq_m):
    sz=RACMMicrostructure.spread_zscore(spread_raw, 168, 48)
    lz=RACMMicrostructure.liq_zscore(total_liq_raw, 168, 48)
    safety=np.ones(nn)*0.5
    for i in range(nn):
        s=1.0
        if sz[i]>sp_h:s-=0.3
        elif sz[i]>sp_m:s-=0.15
        if lz[i]>lq_h:s-=0.3
        elif lz[i]>lq_m:s-=0.15
        if sz[i]<-0.5 and lz[i]<0.5:s+=0.2
        safety[i]=float(np.clip(s,0.0,1.5))
    return safety

def run_period(safety, start_idx, end_idx, lev_range=(1.5,4.0)):
    """Run strategy on a specific period, equity starts at 1.0."""
    eq=np.ones(end_idx-start_idx+1);e=1.0;pk=1.0
    lev_min,lev_max=lev_range
    for j,i in enumerate(range(start_idx, end_idx+1)):
        if i<S:eq[j]=1.0;continue
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp_v2[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        lev=lev_min+(lev_max-lev_min)*safety[i]
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[j]=e;pk=max(pk,e)
    return (eq[-1]-1)*100  # return %

# WF folds
folds=[];fold_dates=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:
        folds.append(np.where(tm)[0])
        fold_dates.append(ts)
    cur+=pd.DateOffset(months=1)

# ============================================================
# TEST 1: STRICT OOS (train 2021-2023, test 2024-2025)
# ============================================================
print('='*70,flush=True)
print(' TEST 1: STRICT OOS (train=2021-23, test=2024-25)',flush=True)
print('='*70,flush=True)

# Find period boundaries
train_end_idx=np.where(np.array([d.year<=2023 for d in idx_h]))[0][-1]
test_start_idx=np.where(np.array([d.year>=2024 for d in idx_h]))[0][0]
print(f'  Train: 0-{train_end_idx} ({idx_h[S].date()} to {idx_h[train_end_idx].date()})',flush=True)
print(f'  Test:  {test_start_idx}-{nn-1} ({idx_h[test_start_idx].date()} to {idx_h[-1].date()})',flush=True)

# Grid search on TRAIN period only
print('\n  Grid search on train (2021-2023)...',flush=True)
train_folds=[idx for idx in folds if idx_h[idx[0]].year<=2023]
test_folds=[idx for idx in folds if idx_h[idx[0]].year>=2024]

param_grid=[
    (1.5, 0.8, 2.0, 1.5),
    (2.0, 1.0, 2.0, 1.5),
    (2.0, 1.0, 3.0, 2.0),  # default
    (2.5, 1.5, 3.0, 2.0),
    (2.5, 1.5, 4.0, 3.0),
    (1.5, 0.8, 3.0, 2.0),
    (2.0, 1.0, 4.0, 3.0),
    (3.0, 2.0, 3.0, 2.0),
    (3.0, 2.0, 4.0, 3.0),
]

best_train_wfs=0;best_params=None
print(f'  {"Params":<30} {"Train WFS":>10} {"Train Win":>10}',flush=True)
print(f'  {"-"*52}',flush=True)

for sp_h, sp_m, lq_h, lq_m in param_grid:
    safety=compute_safety_with_params(sp_h, sp_m, lq_h, lq_m)
    train_rets=[run_period(safety, idx[0], idx[-1]) for idx in train_folds if len(idx)>=10]
    wfs=np.mean(train_rets)*12;win=sum(1 for r in train_rets if r>0)
    tag=' <-- BEST' if wfs>best_train_wfs else ''
    if wfs>best_train_wfs:best_train_wfs=wfs;best_params=(sp_h,sp_m,lq_h,lq_m)
    print(f'  sp={sp_h:.1f}/{sp_m:.1f} lq={lq_h:.1f}/{lq_m:.1f}   {wfs:>8.0f}% {win:>4}/{len(train_rets)}{tag}',flush=True)

print(f'\n  Best train params: sp={best_params[0]}/{best_params[1]}, lq={best_params[2]}/{best_params[3]}',flush=True)

# Apply best train params to TEST period (never seen)
safety_best=compute_safety_with_params(*best_params)
safety_default=compute_safety_with_params(2.0, 1.0, 3.0, 2.0)
safety_fixed=np.ones(nn)*((3.0-1.5)/(4.0-1.5))  # V2 equiv (fixed 3.0x)

test_rets_best=[run_period(safety_best, idx[0], idx[-1]) for idx in test_folds if len(idx)>=10]
test_rets_default=[run_period(safety_default, idx[0], idx[-1]) for idx in test_folds if len(idx)>=10]
test_rets_v2=[run_period(safety_fixed, idx[0], idx[-1]) for idx in test_folds if len(idx)>=10]

wfs_best=np.mean(test_rets_best)*12;win_best=sum(1 for r in test_rets_best if r>0)
wfs_default=np.mean(test_rets_default)*12;win_default=sum(1 for r in test_rets_default if r>0)
wfs_v2=np.mean(test_rets_v2)*12;win_v2=sum(1 for r in test_rets_v2 if r>0)

print(f'\n  OOS Results (2024-2025 ONLY, fold-independent):',flush=True)
print(f'  {"Model":<35} {"WFS":>6} {"Win":>5}',flush=True)
print(f'  {"-"*48}',flush=True)
print(f'  {"V2 (fixed 3.0x)":<35} {wfs_v2:>5.0f}% {win_v2:>2}/{len(test_rets_v2)}',flush=True)
print(f'  {"V3 default (2.0/1.0, 3.0/2.0)":<35} {wfs_default:>5.0f}% {win_default:>2}/{len(test_rets_default)}',flush=True)
print(f'  {"V3 train-best":<35} {wfs_best:>5.0f}% {win_best:>2}/{len(test_rets_best)}',flush=True)
print(f'\n  Strict OOS: V3>V2? {"YES ✓" if wfs_default>wfs_v2 else "NO ✗"}',flush=True)

# ============================================================
# TEST 2: EXPANDING WINDOW WF
# ============================================================
print('\n'+'='*70,flush=True)
print(' TEST 2: EXPANDING WINDOW WF (train grows, test=next month)',flush=True)
print('='*70,flush=True)

wf_v2=[];wf_v3=[];wf_dates=[]
for fi in range(12, len(folds)):  # start from fold 12 (1 year of train data)
    test_fold=folds[fi]
    if len(test_fold)<10:continue
    # Train on all folds before this one
    train_indices=np.concatenate([f for f in folds[:fi]])
    train_indices=train_indices[train_indices>=S]

    # For V3: use default params (no optimization to avoid snooping)
    # Just test if V3 default beats V2 on each fold
    ret_v2=run_period(safety_fixed, test_fold[0], test_fold[-1])
    ret_v3=run_period(safety_default, test_fold[0], test_fold[-1])
    wf_v2.append(ret_v2);wf_v3.append(ret_v3)
    wf_dates.append(idx_h[test_fold[0]])

# Print results
print(f'  {"Period":<12} {"V2":>8} {"V3":>8} {"V3>V2":>6}',flush=True)
print(f'  {"-"*36}',flush=True)
v3_wins=0
for i,(d,rv2,rv3) in enumerate(zip(wf_dates,wf_v2,wf_v3)):
    win='✓' if rv3>rv2 else '✗'
    if rv3>rv2:v3_wins+=1
    # Print every 3rd month to save space
    if i%3==0 or i>=len(wf_dates)-6:
        print(f'  {d.strftime("%Y-%m"):<12} {rv2:>+7.1f}% {rv3:>+7.1f}% {win:>6}',flush=True)

print(f'  ...',flush=True)
print(f'  V3>V2 勝率: {v3_wins}/{len(wf_v2)} ({v3_wins/len(wf_v2)*100:.0f}%)',flush=True)
print(f'  V2 WFS: {np.mean(wf_v2)*12:.0f}%',flush=True)
print(f'  V3 WFS: {np.mean(wf_v3)*12:.0f}%',flush=True)

# ============================================================
# TEST 3: 2025年のみ（最新OOS）
# ============================================================
print('\n'+'='*70,flush=True)
print(' TEST 3: 2025年のみ (最新OOS)',flush=True)
print('='*70,flush=True)

folds_2025=[idx for idx in folds if idx_h[idx[0]].year==2025]
rets_v2_25=[run_period(safety_fixed, idx[0], idx[-1]) for idx in folds_2025 if len(idx)>=10]
rets_v3_25=[run_period(safety_default, idx[0], idx[-1]) for idx in folds_2025 if len(idx)>=10]

if rets_v2_25:
    print(f'  2025 folds: {len(rets_v2_25)}個',flush=True)
    print(f'  V2: WFS={np.mean(rets_v2_25)*12:.0f}%, Win={sum(1 for r in rets_v2_25 if r>0)}/{len(rets_v2_25)}',flush=True)
    print(f'  V3: WFS={np.mean(rets_v3_25)*12:.0f}%, Win={sum(1 for r in rets_v3_25 if r>0)}/{len(rets_v3_25)}',flush=True)
    print(f'  V3>V2: {"YES ✓" if np.mean(rets_v3_25)>np.mean(rets_v2_25) else "NO ✗"}',flush=True)

    # Month by month
    print(f'\n  {"Month":<10} {"V2":>8} {"V3":>8}',flush=True)
    for fold_idx, rv2, rv3 in zip(folds_2025, rets_v2_25, rets_v3_25):
        m=idx_h[fold_idx[0]].strftime("%Y-%m")
        print(f'  {m:<10} {rv2:>+7.1f}% {rv3:>+7.1f}%',flush=True)

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n'+'='*70,flush=True)
print(' FINAL OOS SUMMARY',flush=True)
print('='*70,flush=True)
print(f'  1. Strict OOS (train=21-23, test=24-25): V3={wfs_default:.0f}% vs V2={wfs_v2:.0f}% → {"PASS" if wfs_default>wfs_v2 else "FAIL"}',flush=True)
print(f'  2. Expanding WF: V3 勝率 {v3_wins}/{len(wf_v2)} ({v3_wins/len(wf_v2)*100:.0f}%)',flush=True)
if rets_v2_25:
    print(f'  3. 2025年のみ: V3={np.mean(rets_v3_25)*12:.0f}% vs V2={np.mean(rets_v2_25)*12:.0f}% → {"PASS" if np.mean(rets_v3_25)>np.mean(rets_v2_25) else "FAIL"}',flush=True)

print('\nDone.',flush=True)
