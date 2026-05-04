"""OOS 2025: +93% → +315% の差分を正直に分解
==================================================
V24報告時: OOS 2025 = +93% (fold-independent)
V3現在:    OOS 2025 = +315% (continuous)

これは本当の改善か？数字のトリックか？
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  OOS 2025: +93% → +315% の正直な分解',flush=True)
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
sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety=RACMMicrostructure.safety_score(sz,lz)

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
print('Data ready.\n',flush=True)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)
folds_2025=[idx for idx in folds if idx_h[idx[0]].year==2025]

# ============================================================
# 計測方法の違い
# ============================================================
def run_continuous(safety_arr, lev_min, lev_max, dd_levels):
    """連続equity — DD制御が前年からの利益を引き継ぐ"""
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

def run_fold_independent_2025(safety_arr, lev_min, lev_max, dd_levels):
    """Fold-independent — 各月でequity=1.0リセット"""
    fold_rets=[]
    for fold_idx in folds_2025:
        if len(fold_idx)<10:continue
        e=1.0;pk=1.0
        for i in fold_idx:
            if i<S:continue
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
            e*=(1+pnl);pk=max(pk,e)
        fold_rets.append((e-1)*100)
    return fold_rets

def year_ret_continuous(eq, yr):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:return 0
    return (eq[iy[-1]]/eq[max(0,iy[0]-1)]-1)*100

# ============================================================
# DECOMPOSITION
# ============================================================
print('='*70,flush=True)
print(' STEP 1: 計測方法の違い (Continuous vs Fold-Independent)',flush=True)
print('='*70,flush=True)

# V3 (1.5-3.0x)
eq_v3_cont=run_continuous(safety, 1.5, 3.0, dd_default)
fi_v3=run_fold_independent_2025(safety, 1.5, 3.0, dd_default)
oos_v3_cont=year_ret_continuous(eq_v3_cont, 2025)
oos_v3_fi=np.mean(fi_v3)*12 if fi_v3 else 0

# V2 (fixed 3.0x)
eq_v2_cont=run_continuous(np.ones(nn)*1.0, 3.0, 3.0, dd_default)
fi_v2=run_fold_independent_2025(np.ones(nn)*1.0, 3.0, 3.0, dd_default)
oos_v2_cont=year_ret_continuous(eq_v2_cont, 2025)
oos_v2_fi=np.mean(fi_v2)*12 if fi_v2 else 0

print(f'  {"計測方法":<25} {"V2 (3.0x)":>10} {"V3 (1.5-3.0x)":>14}',flush=True)
print(f'  {"-"*52}',flush=True)
print(f'  {"Continuous (報告値)":<25} {oos_v2_cont:>+9.0f}% {oos_v3_cont:>+13.0f}%',flush=True)
print(f'  {"Fold-Independent":<25} {oos_v2_fi:>+9.0f}% {oos_v3_fi:>+13.0f}%',flush=True)
print(f'  {"差分":<25} {oos_v2_cont-oos_v2_fi:>+9.0f}% {oos_v3_cont-oos_v3_fi:>+13.0f}%',flush=True)

print(f'\n  ⚠️ Continuous版は2024年までの利益で peak_equity が膨大',flush=True)
print(f'     → DD制御の閾値(-15%/-22%/-30%)が実質的に発火しない',flush=True)
print(f'     → 2025年で-15%落ちても、全体では+3000%なので制御不要',flush=True)
print(f'     → これが「フルレバのまま走れる」= 高リターンの原因',flush=True)

# ============================================================
# STEP 2: 各fold詳細 (fold-independent)
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 2: 2025年 月別 (fold-independent)',flush=True)
print('='*70,flush=True)

print(f'  {"Month":<10} {"V2":>8} {"V3":>8} {"V3-V2":>8}',flush=True)
print(f'  {"-"*36}',flush=True)

fi_v2_detail=[]
fi_v3_detail=[]
for fold_idx in folds_2025:
    if len(fold_idx)<10:continue
    # V2
    e=1.0;pk=1.0
    for i in fold_idx:
        if i<S:continue
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp_v2[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*3.0+fra[i]*3.0
        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);pk=max(pk,e)
    fi_v2_detail.append((e-1)*100)
    # V3
    e=1.0;pk=1.0
    for i in fold_idx:
        if i<S:continue
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
        e*=(1+pnl);pk=max(pk,e)
    fi_v3_detail.append((e-1)*100)

    m=idx_h[fold_idx[0]].strftime('%Y-%m')
    print(f'  {m:<10} {fi_v2_detail[-1]:>+7.1f}% {fi_v3_detail[-1]:>+7.1f}% {fi_v3_detail[-1]-fi_v2_detail[-1]:>+7.1f}%',flush=True)

wfs_v2_fi=np.mean(fi_v2_detail)*12
wfs_v3_fi=np.mean(fi_v3_detail)*12
win_v2=sum(1 for r in fi_v2_detail if r>0)
win_v3=sum(1 for r in fi_v3_detail if r>0)

print(f'\n  V2 FI-WFS: {wfs_v2_fi:.0f}%, Win: {win_v2}/{len(fi_v2_detail)}',flush=True)
print(f'  V3 FI-WFS: {wfs_v3_fi:.0f}%, Win: {win_v3}/{len(fi_v3_detail)}',flush=True)

# ============================================================
# STEP 3: V24 vs V2 vs V3 の正確な比較
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 3: 教授報告値との正確な比較',flush=True)
print('='*70,flush=True)

print(f'  教授に報告した V24 (fold-independent):',flush=True)
print(f'    OOS 2025: +93%, Win: 8/12',flush=True)
print(f'    WFS (IS): 367%',flush=True)
print(f'    Bear 2022: +121%',flush=True)
print(f'',flush=True)
print(f'  現在の V3 — 同じ計測方法 (fold-independent):',flush=True)
print(f'    OOS 2025 FI-WFS: {wfs_v3_fi:.0f}%, Win: {win_v3}/{len(fi_v3_detail)}',flush=True)

# Full WFS fold-independent
fi_all_v3=[]
for fold_idx in folds:
    if len(fold_idx)<10:continue
    e=1.0;pk=1.0
    for i in fold_idx:
        if i<S:continue
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
        e*=(1+pnl);pk=max(pk,e)
    fi_all_v3.append((e-1)*100)

wfs_fi_all=np.mean(fi_all_v3)*12
win_fi_all=sum(1 for r in fi_all_v3 if r>0)

# Bear 2022 fold-independent
fi_2022=[]
folds_2022=[idx for idx in folds if idx_h[idx[0]].year==2022]
for fold_idx in folds_2022:
    if len(fold_idx)<10:continue
    e=1.0;pk=1.0
    for i in fold_idx:
        if i<S:continue
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
        e*=(1+pnl);pk=max(pk,e)
    fi_2022.append((e-1)*100)
b22_fi=np.mean(fi_2022)*12 if fi_2022 else 0

print(f'    WFS (全fold FI): {wfs_fi_all:.0f}%, Win: {win_fi_all}/{len(fi_all_v3)}',flush=True)
print(f'    Bear 2022 FI-WFS: {b22_fi:.0f}%',flush=True)

# ============================================================
# FINAL: 正直な比較表
# ============================================================
print('\n'+'='*70,flush=True)
print(' FINAL: 同じ計測方法での比較',flush=True)
print('='*70,flush=True)

print(f'  全てFold-Independent (equity=1.0リセット):',flush=True)
print(f'',flush=True)
print(f'  {"指標":<20} {"V24(教授報告)":>14} {"V3(現在)":>12} {"改善":>10}',flush=True)
print(f'  {"-"*58}',flush=True)
print(f'  {"WFS (全fold)":<20} {"367%":>14} {wfs_fi_all:>11.0f}% {wfs_fi_all-367:>+9.0f}%',flush=True)
print(f'  {"OOS 2025 WFS":<20} {"81%":>14} {wfs_v3_fi:>11.0f}% {wfs_v3_fi-81:>+9.0f}%',flush=True)
print(f'  {"Bear 2022 WFS":<20} {"+121% (IS)":>14} {b22_fi:>11.0f}%',flush=True)
print(f'  {"Win (全fold)":<20} {"39/45 (87%)":>14} {win_fi_all:>8}/{len(fi_all_v3)} ({win_fi_all/len(fi_all_v3)*100:.0f}%)',flush=True)
print(f'',flush=True)
print(f'  ⚠️ Continuous版で報告すると:',flush=True)
print(f'  {"OOS 2025 (cont)":<20} {"+93%(FI)/+177%(C)":>14} {oos_v3_cont:>+11.0f}%',flush=True)
print(f'  → Continuous +315%は2024年までの利益でDD制御が無効化されたバイアス込み',flush=True)

print('\nDone.',flush=True)
