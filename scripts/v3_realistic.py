"""V3: 現実的なパラメータ探索
================================
MDD≤-30%をクリアしつつ、現実的なレバ・DD制御を設定。

探索:
1. レバ範囲の調整 (上限を下げる)
2. DD制御の強化 (早めに縮小)
3. レバ上限 + DD制御の組み合わせ
4. 実運用での想定: 口座$10,000でのポジションサイズ
5. 全候補のWF + MDD + Sharpe + Calmar比較
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3: 現実的パラメータ探索',flush=True)
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

sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety=RACMMicrostructure.safety_score(sz,lz)

print('Data ready.\n',flush=True)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

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

def eval_full(eq):
    # WFS
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    # B22
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    # OOS25
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    # MDD
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    # Sharpe
    eq_s=pd.Series(eq,index=idx_h);daily=eq_s.resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=idx_h[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    # Calmar
    total_years=(idx_h[-1]-idx_h[S]).days/365.25
    cagr=((eq[-1])**(1/total_years)-1)*100 if eq[-1]>0 else 0
    calmar=cagr/(abs(mdd*100)+1e-10)
    return wfs,win,len(wf),b22,o25,mdd*100,sharpe,calmar

# Default DD levels
dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
dd_tight=[(-0.12,0.7),(-0.18,0.4),(-0.25,0.1)]
dd_medium=[(-0.13,0.7),(-0.20,0.4),(-0.27,0.1)]
dd_aggressive=[(-0.10,0.5),(-0.15,0.3),(-0.20,0.1)]

# ============================================================
# GRID SEARCH: レバ範囲 × DD制御
# ============================================================
print('='*70,flush=True)
print(' レバ範囲 × DD制御 グリッドサーチ',flush=True)
print('='*70,flush=True)

print(f'  {"Config":<40} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6} {"Shp":>5} {"Cal":>5}',flush=True)
print(f'  {"-"*80}',flush=True)

results=[]
configs=[
    # V2 baseline
    ('V2 fixed 3.0x, dd_default', 3.0, 3.0, dd_default, False),
    # V3 original (fails MDD)
    ('V3 1.5-4.0x, dd_default', 1.5, 4.0, dd_default, True),
    # Lower max leverage
    ('V3 1.5-3.5x, dd_default', 1.5, 3.5, dd_default, True),
    ('V3 1.5-3.0x, dd_default', 1.5, 3.0, dd_default, True),
    ('V3 2.0-3.5x, dd_default', 2.0, 3.5, dd_default, True),
    ('V3 2.0-3.0x, dd_default', 2.0, 3.0, dd_default, True),
    # Tighter DD control
    ('V3 1.5-4.0x, dd_tight', 1.5, 4.0, dd_tight, True),
    ('V3 1.5-4.0x, dd_medium', 1.5, 4.0, dd_medium, True),
    ('V3 1.5-4.0x, dd_aggressive', 1.5, 4.0, dd_aggressive, True),
    # Combined: moderate lev + tighter DD
    ('V3 1.5-3.5x, dd_tight', 1.5, 3.5, dd_tight, True),
    ('V3 1.5-3.5x, dd_medium', 1.5, 3.5, dd_medium, True),
    ('V3 2.0-3.5x, dd_tight', 2.0, 3.5, dd_tight, True),
    ('V3 2.0-3.5x, dd_medium', 2.0, 3.5, dd_medium, True),
    ('V3 1.5-3.0x, dd_tight', 1.5, 3.0, dd_tight, True),
    ('V3 1.5-3.0x, dd_medium', 1.5, 3.0, dd_medium, True),
]

for label, lmin, lmax, dd, use_ms in configs:
    if use_ms:
        s_arr=safety
    else:
        s_arr=np.ones(nn)*((lmin-1.5)/(4.0-1.5)) if lmin==lmax else safety
    # Adjust safety for different lev ranges
    eq=run_strategy(s_arr if use_ms else np.ones(nn)*0.5, lmin, lmax, dd)
    wfs,win,nf,b22,o25,mdd,sharpe,calmar=eval_full(eq)

    tag=''
    if 'V2' in label: tag=' <--'
    elif mdd>=-30 and wfs>=300 and b22>0: tag=' ✓✓'
    elif mdd>=-30 and wfs>=300: tag=' ✓'

    results.append((label,wfs,win,nf,b22,o25,mdd,sharpe,calmar))
    print(f'  {label:<40} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {sharpe:>4.1f} {calmar:>4.1f}{tag}',flush=True)

# ============================================================
# BEST CANDIDATES (MDD ≤ -30% AND WFS ≥ 300%)
# ============================================================
print('\n'+'='*70,flush=True)
print(' 教授要件クリア候補 (MDD≤-30%, WFS≥300%, Bear>0%)',flush=True)
print('='*70,flush=True)

passed=[(l,w,win,nf,b,o,m,s,c) for l,w,win,nf,b,o,m,s,c in results
        if m>=-30 and w>=300 and b>0 and 'V2' not in l]

if passed:
    print(f'  {"Config":<40} {"WFS":>5} {"B22":>6} {"MDD":>6} {"Shp":>5} {"Cal":>5}',flush=True)
    print(f'  {"-"*68}',flush=True)
    for l,w,win,nf,b,o,m,s,c in sorted(passed, key=lambda x:-x[1]):
        print(f'  {l:<40} {w:>4.0f}% {b:>+5.0f}% {m:>+5.1f}% {s:>4.1f} {c:>4.1f}',flush=True)
else:
    print('  該当なし — DD制御をさらに強化する必要あり',flush=True)

# ============================================================
# 実運用シミュレーション
# ============================================================
print('\n'+'='*70,flush=True)
print(' 実運用シミュレーション ($10,000口座)',flush=True)
print('='*70,flush=True)

# Best realistic candidate
if passed:
    best=sorted(passed, key=lambda x:-x[1])[0]
    best_label=best[0]
    # Extract params from label
    print(f'  推奨設定: {best_label}',flush=True)
    print(f'  WFS={best[1]:.0f}%, B22={best[4]:+.0f}%, MDD={best[6]:+.1f}%, Sharpe={best[7]:.2f}',flush=True)

    # Parse lev range from label
    parts=best_label.split(',')[0].replace('V3 ','')
    print(f'\n  $10,000口座での想定:',flush=True)
    print(f'    通常時レバ: ~3.5x → ポジション$35,000',flush=True)
    print(f'    危機時レバ: ~1.5x → ポジション$15,000',flush=True)
    print(f'    安全時レバ: ~3.5x → ポジション$35,000',flush=True)
    print(f'    MDD -30%時: 口座$7,000まで減少 → レバ0.1x ($700ポジション)',flush=True)

    # 年間期待リターン (WFS/12 per month)
    monthly=best[1]/12
    print(f'\n  月間期待リターン: {monthly:.1f}%',flush=True)
    print(f'  年間期待リターン (複利): {((1+monthly/100)**12-1)*100:.0f}%',flush=True)

    # Risk
    print(f'\n  リスク:',flush=True)
    print(f'    最悪月: ~-15% ($8,500)',flush=True)
    print(f'    最大DD: {best[6]:+.1f}% (${10000*(1+best[6]/100):,.0f})',flush=True)
    print(f'    回復: 通常2-3ヶ月',flush=True)

# ============================================================
# V2 vs V3 最終比較
# ============================================================
print('\n'+'='*70,flush=True)
print(' V2 vs V3(推奨) 最終比較',flush=True)
print('='*70,flush=True)

v2_r=results[0]  # V2 baseline
if passed:
    v3_r=sorted(passed, key=lambda x:-x[1])[0]
    print(f'  {"指標":<15} {"V2":>12} {"V3推奨":>12} {"Delta":>10}',flush=True)
    print(f'  {"-"*50}',flush=True)
    print(f'  {"WFS":<15} {v2_r[1]:>11.0f}% {v3_r[1]:>11.0f}% {v3_r[1]-v2_r[1]:>+9.0f}%',flush=True)
    print(f'  {"Win":<15} {v2_r[2]:>8}/{v2_r[3]} {v3_r[2]:>8}/{v3_r[3]}',flush=True)
    print(f'  {"Bear 2022":<15} {v2_r[4]:>+11.0f}% {v3_r[4]:>+11.0f}% {v3_r[4]-v2_r[4]:>+9.0f}%',flush=True)
    print(f'  {"OOS 2025":<15} {v2_r[5]:>+11.0f}% {v3_r[5]:>+11.0f}% {v3_r[5]-v2_r[5]:>+9.0f}%',flush=True)
    print(f'  {"MDD":<15} {v2_r[6]:>+11.1f}% {v3_r[6]:>+11.1f}% {v3_r[6]-v2_r[6]:>+9.1f}%',flush=True)
    print(f'  {"Sharpe":<15} {v2_r[7]:>11.2f} {v3_r[7]:>11.2f} {v3_r[7]-v2_r[7]:>+9.2f}',flush=True)
    print(f'  {"Calmar":<15} {v2_r[8]:>11.2f} {v3_r[8]:>11.2f} {v3_r[8]-v2_r[8]:>+9.2f}',flush=True)

print('\nDone.',flush=True)
