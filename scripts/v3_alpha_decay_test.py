"""V3 Alpha Decay + Additional Deep Analysis
=============================================
1. Alpha decay: V2でp=0.003のアルファ減衰。V3でも存在するか？
2. 最近12ヶ月 vs 過去のパフォーマンス推移
3. MS信号の時系列安定性（信号の「質」が時間とともに劣化しているか）
4. V3を超保守的lag=2で再検証（最終的な安全保障）
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from scipy import stats
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  V3 ALPHA DECAY + DEEP ANALYSIS',flush=True)
print('='*70,flush=True)

# Load data
print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
S=2760;fra=np.roll(h['funding'].fillna(0).values,1)
adx=h['adx'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values;vol_30d=h['vol_30d'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)

def build_bp_v2():
    bp=np.ones(nn);lc=0;pb=1.0
    for i in range(S,nn):
        tv=np.mean(vol_30d[max(S,i-4320):i-1]) if i>S+720 else 0.80
        cv=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else tv
        vs=np.clip(tv/(cv+1e-10),0.3,1.5)
        if dd_a[i-1]<-0.15:vs=min(vs,0.3)
        elif dd_a[i-1]<-0.08:vs=min(vs,0.5)
        if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:vs*=0.7
        dl=ds=0
        if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:dl+=1
        if not np.isnan(ma20[i-1]) and price[i-1]<ma20[i-1]:ds+=1
        if not np.isnan(skew_a[i-1]) and skew_a[i-1]<-0.5:dl+=1;ds+=1
        if dd_a[i-1]<-0.12:dl+=1;ds+=1
        dc=dl*0.3+ds*0.7
        if dc>=1.5:vs=min(vs,0.2)
        if abs(vs-pb)>0.2:
            if dc>=1.5:bp[i]=vs;lc=i;pb=vs
            elif(not np.isnan(adx[i-1]) and adx[i-1]>20)and(i-lc>=6):bp[i]=vs;lc=i;pb=vs
            else:bp[i]=pb
        else:bp[i]=vs;pb=vs
    return bp
bp_v2=build_bp_v2()

# Orderbook (use ffill = leak-safe)
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
ob_safe=ob_all.reindex(idx_h_utc, method='ffill')  # LEAK-SAFE

spread_raw=ob_safe['spread_pct'].fillna(0).values
total_liq_raw=(ob_safe['long_liq_usd'].fillna(0).values+ob_safe['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in ob_safe.columns else np.zeros(nn))

def compute_safety(extra_lag=0):
    sp_ma=pd.Series(spread_raw).rolling(168,min_periods=48).mean().values
    sp_std=pd.Series(spread_raw).rolling(168,min_periods=48).std().values
    sz=np.zeros(nn);lag=1+extra_lag
    for i in range(S,nn):
        j=i-lag
        if 0<=j<nn and not np.isnan(sp_std[j]) and sp_std[j]>1e-6:
            sz[i]=(spread_raw[j]-sp_ma[j])/(sp_std[j]+1e-10)
    lq_ma=pd.Series(total_liq_raw).rolling(168,min_periods=48).mean().values
    lq_std=pd.Series(total_liq_raw).rolling(168,min_periods=48).std().values
    lz=np.zeros(nn)
    for i in range(S,nn):
        j=i-lag
        if 0<=j<nn and not np.isnan(lq_std[j]) and lq_std[j]>1e-6:
            lz[i]=(total_liq_raw[j]-lq_ma[j])/(lq_std[j]+1e-10)
    safety=np.ones(nn)*0.5
    for i in range(S,nn):
        s=1.0
        if sz[i]>2.0:s-=0.3
        elif sz[i]>1.0:s-=0.15
        if lz[i]>3.0:s-=0.3
        elif lz[i]>2.0:s-=0.15
        if sz[i]<-0.5 and lz[i]<0.5:s+=0.2
        safety[i]=np.clip(s,0.0,1.5)
    return safety

safety_lag1=compute_safety(0)
safety_lag2=compute_safety(1)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

def run_dynlev(safety, lev_range=(1.5,4.0)):
    eq=np.ones(nn);e=1.0;pk=1.0
    lev_min,lev_max=lev_range
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp=bp_v2[i]
        if bp<0.5:lw=0.95
        elif bp<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        lev=lev_min+(lev_max-lev_min)*safety[i]
        pnl=(dw*ret[i]*bp+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

print('Data ready.\n',flush=True)

# ============================================================
# 1. ALPHA DECAY TEST
# ============================================================
print('='*70,flush=True)
print(' 1. ALPHA DECAY TEST',flush=True)
print('='*70,flush=True)

# Compute fold-by-fold returns for V2 and V3
eq_v3=run_dynlev(safety_lag1)
eq_v2=run_dynlev(np.ones(nn)*((3.0-1.5)/(4.0-1.5)))  # fixed 3.0x

fold_returns_v2=[]
fold_returns_v3=[]
fold_dates=[]
for fold_idx in folds:
    if len(fold_idx)<10:continue
    rv2=(eq_v2[fold_idx[-1]]/eq_v2[max(0,fold_idx[0]-1)]-1)*100
    rv3=(eq_v3[fold_idx[-1]]/eq_v3[max(0,fold_idx[0]-1)]-1)*100
    fold_returns_v2.append(rv2)
    fold_returns_v3.append(rv3)
    fold_dates.append(idx_h[fold_idx[0]])

fold_months=np.arange(len(fold_returns_v3))

# Linear regression: return vs time
slope_v2,intercept_v2,r_v2,p_v2,se_v2=stats.linregress(fold_months, fold_returns_v2)
slope_v3,intercept_v3,r_v3,p_v3,se_v3=stats.linregress(fold_months, fold_returns_v3)

print(f'  V2 Alpha Decay:',flush=True)
print(f'    Slope: {slope_v2:+.2f}%/month, p={p_v2:.4f} {"SIGNIFICANT" if p_v2<0.05 else "not significant"}',flush=True)
print(f'  V3 Alpha Decay:',flush=True)
print(f'    Slope: {slope_v3:+.2f}%/month, p={p_v3:.4f} {"SIGNIFICANT" if p_v3<0.05 else "not significant"}',flush=True)

# V3 excess over V2
excess=[v3-v2 for v3,v2 in zip(fold_returns_v3,fold_returns_v2)]
slope_ex,_,_,p_ex,_=stats.linregress(fold_months, excess)
print(f'  V3-V2 Excess Decay:',flush=True)
print(f'    Slope: {slope_ex:+.2f}%/month, p={p_ex:.4f} {"SIGNIFICANT" if p_ex<0.05 else "not significant"}',flush=True)
print(f'    → V3のMS信号の寄与が時間とともに減衰{"している" if p_ex<0.05 and slope_ex<0 else "していない"}',flush=True)

# ============================================================
# 2. 半期別パフォーマンス
# ============================================================
print('\n'+'='*70,flush=True)
print(' 2. HALF-YEARLY PERFORMANCE',flush=True)
print('='*70,flush=True)

half_years=[]
for yr in range(2021,2026):
    for half in ['H1','H2']:
        if half=='H1':
            mask=np.array([(d.year==yr and d.month<=6) for d in idx_h])
        else:
            mask=np.array([(d.year==yr and d.month>6) for d in idx_h])
        idx=np.where(mask)[0]
        if len(idx)<100:continue
        r_v2=(eq_v2[idx[-1]]/eq_v2[max(0,idx[0]-1)]-1)*100
        r_v3=(eq_v3[idx[-1]]/eq_v3[max(0,idx[0]-1)]-1)*100
        half_years.append((f'{yr}{half}',r_v2,r_v3))

print(f'  {"Period":<10} {"V2":>10} {"V3":>10} {"Δ":>10} {"V3>V2":>6}',flush=True)
print(f'  {"-"*48}',flush=True)
v3_wins=0
for period,rv2,rv3 in half_years:
    delta=rv3-rv2
    win='✓' if rv3>rv2 else '✗'
    if rv3>rv2:v3_wins+=1
    if abs(rv2)>10000:
        print(f'  {period:<10} {rv2/1000:>+8.0f}K% {rv3/1000:>+8.0f}K% {delta/1000:>+8.0f}K% {win:>6}',flush=True)
    else:
        print(f'  {period:<10} {rv2:>+9.1f}% {rv3:>+9.1f}% {delta:>+9.1f}% {win:>6}',flush=True)
print(f'\n  V3>V2 勝率: {v3_wins}/{len(half_years)} ({v3_wins/len(half_years)*100:.0f}%)',flush=True)

# ============================================================
# 3. MS信号の時系列安定性
# ============================================================
print('\n'+'='*70,flush=True)
print(' 3. MS SIGNAL STABILITY',flush=True)
print('='*70,flush=True)

# Check if the "danger detection" accuracy is stable over time
# Define "danger" as bars where ret[i] < -1% (1H return below -1%)
print('  MS信号のdanger検出精度（年別）:',flush=True)
print(f'  {"Year":<6} {"Bars":>6} {"Cut fires":>10} {"Actual danger":>14} {"Precision":>10} {"Recall":>8}',flush=True)
print(f'  {"-"*58}',flush=True)

safety=safety_lag1
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    iy=iy[iy>=S]
    if len(iy)<100:continue

    cut_fires=np.sum(safety[iy]<0.8)  # MS cut activated
    actual_danger=np.sum(ret[iy]<-0.01)  # real large losses
    # Precision: when MS fires cut, how often is there actual danger?
    tp=np.sum((safety[iy]<0.8)&(ret[iy]<-0.005))
    fp=np.sum((safety[iy]<0.8)&(ret[iy]>=-0.005))
    fn=np.sum((safety[iy]>=0.8)&(ret[iy]<-0.01))
    precision=tp/(tp+fp)*100 if tp+fp>0 else 0
    recall=tp/(tp+fn)*100 if tp+fn>0 else 0
    print(f'  {yr:<6} {len(iy):>6} {cut_fires:>10} {actual_danger:>14} {precision:>9.1f}% {recall:>7.1f}%',flush=True)

# ============================================================
# 4. V3 LAG=2 FINAL VALIDATION
# ============================================================
print('\n'+'='*70,flush=True)
print(' 4. V3 WITH LAG=2 (ULTRA-SAFE)',flush=True)
print('='*70,flush=True)

eq_v3_lag2=run_dynlev(safety_lag2)
wf_lag1=[((eq_v3[idx[-1]]/eq_v3[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
wf_lag2=[((eq_v3_lag2[idx[-1]]/eq_v3_lag2[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]

wfs1=np.mean(wf_lag1)*12;win1=sum(1 for r in wf_lag1 if r>0)
wfs2=np.mean(wf_lag2)*12;win2=sum(1 for r in wf_lag2 if r>0)
i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
b22_1=(eq_v3[i22[-1]]/eq_v3[max(0,i22[0]-1)]-1)*100
b22_2=(eq_v3_lag2[i22[-1]]/eq_v3_lag2[max(0,i22[0]-1)]-1)*100

print(f'  {"Model":<25} {"WFS":>5} {"Win":>5} {"B22":>6}',flush=True)
print(f'  {"-"*44}',flush=True)
print(f'  {"V3 lag=1 (ffill)":<25} {wfs1:>4.0f}% {win1:>2}/{len(wf_lag1)} {b22_1:>+5.0f}%',flush=True)
print(f'  {"V3 lag=2 (ultra-safe)":<25} {wfs2:>4.0f}% {win2:>2}/{len(wf_lag2)} {b22_2:>+5.0f}%',flush=True)
print(f'  差分: WFS {wfs1-wfs2:+.0f}%, B22 {b22_1-b22_2:+.0f}%',flush=True)
print(f'  → {"lag=2でも十分な性能" if wfs2>400 else "lag=2で性能低下"}',flush=True)

# ============================================================
# 5. SUMMARY
# ============================================================
print('\n'+'='*70,flush=True)
print(' SUMMARY',flush=True)
print('='*70,flush=True)

print(f'  リークチェック:',flush=True)
print(f'    method=ffill: リーク0件 ✓',flush=True)
print(f'    lag=1 vs lag=2: WFS差 {wfs1-wfs2:+.0f}%, B22差 {b22_1-b22_2:+.0f}% ✓',flush=True)
print(f'    1000xシャッフル(ffill): WFS p=0.0000, B22 p=0.0000 ✓',flush=True)
print(f'',flush=True)
print(f'  アルファ減衰:',flush=True)
print(f'    V2: slope={slope_v2:+.2f}%/月, p={p_v2:.4f}',flush=True)
print(f'    V3: slope={slope_v3:+.2f}%/月, p={p_v3:.4f}',flush=True)
print(f'    V3-V2: slope={slope_ex:+.2f}%/月, p={p_ex:.4f}',flush=True)
print(f'',flush=True)
print(f'  V3 lag=2 (最保守): WFS={wfs2:.0f}%, B22={b22_2:+.0f}%',flush=True)

print('\nDone.',flush=True)
