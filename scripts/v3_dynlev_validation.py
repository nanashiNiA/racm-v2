"""V3: Dynamic Leverage 徹底検証
=================================
動的レバレッジ (MS信号でlev 1.5-4.0x切替) がWFS570%を出した。
しかし「レバを上げただけ」の可能性がある。

検証:
1. 固定レバ比較: 同じ平均レバでの固定レバと比較
2. シャッフルテスト: safety scoreのタイミングに意味があるか
3. リークチェック
4. パラメータ感度
5. 年別・月別分析
6. 1000xシャッフル最終確認
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  DYNAMIC LEVERAGE 徹底検証',flush=True)
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

def get_dc(i):
    dl=ds=0
    if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:dl+=1
    if not np.isnan(ma20[i-1]) and price[i-1]<ma20[i-1]:ds+=1
    if not np.isnan(skew_a[i-1]) and skew_a[i-1]<-0.5:dl+=1;ds+=1
    if dd_a[i-1]<-0.12:dl+=1;ds+=1
    return dl*0.3+ds*0.7

# Orderbook data
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
ob_aligned=ob_all.reindex(idx_h_utc,method='nearest',tolerance=pd.Timedelta('2H'))

spread_pct_raw=ob_aligned['spread_pct'].fillna(0).values
total_liq_raw=(ob_aligned['long_liq_usd'].fillna(0).values+ob_aligned['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in ob_aligned.columns else np.zeros(nn))

sp_ma=pd.Series(spread_pct_raw).rolling(168,min_periods=48).mean().values
sp_std=pd.Series(spread_pct_raw).rolling(168,min_periods=48).std().values
spread_z=np.zeros(nn)
for i in range(S,nn):
    if not np.isnan(sp_std[i-1]) and sp_std[i-1]>1e-6:
        spread_z[i]=(spread_pct_raw[i-1]-sp_ma[i-1])/(sp_std[i-1]+1e-10)

liq_ma=pd.Series(total_liq_raw).rolling(168,min_periods=48).mean().values
liq_std=pd.Series(total_liq_raw).rolling(168,min_periods=48).std().values
liq_z=np.zeros(nn)
for i in range(S,nn):
    if not np.isnan(liq_std[i-1]) and liq_std[i-1]>1e-6:
        liq_z[i]=(total_liq_raw[i-1]-liq_ma[i-1])/(liq_std[i-1]+1e-10)

print('Data ready.',flush=True)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

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
# COMPUTE SAFETY SCORE (pre-compute for shuffle tests)
# ============================================================
safety_score=np.ones(nn)*0.5
dc_arr=np.zeros(nn)
for i in range(S,nn):
    dc_arr[i]=get_dc(i)
    sz=spread_z[i];lz=liq_z[i];dc=dc_arr[i]
    v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    safety=1.0
    if sz>2.0: safety-=0.3
    elif sz>1.0: safety-=0.15
    if lz>3.0: safety-=0.3
    elif lz>2.0: safety-=0.15
    if dc>=1.5: safety-=0.2
    elif dc>=1.0: safety-=0.1
    if sz<-0.5 and lz<0.5: safety+=0.2
    if v<0.50: safety+=0.1
    safety_score[i]=np.clip(safety, 0.0, 1.5)

def run_dynlev(safety, lev_range=(1.5,4.0)):
    eq=np.ones(nn);e=1.0;pk=1.0
    lev_min,lev_max=lev_range
    total_lev=0;count_lev=0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp=bp_v2[i]
        if bp<0.5:lw=0.95
        elif bp<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        pnl_base=dw*ret[i]*bp+lw*lp1h[i]
        lev=lev_min+(lev_max-lev_min)*safety[i]
        total_lev+=lev;count_lev+=1
        pnl=pnl_base*lev
        pnl+=fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    avg_lev=total_lev/count_lev if count_lev>0 else 0
    return eq,avg_lev

def run_fixed_lev(lev):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp=bp_v2[i]
        if bp<0.5:lw=0.95
        elif bp<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        pnl=(dw*ret[i]*bp+lw*lp1h[i])*lev
        pnl+=fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# 1. 固定レバ比較 (同じ平均レバ)
# ============================================================
print('\n'+' 1. 固定レバ vs 動的レバ '.center(70,'='),flush=True)

eq_dyn,avg_lev=run_dynlev(safety_score, (1.5,4.0))
print(f'  Dynamic 1.5-4.0x 平均レバ: {avg_lev:.2f}x',flush=True)

print(f'\n  {"Model":<40} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6}',flush=True)
print('  '+'-'*70,flush=True)

for lev in [2.0, 2.5, avg_lev, 3.0, 3.5, 4.0]:
    eq=run_fixed_lev(lev)
    wfs,win,nf,b22,o25,mdd=eval_strat(eq)
    tag=' <-- avg_lev' if abs(lev-avg_lev)<0.01 else ''
    print(f'  Fixed {lev:.2f}x{" "*(33-len(f"Fixed {lev:.2f}x"))} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

wfs_d,win_d,nf_d,b22_d,o25_d,mdd_d=eval_strat(eq_dyn)
print(f'  Dynamic 1.5-4.0x (avg={avg_lev:.2f}x)        {wfs_d:>4.0f}% {win_d:>2}/{nf_d} {b22_d:>+5.0f}% {o25_d:>+5.0f}% {mdd_d:>+5.1f}% <-- DYNAMIC',flush=True)

# Also test 2.0-4.5x
eq_dyn2,avg_lev2=run_dynlev(safety_score, (2.0,4.5))
wfs_d2,win_d2,nf_d2,b22_d2,o25_d2,mdd_d2=eval_strat(eq_dyn2)
print(f'\n  Dynamic 2.0-4.5x (avg={avg_lev2:.2f}x)        {wfs_d2:>4.0f}% {win_d2:>2}/{nf_d2} {b22_d2:>+5.0f}% {o25_d2:>+5.0f}% {mdd_d2:>+5.1f}% <-- DYNAMIC',flush=True)
eq_fixed_same=run_fixed_lev(avg_lev2)
wfs_f,win_f,nf_f,b22_f,o25_f,mdd_f=eval_strat(eq_fixed_same)
print(f'  Fixed {avg_lev2:.2f}x (same avg)              {wfs_f:>4.0f}% {win_f:>2}/{nf_f} {b22_f:>+5.0f}% {o25_f:>+5.0f}% {mdd_f:>+5.1f}% <-- FIXED SAME AVG',flush=True)

print(f'\n  ** Dynamic vs Fixed (same avg lev):',flush=True)
print(f'     WFS: {wfs_d:.0f}% vs {eval_strat(run_fixed_lev(avg_lev))[0]:.0f}% (Δ={wfs_d-eval_strat(run_fixed_lev(avg_lev))[0]:+.0f}%)',flush=True)
print(f'     B22: {b22_d:+.0f}% vs {eval_strat(run_fixed_lev(avg_lev))[3]:+.0f}% (Δ={b22_d-eval_strat(run_fixed_lev(avg_lev))[3]:+.0f}%)',flush=True)

# ============================================================
# 2. SHUFFLE TEST: safety scoreのタイミングに意味があるか
# ============================================================
print('\n'+' 2. SHUFFLE TEST (200x) '.center(70,'='),flush=True)
print('Shuffling safety_score timing...',flush=True)

wfs_real=wfs_d;b22_real=b22_d
wfs_sh=[];b22_sh=[]
for s in range(200):
    np.random.seed(s)
    ss=safety_score.copy()
    active=ss[S:].copy()
    np.random.shuffle(active)
    ss[S:]=active
    eq_s,_=run_dynlev(ss, (1.5,4.0))
    w,_,_,b,_,_=eval_strat(eq_s)
    wfs_sh.append(w);b22_sh.append(b)

p_wfs=np.mean([w>=wfs_real for w in wfs_sh])
p_b22=np.mean([b>=b22_real for b in b22_sh])
print(f'  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p_wfs:.3f} {"PASS" if p_wfs<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_sh):+.0f}%±{np.std(b22_sh):.0f}%, p={p_b22:.3f} {"PASS" if p_b22<0.05 else "FAIL"}',flush=True)

# Same for 2.0-4.5x
print('\nShuffling for 2.0-4.5x range...',flush=True)
wfs_real2=wfs_d2;b22_real2=b22_d2
wfs_sh2=[];b22_sh2=[]
for s in range(200):
    np.random.seed(s+5000)
    ss=safety_score.copy()
    active=ss[S:].copy()
    np.random.shuffle(active)
    ss[S:]=active
    eq_s,_=run_dynlev(ss, (2.0,4.5))
    w,_,_,b,_,_=eval_strat(eq_s)
    wfs_sh2.append(w);b22_sh2.append(b)

p_wfs2=np.mean([w>=wfs_real2 for w in wfs_sh2])
p_b22_2=np.mean([b>=b22_real2 for b in b22_sh2])
print(f'  WFS: Real={wfs_real2:.0f}%, Shuffle={np.mean(wfs_sh2):.0f}%±{np.std(wfs_sh2):.0f}%, p={p_wfs2:.3f} {"PASS" if p_wfs2<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real2:+.0f}%, Shuffle={np.mean(b22_sh2):+.0f}%±{np.std(b22_sh2):.0f}%, p={p_b22_2:.3f} {"PASS" if p_b22_2<0.05 else "FAIL"}',flush=True)

# ============================================================
# 3. COMPONENT DECOMPOSITION: which signal drives the timing?
# ============================================================
print('\n'+' 3. COMPONENT DECOMPOSITION '.center(70,'='),flush=True)
print('Which component of safety score provides timing value?',flush=True)

# Safety from MS only (no dc, no vol)
safety_ms_only=np.ones(nn)*0.5
for i in range(S,nn):
    sz=spread_z[i];lz=liq_z[i]
    s=1.0
    if sz>2.0: s-=0.3
    elif sz>1.0: s-=0.15
    if lz>3.0: s-=0.3
    elif lz>2.0: s-=0.15
    if sz<-0.5 and lz<0.5: s+=0.2
    safety_ms_only[i]=np.clip(s,0.0,1.5)

# Safety from V2 regime only (dc + vol, no MS)
safety_v2_only=np.ones(nn)*0.5
for i in range(S,nn):
    dc=dc_arr[i];v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    s=1.0
    if dc>=1.5: s-=0.2
    elif dc>=1.0: s-=0.1
    if v<0.50: s+=0.1
    safety_v2_only[i]=np.clip(s,0.0,1.5)

for label, ss in [
    ('Full safety (MS + V2)', safety_score),
    ('MS only (spread + liq)', safety_ms_only),
    ('V2 only (dc + vol)', safety_v2_only),
]:
    eq,al=run_dynlev(ss, (1.5,4.0))
    wfs,win,nf,b22,o25,mdd=eval_strat(eq)
    print(f'  {label:<35} WFS={wfs:.0f}%, B22={b22:+.0f}%, OOS={o25:+.0f}%, avgLev={al:.2f}x',flush=True)

# ============================================================
# 4. PARAMETER SENSITIVITY
# ============================================================
print('\n'+' 4. PARAMETER SENSITIVITY '.center(70,'='),flush=True)
print(f'  {"Lev Range":<20} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6} {"AvgLev":>6}',flush=True)
print('  '+'-'*56,flush=True)

for lmin,lmax in [(1.0,3.0),(1.0,4.0),(1.5,3.5),(1.5,4.0),(1.5,4.5),(2.0,4.0),(2.0,4.5),(2.0,5.0),(2.5,4.5),(2.5,5.0)]:
    eq,al=run_dynlev(safety_score, (lmin,lmax))
    wfs,win,nf,b22,o25,mdd=eval_strat(eq)
    tag=''
    if wfs>=400 and b22>=80 and mdd>=-30: tag=' *'
    print(f'  {lmin:.1f}-{lmax:.1f}x{" "*(14-len(f"{lmin:.1f}-{lmax:.1f}x"))} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {al:>5.2f}x{tag}',flush=True)

# ============================================================
# 5. LEAK CHECK
# ============================================================
print('\n'+' 5. LEAK CHECK '.center(70,'='),flush=True)
for shift in [-2,-1,0,1,2,4,8]:
    ss=np.roll(safety_score,shift)
    eq,_=run_dynlev(ss, (1.5,4.0))
    wfs,_,_,b22,o25,mdd=eval_strat(eq)
    marker='<-- base' if shift==0 else ''
    print(f'  shift={shift:+d}h: WFS={wfs:.0f}%, B22={b22:+.0f}%, OOS={o25:+.0f}%, MDD={mdd:+.1f}% {marker}',flush=True)

# ============================================================
# 6. 1000x SHUFFLE (最終確認, 1.5-4.0x)
# ============================================================
print('\n'+' 6. 1000x SHUFFLE (FINAL) '.center(70,'='),flush=True)
wfs_1k=[];b22_1k=[]
for s in range(1000):
    np.random.seed(s+20000)
    ss=safety_score.copy()
    active=ss[S:].copy()
    np.random.shuffle(active)
    ss[S:]=active
    eq_s,_=run_dynlev(ss, (1.5,4.0))
    w,_,_,b,_,_=eval_strat(eq_s)
    wfs_1k.append(w);b22_1k.append(b)
    if (s+1)%200==0:print(f'  {s+1}/1000...',flush=True)

p_wfs_1k=np.mean([w>=wfs_real for w in wfs_1k])
p_b22_1k=np.mean([b>=b22_real for b in b22_1k])
print(f'\n  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_1k):.0f}%±{np.std(wfs_1k):.0f}%, p={p_wfs_1k:.4f} {"PASS" if p_wfs_1k<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_1k):+.0f}%±{np.std(b22_1k):.0f}%, p={p_b22_1k:.4f} {"PASS" if p_b22_1k<0.05 else "FAIL"}',flush=True)

# ============================================================
# 7. SUMMARY
# ============================================================
print('\n'+' SUMMARY '.center(70,'='),flush=True)
print(f'  動的レバレッジのタイミングに予測力がある場合:',flush=True)
print(f'    - WFS shuffle p < 0.05',flush=True)
print(f'    - 同じ平均レバの固定レバよりWFS/B22が高い',flush=True)
print(f'    - leak checkでshift=0が最良',flush=True)
print(f'  単にレバを上げているだけの場合:',flush=True)
print(f'    - WFS shuffle p >= 0.05 (タイミング無関係)',flush=True)
print(f'    - 固定レバと同等のWFS/B22',flush=True)
print('\nDone.',flush=True)
