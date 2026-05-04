"""V3 Final Candidate: MS-Only Dynamic Leverage
=================================================
発見: MS信号のみ(spread_z + liq_z)で動的レバを制御するのが最良。
V2のdcを安全スコアに含めると二重適用で有害。

最終候補:
  - V3-A Conservative: 1.5-4.0x (MDD < -28%)
  - V3-B Moderate:     1.5-4.5x (MDD < -30%)
  - V3-C Aggressive:   2.0-5.0x (MDD < -31%)

検証:
  1. MS-only vs Full safety (改めて確認)
  2. 1000xシャッフル
  3. パラメータ感度 (全組み合わせ)
  4. リークチェック
  5. 年別・Final資産比較
  6. V2 Baseline との差分分析
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  V3 FINAL CANDIDATE: MS-ONLY DYNAMIC LEVERAGE',flush=True)
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

# MS-Only Safety Score
safety_ms=np.ones(nn)*0.5
for i in range(S,nn):
    sz=spread_z[i];lz=liq_z[i]
    s=1.0
    if sz>2.0: s-=0.3
    elif sz>1.0: s-=0.15
    if lz>3.0: s-=0.3
    elif lz>2.0: s-=0.15
    if sz<-0.5 and lz<0.5: s+=0.2
    safety_ms[i]=np.clip(s, 0.0, 1.5)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

def eval_full(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    yearly={}
    for yr in range(2021,2026):
        iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
        if len(iy)>100:yearly[yr]=(eq[iy[-1]]/eq[max(0,iy[0]-1)]-1)*100
    return wfs,win,len(wf),b22,o25,mdd*100,eq[-1],yearly

def run_dynlev(safety, lev_range=(1.5,4.0)):
    eq=np.ones(nn);e=1.0;pk=1.0
    lev_min,lev_max=lev_range
    levs=np.zeros(nn)
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
        levs[i]=lev
        pnl=pnl_base*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq,np.mean(levs[S:])

def run_fixed(lev):
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
        pnl=(dw*ret[i]*bp+lw*lp1h[i])*lev+fra[i]*abs(lev)
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# 1. THREE CANDIDATES
# ============================================================
print('\n'+' 1. V3 CANDIDATES (MS-Only) '.center(70,'='),flush=True)

candidates=[
    ('V3-A Conservative', (1.5, 4.0)),
    ('V3-B Moderate',     (1.5, 4.5)),
    ('V3-C Aggressive',   (2.0, 5.0)),
]

print(f'  {"Model":<30} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6} {"AvgL":>5} {"Final":>10}',flush=True)
print('  '+'-'*80,flush=True)

# V2 baseline
eq_v2=run_fixed(3.0)
wfs,win,nf,b22,o25,mdd,final,yr=eval_full(eq_v2)
print(f'  {"V2 Baseline (3.0x)":<30} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {3.0:>4.1f}x {final:>10.1f}x',flush=True)

for name, lr in candidates:
    eq,al=run_dynlev(safety_ms, lr)
    wfs,win,nf,b22,o25,mdd,final,yr=eval_full(eq)
    # Same avg fixed for comparison
    eq_f=run_fixed(al)
    wfs_f,_,_,b22_f,_,_,_,_=eval_full(eq_f)
    timing_wfs=wfs-wfs_f;timing_b22=b22-b22_f
    print(f'  {name:<30} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {al:>4.1f}x {final:>10.1f}x',flush=True)
    print(f'    (Fixed {al:.1f}x: WFS={wfs_f:.0f}%, B22={b22_f:+.0f}% → timing Δ: WFS{timing_wfs:+.0f}%, B22{timing_b22:+.0f}%)',flush=True)

# ============================================================
# 2. YEAR-BY-YEAR
# ============================================================
print('\n'+' 2. YEAR-BY-YEAR '.center(70,'='),flush=True)
print(f'  {"Year":<6} {"V2(3.0x)":>12} {"V3-A(1.5-4)":>14} {"V3-B(1.5-4.5)":>14} {"V3-C(2-5)":>14}',flush=True)
print('  '+'-'*62,flush=True)

yearly_data=[]
for name, lr in [('V2',None)]+[(n,lr) for n,lr in candidates]:
    if lr is None:
        eq=eq_v2
    else:
        eq,_=run_dynlev(safety_ms, lr)
    _,_,_,_,_,_,_,yr=eval_full(eq)
    yearly_data.append(yr)

for year in range(2021,2026):
    print(f'  {year:<6}', end='',flush=True)
    for yr in yearly_data:
        v=yr.get(year,0)
        if abs(v)>10000:
            print(f' {v/1000:>+12.0f}K%', end='',flush=True)
        else:
            print(f' {v:>+13.1f}%', end='',flush=True)
    print(flush=True)

# ============================================================
# 3. SHUFFLE TEST: MS-Only 1000x for all 3 candidates
# ============================================================
print('\n'+' 3. 1000x SHUFFLE TESTS '.center(70,'='),flush=True)

for name, lr in candidates:
    eq_real,_=run_dynlev(safety_ms, lr)
    wfs_real,_,_,b22_real,_,_,_,_=eval_full(eq_real)
    wfs_sh=[];b22_sh=[]
    for s in range(1000):
        np.random.seed(s+30000)
        ss=safety_ms.copy()
        active=ss[S:].copy()
        np.random.shuffle(active)
        ss[S:]=active
        eq_s,_=run_dynlev(ss, lr)
        w,_,_,b,_,_,_,_=eval_full(eq_s)
        wfs_sh.append(w);b22_sh.append(b)
    p_w=np.mean([w>=wfs_real for w in wfs_sh])
    p_b=np.mean([b>=b22_real for b in b22_sh])
    print(f'  {name}:',flush=True)
    print(f'    WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p_w:.4f} {"PASS" if p_w<0.05 else "FAIL"}',flush=True)
    print(f'    B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_sh):+.0f}%±{np.std(b22_sh):.0f}%, p={p_b:.4f} {"PASS" if p_b<0.05 else "FAIL"}',flush=True)

# ============================================================
# 4. PARAMETER SENSITIVITY (spread/liq thresholds)
# ============================================================
print('\n'+' 4. MS THRESHOLD SENSITIVITY (V3-B 1.5-4.5x) '.center(70,'='),flush=True)
print(f'  {"Config":<45} {"WFS":>5} {"B22":>6} {"OOS":>6}',flush=True)
print('  '+'-'*64,flush=True)

# Test different spread/liq thresholds
configs=[
    ('Default (sp=2/1, liq=3/2)', 2.0, 1.0, 3.0, 2.0),
    ('sp tighter (1.5/0.8)', 1.5, 0.8, 3.0, 2.0),
    ('sp looser (2.5/1.5)', 2.5, 1.5, 3.0, 2.0),
    ('liq tighter (2.0/1.5)', 2.0, 1.0, 2.0, 1.5),
    ('liq looser (4.0/3.0)', 2.0, 1.0, 4.0, 3.0),
    ('both tighter', 1.5, 0.8, 2.0, 1.5),
    ('both looser', 2.5, 1.5, 4.0, 3.0),
    ('boost only (no cut thresh)', 99, 99, 99, 99),  # only boost side
    ('no boost (cut only)', 2.0, 1.0, 3.0, 2.0),  # will test separately
]

for label, sph, spm, lqh, lqm in configs:
    ss=np.ones(nn)*0.5
    for i in range(S,nn):
        sz=spread_z[i];lz=liq_z[i]
        s=1.0
        if sz>sph: s-=0.3
        elif sz>spm: s-=0.15
        if lz>lqh: s-=0.3
        elif lz>lqm: s-=0.15
        if 'boost only' not in label:
            if sz<-0.5 and lz<0.5: s+=0.2
        if 'no boost' in label:
            s=min(s,1.0)  # cap at 1.0, no boost
        ss[i]=np.clip(s,0.0,1.5)
    eq,_=run_dynlev(ss, (1.5,4.5))
    wfs,_,_,b22,o25,_,_,_=eval_full(eq)
    tag=' *' if wfs>=550 and b22>=100 else ''
    print(f'  {label:<45} {wfs:>4.0f}% {b22:>+5.0f}% {o25:>+5.0f}%{tag}',flush=True)

# Test no-boost version specifically
ss_noboost=np.ones(nn)*0.5
for i in range(S,nn):
    sz=spread_z[i];lz=liq_z[i]
    s=1.0
    if sz>2.0: s-=0.3
    elif sz>1.0: s-=0.15
    if lz>3.0: s-=0.3
    elif lz>2.0: s-=0.15
    # NO boost
    ss_noboost[i]=np.clip(s,0.0,1.0)
eq_nb,al_nb=run_dynlev(ss_noboost, (1.5,4.5))
wfs_nb,_,_,b22_nb,o25_nb,_,_,_=eval_full(eq_nb)
print(f'  {"CUT ONLY (no boost, cap=1.0)":<45} {wfs_nb:>4.0f}% {b22_nb:>+5.0f}% {o25_nb:>+5.0f}% (avgLev={al_nb:.2f}x)',flush=True)

# ============================================================
# 5. LEAK CHECK
# ============================================================
print('\n'+' 5. LEAK CHECK (V3-B) '.center(70,'='),flush=True)
for shift in [-4,-2,-1,0,1,2,4,8,24]:
    ss=np.roll(safety_ms,shift)
    eq,_=run_dynlev(ss, (1.5,4.5))
    wfs,_,_,b22,o25,_,_,_=eval_full(eq)
    marker='<-- base' if shift==0 else ''
    print(f'  shift={shift:+3d}h: WFS={wfs:.0f}%, B22={b22:+.0f}%, OOS={o25:+.0f}% {marker}',flush=True)

# ============================================================
# 6. FINAL TABLE
# ============================================================
print('\n'+' ═══════════════════ FINAL COMPARISON ═══════════════════ ',flush=True)
print(f'  {"Model":<28} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS25":>6} {"MDD":>6} {"TimingΔ":>8}',flush=True)
print('  '+'-'*72,flush=True)

# V2 Baseline
eq=run_fixed(3.0)
wfs,win,nf,b22,o25,mdd,_,_=eval_full(eq)
print(f'  {"V2 (fixed 3.0x)":<28} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {"--":>8}',flush=True)

for name, lr in candidates:
    eq,al=run_dynlev(safety_ms, lr)
    wfs,win,nf,b22,o25,mdd,_,_=eval_full(eq)
    eq_f=run_fixed(al)
    wfs_f,_,_,b22_f,_,_,_,_=eval_full(eq_f)
    td_wfs=wfs-wfs_f
    print(f'  {name:<28} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {td_wfs:>+7.0f}%',flush=True)

print('\n  TimingΔ = WFS improvement beyond "just using higher fixed leverage"',flush=True)
print('\nDone.',flush=True)
