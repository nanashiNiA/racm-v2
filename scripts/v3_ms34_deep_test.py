"""V3: MS3+MS4 Deep Validation
MS3 (Spread Regime) + MS4 (Liq Cascade) を深掘り検証
- 組み合わせシャッフルテスト (200x)
- パラメータ感度分析
- 年別分析
- 1000xシャッフル (最終確認)
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  MS3+MS4 DEEP VALIDATION',flush=True)
print('='*70,flush=True)

# Load data (same as parent script)
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

# V2 regime
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

# Load orderbook data
print('Loading orderbook data...',flush=True)
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
                ohlc[['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']]=\
                    ohlc[['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']].fillna(0)
            ob_frames.append(ohlc)

ob_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ob_all=ob_all.drop_duplicates('timestamp').set_index('timestamp')
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
ob_aligned=ob_all.reindex(idx_h_utc,method='nearest',tolerance=pd.Timedelta('2H'))

# Extract raw features
spread_pct_raw=ob_aligned['spread_pct'].fillna(0).values
long_liq_raw=ob_aligned['long_liq_usd'].fillna(0).values if 'long_liq_usd' in ob_aligned.columns else np.zeros(nn)
short_liq_raw=ob_aligned['short_liq_usd'].fillna(0).values if 'short_liq_usd' in ob_aligned.columns else np.zeros(nn)
total_liq_raw=long_liq_raw+short_liq_raw

print('Data ready.',flush=True)

# ============================================================
# PARAMETERIZED MS FEATURES
# ============================================================
def compute_spread_z(window=168, min_periods=48):
    sp_ma=pd.Series(spread_pct_raw).rolling(window,min_periods=min_periods).mean().values
    sp_std=pd.Series(spread_pct_raw).rolling(window,min_periods=min_periods).std().values
    sz=np.zeros(nn)
    for i in range(S,nn):
        if not np.isnan(sp_std[i-1]) and sp_std[i-1]>1e-6:
            sz[i]=(spread_pct_raw[i-1]-sp_ma[i-1])/(sp_std[i-1]+1e-10)
    return sz

def compute_liq_z(window=168, min_periods=48):
    liq_ma=pd.Series(total_liq_raw).rolling(window,min_periods=min_periods).mean().values
    liq_std=pd.Series(total_liq_raw).rolling(window,min_periods=min_periods).std().values
    lz=np.zeros(nn)
    for i in range(S,nn):
        if not np.isnan(liq_std[i-1]) and liq_std[i-1]>1e-6:
            lz[i]=(total_liq_raw[i-1]-liq_ma[i-1])/(liq_std[i-1]+1e-10)
    return lz

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
    eq_s=pd.Series(eq,index=idx_h);dr=eq_s.resample('1D').last().pct_change().dropna()*100
    dr=dr[dr.index>=idx_h[S]];dw=int((dr>0).sum());dl=int((dr<0).sum())
    # Year-by-year
    yearly={}
    for yr in range(2021,2026):
        iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
        if len(iy)>100:
            yearly[yr]=(eq[iy[-1]]/eq[max(0,iy[0]-1)]-1)*100
    mdd=0;pk=1
    for i in range(S,nn):
        pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return wfs,win,len(wf),b22,o25,dw/(dw+dl)*100 if dw+dl>0 else 50,mdd*100,yearly

def run_ms(spread_z, liq_z, sp_thresh_high=2.0, sp_thresh_med=1.0,
           sp_cut_high=0.5, sp_cut_med=0.8,
           liq_thresh_high=3.0, liq_thresh_med=2.0,
           liq_cut_high=0.3, liq_cut_med=0.6):
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
        pnl=dw*ret[i]*bp+lw*lp1h[i]

        ms_mult=1.0
        sz=spread_z[i]
        if sz>sp_thresh_high: ms_mult *= sp_cut_high
        elif sz>sp_thresh_med: ms_mult *= sp_cut_med

        lz=liq_z[i]
        if lz>liq_thresh_high: ms_mult *= liq_cut_high
        elif lz>liq_thresh_med: ms_mult *= liq_cut_med

        pnl *= ms_mult * 3.0
        pnl += fra[i]*3.0

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# 1. DEFAULT MS3+MS4
# ============================================================
print('\n'+' 1. DEFAULT PARAMETERS '.center(70,'='),flush=True)
spread_z_def=compute_spread_z(168,48)
liq_z_def=compute_liq_z(168,48)

eq_v2_base=run_ms(np.zeros(nn), np.zeros(nn))  # no MS
eq_ms34=run_ms(spread_z_def, liq_z_def)

wfs_b,win_b,nf_b,b22_b,o25_b,dw_b,mdd_b,yr_b=eval_strat(eq_v2_base)
wfs_m,win_m,nf_m,b22_m,o25_m,dw_m,mdd_m,yr_m=eval_strat(eq_ms34)

print(f'  V2 Baseline:  WFS={wfs_b:.0f}%, Win={win_b}/{nf_b}, B22={b22_b:+.0f}%, OOS25={o25_b:+.0f}%, MDD={mdd_b:+.1f}%',flush=True)
print(f'  MS3+MS4:      WFS={wfs_m:.0f}%, Win={win_m}/{nf_m}, B22={b22_m:+.0f}%, OOS25={o25_m:+.0f}%, MDD={mdd_m:+.1f}%',flush=True)

# ============================================================
# 2. YEAR-BY-YEAR BREAKDOWN
# ============================================================
print('\n'+' 2. YEAR-BY-YEAR '.center(70,'='),flush=True)
print(f'  {"Year":<6} {"V2":>8} {"MS3+4":>8} {"Delta":>8}',flush=True)
print('  '+'-'*30,flush=True)
for yr in sorted(set(yr_b.keys())&set(yr_m.keys())):
    delta=yr_m[yr]-yr_b[yr]
    print(f'  {yr:<6} {yr_b[yr]:>+7.1f}% {yr_m[yr]:>+7.1f}% {delta:>+7.1f}%',flush=True)

# ============================================================
# 3. PARAMETER SENSITIVITY
# ============================================================
print('\n'+' 3. PARAMETER SENSITIVITY '.center(70,'='),flush=True)
print(f'  {"Config":<40} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS25":>6}',flush=True)
print('  '+'-'*64,flush=True)

param_tests=[
    ('Default (sp=2/1, liq=3/2)', dict()),
    # Spread threshold sensitivity
    ('sp_thresh: 1.5/0.8 (tighter)', dict(sp_thresh_high=1.5, sp_thresh_med=0.8)),
    ('sp_thresh: 2.5/1.5 (looser)', dict(sp_thresh_high=2.5, sp_thresh_med=1.5)),
    ('sp_thresh: 3.0/2.0 (very loose)', dict(sp_thresh_high=3.0, sp_thresh_med=2.0)),
    # Spread cut sensitivity
    ('sp_cut: 0.3/0.6 (stronger)', dict(sp_cut_high=0.3, sp_cut_med=0.6)),
    ('sp_cut: 0.7/0.9 (weaker)', dict(sp_cut_high=0.7, sp_cut_med=0.9)),
    # Liq threshold sensitivity
    ('liq_thresh: 2.0/1.5 (tighter)', dict(liq_thresh_high=2.0, liq_thresh_med=1.5)),
    ('liq_thresh: 4.0/3.0 (looser)', dict(liq_thresh_high=4.0, liq_thresh_med=3.0)),
    # Liq cut sensitivity
    ('liq_cut: 0.1/0.4 (stronger)', dict(liq_cut_high=0.1, liq_cut_med=0.4)),
    ('liq_cut: 0.5/0.8 (weaker)', dict(liq_cut_high=0.5, liq_cut_med=0.8)),
    # Z-score window sensitivity
    ('window=120 (5d)', None),  # special handling
    ('window=336 (14d)', None),
    ('window=720 (30d)', None),
]

for label, kwargs in param_tests:
    if kwargs is None:  # window variation
        w=int(label.split('=')[1].split(' ')[0])
        sz=compute_spread_z(w,w//3)
        lz=compute_liq_z(w,w//3)
        eq=run_ms(sz,lz)
    else:
        eq=run_ms(spread_z_def, liq_z_def, **kwargs)
    wfs,win,nf,b22,o25,_,_,_=eval_strat(eq)
    tag=''
    if 'Default' in label: tag=' <--'
    elif wfs>=366 and b22>=80: tag=' *'
    print(f'  {label:<40} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}%{tag}',flush=True)

# ============================================================
# 4. COMBINED SHUFFLE TEST (200x)
# ============================================================
print('\n'+' 4. COMBINED SHUFFLE TEST (200x) '.center(70,'='),flush=True)
print('Shuffling BOTH spread_z AND liq_z simultaneously...',flush=True)

wfs_real,_,_,b22_real,_,_,_,_=eval_strat(eq_ms34)
wfs_shuffled=[];b22_shuffled=[]
for s in range(200):
    np.random.seed(s)
    sz_s=spread_z_def.copy();lz_s=liq_z_def.copy()
    np.random.shuffle(sz_s[S:]);np.random.shuffle(lz_s[S:])
    eq_s=run_ms(sz_s,lz_s)
    w,_,_,b,_,_,_,_=eval_strat(eq_s)
    wfs_shuffled.append(w);b22_shuffled.append(b)

p_wfs=np.mean([w>=wfs_real for w in wfs_shuffled])
p_b22=np.mean([b>=b22_real for b in b22_shuffled])
print(f'  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_shuffled):.0f}%±{np.std(wfs_shuffled):.0f}%, p={p_wfs:.3f} {"PASS" if p_wfs<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_shuffled):+.0f}%±{np.std(b22_shuffled):.0f}%, p={p_b22:.3f} {"PASS" if p_b22<0.05 else "FAIL"}',flush=True)

# ============================================================
# 5. 1000x SHUFFLE (Final confirmation)
# ============================================================
print('\n'+' 5. 1000x SHUFFLE (FINAL) '.center(70,'='),flush=True)
print('Running 1000 shuffles...',flush=True)
wfs_1k=[];b22_1k=[]
for s in range(1000):
    np.random.seed(s+10000)
    sz_s=spread_z_def.copy();lz_s=liq_z_def.copy()
    np.random.shuffle(sz_s[S:]);np.random.shuffle(lz_s[S:])
    eq_s=run_ms(sz_s,lz_s)
    w,_,_,b,_,_,_,_=eval_strat(eq_s)
    wfs_1k.append(w);b22_1k.append(b)
    if (s+1)%200==0:print(f'  {s+1}/1000...',flush=True)

p_wfs_1k=np.mean([w>=wfs_real for w in wfs_1k])
p_b22_1k=np.mean([b>=b22_real for b in b22_1k])
print(f'\n  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_1k):.0f}%±{np.std(wfs_1k):.0f}%, p={p_wfs_1k:.4f} {"PASS" if p_wfs_1k<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_1k):+.0f}%±{np.std(b22_1k):.0f}%, p={p_b22_1k:.4f} {"PASS" if p_b22_1k<0.05 else "FAIL"}',flush=True)

# ============================================================
# 6. LEAK CHECK: Feature lead/lag analysis
# ============================================================
print('\n'+' 6. LEAK CHECK '.center(70,'='),flush=True)
print('Testing sensitivity to lead/lag shifts...',flush=True)
for shift in [-2, -1, 0, 1, 2, 4, 8]:
    sz=np.roll(spread_z_def, shift)
    lz=np.roll(liq_z_def, shift)
    eq=run_ms(sz,lz)
    wfs,_,_,b22,o25,_,_,_=eval_strat(eq)
    marker='<-- base' if shift==0 else ''
    print(f'  shift={shift:+d}h: WFS={wfs:.0f}%, B22={b22:+.0f}%, OOS25={o25:+.0f}% {marker}',flush=True)

# ============================================================
# 7. WHEN DOES MS FIRE?
# ============================================================
print('\n'+' 7. ACTIVATION ANALYSIS '.center(70,'='),flush=True)
sz=spread_z_def;lz=liq_z_def
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    sp_high=np.sum(sz[iy]>2.0);sp_med=np.sum((sz[iy]>1.0)&(sz[iy]<=2.0))
    lq_high=np.sum(lz[iy]>3.0);lq_med=np.sum((lz[iy]>2.0)&(lz[iy]<=3.0))
    total_bars=len(iy)
    print(f'  {yr}: Spread high={sp_high} ({sp_high/total_bars*100:.1f}%), med={sp_med} ({sp_med/total_bars*100:.1f}%)'
          f'  |  Liq high={lq_high} ({lq_high/total_bars*100:.1f}%), med={lq_med} ({lq_med/total_bars*100:.1f}%)',flush=True)

print('\nDone.',flush=True)
