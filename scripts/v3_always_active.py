"""V3 Always-Active: 5 new approaches that work EVERY bar
Different from v3_daily_active.py (which tested: LS spread sizing, Rolling Sharpe,
Dispersion, Funding carry [BUG], Adaptive Kelly — all failed).

NEW approaches:
A) Continuous regime score — no binary switching, eliminates whipsaw by design
B) Multi-TF momentum agreement — 1H/8H/24H/7D consensus as position weight
C) Adaptive LS weight by momentum quality — z-score of LS spread filters bad periods
D) Crypto-native signal tilt — funding z + OI divergence (from swin_v26 concepts)
E) Hurst-based regime — mean-reverting vs trending detection for sizing

Each tested standalone, then best combos vs V2 baseline.
"""
import sys,os
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams

print('='*70,flush=True)
print('  V3 ALWAYS-ACTIVE: 5 NEW APPROACHES',flush=True)
print('='*70,flush=True)
print('Loading data...',flush=True)

h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=2760;fra=np.roll(h['funding'].fillna(0).values,1)
adx=h['adx'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
ma50=h['ma50'].values;ma200=h['ma200'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values;vol_30d=h['vol_30d'].values
hi=h['high'].values;lo=h['low'].values

# 8H LS
a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)

# 1H assets
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

print('Data loaded. Computing features...',flush=True)

# ============================================================
# V2 baseline regime (for comparison)
# ============================================================
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

# ============================================================
# FEATURE A: Continuous Regime Score (0-1, no binary switch)
# ============================================================
# Instead of binary danger thresholds, compute a smooth score
# that scales position continuously. Eliminates whipsaw by design.
print('  A) Continuous regime score...',flush=True)
cont_regime=np.ones(nn)*0.5  # 0=max danger, 1=max safe
for i in range(S,nn):
    score=0.5  # neutral

    # 1) MA proximity: distance from MA110 (smooth, not binary)
    if not np.isnan(ma110[i-1]):
        ma_dist=(price[i-1]-ma110[i-1])/(ma110[i-1]+1e-10)
        # Sigmoid-like: above MA → positive, below → negative
        ma_score=1.0/(1.0+np.exp(-ma_dist*20))  # steepness 20
        score=0.3*ma_score

    # 2) Skewness contribution (smooth)
    if not np.isnan(skew_a[i-1]):
        skew_score=np.clip((skew_a[i-1]+1.0)/2.0, 0, 1)  # -1→0, +1→1
        score+=0.2*skew_score
    else:
        score+=0.1

    # 3) Drawdown contribution (smooth)
    dd_score=np.clip((dd_a[i-1]+0.20)/0.20, 0, 1)  # -20%→0, 0%→1
    score+=0.2*dd_score

    # 4) Vol regime (inverse vol, smooth)
    tv=np.mean(vol_30d[max(S,i-4320):i-1]) if i>S+720 else 0.80
    cv=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else tv
    vol_score=np.clip(tv/(cv+1e-10), 0.3, 1.5)
    score+=0.3*np.clip(vol_score/1.5, 0, 1)

    cont_regime[i]=np.clip(score, 0.05, 1.0)

# ============================================================
# FEATURE B: Multi-TF Momentum Agreement
# ============================================================
# Agreement across 4 timeframes: 1H, 8H, 24H, 7D
# When all agree → strong position, when divergent → reduce
print('  B) Multi-TF momentum agreement...',flush=True)
mtf_agreement=np.zeros(nn)
ret_1h=pd.Series(price).pct_change(1).values
ret_8h=pd.Series(price).pct_change(8).values
ret_24h=pd.Series(price).pct_change(24).values
ret_7d=pd.Series(price).pct_change(168).values

for i in range(S,nn):
    signs=[]
    for r in [ret_1h[i-1], ret_8h[i-1], ret_24h[i-1], ret_7d[i-1]]:
        if not np.isnan(r):
            signs.append(np.sign(r))
    if len(signs)>=3:
        agreement=abs(np.mean(signs))  # 0=mixed, 1=all agree
        mtf_agreement[i]=agreement

# ============================================================
# FEATURE C: Adaptive LS Weight by Momentum Quality
# ============================================================
# Use z-score of LS spread to identify when momentum is "working"
# High z-score → momentum dispersion is strong → increase LS weight
# Low z-score → momentum is flat → decrease LS weight
print('  C) Adaptive LS weight by momentum quality...',flush=True)
ls_quality=np.zeros(nn)  # -1 to +1
na=len(ar8)
# Compute LS spread at each 8H bar
ls_spread_raw=np.zeros(len(c8))
for lb in [60,90]:
    lb_b=lb*3;w=0.5
    am={n:np.roll(pd.Series(r).rolling(lb_b,min_periods=lb_b//3).sum().values,1) for n,r in ar8.items()}
    av={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in ar8.items()}
    for i in range(200,len(c8)):
        ms=[(am[n][i]/(av[n][i]+1e-10),n) for n in ar8 if not np.isnan(am[n][i])]
        if len(ms)<3:continue
        ms.sort(key=lambda x:x[0],reverse=True)
        ls_spread_raw[i]+=abs(ms[0][0]-ms[-1][0])*w

# Z-score of LS spread (expanding window)
ls_spread_1h=np.zeros(nn)
for i in range(len(c8)):
    ts8=c8[i];te8=c8[i+1] if i+1<len(c8) else ts8+pd.Timedelta(hours=8)
    idxs=np.where((idx_h>=ts8)&(idx_h<te8))[0]
    for ii in idxs:
        ls_spread_1h[ii]=ls_spread_raw[i]

for i in range(S+720,nn):
    past=ls_spread_1h[max(S,i-4320):i]
    past=past[past>0]
    if len(past)>50:
        mu=np.mean(past);sd=np.std(past)
        ls_quality[i]=np.clip((ls_spread_1h[i]-mu)/(sd+1e-10), -2, 2)/2  # [-1, 1]

# ============================================================
# FEATURE D: Crypto-Native Signal Tilt (from swin_v26 concepts)
# ============================================================
# Uses funding z-score + OI divergence as position tilt
# NOT for direct trading, but as regime confidence overlay
print('  D) Crypto-native signal tilt...',flush=True)
crypto_tilt=np.zeros(nn)  # -1 to +1

# Funding z-score (already in h)
funding_z=np.zeros(nn)
funding_raw=h['funding'].fillna(0).values
f_rolling_mean=pd.Series(funding_raw).rolling(168,min_periods=48).mean().values
f_rolling_std=pd.Series(funding_raw).rolling(168,min_periods=48).std().values
for i in range(S,nn):
    if not np.isnan(f_rolling_std[i-1]) and f_rolling_std[i-1]>1e-10:
        funding_z[i]=(funding_raw[i-1]-f_rolling_mean[i-1])/(f_rolling_std[i-1]+1e-10)

# OI change
oi_raw=h['oi'].fillna(0).values if 'oi' in h.columns else np.zeros(nn)
oi_chg_24=np.zeros(nn)
for i in range(S,nn):
    if i>=24 and oi_raw[i-1]>0 and oi_raw[i-25]>0:
        oi_chg_24[i]=(oi_raw[i-1]-oi_raw[i-25])/(oi_raw[i-25]+1e-10)

for i in range(S,nn):
    tilt=0.0
    fz=funding_z[i]
    oc=oi_chg_24[i]
    r1d=ret_24h[i-1] if i>0 and not np.isnan(ret_24h[i-1]) else 0

    # Negative funding = shorts paying longs = contrarian long signal
    if fz<-1.5: tilt+=0.3
    elif fz<-0.5: tilt+=0.1
    elif fz>1.5: tilt-=0.3
    elif fz>0.5: tilt-=0.1

    # OI divergence: OI up + price down = squeeze building (long)
    if oc>0.02 and r1d<-0.02: tilt+=0.2
    # OI flush: OI down + price down = forced liquidations done (long)
    elif oc<-0.03 and r1d<-0.03: tilt+=0.15
    # OI up + price up = crowded (caution)
    elif oc>0.03 and r1d>0.02: tilt-=0.1

    crypto_tilt[i]=np.clip(tilt, -0.5, 0.5)

# ============================================================
# FEATURE E: Hurst-based Regime
# ============================================================
# Hurst exponent proxy to detect mean-reverting vs trending
# H<0.5 → mean-reverting → reduce trend-following
# H>0.5 → trending → increase position
# Uses variance ratio method (from swin_v26)
print('  E) Hurst-based regime...',flush=True)
hurst_proxy=np.full(nn, 0.5)
log_price=np.log(np.clip(price, 1, None))
ret_log=np.diff(log_price, prepend=log_price[0])

for i in range(S,nn):
    window=min(720, i-S)  # 30 days
    if window<168: continue
    r=ret_log[i-window:i]
    var1=np.var(r)
    var2=np.var(np.diff(r))
    if var1>1e-15:
        hurst_proxy[i]=np.clip(1.0 - var2/(2*var1+1e-15), 0.1, 0.9)

# Smooth with EMA
hurst_smooth=pd.Series(hurst_proxy).ewm(span=48,min_periods=12).mean().values

print('Features computed.',flush=True)

# ============================================================
# Walk-Forward Folds
# ============================================================
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

# ============================================================
# Evaluation
# ============================================================
def eval_strat(eq, label=''):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    eq_s=pd.Series(eq,index=idx_h);dr=eq_s.resample('1D').last().pct_change().dropna()*100
    dr=dr[dr.index>=idx_h[S]];dw=int((dr>0).sum());dl=int((dr<0).sum())
    # Whipsaw count (regime changes per day for continuous approaches)
    return wfs,win,len(wf),b22,o25,dw/(dw+dl)*100 if dw+dl>0 else 50

# ============================================================
# Backtest Runner
# ============================================================
def run_backtest(bp_func, label='', use_cont_regime=False, use_mtf=False,
                 use_ls_quality=False, use_crypto_tilt=False, use_hurst=False):
    """Run backtest with configurable always-active features."""
    eq=np.ones(nn);e=1.0;pk=1.0

    for i in range(S,nn):
        # Base: V2 regime-based LW selection
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp=bp_func[i]

        if bp<0.5:lw=0.95
        elif bp<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw_ratio=max(0,1-lw)
        pnl=dw_ratio*ret[i]*bp+lw*lp1h[i]

        pos_mult=1.0

        # A) Continuous regime: replace binary bp with smooth score
        if use_cont_regime:
            # Scale position by continuous regime (0.05 to 1.0)
            cr=cont_regime[i]
            pos_mult *= np.clip(cr*2, 0.1, 1.5)  # 0→0.1, 0.5→1.0, 0.75→1.5

        # B) Multi-TF agreement: boost when all TFs agree
        if use_mtf:
            agr=mtf_agreement[i]
            # High agreement → scale up, low → scale down
            pos_mult *= np.clip(0.6 + agr*0.6, 0.6, 1.2)

        # C) LS quality: adjust LS weight based on momentum quality
        if use_ls_quality:
            lq=ls_quality[i]
            # High quality → more LS weight
            if lq>0.3:
                lw_adj=min(lw+0.05, 0.98)  # increase LS weight
            elif lq<-0.3:
                lw_adj=max(lw-0.05, 0.60)  # decrease LS weight
            else:
                lw_adj=lw
            # Recompute pnl with adjusted weights
            dw_adj=max(0,1-lw_adj)
            pnl=dw_adj*ret[i]*bp+lw_adj*lp1h[i]

        # D) Crypto-native tilt
        if use_crypto_tilt:
            ct=crypto_tilt[i]
            pos_mult *= (1.0 + ct)  # -0.5→0.5x, 0→1.0x, +0.5→1.5x

        # E) Hurst regime: scale position by trending/MR detection
        if use_hurst:
            h_val=hurst_smooth[i-1] if not np.isnan(hurst_smooth[i-1]) else 0.5
            # H>0.5 trending → good for momentum → scale up
            # H<0.5 mean-reverting → bad for momentum → scale down
            hurst_mult = np.clip(0.5 + h_val, 0.6, 1.3)
            pos_mult *= hurst_mult

        pnl *= pos_mult

        # Fixed leverage + funding
        lev=3.0
        pnl *= lev
        pnl += fra[i]*abs(lev*pos_mult)

        # DD control
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7

        e*=(1+pnl);eq[i]=e;pk=max(pk,e)

    return eq

# ============================================================
# RESULTS
# ============================================================
print('\n'+' RESULTS '.center(70,'='),flush=True)
fmt='  %-38s %5s %5s %6s %6s %5s'
print(fmt%('Model','WFS','Win','B22','OOS25','DayW%'),flush=True)
print('  '+'-'*68,flush=True)

tests=[
    ('V2 Baseline (binary regime)',
     dict()),
    ('A) Continuous Regime Score',
     dict(use_cont_regime=True)),
    ('B) Multi-TF Momentum Agreement',
     dict(use_mtf=True)),
    ('C) Adaptive LS Quality Weight',
     dict(use_ls_quality=True)),
    ('D) Crypto-Native Signal Tilt',
     dict(use_crypto_tilt=True)),
    ('E) Hurst-Based Regime',
     dict(use_hurst=True)),
    # Combinations
    ('A+B (cont regime + MTF)',
     dict(use_cont_regime=True, use_mtf=True)),
    ('A+E (cont regime + Hurst)',
     dict(use_cont_regime=True, use_hurst=True)),
    ('B+C (MTF + LS quality)',
     dict(use_mtf=True, use_ls_quality=True)),
    ('B+D (MTF + crypto tilt)',
     dict(use_mtf=True, use_crypto_tilt=True)),
    ('C+D (LS quality + crypto tilt)',
     dict(use_ls_quality=True, use_crypto_tilt=True)),
    ('A+C+D (regime + LS + crypto)',
     dict(use_cont_regime=True, use_ls_quality=True, use_crypto_tilt=True)),
    ('B+C+D (MTF + LS + crypto)',
     dict(use_mtf=True, use_ls_quality=True, use_crypto_tilt=True)),
    ('A+B+E (regime + MTF + Hurst)',
     dict(use_cont_regime=True, use_mtf=True, use_hurst=True)),
    ('ALL FIVE (A+B+C+D+E)',
     dict(use_cont_regime=True, use_mtf=True, use_ls_quality=True,
          use_crypto_tilt=True, use_hurst=True)),
]

for label, kwargs in tests:
    eq=run_backtest(bp_v2, label, **kwargs)
    wfs,win,nf,b22,o25,dw_pct=eval_strat(eq, label)
    tag=''
    if 'Baseline' in label: tag=' <--'
    elif wfs>370 and b22>70: tag=' ***'
    elif wfs>366: tag=' +'
    print('  %-38s %4.0f%% %2d/%d %+5.0f%% %+5.0f%% %4.1f%%%s'%(
        label,wfs,win,nf,b22,o25,dw_pct,tag),flush=True)

# ============================================================
# LEAK CHECK: Shift all signals by 1 extra bar
# ============================================================
print('\n'+' LEAK CHECK '.center(70,'='),flush=True)
print('Shifting all signals by +1 bar (extra lag)...',flush=True)

# Save originals
cont_regime_orig=cont_regime.copy()
mtf_agreement_orig=mtf_agreement.copy()
ls_quality_orig=ls_quality.copy()
crypto_tilt_orig=crypto_tilt.copy()
hurst_smooth_orig=hurst_smooth.copy()

# Shift by 1
cont_regime=np.roll(cont_regime,1);cont_regime[0]=0.5
mtf_agreement=np.roll(mtf_agreement,1);mtf_agreement[0]=0
ls_quality=np.roll(ls_quality,1);ls_quality[0]=0
crypto_tilt=np.roll(crypto_tilt,1);crypto_tilt[0]=0
hurst_smooth=np.roll(hurst_smooth,1);hurst_smooth[0]=0.5

# Re-run best performers (check if extra lag destroys performance)
leak_tests=[
    ('A) Cont Regime (lag+1)', dict(use_cont_regime=True)),
    ('D) Crypto Tilt (lag+1)', dict(use_crypto_tilt=True)),
    ('E) Hurst (lag+1)', dict(use_hurst=True)),
]
for label, kwargs in leak_tests:
    eq=run_backtest(bp_v2, label, **kwargs)
    wfs,win,nf,b22,o25,dw_pct=eval_strat(eq, label)
    print('  %-38s %4.0f%% %2d/%d %+5.0f%% %+5.0f%% %4.1f%%'%(
        label,wfs,win,nf,b22,o25,dw_pct),flush=True)

# Restore originals
cont_regime=cont_regime_orig
mtf_agreement=mtf_agreement_orig
ls_quality=ls_quality_orig
crypto_tilt=crypto_tilt_orig
hurst_smooth=hurst_smooth_orig

# ============================================================
# SHUFFLE TEST: Best performing approach
# ============================================================
print('\n'+' SHUFFLE TEST (200x) '.center(70,'='),flush=True)
print('Shuffling signal arrays to check if timing matters...',flush=True)

def shuffle_test(eq_real, feature_arr, feature_name, run_kwargs, n_shuffles=200):
    """Shuffle a feature array and compare WFS distribution to real."""
    wfs_real,_,_,_,_,_=eval_strat(eq_real)
    wfs_shuffled=[]

    orig=globals()[feature_arr].copy()
    for s in range(n_shuffles):
        np.random.seed(s)
        shuffled=orig.copy()
        np.random.shuffle(shuffled[S:])
        globals()[feature_arr]=shuffled
        eq_s=run_backtest(bp_v2, f'shuffle_{s}', **run_kwargs)
        w,_,_,_,_,_=eval_strat(eq_s)
        wfs_shuffled.append(w)

    globals()[feature_arr]=orig  # restore
    p_value=np.mean([w>=wfs_real for w in wfs_shuffled])
    print(f'  {feature_name}: Real WFS={wfs_real:.0f}%, '
          f'Shuffle mean={np.mean(wfs_shuffled):.0f}%±{np.std(wfs_shuffled):.0f}%, '
          f'p={p_value:.3f} {"PASS" if p_value<0.05 else "FAIL"}',flush=True)
    return p_value

# Run shuffle tests on each individual approach
for feat_name, feat_arr, kwargs in [
    ('Continuous Regime', 'cont_regime', dict(use_cont_regime=True)),
    ('Multi-TF Agreement', 'mtf_agreement', dict(use_mtf=True)),
    ('LS Quality', 'ls_quality', dict(use_ls_quality=True)),
    ('Crypto Tilt', 'crypto_tilt', dict(use_crypto_tilt=True)),
    ('Hurst Regime', 'hurst_smooth', dict(use_hurst=True)),
]:
    eq=run_backtest(bp_v2, feat_name, **kwargs)
    shuffle_test(eq, feat_arr, feat_name, kwargs, n_shuffles=200)

print('\n'+' ANALYSIS '.center(70,'='),flush=True)
print('Compare each approach vs V2 baseline (WFS=366%, Win=51/57, B22=+68%)',flush=True)
print('Any approach with WFS>366% AND B22>68% AND shuffle p<0.05 is a candidate.',flush=True)
print('\nDone.',flush=True)
