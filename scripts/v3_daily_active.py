"""V3 Daily-Active: 5 approaches that work EVERY bar, not just crises
A) LS spread-weighted sizing
B) Rolling Sharpe scaling
C) Cross-asset dispersion sizing
D) Funding carry optimization
E) Adaptive Kelly (escape cap)
Test each individually, then best combos.
"""
import sys,os
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams,RACMRegime
print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=2760;fra=np.roll(h['funding'].fillna(0).values,1)
adx=h['adx'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values;vol_30d=h['vol_30d'].values

# 8H LS + assets
a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)

# 1H asset returns (for dispersion)
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
print('Data ready',flush=True)

# Precompute features for daily-active approaches

# A) LS spread magnitude at 8H (mapped to 1H)
print('Computing features...',flush=True)
ls_spread_mag=np.zeros(nn)
na=len(ar8)
for lb in [60,90]:
    lb_b=lb*3;w=0.5
    am={n:np.roll(pd.Series(r).rolling(lb_b,min_periods=lb_b//3).sum().values,1) for n,r in ar8.items()}
    av={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in ar8.items()}
    for i in range(200,len(c8)):
        ms=[(am[n][i]/(av[n][i]+1e-10),n) for n in ar8 if not np.isnan(am[n][i])]
        if len(ms)<3:continue
        ms.sort(key=lambda x:x[0],reverse=True)
        spread=ms[0][0]-ms[-1][0]  # top - bottom score
        # Map to 1H
        ts8=c8[i];te8=c8[i+1] if i+1<len(c8) else ts8+pd.Timedelta(hours=8)
        idxs=np.where((idx_h>=ts8)&(idx_h<te8))[0]
        for ii in idxs:ls_spread_mag[ii]+=spread*w

# Normalize spread to [0,1] using expanding percentile
spread_pctrank=np.zeros(nn)
for i in range(S,nn):
    past=ls_spread_mag[max(S,i-4320):i]
    past=past[past!=0]
    if len(past)>100:
        spread_pctrank[i]=np.sum(past<=ls_spread_mag[i])/len(past)

# B) Rolling Sharpe (30d = 720 bars)
rolling_sharpe=np.zeros(nn)
# Need base_raw first - compute V2 regime
def build_bp():
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
bp2=build_bp()
# Base raw for Sharpe calc
base_v2=np.zeros(nn)
for i in range(S,nn):
    v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    if bp2[i]<0.5:lw=0.95
    elif bp2[i]<0.8:lw=0.90
    elif v>1.0:lw=0.90
    elif v>0.50:lw=0.80
    else:lw=0.70
    dw=max(0,1-lw)
    base_v2[i]=dw*ret[i]*bp2[i]+lw*lp1h[i]
# Rolling Sharpe
for i in range(S+720,nn):
    w=base_v2[i-720:i]
    mu=np.mean(w);sd=np.std(w)
    rolling_sharpe[i]=mu/(sd+1e-10)*np.sqrt(365*24) if sd>0 else 0

# C) Cross-asset dispersion (std of 6 asset returns per bar)
dispersion=np.zeros(nn)
asset_names=list(ar_al.keys())
for i in range(S,nn):
    rets_i=[ar_al[a][i] for a in asset_names if ar_al[a][i]!=0]
    if len(rets_i)>=3:
        dispersion[i]=np.std(rets_i)
# Rolling dispersion (smooth over 24H)
disp_smooth=np.roll(pd.Series(dispersion).rolling(24,min_periods=6).mean().values,1)
disp_pctrank=np.zeros(nn)
for i in range(S,nn):
    past=disp_smooth[max(S,i-4320):i]
    past=past[past>0]
    if len(past)>100:
        disp_pctrank[i]=np.sum(past<=disp_smooth[i])/len(past)

# D) Funding by asset (for carry optimization)
# Already have fra for BTC. For LS tilt we'd need per-asset funding
# Simplified: just use BTC funding direction
funding_dir=np.roll(np.sign(h['funding'].fillna(0).values),1)

print('Features computed',flush=True)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

def run(bp, spread_scale=False, sharpe_scale=False, disp_scale=False,
        funding_tilt=False, adaptive_kelly=False, label=''):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        if bp[i]<0.5:lw=0.95
        elif bp[i]<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        pnl=dw*ret[i]*bp[i]+lw*lp1h[i]

        # A) LS spread scaling
        pos_mult = 1.0
        if spread_scale:
            sp=spread_pctrank[i]
            pos_mult *= np.clip(0.5 + sp, 0.5, 1.5)  # 0.5x when weak, 1.5x when strong

        # B) Rolling Sharpe scaling
        if sharpe_scale:
            rs=rolling_sharpe[i]
            if rs > 2.0: pass  # full
            elif rs > 1.0: pos_mult *= 0.8
            elif rs > 0: pos_mult *= 0.6
            else: pos_mult *= 0.3  # negative Sharpe -> big reduction

        # C) Dispersion scaling
        if disp_scale:
            dp=disp_pctrank[i]
            pos_mult *= np.clip(0.5 + dp, 0.5, 1.5)  # high disp = good for LS

        # D) Funding tilt (add carry bonus)
        if funding_tilt:
            fd=funding_dir[i]
            pnl += abs(fra[i]) * 0.5  # bonus: actively capture funding

        pnl *= pos_mult

        # E) Adaptive Kelly
        if adaptive_kelly:
            rs=rolling_sharpe[i]
            # Sharpe-based leverage instead of mu/var
            lev = np.clip(rs * 0.5, 0.5, 3.0) if rs > 0 else 0.5
        else:
            lev = 3.0  # fixed cap

        pnl *= lev
        pnl += fra[i] * abs(lev * pos_mult)

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0
    # Daily win rate
    eq_s=pd.Series(eq,index=idx_h);dr=eq_s.resample('1D').last().pct_change().dropna()*100
    dr=dr[dr.index>=idx_h[S]];dw=int((dr>0).sum());dl=int((dr<0).sum())
    return wfs,win,len(wf),b22,o25,dw,dl

print(flush=True)
print('RESULTS:',flush=True)
fmt='  %-40s %5s %5s %6s %6s %5s'
print(fmt%('Model','WFS','Win','B22','OOS25','DayW%'),flush=True)
print('  '+'-'*70,flush=True)

tests=[
('V2 Baseline',False,False,False,False,False),
('A) LS spread sizing',True,False,False,False,False),
('B) Rolling Sharpe scaling',False,True,False,False,False),
('C) Dispersion sizing',False,False,True,False,False),
('D) Funding carry tilt',False,False,False,True,False),
('E) Adaptive Kelly',False,False,False,False,True),
('A+C (spread+disp)',True,False,True,False,False),
('B+C (sharpe+disp)',False,True,True,False,False),
('A+B+C',True,True,True,False,False),
('A+C+E (spread+disp+kelly)',True,False,True,False,True),
('ALL FIVE',True,True,True,True,True),
]

for label,a,b,c,d,e_flag in tests:
    wfs,win,nf,b22,o25,dw,dl=run(bp2,a,b,c,d,e_flag,label)
    dr=dw/(dw+dl)*100
    tag=''
    if label=='V2 Baseline':tag=' <--'
    elif wfs>370 and b22>70 and dr>65:tag=' ***'
    print('  %-40s %4.0f%% %2d/%d %+5.0f%% %+5.0f%% %4.1f%%%s'%(label,wfs,win,nf,b22,o25,dr,tag),flush=True)

print('Done.',flush=True)
