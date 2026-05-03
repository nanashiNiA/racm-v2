"""V3 ALL remaining tests: 1H LS, Macro, GARCH, Crash classifier"""
import sys, os
sys.path.insert(0, 'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0, 'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from v2core.data_loader import build_common_1h, load_6assets_8h, load_6assets_1h
from src.racm_core import RACMLS, RACMKelly, RACMDDControl, RACMParams, RACMRegime
print('Loading...', flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=2760;fra=np.roll(h['funding'].fillna(0).values,1)
adx=h['adx'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values;vol_30d=h['vol_30d'].values
# 8H LS
a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h_8h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
print('8H LS ready',flush=True)
# 1H LS
a1=load_6assets_1h();c1=list(a1.values())[0].index
for df in a1.values():c1=c1.intersection(df.index)
c1=c1.intersection(idx_h);na=len(a1)
ar1={a:a1[a].loc[c1,'return'].values for a in a1}
av1={n:np.roll(pd.Series(np.abs(r)).rolling(720,min_periods=240).mean().values,1) for n,r in ar1.items()}
lp1d=np.zeros(len(c1))
for lb in [60,90]:
    lb_b=lb*24;w=0.5
    am={n:np.roll(pd.Series(r).rolling(lb_b,min_periods=lb_b//3).sum().values,1) for n,r in ar1.items()}
    for i in range(max(lb_b+10,2200),len(c1)):
        ms=[(am[n][i]/(av1[n][i]+1e-10),n,ar1[n][i]) for n in ar1 if not np.isnan(am[n][i])]
        if len(ms)<3:continue
        ms.sort(key=lambda x:x[0],reverse=True)
        lp1d[i]+=(ms[0][2]/na-ms[-1][2]/na)*w
lp1h_1h=np.zeros(nn)
for i,ts in enumerate(c1):
    j=np.searchsorted(idx_h,ts)
    if j<nn:lp1h_1h[j]=lp1d[i]
print(f'1H LS: {np.count_nonzero(lp1h_1h)} values',flush=True)
# Correlation
ar_al={};
for a in a1:
    arr=np.zeros(nn)
    for i,ts in enumerate(c1):
        j=np.searchsorted(idx_h,ts)
        if j<nn:arr[j]=ar1[a][i]
    ar_al[a]=arr
corr1h=np.roll(pd.Series(ar_al.get('BTC',np.zeros(nn))).rolling(720,min_periods=240).corr(pd.Series(ar_al.get('ETH',np.zeros(nn)))).values,1)
print('Corr ready',flush=True)
# Macro
DATA='C:/Users/A701/Documents/nia/prediction_model_project/data/external'
try:
    mc=pd.read_csv(f'{DATA}/alternative/macro_data.csv',parse_dates=[0],index_col=0)
    def ald(s,idx):r=s.reindex(idx.normalize(),method='ffill');r.index=idx;return np.roll(r.values,24)
    vix=ald(mc['vix'],idx_h);dxy=ald(mc['dxy'],idx_h)
    dxy5=np.zeros(nn)
    for i in range(120,nn):dxy5[i]=dxy[i]-dxy[i-120]
except:vix=np.full(nn,20.0);dxy5=np.zeros(nn)
print('Macro ready',flush=True)
# GARCH vol
vf=np.zeros(nn);a_g=0.06
for i in range(S,nn):
    if i==S:vf[i]=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    else:vf[i]=a_g*abs(ret[i-1])*np.sqrt(365*24)+(1-a_g)*vf[i-1]
print('GARCH ready',flush=True)
# V2 bp
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
            elif(not np.isnan(adx[i-1]) and adx[i-1]>20) and(i-lc>=6):bp[i]=vs;lc=i;pb=vs
            else:bp[i]=pb
        else:bp[i]=vs;pb=vs
    return bp
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)
def run(bp,ls,macro=False,garch=False,crash=False,lw_f=None):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        if lw_f:lw=lw_f
        elif bp[i]<0.5:lw=0.95
        elif bp[i]<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw);bpe=bp[i]
        if crash and bp[i]<=0.2:
            c=corr1h[i] if not np.isnan(corr1h[i]) else 0.5
            lp=ls[i-1] if i>0 else 0
            if lp>0:bpe=0.5;lw=0.95;dw=0.05
            elif c>0.85:bpe=0.0
        pnl=dw*ret[i]*bpe+lw*ls[i]
        if macro and 0.9<bpe<1.1:
            vx=vix[i] if not np.isnan(vix[i]) else 20
            if vx<20 and dxy5[i]<0:pnl*=1.3
            elif vx>30 and dxy5[i]>0:pnl*=0.7
        if garch and vf[i]>0:
            sc=np.clip(0.80/(vf[i]+1e-10),0.3,2.0);pnl*=sc
        pnl*=3.0;pnl+=fra[i]*3.0
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
    return wfs,win,len(wf),b22,o25
bp1=RACMRegime.compute(price,ret,params);bp2=build_bp()
print(flush=True)
print('RESULTS:',flush=True)
h_fmt='  %-40s %6s %6s %7s %7s'
print(h_fmt%('Model','WFS','Win','B22','OOS25'),flush=True)
print('  '+'-'*68,flush=True)
tests=[
('V1 Baseline (8H LS)',bp1,lp1h_8h,False,False,False,0.80),
('V2 (8H LS+vol)',bp2,lp1h_8h,False,False,False,None),
('+ 1H LS direct',bp2,lp1h_1h,False,False,False,None),
('+ Macro (VIX/DXY)',bp2,lp1h_8h,True,False,False,None),
('+ GARCH vol forecast',bp2,lp1h_8h,False,True,False,None),
('+ Crash cls (1H corr)',bp2,lp1h_8h,False,False,True,None),
('+ 1H LS + Macro',bp2,lp1h_1h,True,False,False,None),
('+ 1H LS + GARCH',bp2,lp1h_1h,False,True,False,None),
('+ 1H LS + Crash',bp2,lp1h_1h,False,False,True,None),
('+ 1H LS + Macro + GARCH',bp2,lp1h_1h,True,True,False,None),
('+ ALL (1H+Macro+GARCH+Crash)',bp2,lp1h_1h,True,True,True,None),
]
for label,bp,ls,mc,gc,cr,lf in tests:
    wfs,win,nf,b22,o25=run(bp,ls,mc,gc,cr,lf)
    tag=''
    if 'V1' in label:tag=' <--v1'
    elif 'ALL' in label:tag=' <<<'
    elif wfs>300 and b22>70:tag=' ***'
    print('  %-40s %5.0f%% %2d/%d %+6.0f%% %+6.0f%%%s'%(label,wfs,win,nf,b22,o25,tag),flush=True)
print('Done.',flush=True)
