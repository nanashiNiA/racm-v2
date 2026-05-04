"""High-Frequency Strategies Backtest
#1 Mean Reversion (1H BB/RSI)
#3 Funding Rate Arbitrage (per-asset)
#4 Cross-Exchange Basis (Binance vs bitFlyer)
#5 Intraday Momentum (session breakout)
Each tested standalone + combined with RACM V2
"""
import sys,os
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd,json,urllib.request
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams

print('='*70,flush=True)
print('  HIGH-FREQUENCY STRATEGIES BACKTEST',flush=True)
print('='*70,flush=True)

h=build_common_1h();nn=len(h);idx_h=h.index
ret=h['ret'].values;price=h['close'].values;S=2760
fra=np.roll(h['funding'].fillna(0).values,1)

# 1H cached assets
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

# V2 LS for combination
a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h_racm=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)

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
    trades=sum(1 for i in range(S+1,nn) if abs(eq[i]/eq[i-1]-1)>0.0001)
    tpd=trades/((nn-S)/24)
    return wfs,win,len(wf),b22,o25,dw/(dw+dl)*100 if dw+dl>0 else 50,tpd

print('\nComputing strategies...',flush=True)

# ============================================================
# #1 MEAN REVERSION (1H Bollinger Band)
# ============================================================
bb_mean=pd.Series(price).rolling(480,min_periods=240).mean().values  # 20d
bb_std=pd.Series(price).rolling(480,min_periods=240).std().values
rsi_vals=h['rsi_14d'].values if 'rsi_14d' in h.columns else np.full(nn,50)

mr_signal=np.zeros(nn)  # -1 to +1
for i in range(S,nn):
    if np.isnan(bb_mean[i-1]) or bb_std[i-1]<=0:continue
    z=(price[i-1]-bb_mean[i-1])/(bb_std[i-1]*2+1e-10)  # BB z-score
    rsi=rsi_vals[i-1] if not np.isnan(rsi_vals[i-1]) else 50
    # Mean reversion: buy when oversold, sell when overbought
    if z<-1 and rsi<30:mr_signal[i]=1.0   # oversold -> long
    elif z>1 and rsi>70:mr_signal[i]=-1.0  # overbought -> short
    elif z<-0.5 and rsi<40:mr_signal[i]=0.5
    elif z>0.5 and rsi>60:mr_signal[i]=-0.5

# MR equity (standalone, 1x leverage, 3bps cost per trade)
eq_mr=np.ones(nn);e=1.0;prev_sig=0;cost_bps=0.0003
for i in range(S,nn):
    pnl=ret[i]*mr_signal[i]
    if mr_signal[i]!=prev_sig:pnl-=cost_bps*abs(mr_signal[i]-prev_sig)
    prev_sig=mr_signal[i]
    e*=(1+pnl);eq_mr[i]=e

# ============================================================
# #3 FUNDING RATE ARBITRAGE (per-asset differential)
# ============================================================
# Fetch per-asset funding rates
print('Fetching per-asset funding...',flush=True)
asset_funding={}
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']:
    try:
        url=f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=500'
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'V2'}),timeout=15) as r:
            d=json.loads(r.read())
        df=pd.DataFrame(d);df['fundingTime']=pd.to_datetime(df['fundingTime'],unit='ms')
        df['fundingRate']=df['fundingRate'].astype(float)
        asset_funding[sym.replace('USDT','')]=df.set_index('fundingTime').sort_index()
    except:pass
print(f'  Got funding for: {list(asset_funding.keys())}',flush=True)

# Funding arb: long asset with lowest funding, short asset with highest
# (earn carry from both sides)
fa_signal=np.zeros(nn)
for i in range(S,nn):
    ts=idx_h[i]
    rates={}
    for a,fdf in asset_funding.items():
        closest=fdf.index[fdf.index<=ts]
        if len(closest)>0:
            rates[a]=fdf.loc[closest[-1],'fundingRate']
    if len(rates)>=3:
        sorted_rates=sorted(rates.items(),key=lambda x:x[1])
        long_asset=sorted_rates[0][0]   # lowest funding -> earn by longing
        short_asset=sorted_rates[-1][0]  # highest funding -> earn by shorting
        # PnL = long return - short return + carry
        la_ret=ar_al.get(long_asset,np.zeros(nn))[i]
        sa_ret=ar_al.get(short_asset,np.zeros(nn))[i]
        carry=sorted_rates[-1][1]-sorted_rates[0][1]  # funding spread
        fa_signal[i]=la_ret-sa_ret+carry

eq_fa=np.ones(nn);e=1.0
for i in range(S,nn):e*=(1+fa_signal[i]);eq_fa[i]=e

# ============================================================
# #5 INTRADAY MOMENTUM (session breakout)
# ============================================================
# Asian range (UTC 0-8), breakout during London/NY (UTC 8-21)
asian_high=np.zeros(nn);asian_low=np.zeros(nn)
hi=h['high'].values;lo=h['low'].values
for i in range(S,nn):
    hr=idx_h[i].hour
    if hr==8:  # end of Asian session: compute range from last 8 bars
        asian_high[i]=np.max(hi[max(0,i-8):i])
        asian_low[i]=np.min(lo[max(0,i-8):i])
    elif i>0:
        asian_high[i]=asian_high[i-1];asian_low[i]=asian_low[i-1]

im_signal=np.zeros(nn)
for i in range(S,nn):
    hr=idx_h[i].hour
    if 8<=hr<=20 and asian_high[i]>0 and asian_low[i]>0:
        if price[i-1]>asian_high[i]*1.001:im_signal[i]=1.0   # breakout up
        elif price[i-1]<asian_low[i]*0.999:im_signal[i]=-1.0  # breakout down
        # Hold until end of day (UTC 0)
    if hr<8 or hr>20:im_signal[i]=0  # flat outside session

eq_im=np.ones(nn);e=1.0;prev_sig=0
for i in range(S,nn):
    pnl=ret[i]*im_signal[i]
    if im_signal[i]!=prev_sig:pnl-=cost_bps*abs(im_signal[i]-prev_sig)
    prev_sig=im_signal[i]
    e*=(1+pnl);eq_im[i]=e

# ============================================================
# #4 CROSS-EXCHANGE BASIS (Binance vs bitFlyer)
# ============================================================
# Check if bitFlyer data available
bf_path='Y:/tardis_data/bitflyer/spot'
has_bf=os.path.exists(bf_path)
print(f'bitFlyer data: {"Available" if has_bf else "Not available"}',flush=True)
eq_ce=np.ones(nn)  # placeholder if no data

# ============================================================
# COMBINED: RACM V2 + each HF strategy
# ============================================================
# V2 base return
vol_30d=h['vol_30d'].values;adx=h['adx'].values;ma110=h['ma110'].values
ma20=h['ma20'].values;skew_a=h['skew_30d'].values;dd_a=h['dd'].values
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

def run_combined(hf_pnl, hf_weight, label):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        if bp2[i]<0.5:lw=0.95
        elif bp2[i]<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        racm_pnl=dw*ret[i]*bp2[i]+lw*lp1h_racm[i]
        # Blend RACM + HF
        pnl=racm_pnl*(1-hf_weight)+hf_pnl[i]*hf_weight
        pnl*=3.0;pnl+=fra[i]*3.0
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# HF PnL arrays
mr_pnl=np.zeros(nn)
for i in range(S,nn):mr_pnl[i]=ret[i]*mr_signal[i]
im_pnl=np.zeros(nn)
for i in range(S,nn):im_pnl[i]=ret[i]*im_signal[i]

# V2 baseline
eq_v2=run_combined(np.zeros(nn),0.0,'V2')

print('\nRESULTS:',flush=True)
print(f'  {"Strategy":<35} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS25":>6} {"DayW":>5} {"T/d":>5}',flush=True)
print('  '+'-'*70,flush=True)

tests=[
    ('V2 RACM (baseline)',eq_v2),
    ('#1 Mean Reversion (standalone)',eq_mr),
    ('#3 Funding Arb (standalone)',eq_fa),
    ('#5 Intraday Mom (standalone)',eq_im),
    ('V2 + 10% MeanRev',run_combined(mr_pnl,0.10,'V2+MR10')),
    ('V2 + 20% MeanRev',run_combined(mr_pnl,0.20,'V2+MR20')),
    ('V2 + 10% FundingArb',run_combined(fa_signal,0.10,'V2+FA10')),
    ('V2 + 20% FundingArb',run_combined(fa_signal,0.20,'V2+FA20')),
    ('V2 + 10% IntradayMom',run_combined(im_pnl,0.10,'V2+IM10')),
    ('V2 + 20% IntradayMom',run_combined(im_pnl,0.20,'V2+IM20')),
    ('V2 + 10% MR + 10% IM',run_combined(mr_pnl+im_pnl,0.10,'V2+MR+IM')),
]

for label,eq in tests:
    wfs,win,nf,b22,o25,dw,tpd=eval_strat(eq)
    tag=''
    if 'baseline' in label:tag=' <--'
    elif wfs>370 and b22>70:tag=' ***'
    print(f'  {label:<35} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {dw:>4.1f}% {tpd:>4.1f}',flush=True)

print('\nDone.',flush=True)
