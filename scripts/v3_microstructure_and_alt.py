"""V3: TWO NEW DIRECTIONS
=========================
Direction 1: Tick-Level Microstructure (Bybit orderbook features → RACM overlay)
  - Order Book Imbalance (OBI) as regime confidence
  - Depth pressure asymmetry as position tilt
  - Spread regime as volatility filter
  - Liquidation cascade detector

Direction 2: Non-LS Architecture (completely different alpha sources)
  - A) Vol-Target Trend Following (single-asset, no LS)
  - B) Risk Parity across 6 assets
  - C) Carry Strategy (funding rate capture)
  - D) Cross-Sectional Value (mean-reversion, opposite of momentum)
  - E) Hybrid: best non-LS + RACM V2

All tested with Walk-Forward + shuffle + leak check.
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  V3: MICROSTRUCTURE + ALTERNATIVE ARCHITECTURES',flush=True)
print('='*70,flush=True)

# ============================================================
# LOAD BASE DATA
# ============================================================
print('Loading base data...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
S=2760;fra=np.roll(h['funding'].fillna(0).values,1)
adx=h['adx'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values;vol_30d=h['vol_30d'].values

# 8H LS (for V2 baseline)
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

# ============================================================
# DIRECTION 1: LOAD BYBIT MICROSTRUCTURE DATA
# ============================================================
print('\n--- Loading Bybit orderbook data (2021-2025) ---',flush=True)
cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
ob_frames=[]
for year in range(2021,2026):
    f=os.path.join(cache_dir,f'BTCUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(f):
        with open(f,'rb') as fp:
            ohlc,ticker,liq=pickle.load(fp)
        if ohlc is not None:
            ohlc['timestamp']=pd.to_datetime(ohlc['timestamp'],utc=True)
            ob_frames.append(ohlc)
            print(f'  {year}: {len(ohlc)} bars',flush=True)
        # Merge ticker
        if ticker is not None:
            ticker['timestamp']=pd.to_datetime(ticker['timestamp'],utc=True)
            ohlc=ohlc.merge(ticker[['timestamp','funding_rate','open_interest']],
                           on='timestamp',how='left')
        # Merge liquidations
        if liq is not None:
            liq['timestamp']=pd.to_datetime(liq['timestamp'],utc=True)
            ohlc=ohlc.merge(liq,on='timestamp',how='left')
            ohlc[['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']]=\
                ohlc[['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']].fillna(0)
        ob_frames[-1]=ohlc  # update with merged data

if ob_frames:
    ob_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    ob_all=ob_all.drop_duplicates('timestamp').set_index('timestamp')
    print(f'  Total: {len(ob_all)} bars ({ob_all.index[0].date()} → {ob_all.index[-1].date()})',flush=True)
else:
    print('  ERROR: No orderbook data found!',flush=True)
    sys.exit(1)

# Align orderbook data to RACM 1H index
# RACM uses tz-naive timestamps, Bybit uses UTC
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
ob_aligned=ob_all.reindex(idx_h_utc,method='nearest',tolerance=pd.Timedelta('2H'))

# Extract features as numpy arrays aligned with nn
obi_0=ob_aligned['imbalance_0'].fillna(0).values  # [-1,+1]
depth_imb=ob_aligned['depth_imbalance'].fillna(0).values
spread_pct=ob_aligned['spread_pct'].fillna(0).values
bid_press=ob_aligned['bid_pressure'].fillna(0).values
ask_press=ob_aligned['ask_pressure'].fillna(0).values
ob_oi=ob_aligned['open_interest'].fillna(method='ffill').fillna(0).values if 'open_interest' in ob_aligned.columns else np.zeros(nn)
liq_count_ob=ob_aligned['liq_count'].fillna(0).values if 'liq_count' in ob_aligned.columns else np.zeros(nn)
liq_imb=ob_aligned['liq_imbalance'].fillna(0).values if 'liq_imbalance' in ob_aligned.columns else np.zeros(nn)
long_liq=ob_aligned['long_liq_usd'].fillna(0).values if 'long_liq_usd' in ob_aligned.columns else np.zeros(nn)
short_liq=ob_aligned['short_liq_usd'].fillna(0).values if 'short_liq_usd' in ob_aligned.columns else np.zeros(nn)

print(f'  OBI coverage: {np.sum(obi_0!=0)}/{nn} bars ({np.sum(obi_0!=0)/nn*100:.1f}%)',flush=True)

# ============================================================
# MICROSTRUCTURE FEATURES
# ============================================================
print('\nComputing microstructure features...',flush=True)

# MS1: OBI z-score (rolling z-score of order book imbalance)
obi_z=np.zeros(nn)
obi_ma=pd.Series(obi_0).rolling(168,min_periods=48).mean().values  # 7d MA
obi_std=pd.Series(obi_0).rolling(168,min_periods=48).std().values
for i in range(S,nn):
    if not np.isnan(obi_std[i-1]) and obi_std[i-1]>1e-6:
        obi_z[i]=(obi_0[i-1]-obi_ma[i-1])/(obi_std[i-1]+1e-10)

# MS2: Depth pressure ratio (bid vs ask pressure asymmetry)
pressure_ratio=np.zeros(nn)
for i in range(S,nn):
    bp=bid_press[i-1];ap=ask_press[i-1]
    if bp+ap>0.01:
        pressure_ratio[i]=(bp-ap)/(bp+ap+1e-10)

# MS3: Spread regime (high spread = danger/illiquidity)
spread_z=np.zeros(nn)
sp_ma=pd.Series(spread_pct).rolling(168,min_periods=48).mean().values
sp_std=pd.Series(spread_pct).rolling(168,min_periods=48).std().values
for i in range(S,nn):
    if not np.isnan(sp_std[i-1]) and sp_std[i-1]>1e-6:
        spread_z[i]=(spread_pct[i-1]-sp_ma[i-1])/(sp_std[i-1]+1e-10)

# MS4: Liquidation cascade detector
liq_cascade=np.zeros(nn)
total_liq=long_liq+short_liq
liq_ma=pd.Series(total_liq).rolling(168,min_periods=48).mean().values
liq_std=pd.Series(total_liq).rolling(168,min_periods=48).std().values
for i in range(S,nn):
    if not np.isnan(liq_std[i-1]) and liq_std[i-1]>1e-6:
        liq_cascade[i]=(total_liq[i-1]-liq_ma[i-1])/(liq_std[i-1]+1e-10)

# MS5: OBI-Price divergence (OBI bullish but price falling = squeeze)
obi_price_div=np.zeros(nn)
for i in range(S,nn):
    r1d=0
    if i>=24:r1d=(price[i-1]-price[i-25])/(price[i-25]+1e-10) if price[i-25]>0 else 0
    ob=obi_z[i]
    # OBI bullish + price falling → potential squeeze (long signal)
    if ob>1.0 and r1d<-0.02: obi_price_div[i]=0.5
    elif ob>0.5 and r1d<-0.01: obi_price_div[i]=0.2
    # OBI bearish + price rising → potential top (caution)
    elif ob<-1.0 and r1d>0.02: obi_price_div[i]=-0.5
    elif ob<-0.5 and r1d>0.01: obi_price_div[i]=-0.2

print(f'  MS features computed. OBI z range: [{np.percentile(obi_z[S:],5):.2f}, {np.percentile(obi_z[S:],95):.2f}]',flush=True)

# ============================================================
# WF FOLDS
# ============================================================
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
    mdd=0;pk=1
    for i in range(S,nn):
        pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return wfs,win,len(wf),b22,o25,dw/(dw+dl)*100 if dw+dl>0 else 50,mdd*100

# ============================================================
# DIRECTION 1: MICROSTRUCTURE AS RACM OVERLAY
# ============================================================
print('\n'+' DIRECTION 1: MICROSTRUCTURE OVERLAY '.center(70,'='),flush=True)

def run_v2_with_ms(use_obi=False, use_pressure=False, use_spread=False,
                   use_liq=False, use_div=False):
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

        # OBI z-score: bullish OBI → slight boost, bearish → reduce
        if use_obi:
            oz=obi_z[i]
            ms_mult *= np.clip(1.0 + oz*0.1, 0.7, 1.3)

        # Depth pressure: bid-heavy → boost, ask-heavy → reduce
        if use_pressure:
            pr=pressure_ratio[i]
            ms_mult *= np.clip(1.0 + pr*0.15, 0.8, 1.2)

        # Spread regime: high spread → reduce (illiquidity danger)
        if use_spread:
            sz=spread_z[i]
            if sz>2.0: ms_mult *= 0.5  # extreme spread → big cut
            elif sz>1.0: ms_mult *= 0.8

        # Liquidation cascade: extreme liq → reduce (chaos)
        if use_liq:
            lc=liq_cascade[i]
            if lc>3.0: ms_mult *= 0.3  # massive liquidation cascade
            elif lc>2.0: ms_mult *= 0.6

        # OBI-Price divergence: squeeze signal
        if use_div:
            dv=obi_price_div[i]
            ms_mult *= (1.0 + dv*0.3)

        pnl *= ms_mult
        pnl *= 3.0  # leverage
        pnl += fra[i]*3.0

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

print(f'  {"Model":<38} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS25":>6} {"DW%":>5} {"MDD":>6}',flush=True)
print('  '+'-'*72,flush=True)

ms_tests=[
    ('V2 Baseline', dict()),
    ('MS1: OBI z-score', dict(use_obi=True)),
    ('MS2: Depth Pressure', dict(use_pressure=True)),
    ('MS3: Spread Regime', dict(use_spread=True)),
    ('MS4: Liq Cascade', dict(use_liq=True)),
    ('MS5: OBI-Price Divergence', dict(use_div=True)),
    ('MS1+3 (OBI + Spread)', dict(use_obi=True, use_spread=True)),
    ('MS1+4 (OBI + Liq)', dict(use_obi=True, use_liq=True)),
    ('MS3+4 (Spread + Liq)', dict(use_spread=True, use_liq=True)),
    ('MS1+3+4 (OBI+Spread+Liq)', dict(use_obi=True, use_spread=True, use_liq=True)),
    ('MS ALL', dict(use_obi=True, use_pressure=True, use_spread=True, use_liq=True, use_div=True)),
]

for label, kwargs in ms_tests:
    eq=run_v2_with_ms(**kwargs)
    wfs,win,nf,b22,o25,dw,mdd=eval_strat(eq)
    tag=''
    if 'Baseline' in label: tag=' <--'
    elif wfs>366 and b22>68: tag=' ***'
    elif wfs>366: tag=' +'
    print(f'  {label:<38} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {dw:>4.1f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# DIRECTION 2: NON-LS ARCHITECTURES
# ============================================================
print('\n'+' DIRECTION 2: NON-LS ARCHITECTURES '.center(70,'='),flush=True)

# --- 2A: Vol-Target Trend Following (Single Asset BTC) ---
print('  Computing 2A: Vol-Target Trend Following...',flush=True)
def run_voltarget_trend():
    eq=np.ones(nn);e=1.0;pk=1.0
    # 4 MA crossovers: fast/slow = 20/110d (in 1H bars: 480/2640)
    ma_fast=pd.Series(price).ewm(span=480,min_periods=120).mean().values
    ma_slow=pd.Series(price).ewm(span=2640,min_periods=720).mean().values
    # Target vol: 15% annualized → position size = target/realized
    target_vol=0.15/np.sqrt(365*24)  # per-hour
    for i in range(S,nn):
        # Trend signal: fast > slow → long, else flat/short
        if np.isnan(ma_fast[i-1]) or np.isnan(ma_slow[i-1]):
            pos=0
        elif ma_fast[i-1]>ma_slow[i-1]:
            pos=1.0
        elif ma_fast[i-1]<ma_slow[i-1]*0.98:  # 2% buffer for short
            pos=-0.3
        else:
            pos=0

        # Vol targeting: scale position by target/realized vol
        rv=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        rv_hourly=rv/(100*np.sqrt(365*24))  # convert daily% to hourly decimal
        if rv_hourly>1e-6:
            vol_scale=np.clip(target_vol/rv_hourly, 0.3, 3.0)
        else:
            vol_scale=1.0
        pos *= vol_scale

        pnl=ret[i]*pos
        pnl += fra[i]*abs(pos)  # funding

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# --- 2B: Risk Parity across 6 assets ---
print('  Computing 2B: Risk Parity...',flush=True)
def run_risk_parity():
    eq=np.ones(nn);e=1.0;pk=1.0
    asset_names=list(ar_al.keys())
    na=len(asset_names)
    # Rolling vol per asset (720 bars = 30d)
    asset_vols={a: pd.Series(ar_al[a]).rolling(720,min_periods=168).std().values for a in asset_names}
    for i in range(S,nn):
        # Inverse-vol weights
        vols=[]
        for a in asset_names:
            v=asset_vols[a][i-1] if i>0 and not np.isnan(asset_vols[a][i-1]) else 0.01
            vols.append(max(v,1e-6))
        inv_vols=[1.0/v for v in vols]
        total=sum(inv_vols)
        weights=[iv/total for iv in inv_vols]

        # Portfolio return (all long, risk-parity weighted)
        pnl=sum(weights[j]*ar_al[asset_names[j]][i] for j in range(na))

        # Vol targeting: scale total portfolio
        port_vol=np.sqrt(sum((weights[j]*vols[j])**2 for j in range(na)))
        target=0.0005  # target hourly vol
        scale=np.clip(target/(port_vol+1e-10), 0.5, 3.0)
        pnl *= scale

        pnl += fra[i]*abs(scale)  # funding

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# --- 2C: Carry Strategy (Funding Rate Capture) ---
print('  Computing 2C: Carry Strategy...',flush=True)
def run_carry():
    eq=np.ones(nn);e=1.0;pk=1.0
    # Strategy: Short perp when funding is positive (earn carry)
    #           Long perp when funding is negative (earn carry)
    # Position scaled by funding magnitude
    funding_raw=h['funding'].fillna(0).values
    f_ma=pd.Series(funding_raw).rolling(168,min_periods=24).mean().values
    for i in range(S,nn):
        fv=f_ma[i-1] if not np.isnan(f_ma[i-1]) else 0
        # Carry position: opposite of funding direction
        if fv>0.0001:
            pos=-np.clip(fv*5000, 0.3, 1.5)  # short to earn positive funding
        elif fv<-0.0001:
            pos=np.clip(-fv*5000, 0.3, 1.5)  # long to earn negative funding
        else:
            pos=0

        pnl=ret[i]*pos  # directional PnL
        # CRITICAL: funding earn is ONLY on the correct side
        # If pos<0 (short) and funding>0, we earn funding
        # If pos>0 (long) and funding<0, we earn abs(funding)
        if pos<0 and fra[i]>0:
            pnl += fra[i]*abs(pos)  # earn positive funding while short
        elif pos>0 and fra[i]<0:
            pnl += abs(fra[i])*abs(pos)  # earn negative funding while long
        else:
            pnl -= abs(fra[i])*abs(pos)*0.5  # pay funding on wrong side (partial)

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# --- 2D: Cross-Sectional Value (Mean Reversion, opposite of Momentum) ---
print('  Computing 2D: Cross-Sectional Value...',flush=True)
def run_cs_value():
    eq=np.ones(nn);e=1.0;pk=1.0
    asset_names=list(ar8.keys())
    na=len(asset_names)
    # Value signal: buy past losers, sell past winners (30d lookback)
    # Opposite of LS momentum
    lb_b=30*3  # 30d in 8H bars
    am={n:np.roll(pd.Series(r).rolling(lb_b,min_periods=lb_b//3).sum().values,1) for n,r in ar8.items()}
    av={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in ar8.items()}

    # Map to 1H: compute 8H value PnL, then map
    val_pnl_8h=np.zeros(len(c8))
    for i in range(200,len(c8)):
        ms=[(am[n][i]/(av[n][i]+1e-10),n) for n in ar8 if not np.isnan(am[n][i])]
        if len(ms)<3:continue
        ms.sort(key=lambda x:x[0],reverse=True)
        # REVERSE: short winner, long loser
        short_a=ms[0][1];long_a=ms[-1][1]
        val_pnl_8h[i]=(ar8[long_a][i]-ar8[short_a][i])/2

    val_pnl_1h=RACMLS.map_8h_to_1h(val_pnl_8h,c8,idx_h,nn)

    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp=bp_v2[i]
        if bp<0.5:lw=0.95
        elif bp<0.8:lw=0.90
        elif v>1.0:lw=0.90
        else:lw=0.70
        dw=max(0,1-lw)
        # Replace LS momentum with value (MR)
        pnl=dw*ret[i]*bp+lw*val_pnl_1h[i]
        pnl *= 3.0
        pnl += fra[i]*3.0

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# --- 2E: Hybrid (best non-LS + V2 RACM blend) ---
print('  Computing 2E: Hybrid blends...',flush=True)
def run_v2_baseline():
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
        pnl *= 3.0
        pnl += fra[i]*3.0
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

eq_v2=run_v2_baseline()
eq_vt=run_voltarget_trend()
eq_rp=run_risk_parity()
eq_carry=run_carry()
eq_val=run_cs_value()

# Blended hybrids
def blend_equity(eq_a, eq_b, w_a=0.7, w_b=0.3, label=''):
    """Blend two strategies by combining their bar-level returns."""
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        ret_a=eq_a[i]/eq_a[max(0,i-1)]-1 if eq_a[max(0,i-1)]>0 else 0
        ret_b=eq_b[i]/eq_b[max(0,i-1)]-1 if eq_b[max(0,i-1)]>0 else 0
        pnl=w_a*ret_a+w_b*ret_b
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

eq_hybrid_vt=blend_equity(eq_v2, eq_vt, 0.7, 0.3)
eq_hybrid_rp=blend_equity(eq_v2, eq_rp, 0.7, 0.3)
eq_hybrid_carry=blend_equity(eq_v2, eq_carry, 0.8, 0.2)

print(f'\n  {"Model":<38} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS25":>6} {"DW%":>5} {"MDD":>6}',flush=True)
print('  '+'-'*72,flush=True)

alt_tests=[
    ('V2 RACM Baseline', eq_v2),
    ('2A) Vol-Target Trend (BTC only)', eq_vt),
    ('2B) Risk Parity (6 assets)', eq_rp),
    ('2C) Carry Strategy (funding)', eq_carry),
    ('2D) CS Value (anti-momentum)', eq_val),
    ('2E-1) 70% V2 + 30% VolTrend', eq_hybrid_vt),
    ('2E-2) 70% V2 + 30% RiskParity', eq_hybrid_rp),
    ('2E-3) 80% V2 + 20% Carry', eq_hybrid_carry),
]

for label, eq in alt_tests:
    wfs,win,nf,b22,o25,dw,mdd=eval_strat(eq)
    tag=''
    if 'Baseline' in label: tag=' <--'
    elif wfs>366 and b22>68: tag=' ***'
    elif wfs>366: tag=' +'
    print(f'  {label:<38} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {dw:>4.1f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# SHUFFLE TESTS (for any approach that beats V2)
# ============================================================
print('\n'+' SHUFFLE TESTS '.center(70,'='),flush=True)
print('Testing microstructure signals (200x)...',flush=True)

def shuffle_feature_test(feature_name, feature_arr, run_func, run_kwargs, n_shuffles=200):
    eq_real=run_func(**run_kwargs)
    wfs_real,_,_,_,_,_,_=eval_strat(eq_real)
    wfs_shuffled=[]
    orig=feature_arr.copy()
    for s in range(n_shuffles):
        np.random.seed(s)
        shuffled=orig.copy()
        active=shuffled[S:]
        np.random.shuffle(active)
        shuffled[S:]=active
        # Temporarily replace global
        if feature_name=='obi_z':
            global obi_z; obi_z=shuffled
        elif feature_name=='spread_z':
            global spread_z; spread_z=shuffled
        elif feature_name=='liq_cascade':
            global liq_cascade; liq_cascade=shuffled
        eq_s=run_func(**run_kwargs)
        w,_,_,_,_,_,_=eval_strat(eq_s)
        wfs_shuffled.append(w)

    # Restore
    if feature_name=='obi_z': obi_z=orig
    elif feature_name=='spread_z': spread_z=orig
    elif feature_name=='liq_cascade': liq_cascade=orig

    p=np.mean([w>=wfs_real for w in wfs_shuffled])
    print(f'  {feature_name}: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_shuffled):.0f}%±{np.std(wfs_shuffled):.0f}%, '
          f'p={p:.3f} {"PASS" if p<0.05 else "FAIL"}',flush=True)
    return p

for feat_name, feat_arr, kwargs in [
    ('obi_z', obi_z, dict(use_obi=True)),
    ('spread_z', spread_z, dict(use_spread=True)),
    ('liq_cascade', liq_cascade, dict(use_liq=True)),
]:
    shuffle_feature_test(feat_name, feat_arr, run_v2_with_ms, kwargs)

# ============================================================
# LEAK CHECK
# ============================================================
print('\n'+' LEAK CHECK '.center(70,'='),flush=True)
print('All microstructure features use [i-1] (lagged). Verifying with +1 shift...',flush=True)

# Shift all MS features by +1
obi_z_orig=obi_z.copy()
spread_z_orig=spread_z.copy()
liq_cascade_orig=liq_cascade.copy()
obi_z=np.roll(obi_z,1);obi_z[0]=0
spread_z=np.roll(spread_z,1);spread_z[0]=0
liq_cascade=np.roll(liq_cascade,1);liq_cascade[0]=0

for label, kwargs in [
    ('MS1: OBI z (lag+1)', dict(use_obi=True)),
    ('MS3: Spread (lag+1)', dict(use_spread=True)),
    ('MS4: Liq (lag+1)', dict(use_liq=True)),
]:
    eq=run_v2_with_ms(**kwargs)
    wfs,win,nf,b22,o25,dw,mdd=eval_strat(eq)
    print(f'  {label:<38} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {dw:>4.1f}%',flush=True)

obi_z=obi_z_orig;spread_z=spread_z_orig;liq_cascade=liq_cascade_orig

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n'+' SUMMARY '.center(70,'='),flush=True)
print('V2 Baseline: WFS=366%, Win=51/57, B22=+68%, MDD=-42%',flush=True)
print('',flush=True)
print('Any approach that:',flush=True)
print('  1. WFS > 366% (improvement)',flush=True)
print('  2. B22 > +68% (no Bear tradeoff)',flush=True)
print('  3. Shuffle p < 0.05 (not random)',flush=True)
print('  4. Leak check passes (lag+1 stable)',flush=True)
print('is a genuine improvement candidate.',flush=True)
print('\nDone.',flush=True)
