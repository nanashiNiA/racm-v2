"""Binance再現性テスト — 清算データ + ticker のみ (高速版)
==========================================================
板データ(50-370MB/日)の処理が遅すぎるため、以下で代替:
1. liquidations (10-300KB/日): 清算連鎖 → liq_z
2. derivative_ticker (2-4MB/日): mark-last spread → spread proxy

もしliq_zだけでV3の効果が再現できれば、Binance固有性は否定される。
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from datetime import date,timedelta
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  BINANCE再現性: 清算データ + Ticker (高速)',flush=True)
print('='*70,flush=True)

BN_DIR='Y:/tardis_data/binance-futures/perpetual'
SYMBOL='BTCUSDT'

# ============================================================
# Load Binance liquidations + ticker (small files, fast)
# ============================================================
print('Loading Binance liquidations + ticker...',flush=True)

all_liq=[];all_ticker=[]
for year in range(2021,2026):
    for month in range(1,13):
        start=date(year,month,1)
        end=date(year,month+1,1)-timedelta(days=1) if month<12 else date(year,12,31)
        current=start
        while current<=end:
            ds=current.strftime('%Y-%m-%d')
            folder=os.path.join(BN_DIR,str(year),f'{month:02d}')

            # Liquidations (tiny files)
            liq_file=os.path.join(folder,f'binance-futures_perpetual_liquidations_{ds}_{SYMBOL}.csv.gz')
            if os.path.exists(liq_file):
                try:
                    df=pd.read_csv(liq_file,compression='gzip',usecols=['timestamp','side','price','amount'])
                    df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
                    df['liq_usd']=df['price']*df['amount']
                    df=df.set_index('timestamp').sort_index()
                    longs=df[df['side']=='sell'];shorts=df[df['side']=='buy']
                    long_agg=longs.resample('1H').agg({'liq_usd':'sum'}).rename(columns={'liq_usd':'long_liq_usd'})
                    short_agg=shorts.resample('1H').agg({'liq_usd':'sum'}).rename(columns={'liq_usd':'short_liq_usd'})
                    liq_count=df.resample('1H').size().to_frame('liq_count')
                    liq_df=pd.concat([long_agg,short_agg,liq_count],axis=1).fillna(0)
                    all_liq.append(liq_df.reset_index())
                except:pass

            # Ticker (small files, has mark_price and last_price for spread proxy)
            tick_file=os.path.join(folder,f'binance-futures_perpetual_derivative_ticker_{ds}_{SYMBOL}.csv.gz')
            if os.path.exists(tick_file):
                try:
                    df=pd.read_csv(tick_file,compression='gzip',
                                  usecols=['timestamp','funding_rate','open_interest','last_price','mark_price'])
                    df['timestamp']=pd.to_datetime(df['timestamp'],unit='us',utc=True)
                    # Spread proxy: |mark_price - last_price| / last_price * 100
                    df['spread_proxy']=((df['mark_price']-df['last_price']).abs()/(df['last_price']+1e-10)*100)
                    df=df.set_index('timestamp').sort_index()
                    resampled=df.resample('1H').agg({
                        'spread_proxy':'mean',
                        'funding_rate':'last',
                        'open_interest':'last'
                    }).dropna(how='all')
                    all_ticker.append(resampled.reset_index())
                except:pass
            current+=timedelta(days=1)
    print(f'  {year} done',flush=True)

bn_liq=pd.concat(all_liq,ignore_index=True).sort_values('timestamp') if all_liq else None
bn_tick=pd.concat(all_ticker,ignore_index=True).sort_values('timestamp') if all_ticker else None

if bn_liq is not None:
    bn_liq=bn_liq.drop_duplicates('timestamp').set_index('timestamp')
    print(f'  Binance liquidations: {len(bn_liq)} bars',flush=True)
if bn_tick is not None:
    bn_tick=bn_tick.drop_duplicates('timestamp').set_index('timestamp')
    print(f'  Binance ticker: {len(bn_tick)} bars',flush=True)

# ============================================================
# Load RACM base data
# ============================================================
print('Loading RACM base...',flush=True)
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

idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h

# Align Binance data
bn_liq_aligned=bn_liq.reindex(idx_h_utc,method='ffill') if bn_liq is not None else None
bn_tick_aligned=bn_tick.reindex(idx_h_utc,method='ffill') if bn_tick is not None else None

# Extract Binance features
bn_total_liq=(bn_liq_aligned['long_liq_usd'].fillna(0).values+bn_liq_aligned['short_liq_usd'].fillna(0).values
              if bn_liq_aligned is not None else np.zeros(nn))
bn_spread=(bn_tick_aligned['spread_proxy'].fillna(0).values if bn_tick_aligned is not None else np.zeros(nn))

# Also load Bybit for comparison
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

by_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
by_all=by_all.drop_duplicates('timestamp').set_index('timestamp')
by_aligned=by_all.reindex(idx_h_utc,method='ffill')

by_spread=by_aligned['spread_pct'].fillna(0).values
by_total_liq=(by_aligned['long_liq_usd'].fillna(0).values+by_aligned['short_liq_usd'].fillna(0).values
              if 'long_liq_usd' in by_aligned.columns else np.zeros(nn))

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print('\n'+' CORRELATION: Bybit vs Binance '.center(70,'='),flush=True)

mask=(by_total_liq[S:]>0)&(bn_total_liq[S:]>0)
if mask.sum()>100:
    corr_liq=np.corrcoef(by_total_liq[S:][mask], bn_total_liq[S:][mask])[0,1]
    print(f'  Liquidation volume correlation: r={corr_liq:.3f}',flush=True)
else:
    print(f'  Insufficient overlap for liquidation correlation',flush=True)
    corr_liq=0

mask2=(by_spread[S:]>0)&(bn_spread[S:]>0)
if mask2.sum()>100:
    corr_sp=np.corrcoef(by_spread[S:][mask2], bn_spread[S:][mask2])[0,1]
    print(f'  Spread proxy correlation: r={corr_sp:.3f}',flush=True)
else:
    print(f'  Insufficient overlap for spread correlation',flush=True)
    corr_sp=0

# ============================================================
# COMPUTE Z-SCORES AND SAFETY
# ============================================================
def compute_safety_from_raw(spread_raw, liq_raw, extra_lag=0):
    sp_ma=pd.Series(spread_raw).rolling(168,min_periods=48).mean().values
    sp_std=pd.Series(spread_raw).rolling(168,min_periods=48).std().values
    sz=np.zeros(nn);lag=1+extra_lag
    for i in range(S,nn):
        j=i-lag
        if 0<=j<nn and not np.isnan(sp_std[j]) and sp_std[j]>1e-6:
            sz[i]=(spread_raw[j]-sp_ma[j])/(sp_std[j]+1e-10)

    lq_ma=pd.Series(liq_raw).rolling(168,min_periods=48).mean().values
    lq_std=pd.Series(liq_raw).rolling(168,min_periods=48).std().values
    lz=np.zeros(nn)
    for i in range(S,nn):
        j=i-lag
        if 0<=j<nn and not np.isnan(lq_std[j]) and lq_std[j]>1e-6:
            lz[i]=(liq_raw[j]-lq_ma[j])/(lq_std[j]+1e-10)

    safety=np.ones(nn)*0.5
    for i in range(S,nn):
        s=1.0
        if sz[i]>2.0:s-=0.3
        elif sz[i]>1.0:s-=0.15
        if lz[i]>3.0:s-=0.3
        elif lz[i]>2.0:s-=0.15
        if sz[i]<-0.5 and lz[i]<0.5:s+=0.2
        safety[i]=np.clip(s,0.0,1.5)
    return safety,sz,lz

safety_bybit,sz_by,lz_by=compute_safety_from_raw(by_spread, by_total_liq)
safety_binance,sz_bn,lz_bn=compute_safety_from_raw(bn_spread, bn_total_liq)

# Liq-only safety (spread not available from Binance properly)
def compute_safety_liq_only(liq_raw, extra_lag=0):
    lq_ma=pd.Series(liq_raw).rolling(168,min_periods=48).mean().values
    lq_std=pd.Series(liq_raw).rolling(168,min_periods=48).std().values
    lz=np.zeros(nn);lag=1+extra_lag
    for i in range(S,nn):
        j=i-lag
        if 0<=j<nn and not np.isnan(lq_std[j]) and lq_std[j]>1e-6:
            lz[i]=(liq_raw[j]-lq_ma[j])/(lq_std[j]+1e-10)
    safety=np.ones(nn)*0.5
    for i in range(S,nn):
        s=1.0
        if lz[i]>3.0:s-=0.3
        elif lz[i]>2.0:s-=0.15
        safety[i]=np.clip(s,0.0,1.5)
    return safety,lz

safety_bn_liq,lz_bn_only=compute_safety_liq_only(bn_total_liq)
safety_by_liq,lz_by_only=compute_safety_liq_only(by_total_liq)

# Z-score correlation
mask_z=(lz_by[S:]!=0)&(lz_bn[S:]!=0)
if mask_z.sum()>100:
    corr_lz=np.corrcoef(lz_by[S:][mask_z], lz_bn[S:][mask_z])[0,1]
    print(f'  liq_z correlation: r={corr_lz:.3f}',flush=True)

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

def eval_strat(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    return wfs,win,len(wf),b22,o25

# ============================================================
# PERFORMANCE COMPARISON
# ============================================================
print('\n'+' PERFORMANCE '.center(70,'='),flush=True)
fmt=f'  {"Model":<40} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6}'
print(fmt,flush=True)
print('  '+'-'*64,flush=True)

eq_fixed=run_dynlev(np.ones(nn)*((3.0-1.5)/(4.0-1.5)))
tests=[
    ('V2 Fixed 3.0x', eq_fixed),
    ('Bybit spread+liq (original)', run_dynlev(safety_bybit)),
    ('Bybit liq-only', run_dynlev(safety_by_liq)),
    ('Binance spread_proxy+liq', run_dynlev(safety_binance)),
    ('Binance liq-only', run_dynlev(safety_bn_liq)),
]

for label,eq in tests:
    wfs,win,nf,b22,o25=eval_strat(eq)
    tag=''
    if 'original' in label: tag=' ← ORIGINAL'
    elif 'Fixed' in label: tag=' ← BASE'
    print(f'  {label:<40} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}%{tag}',flush=True)

# ============================================================
# SHUFFLE TEST: Binance liq_z (200x)
# ============================================================
print('\n'+' SHUFFLE: Binance liq-only (200x) '.center(70,'='),flush=True)
eq_bn_real=run_dynlev(safety_bn_liq)
wfs_real,_,_,b22_real,_=eval_strat(eq_bn_real)
wfs_sh=[];b22_sh=[]
for s in range(200):
    np.random.seed(s+80000)
    ss=safety_bn_liq.copy()
    active=ss[S:].copy()
    np.random.shuffle(active)
    ss[S:]=active
    eq_s=run_dynlev(ss)
    w,_,_,b,_=eval_strat(eq_s)
    wfs_sh.append(w);b22_sh.append(b)

p_wfs=np.mean([w>=wfs_real for w in wfs_sh])
p_b22=np.mean([b>=b22_real for b in b22_sh])
print(f'  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p_wfs:.3f} {"PASS" if p_wfs<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_sh):+.0f}%±{np.std(b22_sh):.0f}%, p={p_b22:.3f} {"PASS" if p_b22<0.05 else "FAIL"}',flush=True)

# ============================================================
# SUMMARY
# ============================================================
print('\n'+' SUMMARY '.center(70,'='),flush=True)
wfs_bn,_,_,b22_bn,_=eval_strat(run_dynlev(safety_bn_liq))
wfs_by,_,_,b22_by,_=eval_strat(run_dynlev(safety_by_liq))
wfs_v2,_,_,b22_v2,_=eval_strat(eq_fixed)

print(f'  Bybit liq-only:   WFS={wfs_by:.0f}%, B22={b22_by:+.0f}%',flush=True)
print(f'  Binance liq-only: WFS={wfs_bn:.0f}%, B22={b22_bn:+.0f}%',flush=True)
print(f'  V2 baseline:      WFS={wfs_v2:.0f}%, B22={b22_v2:+.0f}%',flush=True)
print(f'',flush=True)
if wfs_bn>wfs_v2 and b22_bn>b22_v2:
    print(f'  ✓ Binanceでも清算データの効果が再現 → 取引所固有ではない',flush=True)
else:
    print(f'  ⚠️ Binanceでは効果が再現されない',flush=True)
print(f'  清算量相関 (Bybit vs Binance): r={corr_liq:.3f}',flush=True)
print(f'  → r>0.7なら「市場全体の現象」、r<0.3なら「取引所固有」',flush=True)

print('\nDone.',flush=True)
