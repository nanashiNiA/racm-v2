"""V3: Binance先物でのMS信号再現性テスト
==========================================
Bybitで発見したMS3+MS4 (spread regime + liq cascade)が
Binance先物データでも同じ効果を持つか検証。

テスト:
1. Binance spread_z / liq_z の統計量比較
2. Binanceデータでの動的レバV3性能
3. Bybit vs Binance の信号相関
4. Binanceデータでのシャッフルテスト
5. クロス検証: Bybit信号でBinance、Binance信号でBybit
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  V3: BINANCE先物 再現性テスト',flush=True)
print('='*70,flush=True)

# Load base data
print('Loading base data...',flush=True)
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

cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h

# Load both exchanges
def load_exchange_data(prefix):
    frames=[]
    for year in range(2021,2026):
        f=os.path.join(cache_dir,f'{prefix}_{year}0101_{year}1231_1H.pkl')
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
                frames.append(ohlc)
    if not frames:
        return None
    df=pd.concat(frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    df=df.drop_duplicates('timestamp').set_index('timestamp')
    return df

print('Loading Bybit...',flush=True)
bybit=load_exchange_data('BTCUSDT')
print(f'  Bybit: {len(bybit)} bars',flush=True)

print('Loading Binance...',flush=True)
binance=load_exchange_data('BINANCE_BTCUSDT')
if binance is None:
    print('  ERROR: Binance cache not found! Run v3_binance_build_cache.py first.',flush=True)
    sys.exit(1)
print(f'  Binance: {len(binance)} bars',flush=True)

# Align both to RACM index
def align_and_compute(ob_df, label):
    aligned=ob_df.reindex(idx_h_utc,method='nearest',tolerance=pd.Timedelta('2H'))
    spread=aligned['spread_pct'].fillna(0).values
    total_liq=(aligned['long_liq_usd'].fillna(0).values+aligned['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in aligned.columns else np.zeros(nn))

    sp_ma=pd.Series(spread).rolling(168,min_periods=48).mean().values
    sp_std=pd.Series(spread).rolling(168,min_periods=48).std().values
    sz=np.zeros(nn)
    for i in range(S,nn):
        if not np.isnan(sp_std[i-1]) and sp_std[i-1]>1e-6:
            sz[i]=(spread[i-1]-sp_ma[i-1])/(sp_std[i-1]+1e-10)

    lq_ma=pd.Series(total_liq).rolling(168,min_periods=48).mean().values
    lq_std=pd.Series(total_liq).rolling(168,min_periods=48).std().values
    lz=np.zeros(nn)
    for i in range(S,nn):
        if not np.isnan(lq_std[i-1]) and lq_std[i-1]>1e-6:
            lz[i]=(total_liq[i-1]-lq_ma[i-1])/(lq_std[i-1]+1e-10)

    safety=np.ones(nn)*0.5
    for i in range(S,nn):
        s=1.0
        if sz[i]>2.0: s-=0.3
        elif sz[i]>1.0: s-=0.15
        if lz[i]>3.0: s-=0.3
        elif lz[i]>2.0: s-=0.15
        if sz[i]<-0.5 and lz[i]<0.5: s+=0.2
        safety[i]=np.clip(s,0.0,1.5)

    coverage=np.sum(spread!=0)
    print(f'  {label}: coverage={coverage}/{nn} ({coverage/nn*100:.1f}%)',flush=True)
    print(f'  {label}: spread_z range [{np.percentile(sz[S:],5):.2f}, {np.percentile(sz[S:],95):.2f}]',flush=True)
    print(f'  {label}: liq_z range [{np.percentile(lz[S:],5):.2f}, {np.percentile(lz[S:],95):.2f}]',flush=True)
    return sz,lz,safety,spread,total_liq

print('\nComputing features...',flush=True)
sz_by,lz_by,safety_by,sp_by,liq_by=align_and_compute(bybit,'Bybit')
sz_bn,lz_bn,safety_bn,sp_bn,liq_bn=align_and_compute(binance,'Binance')

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
        pnl=(dw*ret[i]*bp+lw*lp1h[i])*(lev_min+(lev_max-lev_min)*safety[i])
        lev=lev_min+(lev_max-lev_min)*safety[i]
        pnl+=fra[i]*abs(lev)
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
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return wfs,win,len(wf),b22,o25,mdd*100

# ============================================================
# 1. 信号の統計比較
# ============================================================
print('\n'+' 1. SIGNAL STATISTICS '.center(70,'='),flush=True)

# Correlation between Bybit and Binance signals
mask=(sz_by[S:]!=0)&(sz_bn[S:]!=0)
if mask.sum()>100:
    corr_sz=np.corrcoef(sz_by[S:][mask],sz_bn[S:][mask])[0,1]
    corr_lz=np.corrcoef(lz_by[S:][mask],lz_bn[S:][mask])[0,1]
    corr_safety=np.corrcoef(safety_by[S:][mask],safety_bn[S:][mask])[0,1]
    print(f'  Bybit-Binance相関:',flush=True)
    print(f'    spread_z: r={corr_sz:.3f}',flush=True)
    print(f'    liq_z:    r={corr_lz:.3f}',flush=True)
    print(f'    safety:   r={corr_safety:.3f}',flush=True)

# ============================================================
# 2. Binanceデータでの性能
# ============================================================
print('\n'+' 2. PERFORMANCE COMPARISON '.center(70,'='),flush=True)
fmt=f'  {"Model":<35} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6}'
print(fmt,flush=True)
print('  '+'-'*70,flush=True)

eq_v2=run_dynlev(np.ones(nn)*((3.0-1.5)/(4.0-1.5)), (1.5,4.0))  # fixed 3.0x equiv
eq_by=run_dynlev(safety_by, (1.5,4.0))
eq_bn=run_dynlev(safety_bn, (1.5,4.0))

# Average safety for combined
safety_avg=np.clip((safety_by+safety_bn)/2, 0.0, 1.5)
eq_avg=run_dynlev(safety_avg, (1.5,4.0))

for label,eq in [
    ('V2 Fixed 3.0x equiv',eq_v2),
    ('V3 Bybit signals',eq_by),
    ('V3 Binance signals',eq_bn),
    ('V3 Average (Bybit+Binance)',eq_avg),
]:
    wfs,win,nf,b22,o25,mdd=eval_strat(eq)
    tag=''
    if 'Bybit' in label: tag=' <-- original'
    print(f'  {label:<35} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# 3. Cross-exchange validation
# ============================================================
print('\n'+' 3. CROSS-EXCHANGE VALIDATION '.center(70,'='),flush=True)
print('  Bybit信号で取引 vs Binance信号で取引:',flush=True)
print(f'  → Binance WFS/V2 ratio: {eval_strat(eq_bn)[0]/eval_strat(eq_v2)[0]:.2f}x',flush=True)
print(f'  → Bybit WFS/V2 ratio:   {eval_strat(eq_by)[0]/eval_strat(eq_v2)[0]:.2f}x',flush=True)

# Year-by-year
print(f'\n  Year-by-year:',flush=True)
print(f'  {"Year":<6} {"Bybit":>10} {"Binance":>10} {"Average":>10}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:continue
    r_by=(eq_by[iy[-1]]/eq_by[max(0,iy[0]-1)]-1)*100
    r_bn=(eq_bn[iy[-1]]/eq_bn[max(0,iy[0]-1)]-1)*100
    r_av=(eq_avg[iy[-1]]/eq_avg[max(0,iy[0]-1)]-1)*100
    if abs(r_by)>10000:
        print(f'  {yr:<6} {r_by/1000:>+8.0f}K% {r_bn/1000:>+8.0f}K% {r_av/1000:>+8.0f}K%',flush=True)
    else:
        print(f'  {yr:<6} {r_by:>+9.1f}% {r_bn:>+9.1f}% {r_av:>+9.1f}%',flush=True)

# ============================================================
# 4. Binance shuffle test (200x)
# ============================================================
print('\n'+' 4. BINANCE SHUFFLE TEST (200x) '.center(70,'='),flush=True)
wfs_real_bn=eval_strat(eq_bn)[0]
b22_real_bn=eval_strat(eq_bn)[3]
wfs_sh=[];b22_sh=[]
for s in range(200):
    np.random.seed(s+70000)
    ss=safety_bn.copy()
    active=ss[S:].copy()
    np.random.shuffle(active)
    ss[S:]=active
    eq_s=run_dynlev(ss,(1.5,4.0))
    w,_,_,b,_,_=eval_strat(eq_s)
    wfs_sh.append(w);b22_sh.append(b)

p_wfs=np.mean([w>=wfs_real_bn for w in wfs_sh])
p_b22=np.mean([b>=b22_real_bn for b in b22_sh])
print(f'  WFS: Real={wfs_real_bn:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p_wfs:.3f} {"PASS" if p_wfs<0.05 else "FAIL"}',flush=True)
print(f'  B22: Real={b22_real_bn:+.0f}%, Shuffle={np.mean(b22_sh):+.0f}%±{np.std(b22_sh):.0f}%, p={p_b22:.3f} {"PASS" if p_b22<0.05 else "FAIL"}',flush=True)

# ============================================================
# 5. SUMMARY
# ============================================================
print('\n'+' SUMMARY '.center(70,'='),flush=True)
print(f'  Binance再現性: {"CONFIRMED" if eval_strat(eq_bn)[0]>eval_strat(eq_v2)[0] else "FAILED"}',flush=True)
print(f'  信号相関: spread_z r={corr_sz:.3f}, liq_z r={corr_lz:.3f}',flush=True)
print(f'  → r>0.7なら「市場全体の現象」、r<0.3なら「取引所固有」',flush=True)

print('\nDone.',flush=True)
