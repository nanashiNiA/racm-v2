"""V4: Multi-Factor Architecture Research
==========================================
V3の限界: 単一alpha (LS momentum) に依存、decay -1.81%/月

V4構想: 5 factorの低相関ポートフォリオ
  F1: Momentum (existing LS)
  F2: Carry (funding rate capture)
  F3: Microstructure (spread/liq risk signal)
  F4: Value (cross-sectional mean reversion)
  F5: Volatility (vol prediction → sizing)

Step 1: 各factorの独立PnLを計算
Step 2: ファクター間相関を確認
Step 3: 等重み・逆vol重み・動的重みを比較
Step 4: WF + Shuffle + OOS 検証
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V4: MULTI-FACTOR ARCHITECTURE RESEARCH',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=params.warmup_hours
fra=np.roll(h['funding'].fillna(0).values,1)
vol_30d=h['vol_30d'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
bp_v2=RACMRegime.compute(price,ret,params)

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
                    if c in ohlc.columns:ohlc[c]=ohlc[c].fillna(0)
            ob_frames.append(ohlc)
ob_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ob_all=ob_all.drop_duplicates('timestamp').set_index('timestamp')
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
ob_safe=ob_all.reindex(idx_h_utc,method='ffill')
spread_raw=ob_safe['spread_pct'].fillna(0).values
total_liq_raw=(ob_safe['long_liq_usd'].fillna(0).values+ob_safe['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in ob_safe.columns else np.zeros(nn))

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

print('Data ready.\n',flush=True)

# ============================================================
# STEP 1: 各ファクターの独立 PnL
# ============================================================
print('='*70,flush=True)
print(' STEP 1: Factor PnL計算',flush=True)
print('='*70,flush=True)

asset_names=list(ar_al.keys());na=len(asset_names)

# F1: Momentum LS (existing)
f1_pnl=lp1h.copy()
print(f'  F1 Momentum: {np.sum(f1_pnl[S:]!=0)} active bars',flush=True)

# F2: Carry (funding rate capture)
# Long BTC when funding negative (earn carry), short when positive
funding_raw=h['funding'].fillna(0).values
funding_ma=pd.Series(funding_raw).rolling(168,min_periods=24).mean().values
f2_pnl=np.zeros(nn)
for i in range(S,nn):
    fv=funding_ma[i-1] if not np.isnan(funding_ma[i-1]) else 0
    # Carry position: opposite of avg funding direction
    if fv>0.00005:
        # Positive funding → shorts earn → go short BTC (earn carry, bear directional)
        f2_pnl[i]=-ret[i]*0.3 + abs(fra[i])*0.3  # directional + carry
    elif fv<-0.00005:
        # Negative funding → longs earn → go long BTC (earn carry, bull directional)
        f2_pnl[i]=ret[i]*0.3 + abs(fra[i])*0.3
    else:
        f2_pnl[i]=0  # neutral funding → no position
print(f'  F2 Carry: {np.sum(f2_pnl[S:]!=0)} active bars',flush=True)

# F3: Microstructure (spread/liq → risk-adjusted return)
# Not a directional signal but risk scaling: safe periods earn more
sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety=RACMMicrostructure.safety_score(sz,lz)
f3_pnl=np.zeros(nn)
for i in range(S,nn):
    # Scale BTC return by safety: safe → full, danger → cut
    f3_pnl[i]=ret[i]*safety[i]  # risk-adjusted directional
print(f'  F3 Microstructure: {np.sum(f3_pnl[S:]!=0)} active bars',flush=True)

# F4: Value (cross-sectional mean reversion, opposite of momentum)
# Buy 30d losers, sell 30d winners
lb_val=30*3
am_val={n:np.roll(pd.Series(r).rolling(lb_val,min_periods=lb_val//3).sum().values,1) for n,r in ar8.items()}
av_val={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in ar8.items()}
val_pnl_8h=np.zeros(len(c8))
for i in range(200,len(c8)):
    ms=[(am_val[n][i]/(av_val[n][i]+1e-10),n) for n in ar8 if not np.isnan(am_val[n][i])]
    if len(ms)<3:continue
    ms.sort(key=lambda x:x[0],reverse=True)
    val_pnl_8h[i]=(ar8[ms[-1][1]][i]-ar8[ms[0][1]][i])/(2*na)  # long loser, short winner
f4_pnl=RACMLS.map_8h_to_1h(val_pnl_8h,c8,idx_h,nn)
print(f'  F4 Value: {np.sum(f4_pnl[S:]!=0)} active bars',flush=True)

# F5: Volatility targeting (scale by inverse vol)
# Higher vol → smaller position, lower vol → larger position → captures vol risk premium
f5_pnl=np.zeros(nn)
for i in range(S,nn):
    v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    # Target 50% annual vol → scale = 0.50/v
    vol_scale=np.clip(0.50/(v+1e-10), 0.3, 2.0)
    f5_pnl[i]=ret[i]*vol_scale
print(f'  F5 VolTarget: {np.sum(f5_pnl[S:]!=0)} active bars',flush=True)

# ============================================================
# STEP 2: ファクター間相関
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 2: Factor Correlation Matrix',flush=True)
print('='*70,flush=True)

factors={'F1_Mom':f1_pnl[S:],'F2_Carry':f2_pnl[S:],'F3_MS':f3_pnl[S:],
         'F4_Value':f4_pnl[S:],'F5_Vol':f5_pnl[S:],'BTC':ret[S:]}
fdf=pd.DataFrame(factors)
corr=fdf.corr()
print('  Correlation matrix:',flush=True)
for f in corr.columns:
    vals=' '.join([f'{corr.loc[f,g]:+.2f}' for g in corr.columns])
    print(f'    {f:<10} {vals}',flush=True)

print(f'\n  Key correlations:',flush=True)
print(f'    F1(Mom) vs F4(Value): {corr.loc["F1_Mom","F4_Value"]:+.3f} (should be negative = diversification)',flush=True)
print(f'    F1(Mom) vs F2(Carry): {corr.loc["F1_Mom","F2_Carry"]:+.3f}',flush=True)
print(f'    F2(Carry) vs F4(Value): {corr.loc["F2_Carry","F4_Value"]:+.3f}',flush=True)
print(f'    F3(MS) vs BTC: {corr.loc["F3_MS","BTC"]:+.3f}',flush=True)

# ============================================================
# STEP 3: 各ファクターの独立パフォーマンス
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 3: Factor Standalone Performance',flush=True)
print('='*70,flush=True)

def eval_factor_pnl(pnl_arr, lev=3.0):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        p=pnl_arr[i]*lev
        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:p*=0.1
        elif dd<-0.22:p*=0.4
        elif dd<-0.15:p*=0.7
        e*=(1+p);eq[i]=e;pk=max(pk,e)
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    mdd=0;pk_=1
    for i in range(S,nn):pk_=max(pk_,eq[i]);dd_=(eq[i]-pk_)/pk_;mdd=min(mdd,dd_)
    # Sharpe
    daily_ret=pd.Series(eq,index=idx_h).resample('1D').last().pct_change().dropna()
    daily_ret=daily_ret[daily_ret.index>=idx_h[S]]
    sharpe=daily_ret.mean()/daily_ret.std()*np.sqrt(365) if daily_ret.std()>0 else 0
    # Alpha decay
    from scipy import stats
    fold_rets=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    if len(fold_rets)>10:
        slope,_,_,p_decay,_=stats.linregress(range(len(fold_rets)),fold_rets)
    else:
        slope=0;p_decay=1
    return wfs,win,len(wf),mdd*100,sharpe,slope,p_decay

print(f'  {"Factor":<15} {"WFS":>6} {"Win":>6} {"MDD":>7} {"Sharpe":>7} {"Decay/mo":>9} {"p":>6}',flush=True)
print(f'  {"-"*60}',flush=True)

for name,pnl in [('F1 Momentum',f1_pnl),('F2 Carry',f2_pnl),('F3 MS',f3_pnl),
                  ('F4 Value',f4_pnl),('F5 VolTarget',f5_pnl),('BTC B&H',ret)]:
    wfs,win,nf,mdd,sharpe,slope,p_d=eval_factor_pnl(pnl)
    print(f'  {name:<15} {wfs:>5.0f}% {win:>2}/{nf} {mdd:>+6.1f}% {sharpe:>6.2f} {slope:>+8.2f}% {p_d:>5.3f}',flush=True)

# ============================================================
# STEP 4: Multi-Factor Portfolios
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 4: Multi-Factor Portfolio Construction',flush=True)
print('='*70,flush=True)

def run_multifactor(weights, lev=3.0, use_regime=True, use_ms_lev=True):
    """weights: dict of factor_name → weight"""
    eq=np.ones(nn);e=1.0;pk=1.0
    sz_=RACMMicrostructure.spread_zscore(spread_raw,168,48)
    lz_=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
    safety_=RACMMicrostructure.safety_score(sz_,lz_)

    for i in range(S,nn):
        # Base PnL: weighted sum of factors
        pnl_base=0
        for fname,w in weights.items():
            if fname=='F1':pnl_base+=f1_pnl[i]*w
            elif fname=='F2':pnl_base+=f2_pnl[i]*w
            elif fname=='F3':pnl_base+=f3_pnl[i]*w
            elif fname=='F4':pnl_base+=f4_pnl[i]*w
            elif fname=='F5':pnl_base+=f5_pnl[i]*w

        # Regime scaling (optional)
        regime_mult=1.0
        if use_regime:
            bp_i=bp_v2[i]
            if bp_i<0.3:regime_mult=0.2
            elif bp_i<0.5:regime_mult=0.5
            elif bp_i<0.8:regime_mult=0.7
            else:regime_mult=1.0

        # MS dynamic leverage (optional)
        if use_ms_lev:
            actual_lev=1.5+(lev-1.5)*safety_[i]
        else:
            actual_lev=lev

        pnl=pnl_base*regime_mult*actual_lev+fra[i]*abs(actual_lev)*regime_mult

        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_default):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

def eval_eq(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else 0
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    from scipy import stats
    fold_rets=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    slope,_,_,p_d,_=stats.linregress(range(len(fold_rets)),fold_rets) if len(fold_rets)>10 else (0,0,0,1,0)
    return wfs,win,len(wf),b22,o25,mdd*100,slope,p_d

hdr=f'  {"Model":<40} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6} {"Decay":>7} {"p":>5}'
print(hdr,flush=True);print('  '+'-'*82,flush=True)

portfolios=[
    ('V3 (F1 only, reference)', {'F1':1.0}),
    ('F1+F2 (Mom+Carry)', {'F1':0.7,'F2':0.3}),
    ('F1+F4 (Mom+Value)', {'F1':0.7,'F4':0.3}),
    ('F1+F2+F4 (Mom+Carry+Value)', {'F1':0.5,'F2':0.25,'F4':0.25}),
    ('F1+F2+F3+F4 (4-factor)', {'F1':0.4,'F2':0.2,'F3':0.2,'F4':0.2}),
    ('ALL 5 equal', {'F1':0.2,'F2':0.2,'F3':0.2,'F4':0.2,'F5':0.2}),
    ('F1+F4 balanced', {'F1':0.5,'F4':0.5}),
    ('F1+F2+F4+F5 (no MS)', {'F1':0.35,'F2':0.2,'F4':0.25,'F5':0.2}),
    ('F2+F4 (Carry+Value, no Mom)', {'F2':0.5,'F4':0.5}),
    ('F1 heavy + diversifiers', {'F1':0.6,'F2':0.15,'F4':0.15,'F5':0.1}),
]

results=[]
for label,weights in portfolios:
    eq=run_multifactor(weights, lev=3.0)
    wfs,win,nf,b22,o25,mdd,slope,p_d=eval_eq(eq)
    tag=''
    if 'reference' in label:tag=' <--'
    elif wfs>370 and b22>90 and mdd>=-30:tag=' ***'
    elif wfs>369:tag=' +'
    results.append((label,wfs,win,nf,b22,o25,mdd,slope,p_d))
    print(f'  {label:<40} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {slope:>+6.2f} {p_d:>4.3f}{tag}',flush=True)

# ============================================================
# STEP 5: Alpha decay comparison
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 5: Alpha Decay Comparison',flush=True)
print('='*70,flush=True)
print('  Does multi-factor reduce alpha decay?',flush=True)
print(f'\n  {"Model":<40} {"Decay/mo":>9} {"p-value":>8} {"Significant?":>12}',flush=True)
print(f'  {"-"*72}',flush=True)

for label,wfs,win,nf,b22,o25,mdd,slope,p_d in results:
    sig='YES ⚠️' if p_d<0.05 else 'NO ✓'
    tag=' <--' if 'reference' in label else ''
    print(f'  {label:<40} {slope:>+8.2f}% {p_d:>7.4f} {sig:>12}{tag}',flush=True)

# ============================================================
# STEP 6: Year-by-year for best candidates
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 6: Year-by-Year Best Candidates',flush=True)
print('='*70,flush=True)

best_labels=['V3 (F1 only, reference)','F1+F2+F4 (Mom+Carry+Value)','F1 heavy + diversifiers']
print(f'  {"Year":<6}',end='',flush=True)
for l in best_labels:print(f' {l[:18]:>18}',end='',flush=True)
print(flush=True)
print(f'  {"-"*(6+19*len(best_labels))}',flush=True)

for yr in range(2021,2026):
    print(f'  {yr:<6}',end='',flush=True)
    for label in best_labels:
        w={l:w for l,w in portfolios}[label]
        eq=run_multifactor(w)
        iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
        if len(iy)<100:print(f' {"--":>18}',end='',flush=True);continue
        r=(eq[iy[-1]]/eq[max(0,iy[0]-1)]-1)*100
        if abs(r)>10000:print(f' {r/1000:>+16.0f}K%',end='',flush=True)
        else:print(f' {r:>+17.1f}%',end='',flush=True)
    print(flush=True)

# ============================================================
# STEP 7: Shuffle test on best multi-factor
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 7: Shuffle Test (Best Multi-Factor, 200x)',flush=True)
print('='*70,flush=True)

# Find best that beats V3 with MDD ok
best_mf=None;best_mf_wfs=0
for label,wfs,win,nf,b22,o25,mdd,slope,p_d in results:
    if 'reference' not in label and wfs>best_mf_wfs and mdd>=-30:
        best_mf_wfs=wfs;best_mf=label

if best_mf:
    best_w={l:w for l,w in portfolios}[best_mf]
    eq_real=run_multifactor(best_w)
    wfs_real,_,_,b22_real,_,_,_,_=eval_eq(eq_real)
    print(f'  Testing: {best_mf}',flush=True)

    # Shuffle each factor independently
    wfs_sh=[];b22_sh=[]
    for seed in range(200):
        np.random.seed(seed+800000)
        # Shuffle factor PnLs (break timing)
        f1_s=f1_pnl.copy();f2_s=f2_pnl.copy();f4_s=f4_pnl.copy()
        np.random.shuffle(f1_s[S:]);np.random.shuffle(f2_s[S:]);np.random.shuffle(f4_s[S:])
        # Temporarily replace
        f1_orig=f1_pnl.copy();f2_orig=f2_pnl.copy();f4_orig=f4_pnl.copy()
        f1_pnl[:]=f1_s;f2_pnl[:]=f2_s;f4_pnl[:]=f4_s
        eq_s=run_multifactor(best_w)
        f1_pnl[:]=f1_orig;f2_pnl[:]=f2_orig;f4_pnl[:]=f4_orig
        w,_,_,b,_,_,_,_=eval_eq(eq_s)
        wfs_sh.append(w);b22_sh.append(b)

    p_w=np.mean([w>=wfs_real for w in wfs_sh])
    p_b=np.mean([b>=b22_real for b in b22_sh])
    print(f'  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%+/-{np.std(wfs_sh):.0f}%, p={p_w:.3f} {"PASS" if p_w<0.05 else "FAIL"}',flush=True)
    print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_sh):+.0f}%+/-{np.std(b22_sh):.0f}%, p={p_b:.3f} {"PASS" if p_b<0.05 else "FAIL"}',flush=True)

print('\nDone.',flush=True)
