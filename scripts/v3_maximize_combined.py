"""V3+ 組み合わせ価値の最大化
================================
前回: A+C+E = WFS +3%, B22 +6% (小さい)
      B (1H LS) = WFS -27% だがB22 +236% (Bearで圧倒的)

戦略:
1. Bear時だけ1H LS、平常時はMR+Funding (レジーム条件付き)
2. チャンネルサイズ拡大 (0.3x→0.5-1.0x)
3. MDD改善分をレバに転換 (1.5-3.0x → 1.5-3.5x)
4. クロスチャンネル確認で質を向上
5. 全てを組み合わせたフルモデル
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3+ MAXIMIZE COMBINED VALUE',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=params.warmup_hours
fra=np.roll(h['funding'].fillna(0).values,1)
vol_30d=h['vol_30d'].values;adx=h['adx'].values
ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
bp_v2=RACMRegime.compute(price,ret,params)

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

# Orderbook
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
ob_safe=ob_all.reindex(idx_h_utc, method='ffill')
spread_raw=ob_safe['spread_pct'].fillna(0).values
total_liq_raw=(ob_safe['long_liq_usd'].fillna(0).values+ob_safe['short_liq_usd'].fillna(0).values
               if 'long_liq_usd' in ob_safe.columns else np.zeros(nn))
sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety=RACMMicrostructure.safety_score(sz,lz)

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
print('Data ready.\n',flush=True)

# ============================================================
# Pre-compute signals
# ============================================================
print('Computing signals...',flush=True)

# Danger composite
dc_arr=np.zeros(nn)
for i in range(S,nn):
    dl=ds=0
    if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:dl+=1
    if not np.isnan(ma20[i-1]) and price[i-1]<ma20[i-1]:ds+=1
    if not np.isnan(skew_a[i-1]) and skew_a[i-1]<-0.5:dl+=1;ds+=1
    if dd_a[i-1]<-0.12:dl+=1;ds+=1
    dc_arr[i]=dl*0.3+ds*0.7

# A) MR signal
rsi_14=h['rsi_14d'].values if 'rsi_14d' in h.columns else np.full(nn,50)
bb_mean=pd.Series(price).rolling(480,min_periods=240).mean().values
bb_std=pd.Series(price).rolling(480,min_periods=240).std().values
mr_signal=np.zeros(nn)
for i in range(S,nn):
    if np.isnan(bb_mean[i-1]) or bb_std[i-1]<=0:continue
    z=(price[i-1]-bb_mean[i-1])/(bb_std[i-1]*2+1e-10)
    rsi=rsi_14[i-1] if not np.isnan(rsi_14[i-1]) else 50
    if z<-1.0 and rsi<30: mr_signal[i]=1.0
    elif z<-0.5 and rsi<40: mr_signal[i]=0.5
    elif z>1.0 and rsi>70: mr_signal[i]=-1.0
    elif z>0.5 and rsi>60: mr_signal[i]=-0.5

# B) 1H LS
asset_names=list(ar_al.keys());na=len(asset_names)
mom_1h={a:np.roll(pd.Series(ar_al[a]).rolling(720,min_periods=240).sum().values,1) for a in asset_names}
vol_1h={a:np.roll(pd.Series(np.abs(ar_al[a])).rolling(168,min_periods=48).mean().values,1) for a in asset_names}
ls_1h_pnl=np.zeros(nn)
for i in range(S,nn):
    scores=[]
    for a in asset_names:
        m=mom_1h[a][i];v=vol_1h[a][i]
        if not np.isnan(m) and v>1e-10:scores.append((m/v,a,ar_al[a][i]))
    if len(scores)<3:continue
    scores.sort(key=lambda x:x[0],reverse=True)
    ls_1h_pnl[i]=(scores[0][2]-scores[-1][2])/(2*na)

# C) Funding tilt
funding_raw=h['funding'].fillna(0).values
funding_ma=pd.Series(funding_raw).rolling(168,min_periods=24).mean().values
funding_tilt=np.zeros(nn)
for i in range(S,nn):
    fv=funding_ma[i-1] if not np.isnan(funding_ma[i-1]) else 0
    if fv>0.0002: funding_tilt[i]=-1.0
    elif fv>0.0001: funding_tilt[i]=-0.5
    elif fv<-0.0002: funding_tilt[i]=1.0
    elif fv<-0.0001: funding_tilt[i]=0.5

# E) Breakout
donch_high=pd.Series(price).rolling(480,min_periods=120).max().values
donch_low=pd.Series(price).rolling(480,min_periods=120).min().values
atr_24=pd.Series(np.abs(ret)).rolling(24,min_periods=6).mean().values
atr_168=pd.Series(np.abs(ret)).rolling(168,min_periods=48).mean().values
breakout_signal=np.zeros(nn)
for i in range(S,nn):
    vr=atr_24[i-1]/(atr_168[i-1]+1e-10) if atr_168[i-1]>1e-8 else 1
    if vr<0.6:
        if price[i-1]>=donch_high[i-1]*0.995: breakout_signal[i]=1.0
        elif price[i-1]<=donch_low[i-1]*1.005: breakout_signal[i]=-1.0

print('Signals computed.',flush=True)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

# ============================================================
# Full model runner
# ============================================================
def run_full(lev_min, lev_max, dd_levels,
             mr_size=0, fund_size=0, break_size=0,
             use_bear_1h_ls=False, bear_1h_weight=0.5,
             use_regime_adaptive=False):
    eq=np.ones(nn);e=1.0;pk=1.0

    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp_v2[i];dc=dc_arr[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)

        # Core: choose LS source based on regime
        if use_bear_1h_ls and dc>=1.0:
            # Bear regime: blend 8H LS with 1H LS (faster rebalancing)
            ls_pnl=(1-bear_1h_weight)*lp1h[i]+bear_1h_weight*ls_1h_pnl[i]
        else:
            ls_pnl=lp1h[i]

        pnl_base=dw*ret[i]*bp_i+lw*ls_pnl

        # Dynamic leverage
        lev=lev_min+(lev_max-lev_min)*safety[i]

        # Alpha overlay (regime-adaptive sizes)
        overlay=0
        if use_regime_adaptive:
            # Bull/Neutral: MR + Funding + Breakout
            # Bear: larger sizes (more conviction in MR during oversold)
            if dc>=1.0:  # danger
                overlay+=mr_signal[i]*mr_size*1.5  # bigger MR in bear
                overlay+=funding_tilt[i]*fund_size*0.5  # smaller funding in bear
            else:
                overlay+=mr_signal[i]*mr_size
                overlay+=funding_tilt[i]*fund_size
                overlay+=breakout_signal[i]*break_size
        else:
            overlay+=mr_signal[i]*mr_size
            overlay+=funding_tilt[i]*fund_size
            overlay+=breakout_signal[i]*break_size

        pnl=pnl_base*lev + ret[i]*overlay + fra[i]*abs(lev+overlay)

        dd=(e-pk)/pk if pk>0 else 0
        for level,mult in sorted(dd_levels):
            if dd<level:pnl*=mult;break
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

def eval_full(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else 0
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return wfs,win,len(wf),b22,o25,mdd*100

hdr=f'  {"Model":<45} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6}'

# ============================================================
# PHASE 1: Bear-only 1H LS
# ============================================================
print('\n'+' PHASE 1: Bear-only 1H LS '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*75,flush=True)

for label,kwargs in [
    ('V3 Base (1.5-3.0x)', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default)),
    ('+ Bear 1H LS (w=0.3)', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=0.3)),
    ('+ Bear 1H LS (w=0.5)', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=0.5)),
    ('+ Bear 1H LS (w=0.7)', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=0.7)),
    ('+ Bear 1H LS (w=1.0, replace)', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=1.0)),
]:
    eq=run_full(**kwargs)
    wfs,win,nf,b22,o25,mdd=eval_full(eq)
    tag=' <--' if 'Base' in label else (' ***' if wfs>370 and b22>100 else '')
    print(f'  {label:<45} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# PHASE 2: サイズ拡大
# ============================================================
print('\n'+' PHASE 2: Overlay サイズ拡大 '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*75,flush=True)

for label,kwargs in [
    ('V3 Base', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default)),
    ('MR 0.3x + Fund 0.15x', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,mr_size=0.3,fund_size=0.15)),
    ('MR 0.5x + Fund 0.2x', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,mr_size=0.5,fund_size=0.2)),
    ('MR 0.8x + Fund 0.3x', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,mr_size=0.8,fund_size=0.3)),
    ('MR 1.0x + Fund 0.5x + Break 0.3x', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,mr_size=1.0,fund_size=0.5,break_size=0.3)),
]:
    eq=run_full(**kwargs)
    wfs,win,nf,b22,o25,mdd=eval_full(eq)
    tag=' <--' if 'Base' in label else (' ***' if wfs>375 and b22>100 else '')
    print(f'  {label:<45} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# PHASE 3: MDD改善でレバ余地を取り戻す
# ============================================================
print('\n'+' PHASE 3: MDD改善→レバ拡大 '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*75,flush=True)

for label,kwargs in [
    ('V3 Base 1.5-3.0x', dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default)),
    ('1.5-3.5x + BearLS + MR0.5', dict(lev_min=1.5,lev_max=3.5,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=0.5,mr_size=0.5)),
    ('1.5-3.5x + BearLS + MR0.5 + Fund0.2', dict(lev_min=1.5,lev_max=3.5,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=0.5,mr_size=0.5,fund_size=0.2)),
    ('1.5-3.5x + Regime-Adaptive', dict(lev_min=1.5,lev_max=3.5,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=0.5,mr_size=0.5,fund_size=0.2,break_size=0.3,use_regime_adaptive=True)),
    ('2.0-3.5x + BearLS + MR0.5', dict(lev_min=2.0,lev_max=3.5,dd_levels=dd_default,use_bear_1h_ls=True,bear_1h_weight=0.5,mr_size=0.5)),
]:
    eq=run_full(**kwargs)
    wfs,win,nf,b22,o25,mdd=eval_full(eq)
    tag=' <--' if 'Base' in label else (' ***' if wfs>400 and b22>100 and mdd>=-30 else '')
    print(f'  {label:<45} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# PHASE 4: フルモデル候補
# ============================================================
print('\n'+' PHASE 4: FULL MODEL CANDIDATES '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*75,flush=True)

candidates=[
    ('V3 Base (reference)',
     dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default)),
    ('V3+ Conservative',
     dict(lev_min=1.5,lev_max=3.0,dd_levels=dd_default,
          use_bear_1h_ls=True,bear_1h_weight=0.3,mr_size=0.3,fund_size=0.15)),
    ('V3+ Moderate',
     dict(lev_min=1.5,lev_max=3.5,dd_levels=dd_default,
          use_bear_1h_ls=True,bear_1h_weight=0.5,mr_size=0.5,fund_size=0.2,break_size=0.2)),
    ('V3+ Aggressive',
     dict(lev_min=1.5,lev_max=3.5,dd_levels=dd_default,
          use_bear_1h_ls=True,bear_1h_weight=0.5,mr_size=0.8,fund_size=0.3,break_size=0.3,
          use_regime_adaptive=True)),
    ('V3+ Max (lev 2.0-3.5)',
     dict(lev_min=2.0,lev_max=3.5,dd_levels=dd_default,
          use_bear_1h_ls=True,bear_1h_weight=0.5,mr_size=0.5,fund_size=0.2,break_size=0.2,
          use_regime_adaptive=True)),
]

best_eq=None;best_wfs=0;best_label=''
for label,kwargs in candidates:
    eq=run_full(**kwargs)
    wfs,win,nf,b22,o25,mdd=eval_full(eq)
    tag=' <--' if 'reference' in label else ''
    if wfs>best_wfs and mdd>=-30 and b22>0:
        best_wfs=wfs;best_label=label;best_eq=eq;best_kwargs=kwargs
        tag=' ← BEST' if 'reference' not in label else tag
    print(f'  {label:<45} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# ============================================================
# Year-by-year for best
# ============================================================
if best_eq is not None:
    print(f'\n  Best: {best_label}',flush=True)
    eq_base=run_full(lev_min=1.5,lev_max=3.0,dd_levels=dd_default)
    print(f'  {"Year":<6} {"V3 Base":>10} {"V3+ Best":>10} {"Delta":>10}',flush=True)
    print(f'  {"-"*38}',flush=True)
    for yr in range(2021,2026):
        iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
        if len(iy)<100:continue
        rb=(eq_base[iy[-1]]/eq_base[max(0,iy[0]-1)]-1)*100
        rv=(best_eq[iy[-1]]/best_eq[max(0,iy[0]-1)]-1)*100
        def fmt(v):
            if abs(v)>10000:return f'{v/1000:>+7.0f}K%'
            return f'{v:>+9.1f}%'
        print(f'  {yr:<6} {fmt(rb):>10} {fmt(rv):>10} {fmt(rv-rb):>10}',flush=True)

# ============================================================
# Shuffle test on best
# ============================================================
print(f'\n'+' SHUFFLE TEST (Best, 200x) '.center(70,'='),flush=True)
if best_eq is not None:
    wfs_real,_,_,b22_real,_,_=eval_full(best_eq)
    wfs_sh=[];b22_sh=[]
    for seed in range(200):
        np.random.seed(seed+200000)
        # Shuffle all alpha signals simultaneously
        mr_s=mr_signal.copy();ft_s=funding_tilt.copy();br_s=breakout_signal.copy()
        np.random.shuffle(mr_s[S:]);np.random.shuffle(ft_s[S:]);np.random.shuffle(br_s[S:])
        # Temporarily replace
        mr_orig=mr_signal.copy();ft_orig=funding_tilt.copy();br_orig=breakout_signal.copy()
        mr_signal[:]=mr_s;funding_tilt[:]=ft_s;breakout_signal[:]=br_s
        eq_s=run_full(**best_kwargs)
        mr_signal[:]=mr_orig;funding_tilt[:]=ft_orig;breakout_signal[:]=br_orig
        w,_,_,b,_,_=eval_full(eq_s)
        wfs_sh.append(w);b22_sh.append(b)
    p_w=np.mean([w>=wfs_real for w in wfs_sh])
    p_b=np.mean([b>=b22_real for b in b22_sh])
    print(f'  WFS: Real={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%±{np.std(wfs_sh):.0f}%, p={p_w:.3f} {"PASS" if p_w<0.05 else "FAIL"}',flush=True)
    print(f'  B22: Real={b22_real:+.0f}%, Shuffle={np.mean(b22_sh):+.0f}%±{np.std(b22_sh):.0f}%, p={p_b:.3f} {"PASS" if p_b<0.05 else "FAIL"}',flush=True)

print('\nDone.',flush=True)
