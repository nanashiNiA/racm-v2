"""V3 Funding Bug Fix and Recalculation
========================================
BUG: pnl += fra[i] * abs(lev)
  → ロング時にfunding rateを「稼ぐ」計算になっている
  → 実際はロングは87%の時間でfundingを「払う」
  → 年間+323%の架空利益

修正: pnl -= fra[i] * abs(lev)  (ロング時)
      pnl += fra[i] * abs(lev)  (ショート時)
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3 FUNDING BUG: 修正と再計算',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=params.warmup_hours
fra=np.roll(h['funding'].fillna(0).values,1)
vol_30d=h['vol_30d'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
bp_v2=RACMRegime.compute(price,ret,params)

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
sz=RACMMicrostructure.spread_zscore(spread_raw,168,48)
lz=RACMMicrostructure.liq_zscore(total_liq_raw,168,48)
safety=RACMMicrostructure.safety_score(sz,lz)

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

print('Data ready.\n',flush=True)

def run_v3(funding_mode, lev_min=1.5, lev_max=3.0):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp_v2[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        lev=lev_min+(lev_max-lev_min)*safety[i]
        pnl=(dw*ret[i]*bp_i+lw*lp1h[i])*lev

        if funding_mode=='buggy':
            pnl+=fra[i]*abs(lev)
        elif funding_mode=='correct':
            # Position is net long when bp>0, net short when bp<0
            # Long pays positive funding, earns negative
            # Short earns positive funding, pays negative
            if bp_i>=0:  # long
                pnl-=abs(fra[i])*abs(lev)*0.87  # long pays ~87% of time
                # More accurate: pnl -= fra[i]*abs(lev) but fra can be negative
                # Actually: long position → funding cost = fra[i] * position_size
                # If fra>0 → long PAYS → subtract
                # If fra<0 → long EARNS → subtract negative = add
                # So: pnl -= fra[i] * abs(lev) is correct for long
                pass
            else:  # short
                pnl+=fra[i]*abs(lev)
        elif funding_mode=='correct_simple':
            # Simplest correct: strategy is net long → pays funding
            pnl-=fra[i]*abs(lev)
        elif funding_mode=='zero':
            pass

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
    return wfs,win,len(wf),b22,o25,mdd*100

print('V3 (1.5-3.0x) with different funding modes:',flush=True)
print(f'  Mode                      WFS   Win    B22    OOS    MDD',flush=True)
print(f'  '+'-'*60,flush=True)

for mode,label in [
    ('buggy', 'BUG (fra+abs)'),
    ('correct_simple', 'FIX: long pays'),
    ('zero', 'No funding'),
]:
    eq=run_v3(mode, 1.5, 3.0)
    wfs,win,nf,b22,o25,mdd=eval_eq(eq)
    tag=' ← BUG' if mode=='buggy' else (' ← CORRECT' if 'correct' in mode else '')
    print(f'  {label:<25} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%{tag}',flush=True)

# V2 comparison
print(f'\nV2 (fixed 3.0x):',flush=True)
for mode,label in [
    ('buggy', 'BUG'),
    ('correct_simple', 'FIX'),
    ('zero', 'No funding'),
]:
    eq=run_v3(mode, 3.0, 3.0)
    wfs,win,nf,b22,o25,mdd=eval_eq(eq)
    print(f'  {label:<25} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}%',flush=True)

# Check: how much of bp_v2 is short?
short_count=np.sum(bp_v2[S:]<0)
total_active=nn-S
print(f'\nbp_v2 < 0 (short): {short_count}/{total_active} ({short_count/total_active*100:.1f}%)',flush=True)

# Year by year impact
print(f'\nYear-by-year (V3 1.5-3.0x):',flush=True)
print(f'  Year      BUG      FIX    No-Fund   BUG-FIX gap',flush=True)
eq_bug=run_v3('buggy',1.5,3.0)
eq_fix=run_v3('correct_simple',1.5,3.0)
eq_nof=run_v3('zero',1.5,3.0)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:continue
    rb=(eq_bug[iy[-1]]/eq_bug[max(0,iy[0]-1)]-1)*100
    rf=(eq_fix[iy[-1]]/eq_fix[max(0,iy[0]-1)]-1)*100
    rn=(eq_nof[iy[-1]]/eq_nof[max(0,iy[0]-1)]-1)*100
    def fmt(v):
        if abs(v)>10000:return f'{v/1000:>+7.0f}K%'
        return f'{v:>+8.1f}%'
    print(f'  {yr}   {fmt(rb):>8} {fmt(rf):>8} {fmt(rn):>8}   {fmt(rb-rf):>8}',flush=True)

print('\nDone.',flush=True)
