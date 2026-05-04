"""V3: 年代別・要因別パフォーマンス分解
==========================================
各年で「何が」「どれだけ」効いているかを完全分解。

分解軸:
1. V2 Base (LS momentum + regime) at 3.0x
2. V3 MS Dynamic Lev の寄与 (safety score timing)
3. レバ増加分の寄与 (3.0x → avg ~2.5x at 1.5-3.0x range)
4. DD制御の寄与
5. Funding carry の寄与

さらに:
- MS Cut (危機縮小) vs MS Boost (安全ブースト) の分離
- Bear月 vs Bull月 での効果
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V3: 年代別・要因別 完全分解',flush=True)
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
# Build multiple strategy variants for decomposition
# ============================================================
def run_variant(safety_arr, lev_min, lev_max, dd_levels, use_funding=True):
    """Run and return per-bar PnL array (not cumulative)."""
    pnl_arr=np.zeros(nn)
    eq=np.ones(nn);e=1.0;pk=1.0
    lev_arr=np.zeros(nn)
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp_v2[i]
        if bp_i<0.5:lw=0.95
        elif bp_i<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        lev=lev_min+(lev_max-lev_min)*safety_arr[i]
        lev_arr[i]=lev
        pnl_base=(dw*ret[i]*bp_i+lw*lp1h[i])*lev
        pnl_fund=fra[i]*abs(lev) if use_funding else 0
        pnl=pnl_base+pnl_fund
        # DD control
        dd=(e-pk)/pk if pk>0 else 0
        dd_mult=1.0
        for level,mult in sorted(dd_levels):
            if dd<level:dd_mult=mult;break
        pnl*=dd_mult
        pnl_arr[i]=pnl
        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq,pnl_arr,lev_arr

# Variants
safety_fixed=np.ones(nn)*((3.0-1.5)/(3.0-1.5))  # maps to 3.0x in 1.5-3.0
safety_neutral=np.ones(nn)*0.5  # maps to midpoint

# A: V2 baseline (fixed 3.0x, dd_default)
eq_v2, pnl_v2, lev_v2 = run_variant(np.ones(nn)*1.0, 3.0, 3.0, dd_default)

# B: V3 clean (1.5-3.0x, dd_default) — recommended
eq_v3, pnl_v3, lev_v3 = run_variant(safety, 1.5, 3.0, dd_default)

# C: Fixed avg lev of V3 (no MS timing)
avg_lev_v3 = np.mean(lev_v3[S:])
eq_fixed_avg, pnl_fixed_avg, _ = run_variant(np.ones(nn)*1.0, avg_lev_v3, avg_lev_v3, dd_default)

# D: V3 without DD control
eq_v3_nodd, pnl_v3_nodd, _ = run_variant(safety, 1.5, 3.0, [(-1.0,1.0)])

# E: V3 without funding
eq_v3_nofund, pnl_v3_nofund, _ = run_variant(safety, 1.5, 3.0, dd_default, use_funding=False)

# F: V3 cut-only (no boost)
safety_cut_only=np.ones(nn)*0.5
for i in range(nn):
    s=1.0
    if sz[i]>2.0:s-=0.3
    elif sz[i]>1.0:s-=0.15
    if lz[i]>3.0:s-=0.3
    elif lz[i]>2.0:s-=0.15
    # NO boost
    safety_cut_only[i]=float(np.clip(s,0.0,1.0))
eq_v3_cutonly, pnl_v3_cutonly, lev_cutonly = run_variant(safety_cut_only, 1.5, 3.0, dd_default)

# G: V3 boost-only (no cut)
safety_boost_only=np.ones(nn)*0.5
for i in range(nn):
    s=1.0
    if sz[i]<-0.5 and lz[i]<0.5:s+=0.2
    safety_boost_only[i]=float(np.clip(s,0.5,1.5))
eq_v3_boostonly, pnl_v3_boostonly, _ = run_variant(safety_boost_only, 1.5, 3.0, dd_default)

# ============================================================
# YEARLY DECOMPOSITION
# ============================================================
print('='*70,flush=True)
print(' 1. 年別リターン比較',flush=True)
print('='*70,flush=True)

def year_ret(eq, yr):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:return 0
    return (eq[iy[-1]]/eq[max(0,iy[0]-1)]-1)*100

def year_mdd(eq, yr):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:return 0
    pk=eq[iy[0]];mdd=0
    for i in iy:pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    return mdd*100

print(f'  {"Year":<6} {"BTC B&H":>8} {"V2(3.0x)":>10} {"V3(1.5-3)":>10} {"V3 MDD":>8} {"FixAvg":>10} {"TimingΔ":>8}',flush=True)
print(f'  {"-"*64}',flush=True)

for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:continue
    bh=(price[iy[-1]]/price[iy[0]]-1)*100
    rv2=year_ret(eq_v2,yr)
    rv3=year_ret(eq_v3,yr)
    rfa=year_ret(eq_fixed_avg,yr)
    timing=rv3-rfa  # V3 - same avg lev = timing value
    mdd3=year_mdd(eq_v3,yr)

    def fmt(v):
        if abs(v)>10000: return f'{v/1000:>+7.0f}K%'
        return f'{v:>+9.1f}%'

    print(f'  {yr:<6} {fmt(bh):>8} {fmt(rv2):>10} {fmt(rv3):>10} {mdd3:>+7.1f}% {fmt(rfa):>10} {fmt(timing):>8}',flush=True)

# ============================================================
# 2. 要因分解
# ============================================================
print('\n'+'='*70,flush=True)
print(' 2. 年別 要因分解 (V3 1.5-3.0x, dd_default)',flush=True)
print('='*70,flush=True)

print(f'  {"Year":<6} {"V2→V3 Δ":>10} {"= レバ差":>10} {"+ Timing":>10} {"| Cut効果":>10} {"| Boost効果":>10} {"| Funding":>10}',flush=True)
print(f'  {"-"*74}',flush=True)

for yr in range(2021,2026):
    rv2=year_ret(eq_v2,yr)
    rv3=year_ret(eq_v3,yr)
    rfa=year_ret(eq_fixed_avg,yr)
    rcut=year_ret(eq_v3_cutonly,yr)
    rboost=year_ret(eq_v3_boostonly,yr)
    rnofund=year_ret(eq_v3_nofund,yr)

    total_delta=rv3-rv2
    lev_delta=rfa-rv2  # from avg leverage change
    timing_delta=rv3-rfa  # from MS timing
    cut_effect=rcut-rfa  # cut-only vs fixed avg
    boost_effect=rboost-rfa  # boost-only vs fixed avg
    funding_effect=rv3-rnofund  # funding contribution

    def fmt(v):
        if abs(v)>5000: return f'{v/1000:>+7.0f}K%'
        return f'{v:>+9.1f}%'

    print(f'  {yr:<6} {fmt(total_delta):>10} {fmt(lev_delta):>10} {fmt(timing_delta):>10} {fmt(cut_effect):>10} {fmt(boost_effect):>10} {fmt(funding_effect):>10}',flush=True)

# ============================================================
# 3. MS信号の発火状況
# ============================================================
print('\n'+'='*70,flush=True)
print(' 3. MS信号 年別発火状況',flush=True)
print('='*70,flush=True)

print(f'  {"Year":<6} {"Total":>6} {"Cut(sz>1)":>10} {"Cut(lz>2)":>10} {"Boost":>8} {"AvgLev":>8} {"LevRange":>12}',flush=True)
print(f'  {"-"*62}',flush=True)

for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    iy=iy[iy>=S]
    if len(iy)<100:continue

    n_cut_sp=np.sum(sz[iy]>1.0)
    n_cut_lq=np.sum(lz[iy]>2.0)
    n_boost=np.sum((sz[iy]<-0.5)&(lz[iy]<0.5))
    avg_lev=np.mean(lev_v3[iy])
    min_lev=np.min(lev_v3[iy])
    max_lev=np.max(lev_v3[iy])

    print(f'  {yr:<6} {len(iy):>6} {n_cut_sp:>7} ({n_cut_sp/len(iy)*100:.1f}%) {n_cut_lq:>7} ({n_cut_lq/len(iy)*100:.1f}%) {n_boost:>5} ({n_boost/len(iy)*100:.1f}%) {avg_lev:>6.2f}x {min_lev:.1f}-{max_lev:.1f}x',flush=True)

# ============================================================
# 4. 月別ヒートマップ (V3 vs V2)
# ============================================================
print('\n'+'='*70,flush=True)
print(' 4. 月別 V3-V2 差分 (% return)',flush=True)
print('='*70,flush=True)

months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(f'  {"Year":<6}', end='',flush=True)
for m in months:print(f' {m:>5}',end='',flush=True)
print(f' {"Total":>7}',flush=True)
print(f'  {"-"*(6+6*12+7)}',flush=True)

for yr in range(2021,2026):
    print(f'  {yr:<6}', end='',flush=True)
    yr_total=0
    for mo in range(1,13):
        im=np.where(np.array([(d.year==yr and d.month==mo) for d in idx_h]))[0]
        if len(im)<20:
            print(f'    --',end='',flush=True)
            continue
        rv2=(eq_v2[im[-1]]/eq_v2[max(0,im[0]-1)]-1)*100
        rv3=(eq_v3[im[-1]]/eq_v3[max(0,im[0]-1)]-1)*100
        delta=rv3-rv2
        yr_total+=delta
        if abs(delta)>100:
            print(f' {delta/100:>+4.0f}H',end='',flush=True)
        else:
            print(f' {delta:>+4.1f}',end='',flush=True)
    print(f' {yr_total:>+6.0f}%',flush=True)

# ============================================================
# 5. Bear月 vs Bull月
# ============================================================
print('\n'+'='*70,flush=True)
print(' 5. Bear月 vs Bull月 でのV3効果',flush=True)
print('='*70,flush=True)

monthly_v2=[];monthly_v3=[];monthly_btc=[]
for yr in range(2021,2026):
    for mo in range(1,13):
        im=np.where(np.array([(d.year==yr and d.month==mo) for d in idx_h]))[0]
        if len(im)<20:continue
        rv2=(eq_v2[im[-1]]/eq_v2[max(0,im[0]-1)]-1)*100
        rv3=(eq_v3[im[-1]]/eq_v3[max(0,im[0]-1)]-1)*100
        rbtc=(price[im[-1]]/price[im[0]]-1)*100
        monthly_v2.append(rv2);monthly_v3.append(rv3);monthly_btc.append(rbtc)

monthly_v2=np.array(monthly_v2);monthly_v3=np.array(monthly_v3);monthly_btc=np.array(monthly_btc)
delta=monthly_v3-monthly_v2

bear_mask=monthly_btc<0
bull_mask=monthly_btc>=0

print(f'  {"Period":<15} {"N months":>8} {"V2 avg":>8} {"V3 avg":>8} {"V3-V2":>8} {"V3>V2 %":>8}',flush=True)
print(f'  {"-"*58}',flush=True)

for label,mask in [('Bear (BTC<0)',bear_mask),('Bull (BTC≥0)',bull_mask),('All',np.ones(len(monthly_v2),dtype=bool))]:
    n=mask.sum()
    if n==0:continue
    v2m=monthly_v2[mask].mean()
    v3m=monthly_v3[mask].mean()
    dm=delta[mask].mean()
    winpct=np.sum(delta[mask]>0)/n*100
    print(f'  {label:<15} {n:>8} {v2m:>+7.1f}% {v3m:>+7.1f}% {dm:>+7.1f}% {winpct:>7.0f}%',flush=True)

# ============================================================
# 6. SUMMARY
# ============================================================
print('\n'+'='*70,flush=True)
print(' SUMMARY: V3 (1.5-3.0x, dd_default) の効果分解',flush=True)
print('='*70,flush=True)
print(f'  V3のV2に対する改善は以下から構成:',flush=True)
print(f'',flush=True)

total_v2=year_ret(eq_v2,2022)+year_ret(eq_v2,2023)+year_ret(eq_v2,2024)+year_ret(eq_v2,2025)
total_v3=year_ret(eq_v3,2022)+year_ret(eq_v3,2023)+year_ret(eq_v3,2024)+year_ret(eq_v3,2025)
total_fa=year_ret(eq_fixed_avg,2022)+year_ret(eq_fixed_avg,2023)+year_ret(eq_fixed_avg,2024)+year_ret(eq_fixed_avg,2025)

print(f'  2022-2025合計:',flush=True)
print(f'    V2: {total_v2:+.0f}%',flush=True)
print(f'    V3: {total_v3:+.0f}%',flush=True)
print(f'    差分: {total_v3-total_v2:+.0f}%',flush=True)
print(f'      内訳: レバ差 {total_fa-total_v2:+.0f}% + タイミング {total_v3-total_fa:+.0f}%',flush=True)

print('\nDone.',flush=True)
