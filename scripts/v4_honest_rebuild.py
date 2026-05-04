"""V4: 根本的見直し — Fundingバグ修正後のゼロベース再構築
============================================================
発覚した問題:
  pnl += fra[i] * abs(lev) → ロングがfundingを「稼ぐ」計算
  実際: ロングは87%の時間でfundingを「払う」
  影響: WFS 330%分が架空利益

ゼロベースで確認すること:
1. LS Momentum 自体にalpha はあるか？(funding抜き)
2. Funding コストはどれくらいか？
3. Lighter.xyz のfunding構造は？
4. Funding コストを踏まえて成立する戦略は？
5. Spot (現物) なら funding 問題なし → spot戦略は？
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from scipy import stats
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS,RACMParams,RACMRegime,RACMMicrostructure

print('='*70,flush=True)
print('  V4: HONEST REBUILD — Fundingバグ修正後',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
params=RACMParams();S=params.warmup_hours
fra_raw=h['funding'].fillna(0).values
fra=np.roll(fra_raw,1)  # lag-1
vol_30d=h['vol_30d'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)
bp_v2=RACMRegime.compute(price,ret,params)

dd_default=[(-0.15,0.7),(-0.22,0.4),(-0.30,0.1)]
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

print('Data ready.\n',flush=True)

# ============================================================
# STEP 1: Funding Rate の実態
# ============================================================
print('='*70,flush=True)
print(' STEP 1: BTC Perpetual Funding Rate の実態',flush=True)
print('='*70,flush=True)

fra_pos=fra[S:][fra[S:]>0]
fra_neg=fra[S:][fra[S:]<0]
print(f'  Positive funding (ロング払い): {len(fra_pos)}/{nn-S} ({len(fra_pos)/(nn-S)*100:.1f}%)',flush=True)
print(f'  Negative funding (ロング受取): {len(fra_neg)}/{nn-S} ({len(fra_neg)/(nn-S)*100:.1f}%)',flush=True)
print(f'  平均 funding rate: {np.mean(fra[S:])*100:.5f}% per bar',flush=True)
print(f'  年間 funding cost (1x long): {np.mean(fra[S:])*365*24*100:.1f}%',flush=True)
print(f'  年間 funding cost (3x long): {np.mean(fra[S:])*365*24*3*100:.1f}%',flush=True)

# Year by year
print(f'\n  Year-by-year funding cost (1x long):',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
    if len(iy)<100:continue
    avg_f=np.mean(fra[iy])
    annual=avg_f*365*24*100
    print(f'    {yr}: {annual:+.1f}%/year (avg rate: {avg_f*100:.5f}%/bar)',flush=True)

# ============================================================
# STEP 2: LS Momentum Alpha (funding抜き)
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 2: LS Momentum — Funding抜きの真のAlpha',flush=True)
print('='*70,flush=True)

def run_strategy(lw_func, lev, use_funding_cost=False, use_dd=True):
    """Generic strategy runner"""
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp_i=bp_v2[i]
        lw,dw=lw_func(bp_i,v)
        pnl_base=(dw*ret[i]*bp_i+lw*lp1h[i])*lev
        # Correct funding: long pays, short earns
        if use_funding_cost:
            if bp_i>=0:  # net long
                pnl_base-=fra[i]*abs(lev)  # pay funding when long
            else:  # net short
                pnl_base+=fra[i]*abs(lev)  # earn funding when short
        if use_dd:
            dd=(e-pk)/pk if pk>0 else 0
            for level,mult in sorted(dd_default):
                if dd<level:pnl_base*=mult;break
        e*=(1+pnl_base);eq[i]=e;pk=max(pk,e)
    return eq

def lw_v2(bp,v):
    if bp<0.5:lw=0.95
    elif bp<0.8:lw=0.90
    elif v>1.0:lw=0.90
    elif v>0.50:lw=0.80
    else:lw=0.70
    return lw,max(0,1-lw)

def eval_eq(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else 0
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0
    mdd=0;pk_=1
    for i in range(S,nn):pk_=max(pk_,eq[i]);dd_=(eq[i]-pk_)/pk_;mdd=min(mdd,dd_)
    fold_rets=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    if len(fold_rets)>10:
        slope,_,_,p_d,_=stats.linregress(range(len(fold_rets)),fold_rets)
    else:slope=0;p_d=1
    return wfs,win,len(wf),b22,o25,mdd*100,slope,p_d

hdr='  {:<35} {:>5} {:>5} {:>6} {:>6} {:>6} {:>7} {:>5}'
print(hdr.format('Model','WFS','Win','B22','OOS','MDD','Decay','p'),flush=True)
print('  '+'-'*78,flush=True)

# No funding, no DD
eq=run_strategy(lw_v2, 1.0, False, False)
wfs,win,nf,b22,o25,mdd,slope,p_d=eval_eq(eq)
print(f'  {"RACM 1x, no fund, no DD":<35} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {slope:>+6.2f} {p_d:>4.3f}',flush=True)

for label,lev,fund,dd in [
    ('RACM 1x, no fund, +DD', 1.0, False, True),
    ('RACM 2x, no fund, +DD', 2.0, False, True),
    ('RACM 3x, no fund, +DD', 3.0, False, True),
    ('RACM 1x, +fund cost, +DD', 1.0, True, True),
    ('RACM 2x, +fund cost, +DD', 2.0, True, True),
    ('RACM 3x, +fund cost, +DD', 3.0, True, True),
    ('BTC B&H 1x', None, False, False),
]:
    if label.startswith('BTC'):
        eq_bh=np.ones(nn);e=1.0
        for i in range(S,nn):e*=(1+ret[i]);eq_bh[i]=e
        wfs,win,nf,b22,o25,mdd,slope,p_d=eval_eq(eq_bh)
    else:
        eq=run_strategy(lw_v2, lev, fund, dd)
        wfs,win,nf,b22,o25,mdd,slope,p_d=eval_eq(eq)
    print(f'  {label:<35} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {slope:>+6.2f} {p_d:>4.3f}',flush=True)

# ============================================================
# STEP 3: LS Momentum単体のシャッフルテスト (funding抜き)
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 3: LS Momentum Shuffle Test (funding抜き)',flush=True)
print('='*70,flush=True)

eq_real=run_strategy(lw_v2, 1.0, False, True)
wfs_real,_,_,_,_,_,_,_=eval_eq(eq_real)
wfs_sh=[]
for seed in range(500):
    np.random.seed(seed+900000)
    lp1h_s=lp1h.copy()
    np.random.shuffle(lp1h_s[S:])
    # Temporarily replace
    lp1h_orig=lp1h.copy()
    lp1h[:]=lp1h_s
    eq_s=run_strategy(lw_v2, 1.0, False, True)
    lp1h[:]=lp1h_orig
    w,_,_,_,_,_,_,_=eval_eq(eq_s)
    wfs_sh.append(w)

p=np.mean([w>=wfs_real for w in wfs_sh])
print(f'  RACM 1x no-fund: WFS={wfs_real:.0f}%, Shuffle={np.mean(wfs_sh):.0f}%+/-{np.std(wfs_sh):.0f}%, p={p:.3f} {"PASS" if p<0.05 else "FAIL"}',flush=True)

# ============================================================
# STEP 4: Funding-aware戦略の可能性
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 4: Funding Cost を踏まえた戦略設計',flush=True)
print('='*70,flush=True)

# Option A: Spot only (no leverage, no funding)
print('  A) Spot Only (1x, no funding):',flush=True)
eq_spot=run_strategy(lw_v2, 1.0, False, True)
wfs_s,win_s,_,b22_s,o25_s,mdd_s,_,_=eval_eq(eq_spot)
print(f'     WFS={wfs_s:.0f}%, B22={b22_s:+.0f}%, MDD={mdd_s:+.1f}%',flush=True)

# Option B: Low leverage (reduce funding cost impact)
print('  B) Low Leverage (1.5x, with funding):',flush=True)
eq_low=run_strategy(lw_v2, 1.5, True, True)
wfs_l,win_l,_,b22_l,o25_l,mdd_l,_,_=eval_eq(eq_low)
print(f'     WFS={wfs_l:.0f}%, B22={b22_l:+.0f}%, MDD={mdd_l:+.1f}%',flush=True)

# Option C: Carry-aware (short when funding high, long when low)
print('  C) Carry-Aware Position:',flush=True)
funding_ma=pd.Series(fra_raw).rolling(168,min_periods=24).mean().values
eq_carry=np.ones(nn);e=1.0;pk=1.0
for i in range(S,nn):
    v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    bp_i=bp_v2[i]
    lw,dw=lw_v2(bp_i,v)
    pnl_base=(dw*ret[i]*bp_i+lw*lp1h[i])

    # Adjust position by funding direction
    fv=funding_ma[i-1] if not np.isnan(funding_ma[i-1]) else 0
    if fv>0.0003:  # high positive funding → reduce long (pay less)
        pos_mult=0.5
    elif fv>0.0001:
        pos_mult=0.8
    elif fv<-0.0001:  # negative funding → increase long (earn carry)
        pos_mult=1.3
    else:
        pos_mult=1.0

    lev=2.0*pos_mult
    pnl=pnl_base*lev
    # Correct funding
    if bp_i>=0:
        pnl-=fra[i]*abs(lev)
    else:
        pnl+=fra[i]*abs(lev)
    dd=(e-pk)/pk if pk>0 else 0
    for level,mult in sorted(dd_default):
        if dd<level:pnl*=mult;break
    e*=(1+pnl);eq_carry[i]=e;pk=max(pk,e)
wfs_c,win_c,_,b22_c,o25_c,mdd_c,_,_=eval_eq(eq_carry)
print(f'     WFS={wfs_c:.0f}%, B22={b22_c:+.0f}%, MDD={mdd_c:+.1f}%',flush=True)

# Option D: LS only (no BTC directional, pure LS spread)
print('  D) Pure LS (no directional BTC):',flush=True)
eq_purls=np.ones(nn);e=1.0;pk=1.0
for i in range(S,nn):
    pnl=lp1h[i]*3.0  # pure LS, no BTC direction
    # LS is long one asset + short another → net neutral → minimal funding
    dd=(e-pk)/pk if pk>0 else 0
    for level,mult in sorted(dd_default):
        if dd<level:pnl*=mult;break
    e*=(1+pnl);eq_purls[i]=e;pk=max(pk,e)
wfs_p,win_p,_,b22_p,o25_p,mdd_p,_,_=eval_eq(eq_purls)
print(f'     WFS={wfs_p:.0f}%, B22={b22_p:+.0f}%, MDD={mdd_p:+.1f}%',flush=True)

# Option E: Spot + regime (no leverage at all, just timing)
print('  E) Spot + Regime (BTC spot, timing only):',flush=True)
eq_spotregime=np.ones(nn);e=1.0;pk=1.0
for i in range(S,nn):
    bp_i=bp_v2[i]
    if bp_i<0:
        pos=0  # fully out
    elif bp_i<0.5:
        pos=0.2
    elif bp_i<0.8:
        pos=0.5
    else:
        pos=1.0
    pnl=ret[i]*pos
    e*=(1+pnl);eq_spotregime[i]=e;pk=max(pk,e)
wfs_sr,win_sr,_,b22_sr,o25_sr,mdd_sr,_,_=eval_eq(eq_spotregime)
print(f'     WFS={wfs_sr:.0f}%, B22={b22_sr:+.0f}%, MDD={mdd_sr:+.1f}%',flush=True)

# ============================================================
# STEP 5: Summary
# ============================================================
print('\n'+'='*70,flush=True)
print(' SUMMARY: Funding修正後の honest な戦略比較',flush=True)
print('='*70,flush=True)

print(f'  {"Strategy":<35} {"WFS":>5} {"Win":>5} {"B22":>6} {"MDD":>6}',flush=True)
print(f'  {"-"*60}',flush=True)
print(f'  {"BTC B&H":<35}',end='',flush=True)
eq_bh=np.ones(nn);e=1.0
for i in range(S,nn):e*=(1+ret[i]);eq_bh[i]=e
w,wi,_,b,o,m,_,_=eval_eq(eq_bh)
print(f' {w:>4.0f}% {wi:>2}/{nf} {b:>+5.0f}% {m:>+5.1f}%',flush=True)

for label,eq in [
    ('A) Spot RACM (1x, no fund)', eq_spot),
    ('B) Low Lev (1.5x, +fund)', eq_low),
    ('C) Carry-Aware (2x, adj)', eq_carry),
    ('D) Pure LS (3x, neutral)', eq_purls),
    ('E) Spot + Regime (timing)', eq_spotregime),
]:
    w,wi,_,b,o,m,_,_=eval_eq(eq)
    tag=''
    if w>40 and b>0 and m>=-35:tag=' ✓'
    print(f'  {label:<35} {w:>4.0f}% {wi:>2}/{nf} {b:>+5.0f}% {m:>+5.1f}%{tag}',flush=True)

print(f'\n  Lighter.xyz確認事項:',flush=True)
print(f'    - Lighterにfunding rateはあるか？',flush=True)
print(f'    - あるならBinanceと同じ構造か？',flush=True)
print(f'    - Spot取引は可能か？（perpetualのみ？）',flush=True)

print('\nDone.',flush=True)
