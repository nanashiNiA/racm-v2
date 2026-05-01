"""A+B variants: Fix Bear 2022 weakness + explore all remaining ideas
V1: LS vol-scaling excluded (vol only affects BTC regime component)
V2: Asymmetric vol (reduce in high vol, but DON'T reduce in bear trend)
V3: LS boost in bear (increase LS weight when regime is cautious)
V4: Adaptive LW (LW increases when vol is high = LS more important)
V5: Best combination
"""
import sys, os
sys.path.insert(0, 'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0, 'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform == 'win32' else None
import numpy as np, pandas as pd
from v2core.data_loader import build_common_1h, load_6assets_8h
from src.racm_core import RACMLS, RACMKelly, RACMDDControl, RACMParams, RACMRegime

print("="*70)
print("  V2: A+B VARIANTS - Fix Bear + Explore Everything")
print("="*70)

h = build_common_1h()
nn = len(h); idx_h = h.index
ret = h['ret'].values; price = h['close'].values

assets_8h = load_6assets_8h()
params = RACMParams()
c_8h = list(assets_8h.values())[0].index
for df in assets_8h.values(): c_8h = c_8h.intersection(df.index)
a_ret_8h = {a: assets_8h[a].loc[c_8h, 'return'].values for a in assets_8h}
lp_8h, ln, sn = RACMLS.compute_pnl_8h(a_ret_8h, [60, 90], len(c_8h))
lp_1h = RACMLS.map_8h_to_1h(lp_8h, c_8h, idx_h, nn)
fra = np.roll(h['funding'].fillna(0).values, 1)
S = 2760; CAP = 3.0; KF = 0.15
bp_v1 = RACMRegime.compute(price, ret, params)

# Precompute features
adx = h['adx'].values; ma110 = h['ma110'].values; ma20 = h['ma20'].values
skew = h['skew_30d'].values; dd = h['dd'].values
vol_30d = h['vol_30d'].values; vol_ratio = h['vol_ratio'].values

folds = []; cur = pd.Timestamp('2021-01-01')
while cur + pd.DateOffset(months=4) <= idx_h[-1] + pd.DateOffset(days=15):
    te_s = cur + pd.DateOffset(months=3); te_e = te_s + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    te_m = np.array([(d >= te_s and d <= te_e) for d in idx_h])
    if te_m.sum() >= 20: folds.append(np.where(te_m)[0])
    cur += pd.DateOffset(months=1)

def compute_bp_vol(i):
    """Vol-first position for BTC regime component"""
    if i < S + 720: target_vol = 0.80
    else: target_vol = np.mean(vol_30d[max(S, i-4320):i-1])
    current_vol = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else target_vol
    vs = np.clip(target_vol / (current_vol + 1e-10), 0.3, 2.0)
    if dd[i-1] < -0.15: vs = min(vs, 0.3)
    elif dd[i-1] < -0.08: vs = min(vs, 0.5)
    if not np.isnan(ma110[i-1]) and price[i-1] < ma110[i-1]: vs *= 0.7
    return vs

def compute_danger(i):
    dl = 0; ds = 0
    if not np.isnan(ma110[i-1]) and price[i-1] < ma110[i-1]: dl += 1
    if not np.isnan(ma20[i-1]) and price[i-1] < ma20[i-1]: ds += 1
    if not np.isnan(skew[i-1]) and skew[i-1] < -0.5: dl += 1; ds += 1
    if dd[i-1] < -0.12: dl += 1; ds += 1
    return dl * 0.3 + ds * 0.7

def smooth_transition(new_bp, prev_bp, i, last_change, dc):
    if abs(new_bp - prev_bp) > 0.2:
        if dc >= 1.5: return new_bp, i  # emergency
        adx_ok = not np.isnan(adx[i-1]) and adx[i-1] > 20
        cd_ok = i - last_change >= 6
        if adx_ok and cd_ok: return new_bp, i
        return prev_bp, last_change
    return new_bp, last_change

def run_variant(build_pnl_func, label):
    """Run a model variant and return metrics"""
    eq = np.ones(nn); e = 1.0; pk_e = 1.0
    base = np.zeros(nn)
    for i in range(S, nn): base[i] = build_pnl_func(i)
    for i in range(S, nn):
        pnl = base[i]
        past = base[max(S, i-2160):i]
        lev, vt = RACMKelly.compute(past, params)
        total_lev = min(lev * vt, CAP); pnl *= total_lev
        pnl += fra[i] * abs(total_lev)
        dd_mult = RACMDDControl.compute(e, pk_e, params); pnl *= dd_mult
        e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)
    wf_r = [((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs = np.mean(wf_r)*12; win = sum(1 for r in wf_r if r > 0)
    # 1x
    eq1 = np.ones(nn); e1 = 1.0
    for i in range(S, nn):
        pnl1 = build_pnl_func(i)
        # Remove LS for 1x (just regime)
        e1 *= (1 + ret[i] * (build_pnl_func(i) / (ret[i] + 1e-20) if abs(ret[i]) > 1e-10 else 1.0))
    # Simplified 1x: use bp only
    # Per year
    yrs = {}
    for y in range(2021, 2026):
        iy = np.where(np.array([d.year == y for d in idx_h]))[0]
        if len(iy) > 100: yrs[y] = (eq[iy[-1]]/eq[max(0,iy[0]-1)]-1)*100
    eq_s = pd.Series(eq, index=idx_h)
    dr = eq_s.resample('1D').last().pct_change().dropna()*100
    dr = dr[dr.index >= idx_h[S]]
    dw = int((dr>0).sum()); dl = int((dr<0).sum())
    return wfs, win, len(wf_r), dw, dl, yrs

print("\n[1] Running variants...\n")

# Store bp arrays for smoothing
bp_arrays = {}

# V0: V1 baseline
def v0_pnl(i): return 0.20 * ret[i] * bp_v1[i] + 0.80 * lp_1h[i]

# V_AB: Original A+B (vol scales everything)
bp_ab_arr = np.ones(nn); lc = 0; pb = 1.0
for i in range(S, nn):
    vs = compute_bp_vol(i); dc = compute_danger(i)
    if dc >= 1.5: vs = min(vs, 0.2)
    vs, lc = smooth_transition(vs, pb, i, lc, dc); pb = vs
    bp_ab_arr[i] = vs
def v_ab_pnl(i): return 0.20 * ret[i] * bp_ab_arr[i] + 0.80 * lp_1h[i]

# V1: LS excluded from vol scaling (vol only on BTC component)
def v1_pnl(i): return 0.20 * ret[i] * bp_ab_arr[i] + 0.80 * lp_1h[i]  # same, LS not scaled

# V2: LS gets BOOST when regime is bear (more LS weight in bear)
def v2_pnl(i):
    bp = bp_ab_arr[i]
    lw = 0.80
    if bp < 0.5: lw = 0.95  # bear -> almost all LS (market neutral)
    elif bp < 0.8: lw = 0.90
    dw = max(0, 1 - lw)
    return dw * ret[i] * bp + lw * lp_1h[i]

# V3: Adaptive LW based on vol (high vol -> more LS, low vol -> more BTC)
def v3_pnl(i):
    v = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    if v > 1.0: lw = 0.95   # high vol -> LS dominates (market neutral)
    elif v > 0.70: lw = 0.85
    elif v > 0.50: lw = 0.80
    else: lw = 0.70          # low vol -> more BTC directional
    dw = max(0, 1 - lw)
    return dw * ret[i] * bp_ab_arr[i] + lw * lp_1h[i]

# V4: Bear LS boost + vol-adaptive LW combined
def v4_pnl(i):
    bp = bp_ab_arr[i]
    v = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
    # Base LW from vol
    if v > 1.0: lw = 0.90
    elif v > 0.50: lw = 0.80
    else: lw = 0.70
    # Bear boost
    if bp < 0.5: lw = max(lw, 0.95)
    elif bp < 0.8: lw = max(lw, 0.90)
    dw = max(0, 1 - lw)
    return dw * ret[i] * bp + lw * lp_1h[i]

# V5: Vol on BTC only + bear LS boost + no MA110 filter on LS
bp_v5 = np.ones(nn); lc5 = 0; pb5 = 1.0
for i in range(S, nn):
    # Vol scaling for BTC regime ONLY
    if i < S + 720: tv = 0.80
    else: tv = np.mean(vol_30d[max(S, i-4320):i-1])
    cv = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else tv
    vs = np.clip(tv / (cv + 1e-10), 0.3, 1.5)  # tighter upper bound
    if dd[i-1] < -0.15: vs = min(vs, 0.3)
    elif dd[i-1] < -0.08: vs = min(vs, 0.5)
    # NO MA110 filter here (let LS handle bear direction)
    dc = compute_danger(i)
    if dc >= 1.5: vs = min(vs, 0.2)
    vs, lc5 = smooth_transition(vs, pb5, i, lc5, dc); pb5 = vs
    bp_v5[i] = vs
def v5_pnl(i):
    bp = bp_v5[i]
    # Bear boost: increase LS in bear
    if bp < 0.5: lw = 0.95
    elif bp < 0.8: lw = 0.90
    else: lw = 0.80
    dw = max(0, 1 - lw)
    return dw * ret[i] * bp + lw * lp_1h[i]

variants = [
    ('V0: V1 Baseline', v0_pnl),
    ('AB: Original A+B', v_ab_pnl),
    ('V1: LS no vol scale', v1_pnl),
    ('V2: Bear LS boost', v2_pnl),
    ('V3: Vol-adaptive LW', v3_pnl),
    ('V4: V2+V3 combined', v4_pnl),
    ('V5: Full redesign', v5_pnl),
]

print(f"  {'Variant':<25} {'WFS':>6} {'Win':>6} {'DayW%':>6}", end='')
for y in range(2021, 2026): print(f" {y:>7}", end='')
print()
print(f"  {'-'*80}")

for label, pnl_func in variants:
    wfs, win, nf, dw, dl, yrs = run_variant(pnl_func, label)
    tag = ''
    if 'V0' in label: tag = ' <-- v1'
    elif wfs > 344 and yrs.get(2022, 0) > 80: tag = ' ***'
    print(f"  {label:<25} {wfs:>5.0f}% {win:>2}/{nf} {dw/(dw+dl)*100:>5.1f}%", end='')
    for y in range(2021, 2026): print(f" {yrs.get(y,0):>+6.0f}%", end='')
    print(tag)

# BTC B&H
print(f"  {'BTC B&H':<25} {'':>6} {'':>6} {'':>6}", end='')
for y in range(2021, 2026):
    iy = np.where(np.array([d.year == y for d in idx_h]))[0]
    if len(iy) > 100: print(f" {(price[iy[-1]]/price[iy[0]]-1)*100:>+6.0f}%", end='')
    else: print(f" {'N/A':>7}", end='')
print()
