"""Test all 4 v2 candidates on same data, same WF, compare fairly.
A: Regime smoothing (ADX + cooldown + hysteresis)
B: Volatility-first (vol forecast -> position sizing)
C: Multi-timeframe fusion (daily + 8H + 1H independent votes)
D: Ensemble (A+B+C combined)
+ RACM v1.0 as baseline
"""
import sys, os
sys.path.insert(0, 'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0, 'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
import numpy as np, pandas as pd
from v2core.data_loader import build_common_1h, load_6assets_8h
from src.racm_core import RACMLS, RACMKelly, RACMDDControl, RACMParams, safe_val

print("="*70)
print("  RACM V2: ALL CANDIDATES COMPARISON")
print("="*70)

# Load data
print("\n[1] Loading data...")
h = build_common_1h()
nn = len(h); idx_h = h.index
ret = h['ret'].values; price = h['close'].values
print(f"  BTC 1H: {nn} bars ({idx_h[0]} to {idx_h[-1]})")

# Load 8H assets for LS
print("  Loading 6 assets 8H...")
assets_8h = load_6assets_8h()
params = RACMParams()
c_8h = list(assets_8h.values())[0].index
for df in assets_8h.values(): c_8h = c_8h.intersection(df.index)
a_ret_8h = {a: assets_8h[a].loc[c_8h, 'return'].values for a in assets_8h}
lp_8h, ln, sn = RACMLS.compute_pnl_8h(a_ret_8h, [60, 90], len(c_8h))
lp_1h = RACMLS.map_8h_to_1h(lp_8h, c_8h, idx_h, nn)
print(f"  LS ready: {len(c_8h)} 8H bars")

S = 2760  # warmup
fra = np.roll(h['funding'].fillna(0).values, 1)

# WF folds
folds = []; cur = pd.Timestamp('2021-01-01')
while cur + pd.DateOffset(months=4) <= idx_h[-1] + pd.DateOffset(days=15):
    te_s = cur + pd.DateOffset(months=3); te_e = te_s + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    te_m = np.array([(d >= te_s and d <= te_e) for d in idx_h])
    if te_m.sum() >= 20: folds.append(np.where(te_m)[0])
    cur += pd.DateOffset(months=1)

def eval_model(eq):
    wf_r = [((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs = np.mean(wf_r)*12 if wf_r else 0
    win = sum(1 for r in wf_r if r > 0)
    # Daily win rate
    eq_s = pd.Series(eq, index=idx_h)
    dr = eq_s.resample('1D').last().pct_change().dropna()*100
    dr = dr[dr.index >= idx_h[S]]
    dw = int((dr>0).sum()); dl = int((dr<0).sum())
    return wfs, win, len(wf_r), dw, dl

# ============================================================
# CANDIDATE A: Regime Smoothing (ADX + cooldown + hysteresis)
# ============================================================
def candidate_A():
    adx = h['adx'].values
    ma110 = h['ma110'].values
    ma20 = h['ma20'].values
    skew = h['skew_30d'].values
    dd = h['dd'].values
    vol_r = h['vol_ratio'].values

    bp = np.ones(nn); last_change = 0; prev_bp = 1.0
    for i in range(S, nn):
        dl = 0; ds = 0
        if not np.isnan(ma110[i-1]) and price[i-1] < ma110[i-1]: dl += 1
        if not np.isnan(ma20[i-1]) and price[i-1] < ma20[i-1]: ds += 1
        if not np.isnan(skew[i-1]) and skew[i-1] < -0.5: dl += 1; ds += 1
        if dd[i-1] < -0.12: dl += 1; ds += 1
        dc = dl * 0.3 + ds * 0.7

        new_bp = 1.0
        if dc >= 1.5: new_bp = 0.2
        elif dc >= 0.8: new_bp = 0.5
        elif dc >= 0.5: new_bp = 0.7
        else:
            if not np.isnan(vol_r[i-1]) and vol_r[i-1] < 0.7 and i >= 240 and np.sum(ret[i-240:i-1]) > 0:
                new_bp = 1.3
            else: new_bp = 1.0

        # SMOOTHING: require ADX > 20 AND 6H cooldown for regime change
        adx_ok = not np.isnan(adx[i-1]) and adx[i-1] > 20
        cooldown_ok = i - last_change >= 6

        if new_bp != prev_bp:
            if dc >= 1.5:  # emergency: always allow risk-off
                bp[i] = new_bp; last_change = i; prev_bp = new_bp
            elif adx_ok and cooldown_ok:
                bp[i] = new_bp; last_change = i; prev_bp = new_bp
            else:
                bp[i] = prev_bp  # hold previous
        else:
            bp[i] = new_bp
    return bp

# ============================================================
# CANDIDATE B: Volatility-First (position = inverse of forecast vol)
# ============================================================
def candidate_B():
    vol_30d = h['vol_30d'].values
    vol_7d = h['vol_7d'].values
    vol_ratio = h['vol_ratio'].values
    ma110 = h['ma110'].values
    dd = h['dd'].values

    bp = np.ones(nn)
    for i in range(S, nn):
        # Vol forecast: use expanding mean of vol_30d as "normal" vol
        if i < S + 720: target_vol = 0.80  # default 80% annual
        else:
            target_vol = np.mean(vol_30d[S:i-1])

        current_vol = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else target_vol
        # Position = target / current (inverse vol scaling)
        vol_scale = np.clip(target_vol / (current_vol + 1e-10), 0.3, 2.0)

        # Crash override: if DD deep or vol spiking, reduce hard
        if dd[i-1] < -0.15: vol_scale = min(vol_scale, 0.3)
        elif dd[i-1] < -0.08: vol_scale = min(vol_scale, 0.5)

        # Trend filter: below MA110 = reduce
        if not np.isnan(ma110[i-1]) and price[i-1] < ma110[i-1]:
            vol_scale *= 0.7

        bp[i] = vol_scale
    return bp

# ============================================================
# CANDIDATE C: Multi-Timeframe Fusion
# ============================================================
def candidate_C():
    ma110 = h['ma110'].values  # daily trend
    ma20 = h['ma20'].values    # weekly trend
    rsi = h['rsi_14d'].values
    donchian_up = h['donchian_breakout_up'].values
    donchian_dn = h['donchian_breakout_dn'].values
    dd = h['dd'].values
    adx = h['adx'].values

    bp = np.ones(nn)
    for i in range(S, nn):
        votes = 0  # -3 to +3

        # Vote 1: Daily trend (MA110)
        if not np.isnan(ma110[i-1]):
            if price[i-1] > ma110[i-1] * 1.02: votes += 1
            elif price[i-1] < ma110[i-1] * 0.98: votes -= 1

        # Vote 2: Weekly momentum (MA20 slope)
        if i >= 48 and not np.isnan(ma20[i-1]) and not np.isnan(ma20[i-49]):
            slope = (ma20[i-1] - ma20[i-49]) / (ma20[i-49] + 1e-10)
            if slope > 0.01: votes += 1
            elif slope < -0.01: votes -= 1

        # Vote 3: Donchian breakout (1H timing)
        if donchian_up[i-1] > 0.5: votes += 1
        elif donchian_dn[i-1] > 0.5: votes -= 1

        # Map votes to position
        if votes >= 2: bp[i] = 1.3
        elif votes == 1: bp[i] = 1.0
        elif votes == 0: bp[i] = 0.7
        elif votes == -1: bp[i] = 0.5
        else: bp[i] = 0.2

        # Override: deep drawdown = reduce
        if dd[i-1] < -0.15: bp[i] = min(bp[i], 0.3)
    return bp

# ============================================================
# CANDIDATE D: Ensemble (average of A, B, C)
# ============================================================
def candidate_D(bp_a, bp_b, bp_c):
    bp = np.ones(nn)
    for i in range(S, nn):
        bp[i] = (bp_a[i] + bp_b[i] + bp_c[i]) / 3.0
    return bp

# ============================================================
# V1 BASELINE
# ============================================================
def v1_baseline():
    from src.racm_core import RACMRegime
    return RACMRegime.compute(price, ret, params)

# Run all candidates
print("\n[2] Running candidates...")
bp_v1 = v1_baseline()
bp_A = candidate_A()
bp_B = candidate_B()
bp_C = candidate_C()
bp_D = candidate_D(bp_A, bp_B, bp_C)

LW = 0.80; CAP = 3.0; KF = 0.15

def run_with_bp(bp, label):
    dw = max(0, 1 - LW)
    base = np.zeros(nn)
    for i in range(S, nn): base[i] = dw * ret[i] * bp[i] + LW * lp_1h[i]
    eq = np.ones(nn); e = 1.0; pk_e = 1.0
    for i in range(S, nn):
        pnl = base[i]
        past = base[max(S, i-2160):i]
        lev, vt = RACMKelly.compute(past, params)
        total_lev = min(lev * vt, CAP); pnl *= total_lev
        pnl += fra[i] * abs(total_lev)
        dd_mult = RACMDDControl.compute(e, pk_e, params); pnl *= dd_mult
        e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)
    wfs, win, nf, dw_c, dl_c = eval_model(eq)

    # Count regime changes
    changes = sum(1 for i in range(S+1, nn) if bp[i] != bp[i-1])
    changes_day = changes / ((nn - S) / 24)

    # Year by year
    yr_results = {}
    for y in range(2021, 2026):
        mask = np.array([d.year == y for d in idx_h]); iy = np.where(mask)[0]
        if len(iy) > 100:
            yr_results[y] = (eq[iy[-1]] / eq[max(0, iy[0]-1)] - 1) * 100

    return wfs, win, nf, dw_c, dl_c, changes_day, yr_results, eq

# Also test at 1x (no leverage) to check base signal
def run_1x(bp, label):
    base = np.zeros(nn)
    for i in range(S, nn): base[i] = ret[i] * bp[i]  # BTC regime only, 1x
    eq = np.ones(nn); e = 1.0
    for i in range(S, nn): e *= (1 + base[i]); eq[i] = e
    wf_r = [((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs = np.mean(wf_r)*12 if wf_r else 0
    return wfs

# Run all
print("\n[3] Results\n")
print(f"  {'Candidate':<25} {'WFS':>6} {'Win':>6} {'DayW%':>6} {'Chg/d':>6} {'1x WFS':>7}")
print(f"  {'-'*58}")

for label, bp in [('V1 Baseline', bp_v1), ('A: Regime Smoothing', bp_A),
                   ('B: Vol-First', bp_B), ('C: Multi-TF Fusion', bp_C),
                   ('D: Ensemble (A+B+C)', bp_D)]:
    wfs, win, nf, dw, dl, chg, yrs, eq = run_with_bp(bp, label)
    wfs_1x = run_1x(bp, label)
    tag = ' <-- v1' if 'V1' in label else ''
    star = ' ***' if wfs > 379 and wfs_1x > 32 else ''
    print(f"  {label:<25} {wfs:>5.0f}% {win:>2}/{nf} {dw/(dw+dl)*100:>5.1f}% {chg:>5.1f} {wfs_1x:>+6.0f}%{tag}{star}")

# Year by year for each
print(f"\n  Year-by-year:")
print(f"  {'Candidate':<25}", end='')
for y in range(2021, 2026): print(f" {y:>7}", end='')
print()
print(f"  {'-'*60}")
for label, bp in [('V1 Baseline', bp_v1), ('A: Regime Smooth', bp_A),
                   ('B: Vol-First', bp_B), ('C: Multi-TF', bp_C),
                   ('D: Ensemble', bp_D)]:
    _, _, _, _, _, _, yrs, _ = run_with_bp(bp, label)
    print(f"  {label:<25}", end='')
    for y in range(2021, 2026):
        print(f" {yrs.get(y, 0):>+6.0f}%", end='')
    print()

# BTC B&H for reference
print(f"  {'BTC B&H':<25}", end='')
for y in range(2021, 2026):
    mask = np.array([d.year == y for d in idx_h]); iy = np.where(mask)[0]
    if len(iy) > 100:
        yr = (price[iy[-1]] / price[iy[0]] - 1) * 100
        print(f" {yr:>+6.0f}%", end='')
    else: print(f" {'N/A':>7}", end='')
print()
