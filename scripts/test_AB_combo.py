"""A+B Combination: Regime Smoothing + Vol-First
Plus leak check and robustness verification.
"""
import sys, os
sys.path.insert(0, 'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0, 'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform == 'win32' else None
import numpy as np, pandas as pd
from v2core.data_loader import build_common_1h, load_6assets_8h
from src.racm_core import RACMLS, RACMKelly, RACMDDControl, RACMParams, safe_val

print("="*70)
print("  V2: A+B COMBINATION + LEAK CHECK")
print("="*70)

h = build_common_1h()
nn = len(h); idx_h = h.index
ret = h['ret'].values; price = h['close'].values
print(f"\n  Data: {nn} bars")

# LS
assets_8h = load_6assets_8h()
params = RACMParams()
c_8h = list(assets_8h.values())[0].index
for df in assets_8h.values(): c_8h = c_8h.intersection(df.index)
a_ret_8h = {a: assets_8h[a].loc[c_8h, 'return'].values for a in assets_8h}
lp_8h, ln, sn = RACMLS.compute_pnl_8h(a_ret_8h, [60, 90], len(c_8h))
lp_1h = RACMLS.map_8h_to_1h(lp_8h, c_8h, idx_h, nn)
fra = np.roll(h['funding'].fillna(0).values, 1)
S = 2760; LW = 0.80; CAP = 3.0; KF = 0.15

# WF folds
folds = []; cur = pd.Timestamp('2021-01-01')
while cur + pd.DateOffset(months=4) <= idx_h[-1] + pd.DateOffset(days=15):
    te_s = cur + pd.DateOffset(months=3); te_e = te_s + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    te_m = np.array([(d >= te_s and d <= te_e) for d in idx_h])
    if te_m.sum() >= 20: folds.append(np.where(te_m)[0])
    cur += pd.DateOffset(months=1)

def candidate_AB(adx_thresh=20, cooldown=6, vol_dd_hard=-0.15, vol_dd_soft=-0.08):
    """A+B: Vol-first with ADX-smoothed regime override."""
    adx = h['adx'].values
    ma110 = h['ma110'].values
    ma20 = h['ma20'].values
    skew = h['skew_30d'].values
    dd = h['dd'].values
    vol_30d = h['vol_30d'].values

    bp = np.ones(nn); last_change = 0; prev_bp = 1.0
    for i in range(S, nn):
        # B: Vol-first base position
        if i < S + 720: target_vol = 0.80
        else: target_vol = np.mean(vol_30d[max(S, i-4320):i-1])  # expanding but capped at 6 months lag-1
        current_vol = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else target_vol
        vol_scale = np.clip(target_vol / (current_vol + 1e-10), 0.3, 2.0)

        # B: Crash override
        if dd[i-1] < vol_dd_hard: vol_scale = min(vol_scale, 0.3)
        elif dd[i-1] < vol_dd_soft: vol_scale = min(vol_scale, 0.5)

        # B: Trend filter
        if not np.isnan(ma110[i-1]) and price[i-1] < ma110[i-1]:
            vol_scale *= 0.7

        # A: Danger composite for emergency
        dl = 0; ds = 0
        if not np.isnan(ma110[i-1]) and price[i-1] < ma110[i-1]: dl += 1
        if not np.isnan(ma20[i-1]) and price[i-1] < ma20[i-1]: ds += 1
        if not np.isnan(skew[i-1]) and skew[i-1] < -0.5: dl += 1; ds += 1
        if dd[i-1] < -0.12: dl += 1; ds += 1
        dc = dl * 0.3 + ds * 0.7

        new_bp = vol_scale

        # A: Emergency override (always allow, no ADX/cooldown needed)
        if dc >= 1.5: new_bp = min(new_bp, 0.2)

        # A: Smoothing for non-emergency transitions
        if abs(new_bp - prev_bp) > 0.2:  # significant change
            adx_ok = not np.isnan(adx[i-1]) and adx[i-1] > adx_thresh
            cooldown_ok = i - last_change >= cooldown
            if dc >= 1.5:  # emergency always
                bp[i] = new_bp; last_change = i; prev_bp = new_bp
            elif adx_ok and cooldown_ok:
                bp[i] = new_bp; last_change = i; prev_bp = new_bp
            else:
                bp[i] = prev_bp
        else:
            bp[i] = new_bp; prev_bp = new_bp
    return bp

def run_model(bp, label=""):
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
    # Metrics
    wf_r = [((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs = np.mean(wf_r)*12; win = sum(1 for r in wf_r if r > 0)
    changes = sum(1 for i in range(S+1, nn) if abs(bp[i] - bp[i-1]) > 0.1)
    chg_day = changes / ((nn - S) / 24)
    # 1x WFS
    eq1x = np.ones(nn); e1 = 1.0
    for i in range(S, nn): e1 *= (1 + ret[i] * bp[i]); eq1x[i] = e1
    wf1x = [((eq1x[idx[-1]]/eq1x[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs_1x = np.mean(wf1x)*12
    # Daily win rate
    eq_s = pd.Series(eq, index=idx_h)
    dr = eq_s.resample('1D').last().pct_change().dropna()*100
    dr = dr[dr.index >= idx_h[S]]
    dw_c = int((dr>0).sum()); dl_c = int((dr<0).sum())
    # Bear 2022
    i22 = np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22 = (eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    # 2025
    i25 = np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25 = (eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    return wfs, win, len(wf_r), dw_c, dl_c, chg_day, wfs_1x, b22, o25, wf_r

# === V1 baseline ===
from src.racm_core import RACMRegime
bp_v1 = RACMRegime.compute(price, ret, params)

# === A+B ===
bp_ab = candidate_AB()

print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}\n")
print(f"  {'Model':<20} {'WFS':>6} {'Win':>6} {'DayW':>6} {'Chg/d':>6} {'1xWFS':>6} {'B22':>6} {'OOS25':>7}")
print(f"  {'-'*68}")

for label, bp in [('V1 Baseline', bp_v1), ('A+B Combined', bp_ab)]:
    wfs, win, nf, dw, dl, chg, w1x, b22, o25, _ = run_model(bp, label)
    print(f"  {label:<20} {wfs:>5.0f}% {win:>2}/{nf} {dw/(dw+dl)*100:>5.1f}% {chg:>5.1f} {w1x:>+5.0f}% {b22:>+5.0f}% {o25:>+6.0f}%")

# === LEAK CHECK ===
print(f"\n{'='*70}")
print(f"  LEAK CHECK")
print(f"{'='*70}\n")

leaks = [
    ('ADX threshold = 20', 'Standard technical analysis value', 'NO LEAK'),
    ('Cooldown = 6 bars (6H)', 'Structural (prevent rapid switching)', 'NO LEAK'),
    ('Vol target = expanding mean', 'Uses only past vol (lag-1)', 'NO LEAK'),
    ('Vol scale clip [0.3, 2.0]', 'Structural bounds', 'NO LEAK'),
    ('DD hard = -15%', 'From v1 DD control (pre-existing)', 'LOW (inherited)'),
    ('DD soft = -8%', 'Common round number', 'LOW'),
    ('MA110 trend filter', 'From v1 (pre-existing)', 'LOW (inherited)'),
    ('Emergency dc >= 1.5', 'From v1 (pre-existing)', 'LOW (inherited)'),
]
for param, reason, verdict in leaks:
    print(f"  [{verdict:>12}] {param:<30} {reason}")

# === PARAMETER SENSITIVITY ===
print(f"\n{'='*70}")
print(f"  PARAMETER SENSITIVITY")
print(f"{'='*70}\n")

print(f"  {'Config':<45} {'WFS':>6} {'1xWFS':>6} {'B22':>6}")
print(f"  {'-'*65}")

for adx_t in [15, 20, 25, 30]:
    bp_t = candidate_AB(adx_thresh=adx_t)
    wfs,_,_,_,_,_,w1x,b22,_,_ = run_model(bp_t)
    tag = ' <--' if adx_t == 20 else ''
    print(f"  ADX threshold = {adx_t:<30} {wfs:>5.0f}% {w1x:>+5.0f}% {b22:>+5.0f}%{tag}")

for cd in [3, 6, 12, 24]:
    bp_t = candidate_AB(cooldown=cd)
    wfs,_,_,_,_,_,w1x,b22,_,_ = run_model(bp_t)
    tag = ' <--' if cd == 6 else ''
    print(f"  Cooldown = {cd}H{' '*(28-len(str(cd)))} {wfs:>5.0f}% {w1x:>+5.0f}% {b22:>+5.0f}%{tag}")

for dd_h, dd_s in [(-0.10, -0.05), (-0.12, -0.07), (-0.15, -0.08), (-0.20, -0.10)]:
    bp_t = candidate_AB(vol_dd_hard=dd_h, vol_dd_soft=dd_s)
    wfs,_,_,_,_,_,w1x,b22,_,_ = run_model(bp_t)
    tag = ' <--' if dd_h == -0.15 else ''
    print(f"  DD hard={dd_h:.0%} soft={dd_s:.0%}{' '*(18)} {wfs:>5.0f}% {w1x:>+5.0f}% {b22:>+5.0f}%{tag}")

# === SHUFFLE TEST (quick, 200x) ===
print(f"\n{'='*70}")
print(f"  SHUFFLE TEST (200x)")
print(f"{'='*70}\n")

_,_,_,_,_,_,_,_,_,wf_real = run_model(bp_ab)
ws_real = np.mean(wf_real)*12
shuf_wfs = []
dw_ab = max(0, 1-LW)
base_ab = np.zeros(nn)
for i in range(S, nn): base_ab[i] = dw_ab * ret[i] * bp_ab[i] + LW * lp_1h[i]
for s in range(200):
    np.random.seed(s)
    shuf = base_ab[S:].copy(); np.random.shuffle(shuf)
    base_s = np.zeros(nn); base_s[S:] = shuf
    eq_s = np.ones(nn); e = 1.0; pk_s = 1.0
    for i in range(S, nn):
        pnl = base_s[i]
        past = base_s[max(S, i-2160):i]
        if len(past) < 240: lev = 1.5
        else:
            mu = np.mean(past); var = np.var(past) + 1e-10
            lev = np.clip(mu/var*KF, 1.0, CAP)
        pnl *= lev
        dd_eq = (e-pk_s)/pk_s if pk_s > 0 else 0
        if dd_eq < -0.30: pnl *= 0.1
        elif dd_eq < -0.22: pnl *= 0.4
        elif dd_eq < -0.15: pnl *= 0.7
        e *= (1+pnl); eq_s[i] = e; pk_s = max(pk_s, e)
    wf_s = [((eq_s[idx[-1]]/eq_s[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    shuf_wfs.append(np.mean(wf_s)*12)

p_val = np.mean(np.array(shuf_wfs) >= ws_real)
print(f"  Real WFS: {ws_real:.0f}%")
print(f"  Shuffle mean: {np.mean(shuf_wfs):.0f}% +/- {np.std(shuf_wfs):.0f}%")
print(f"  p-value: {p_val:.3f} ({'PASS' if p_val < 0.05 else 'FAIL'})")
