"""V2 Full Verification:
1. Shuffle test (200x3)
2. Parameter sensitivity
3. Alpha decay comparison (V1 vs V2)
4. NEW: Bear tradeoff fix (vol-scale only when correlation high)
"""
import sys, os
sys.path.insert(0, 'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0, 'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform == 'win32' else None
import numpy as np, pandas as pd
from v2core.data_loader import build_common_1h, load_6assets_8h
from src.racm_core import RACMLS, RACMKelly, RACMDDControl, RACMParams, RACMRegime, safe_val
from scipy import stats as st

print("="*70)
print("  V2 FULL VERIFICATION + BEAR FIX")
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

adx = h['adx'].values; ma110 = h['ma110'].values; ma20 = h['ma20'].values
skew_arr = h['skew_30d'].values; dd_arr = h['dd'].values
vol_30d = h['vol_30d'].values

folds = []; cur = pd.Timestamp('2021-01-01')
while cur + pd.DateOffset(months=4) <= idx_h[-1] + pd.DateOffset(days=15):
    te_s = cur + pd.DateOffset(months=3); te_e = te_s + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    te_m = np.array([(d >= te_s and d <= te_e) for d in idx_h])
    if te_m.sum() >= 20: folds.append(np.where(te_m)[0])
    cur += pd.DateOffset(months=1)

# Cross-asset correlation (for Bear fix)
# BTC-ETH rolling correlation at 1H
eth_8h = assets_8h.get('ETH')
if eth_8h is not None:
    eth_1h_ret = np.zeros(nn)
    for j in range(len(c_8h)):
        ts = c_8h[j]; te = c_8h[j+1] if j+1 < len(c_8h) else ts + pd.Timedelta(hours=8)
        idxs = np.where((idx_h >= ts) & (idx_h < te))[0]
        if len(idxs) > 0:
            for ii in idxs: eth_1h_ret[ii] = a_ret_8h['ETH'][j] / len(idxs)
    corr_btc_eth = np.roll(pd.Series(ret).rolling(720, min_periods=240).corr(pd.Series(eth_1h_ret)).values, 1)
else:
    corr_btc_eth = np.full(nn, 0.5)

def build_v2_bp(vol_dd_hard=-0.15, vol_dd_soft=-0.08, adx_thresh=20, cooldown=6):
    bp = np.ones(nn); lc = 0; pb = 1.0
    for i in range(S, nn):
        if i < S+720: tv = 0.80
        else: tv = np.mean(vol_30d[max(S, i-4320):i-1])
        cv = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else tv
        vs = np.clip(tv / (cv + 1e-10), 0.3, 1.5)
        if dd_arr[i-1] < vol_dd_hard: vs = min(vs, 0.3)
        elif dd_arr[i-1] < vol_dd_soft: vs = min(vs, 0.5)
        if not np.isnan(ma110[i-1]) and price[i-1] < ma110[i-1]: vs *= 0.7
        dl=0;ds=0
        if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:dl+=1
        if not np.isnan(ma20[i-1]) and price[i-1]<ma20[i-1]:ds+=1
        if not np.isnan(skew_arr[i-1]) and skew_arr[i-1]<-0.5:dl+=1;ds+=1
        if dd_arr[i-1]<-0.12:dl+=1;ds+=1
        dc=dl*0.3+ds*0.7
        if dc>=1.5:vs=min(vs,0.2)
        if abs(vs-pb)>0.2:
            if dc>=1.5:bp[i]=vs;lc=i;pb=vs
            elif (not np.isnan(adx[i-1]) and adx[i-1]>adx_thresh) and (i-lc>=cooldown):
                bp[i]=vs;lc=i;pb=vs
            else:bp[i]=pb
        else:bp[i]=vs;pb=vs
    return bp

def v2_lw_standard(bp, vol):
    if bp < 0.5: return 0.95
    elif bp < 0.8: return 0.90
    elif vol > 1.0: return 0.90
    elif vol > 0.50: return 0.80
    else: return 0.70

def run_v2(bp, lw_func=v2_lw_standard):
    eq = np.ones(nn); e = 1.0; pk_e = 1.0
    base_arr = np.zeros(nn)
    for i in range(S, nn):
        v = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        lw = lw_func(bp[i], v)
        dw = max(0, 1-lw)
        base_arr[i] = dw*ret[i]*bp[i] + lw*lp_1h[i]
    for i in range(S, nn):
        pnl = base_arr[i]
        past = base_arr[max(S, i-2160):i]
        lev, vt = RACMKelly.compute(past, params)
        total_lev = min(lev*vt, CAP); pnl *= total_lev
        pnl += fra[i]*abs(total_lev)
        dd_mult = RACMDDControl.compute(e, pk_e, params); pnl *= dd_mult
        e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)
    return eq, base_arr

bp_v2 = build_v2_bp()
eq_v2, base_v2 = run_v2(bp_v2)

# === [1] SHUFFLE TEST ===
print("\n[1] SHUFFLE TEST (200x3)")
wf_real = [((eq_v2[idx[-1]]/eq_v2[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
ws_real = np.mean(wf_real)*12

for shuf_type in ['return', 'block30d', 'position']:
    shuf_wfs = []
    for s in range(200):
        np.random.seed(s + {'return':0,'block30d':10000,'position':20000}[shuf_type])
        if shuf_type == 'return':
            shuf = base_v2[S:].copy(); np.random.shuffle(shuf)
            base_s = np.zeros(nn); base_s[S:] = shuf
        elif shuf_type == 'block30d':
            active = base_v2[S:]
            bs = 720; nb = len(active)//bs
            blocks = [active[b*bs:(b+1)*bs] for b in range(nb)]
            if len(active)%bs > 0: blocks.append(active[nb*bs:])
            np.random.shuffle(blocks)
            base_s = np.zeros(nn); base_s[S:S+sum(len(b) for b in blocks)] = np.concatenate(blocks)[:len(active)]
        else:  # position
            bp_shuf = bp_v2.copy(); a = bp_shuf[S:]; np.random.shuffle(a); bp_shuf[S:] = a
            _, base_s = run_v2(bp_shuf)
        eq_s = np.ones(nn); e = 1.0; pk_s = 1.0
        for i in range(S, nn):
            pnl = base_s[i]
            past = base_s[max(S, i-2160):i]
            if len(past) < 240: lev = 1.5
            else:
                mu = np.mean(past); var = np.var(past)+1e-10
                lev = np.clip(mu/var*KF, 1.0, CAP)
            pnl *= lev
            dd_eq = (e-pk_s)/pk_s if pk_s > 0 else 0
            if dd_eq < -0.30: pnl *= 0.1
            elif dd_eq < -0.22: pnl *= 0.4
            elif dd_eq < -0.15: pnl *= 0.7
            e *= (1+pnl); eq_s[i] = e; pk_s = max(pk_s, e)
        wf_s = [((eq_s[idx[-1]]/eq_s[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
        shuf_wfs.append(np.mean(wf_s)*12)
    p = np.mean(np.array(shuf_wfs) >= ws_real)
    print(f"  {shuf_type:<10}: real={ws_real:.0f}% shuf={np.mean(shuf_wfs):.0f}%+/-{np.std(shuf_wfs):.0f}% p={p:.3f} {'PASS' if p<0.05 else 'FAIL'}")

# === [2] PARAMETER SENSITIVITY ===
print("\n[2] PARAMETER SENSITIVITY")
print(f"  {'Config':<40} {'WFS':>6} {'B22':>6}")
print(f"  {'-'*55}")
for adx_t in [15, 20, 25]:
    bp_t = build_v2_bp(adx_thresh=adx_t)
    eq_t, _ = run_v2(bp_t)
    wf_t = [((eq_t[idx[-1]]/eq_t[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    i22 = np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22 = (eq_t[i22[-1]]/eq_t[max(0,i22[0]-1)]-1)*100
    tag = ' <--' if adx_t == 20 else ''
    print(f"  ADX={adx_t}{' '*(36-len(str(adx_t)))} {np.mean(wf_t)*12:>5.0f}% {b22:>+5.0f}%{tag}")
for cd in [3, 6, 12]:
    bp_t = build_v2_bp(cooldown=cd)
    eq_t, _ = run_v2(bp_t)
    wf_t = [((eq_t[idx[-1]]/eq_t[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    b22 = (eq_t[i22[-1]]/eq_t[max(0,i22[0]-1)]-1)*100
    tag = ' <--' if cd == 6 else ''
    print(f"  Cooldown={cd}H{' '*(33-len(str(cd)))} {np.mean(wf_t)*12:>5.0f}% {b22:>+5.0f}%{tag}")
for ddh, dds in [(-0.12,-0.06), (-0.15,-0.08), (-0.20,-0.10)]:
    bp_t = build_v2_bp(vol_dd_hard=ddh, vol_dd_soft=dds)
    eq_t, _ = run_v2(bp_t)
    wf_t = [((eq_t[idx[-1]]/eq_t[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    b22 = (eq_t[i22[-1]]/eq_t[max(0,i22[0]-1)]-1)*100
    tag = ' <--' if ddh == -0.15 else ''
    print(f"  DD={ddh:.0%}/{dds:.0%}{' '*(30)} {np.mean(wf_t)*12:>5.0f}% {b22:>+5.0f}%{tag}")

# === [3] ALPHA DECAY COMPARISON ===
print("\n[3] ALPHA DECAY: V1 vs V2")
bp_v1 = RACMRegime.compute(price, ret, params)
eq_v1_arr = np.ones(nn); e = 1.0; pk_e = 1.0
base_v1 = np.zeros(nn)
for i in range(S, nn): base_v1[i] = 0.20*ret[i]*bp_v1[i] + 0.80*lp_1h[i]
for i in range(S, nn):
    pnl = base_v1[i]
    past = base_v1[max(S, i-2160):i]
    lev, vt = RACMKelly.compute(past, params); total_lev = min(lev*vt, CAP)
    pnl *= total_lev; pnl += fra[i]*abs(total_lev)
    dd_mult = RACMDDControl.compute(e, pk_e, params); pnl *= dd_mult
    e *= (1+pnl); eq_v1_arr[i] = e; pk_e = max(pk_e, e)

# Monthly returns for both
eq1s = pd.Series(eq_v1_arr, index=idx_h); eq2s = pd.Series(eq_v2, index=idx_h)
mr1 = eq1s.resample('ME').last().pct_change().dropna()*100
mr2 = eq2s.resample('ME').last().pct_change().dropna()*100
mr1 = mr1[mr1.index >= idx_h[S]]; mr2 = mr2[mr2.index >= idx_h[S]]
x = np.arange(len(mr1))
slope1, _, r1, p1, _ = st.linregress(x, mr1.values)
x2 = np.arange(len(mr2))
slope2, _, r2, p2, _ = st.linregress(x2, mr2.values)
print(f"  V1: slope={slope1:.2f}%/month p={p1:.4f} {'DECLINING' if p1<0.05 else 'STABLE'}")
print(f"  V2: slope={slope2:.2f}%/month p={p2:.4f} {'DECLINING' if p2<0.05 else 'STABLE'}")
print(f"  V2 decays {'SLOWER' if abs(slope2) < abs(slope1) else 'SAME or FASTER'} than V1")

# === [4] BEAR TRADEOFF FIX: Correlation-aware vol scaling ===
print(f"\n[4] BEAR FIX: Correlation-aware vol scaling")
print(f"  Idea: Only apply vol reduction when BTC-ETH correlation > 0.8 (crash)")
print(f"        When correlation low (LS spread working), keep full position")

def v2_lw_corr_aware(bp, vol, corr):
    """Only reduce position when correlation is high (crash, LS fails)"""
    if bp < 0.5:
        if corr > 0.8: return 0.50  # crash: reduce everything
        else: return 0.95           # productive bear: LS works, keep high
    elif bp < 0.8:
        if corr > 0.8: return 0.70
        else: return 0.90
    elif vol > 1.0: return 0.90
    elif vol > 0.50: return 0.80
    else: return 0.70

def run_v2_corr(bp):
    eq = np.ones(nn); e = 1.0; pk_e = 1.0
    base_arr = np.zeros(nn)
    for i in range(S, nn):
        v = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        c = corr_btc_eth[i] if not np.isnan(corr_btc_eth[i]) else 0.5
        lw = v2_lw_corr_aware(bp[i], v, c)
        dw = max(0, 1-lw)
        base_arr[i] = dw*ret[i]*bp[i] + lw*lp_1h[i]
    for i in range(S, nn):
        pnl = base_arr[i]
        past = base_arr[max(S, i-2160):i]
        lev, vt = RACMKelly.compute(past, params)
        total_lev = min(lev*vt, CAP); pnl *= total_lev
        pnl += fra[i]*abs(total_lev)
        dd_mult = RACMDDControl.compute(e, pk_e, params); pnl *= dd_mult
        e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)
    return eq

eq_v2_corr = run_v2_corr(bp_v2)
wf_corr = [((eq_v2_corr[idx[-1]]/eq_v2_corr[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
i22 = np.where(np.array([d.year==2022 for d in idx_h]))[0]
i25 = np.where(np.array([d.year==2025 for d in idx_h]))[0]
b22_corr = (eq_v2_corr[i22[-1]]/eq_v2_corr[max(0,i22[0]-1)]-1)*100
o25_corr = (eq_v2_corr[i25[-1]]/eq_v2_corr[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0

# Standard V2 for comparison
wf_std = [((eq_v2[idx[-1]]/eq_v2[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
b22_std = (eq_v2[i22[-1]]/eq_v2[max(0,i22[0]-1)]-1)*100
o25_std = (eq_v2[i25[-1]]/eq_v2[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0

# V1 for comparison
wf_v1 = [((eq_v1_arr[idx[-1]]/eq_v1_arr[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
b22_v1 = (eq_v1_arr[i22[-1]]/eq_v1_arr[max(0,i22[0]-1)]-1)*100
o25_v1 = (eq_v1_arr[i25[-1]]/eq_v1_arr[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else 0

print(f"\n  {'Model':<25} {'WFS':>6} {'Win':>6} {'B22':>7} {'OOS25':>7}")
print(f"  {'-'*55}")
print(f"  {'V1 Baseline':<25} {np.mean(wf_v1)*12:>5.0f}% {sum(1 for r in wf_v1 if r>0):>2}/{len(wf_v1)} {b22_v1:>+6.0f}% {o25_v1:>+6.0f}%")
print(f"  {'V2 Standard':<25} {np.mean(wf_std)*12:>5.0f}% {sum(1 for r in wf_std if r>0):>2}/{len(wf_std)} {b22_std:>+6.0f}% {o25_std:>+6.0f}%")
print(f"  {'V2 + Corr-aware':<25} {np.mean(wf_corr)*12:>5.0f}% {sum(1 for r in wf_corr if r>0):>2}/{len(wf_corr)} {b22_corr:>+6.0f}% {o25_corr:>+6.0f}%")

bear_fixed = b22_corr > b22_std + 10
print(f"\n  Bear 2022 fix: {b22_std:+.0f}% -> {b22_corr:+.0f}% ({'IMPROVED' if bear_fixed else 'NOT IMPROVED'})")
print(f"  Correlation-aware approach: {'reduces position only in correlated crashes' if bear_fixed else 'needs more work'}")

# === SUMMARY ===
print(f"\n{'='*70}")
print(f"  FULL VERIFICATION SUMMARY")
print(f"{'='*70}")
