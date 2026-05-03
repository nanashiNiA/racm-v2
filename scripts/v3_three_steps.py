"""V3: Three steps to solve V2's unsolved problems
Step 1: On-chain ensemble (alpha decay)
Step 2: Crash vs productive-bear classifier (bear tradeoff)
Step 3: Macro overlay (1x signal boost)
Each step tested incrementally, then combined.
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
print("  V3: THREE-STEP IMPROVEMENT")
print("="*70)

# === Load base data ===
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

# === Step 1: On-Chain Ensemble ===
print("\n[Step 1] Loading on-chain data...")
DATA = 'C:/Users/A701/Documents/nia/prediction_model_project/data/external'
mvrv = pd.read_csv(f'{DATA}/mvrv.csv', parse_dates=['date'], index_col='date')
fg = pd.read_csv(f'{DATA}/fear_greed_index.csv', parse_dates=['date'], index_col='date')
puell = pd.read_csv(f'{DATA}/puell_multiple.csv', parse_dates=['date'], index_col='date')
sopr = pd.read_csv(f'{DATA}/sopr.csv', parse_dates=['date'], index_col='date')

# Align to 1H (lag-1 day = use previous day's value)
def align_daily_to_1h(daily_series, idx_1h):
    daily_resampled = daily_series.reindex(idx_1h.normalize(), method='ffill')
    daily_resampled.index = idx_1h
    return np.roll(daily_resampled.values, 24)  # lag-1 day = shift 24 hours

mvrv_1h = align_daily_to_1h(mvrv.iloc[:, 0], idx_h)
fg_1h = align_daily_to_1h(fg['value'] if 'value' in fg.columns else fg.iloc[:, 0], idx_h)
puell_1h = align_daily_to_1h(puell.iloc[:, 0], idx_h)
sopr_1h = align_daily_to_1h(sopr.iloc[:, 0], idx_h)

# Z-score each (expanding window, lag-1)
def expanding_zscore(arr, min_periods=365):
    z = np.zeros(len(arr))
    for i in range(min_periods, len(arr)):
        vals = arr[:i]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 10:
            z[i] = (arr[i] - np.mean(vals)) / (np.std(vals) + 1e-10)
    return z

mvrv_z = expanding_zscore(mvrv_1h)
fg_z = expanding_zscore(fg_1h)
puell_z = expanding_zscore(puell_1h)
sopr_z = expanding_zscore(sopr_1h)

# Ensemble = average of available z-scores
onchain_ensemble = np.zeros(nn)
for i in range(nn):
    vals = []
    for z in [mvrv_z, fg_z, puell_z, sopr_z]:
        if not np.isnan(z[i]) and z[i] != 0: vals.append(z[i])
    if len(vals) >= 2:
        onchain_ensemble[i] = np.mean(vals)

print(f"  On-chain ensemble: {np.count_nonzero(onchain_ensemble)} non-zero values")
print(f"  Range: [{np.min(onchain_ensemble[onchain_ensemble!=0]):.2f}, {np.max(onchain_ensemble):.2f}]")

# === Step 2: Crash vs Productive-Bear ===
print("\n[Step 2] Computing LS spread and correlation...")
# LS spread at 8H (lag-1): is the LS making money?
ls_spread_1h = np.zeros(nn)
for j in range(1, len(c_8h)):
    ts = c_8h[j]; te = c_8h[j+1] if j+1 < len(c_8h) else ts + pd.Timedelta(hours=8)
    idxs = np.where((idx_h >= ts) & (idx_h < te))[0]
    # LS spread = previous 8H bar's LS PnL (lag-1)
    for ii in idxs: ls_spread_1h[ii] = lp_8h[j-1]

# 6-asset pairwise correlation (rolling 30d at 8H = 90 bars)
asset_names = list(a_ret_8h.keys())
n8h = len(c_8h)
avg_corr_8h = np.zeros(n8h)
for i in range(90, n8h):
    rets_window = np.array([a_ret_8h[a][i-90:i] for a in asset_names])
    corr_matrix = np.corrcoef(rets_window)
    # Average off-diagonal
    n_assets = len(asset_names)
    total = 0; count = 0
    for a in range(n_assets):
        for b in range(a+1, n_assets):
            if not np.isnan(corr_matrix[a, b]):
                total += corr_matrix[a, b]; count += 1
    avg_corr_8h[i] = total / count if count > 0 else 0.5

# Map to 1H
avg_corr_1h = np.zeros(nn)
for j in range(len(c_8h)):
    ts = c_8h[j]; te = c_8h[j+1] if j+1 < len(c_8h) else ts + pd.Timedelta(hours=8)
    idxs = np.where((idx_h >= ts) & (idx_h < te))[0]
    for ii in idxs: avg_corr_1h[ii] = avg_corr_8h[j]

print(f"  Avg correlation range: [{np.min(avg_corr_1h[S:]):.2f}, {np.max(avg_corr_1h[S:]):.2f}]")

# === Step 3: Macro Overlay ===
print("\n[Step 3] Loading macro data...")
try:
    macro = pd.read_csv(f'{DATA}/alternative/macro_data.csv', parse_dates=[0], index_col=0)
    vix_daily = macro['VIX'] if 'VIX' in macro.columns else None
    dxy_daily = macro['DXY'] if 'DXY' in macro.columns else None
    if vix_daily is not None:
        vix_1h = align_daily_to_1h(vix_daily, idx_h)
        print(f"  VIX: {np.count_nonzero(~np.isnan(vix_1h))} values")
    else:
        vix_1h = np.full(nn, 20); print("  VIX: not found, using default 20")
    if dxy_daily is not None:
        dxy_1h = align_daily_to_1h(dxy_daily, idx_h)
        dxy_5d_change = np.zeros(nn)
        for i in range(120, nn): dxy_5d_change[i] = dxy_1h[i] - dxy_1h[i-120]
        print(f"  DXY: {np.count_nonzero(~np.isnan(dxy_1h))} values")
    else:
        dxy_5d_change = np.zeros(nn); print("  DXY: not found")
except:
    vix_1h = np.full(nn, 20); dxy_5d_change = np.zeros(nn)
    print("  Macro data not available, using defaults")

# === Build V2 baseline regime ===
def build_v2_bp():
    bp = np.ones(nn); lc = 0; pb = 1.0
    for i in range(S, nn):
        if i < S+720: tv = 0.80
        else: tv = np.mean(vol_30d[max(S, i-4320):i-1])
        cv = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else tv
        vs = np.clip(tv/(cv+1e-10), 0.3, 1.5)
        if dd_arr[i-1] < -0.15: vs = min(vs, 0.3)
        elif dd_arr[i-1] < -0.08: vs = min(vs, 0.5)
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
            elif (not np.isnan(adx[i-1]) and adx[i-1]>20) and (i-lc>=6):
                bp[i]=vs;lc=i;pb=vs
            else:bp[i]=pb
        else:bp[i]=vs;pb=vs
    return bp

folds = []; cur = pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

def run_model(bp, lw_func, onchain_mult=False, crash_class=False, macro_overlay=False):
    eq = np.ones(nn); e = 1.0; pk_e = 1.0
    base_arr = np.zeros(nn)
    for i in range(S, nn):
        v = vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        lw = lw_func(bp[i], v)
        dw = max(0, 1-lw)

        # Step 2: Crash classifier modifies bp in bear
        bp_eff = bp[i]
        if crash_class and bp[i] <= 0.2:
            if ls_spread_1h[i] > 0:  # productive bear
                bp_eff = 0.5
            elif avg_corr_1h[i] > 0.85 and ls_spread_1h[i] <= 0:  # crash
                bp_eff = 0.0
            # recalc LW for new bp
            lw = lw_func(bp_eff, v)
            dw = max(0, 1-lw)

        base_arr[i] = dw * ret[i] * bp_eff + lw * lp_1h[i]

        # Step 1: On-chain ensemble multiplier
        if onchain_mult:
            oc = onchain_ensemble[i]
            if oc > 0.7: base_arr[i] *= 0.7
            elif oc < -0.7: base_arr[i] *= 1.3

        # Step 3: Macro overlay (only in neutral)
        if macro_overlay and 0.9 < bp_eff < 1.1:
            vix = vix_1h[i] if not np.isnan(vix_1h[i]) else 20
            dxy_chg = dxy_5d_change[i]
            if vix < 20 and dxy_chg < 0: base_arr[i] *= 1.3
            elif vix > 30 and dxy_chg > 0: base_arr[i] *= 0.7

    for i in range(S, nn):
        pnl = base_arr[i]
        past = base_arr[max(S, i-2160):i]
        lev, vt = RACMKelly.compute(past, params)
        total_lev = min(lev*vt, CAP); pnl *= total_lev
        pnl += fra[i]*abs(total_lev)
        dd_mult = RACMDDControl.compute(e, pk_e, params); pnl *= dd_mult
        e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)
    return eq, base_arr

def v2_lw(bp, vol):
    if bp < 0.5: return 0.95
    elif bp < 0.8: return 0.90
    elif vol > 1.0: return 0.90
    elif vol > 0.50: return 0.80
    else: return 0.70

def eval_full(eq, base_arr, label):
    wf_r = [((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs = np.mean(wf_r)*12; win = sum(1 for r in wf_r if r > 0)
    i22 = np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22 = (eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    i25 = np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25 = (eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    # Alpha decay
    mr = pd.Series(eq, index=idx_h).resample('ME').last().pct_change().dropna()*100
    mr = mr[mr.index >= idx_h[S]]
    x = np.arange(len(mr))
    slope, _, _, p_decay, _ = st.linregress(x, mr.values)
    # 1x WFS
    eq1x = np.ones(nn); e1 = 1.0
    for i in range(S, nn): e1 *= (1+ret[i]*build_v2_bp()[i]); eq1x[i] = e1  # approximate
    return wfs, win, len(wf_r), b22, o25, slope, p_decay

# === Run all variants ===
print("\n[4] Running variants...")
bp_v1 = RACMRegime.compute(price, ret, params)
bp_v2 = build_v2_bp()

configs = [
    ('V1 Baseline', bp_v1, lambda bp,v: 0.80, False, False, False),
    ('V2 (current)', bp_v2, v2_lw, False, False, False),
    ('+Step1 (on-chain)', bp_v2, v2_lw, True, False, False),
    ('+Step2 (crash cls)', bp_v2, v2_lw, False, True, False),
    ('+Step3 (macro)', bp_v2, v2_lw, False, False, True),
    ('+Step1+2', bp_v2, v2_lw, True, True, False),
    ('+Step1+3', bp_v2, v2_lw, True, False, True),
    ('+Step2+3', bp_v2, v2_lw, False, True, True),
    ('V3 (all three)', bp_v2, v2_lw, True, True, True),
]

print(f"\n  {'Model':<22} {'WFS':>6} {'Win':>6} {'B22':>7} {'OOS25':>7} {'Decay':>8} {'p':>6}")
print(f"  {'-'*65}")
for label, bp, lw, oc, cc, mo in configs:
    eq, base = run_model(bp, lw, oc, cc, mo)
    wfs, win, nf, b22, o25, slope, p_d = eval_full(eq, base, label)
    tag = ''
    if 'V1' in label: tag = ' <-- v1'
    elif 'V3' in label: tag = ' <<<'
    p_str = f'{p_d:.3f}' if p_d < 1 else 'N/A'
    print(f"  {label:<22} {wfs:>5.0f}% {win:>2}/{nf} {b22:>+6.0f}% {o25:>+6.0f}% {slope:>+7.2f} {p_str:>6}{tag}")
