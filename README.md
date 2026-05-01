# RACM v2: Whipsaw-Resistant Adaptive Momentum

## Why v2?

RACM v1.0 works (OOS +93%, 39/45 WIN) but has structural weaknesses:

| Problem | v1.0 Impact | Root Cause |
|---------|------------|-----------|
| **Whipsaw** | 4x/day regime changes, ~55% of losses | Regime thresholds too sensitive at 1H |
| **Base signal < B&H at 1x** | +15% vs B&H +32% | Model value is only crash protection + leverage |
| **Alpha decay** | p=0.003, declining | Market evolution |
| **Kelly stuck at cap** | Always 3.0x | mu/var too high for crypto |

## v2 Design Goals

1. **Reduce whipsaw** without losing crash detection speed
2. **Improve base signal at 1x** to beat B&H without leverage
3. **Slow alpha decay** by using more robust/diverse signals
4. **Make Kelly actually adaptive** (not capped-out)

## Available Data (from v1 research)

### Validated (works)
- BTC 1H OHLCV: 2019-2025 (52K+ bars)
- Derivatives 1H: funding, OI, liquidations (2019-2025)
- 6-asset 8H: BTC/ETH/SOL/XRP/DOGE/LINK
- Regime detection: MA+skew+DD composite (needs improvement)
- LS momentum: 60d+90d cross-sectional (works, main alpha)

### Untested (opportunity)
- Tick-level microstructure (Y:\tardis_data, BTC+ETH 2019-2026)
- On-chain data: MVRV/SOPR/NVT (rejected individually but ensemble untested)
- Macro data: S&P/VIX/rates (available but unused)
- Volatility forecasting (more feasible than direction prediction)
- ADX/Donchian for regime confirmation (reduce whipsaw)
- Multi-timeframe cointegration (1H+8H+daily)

### Rejected (don't retry)
- ML direction prediction (LSTM/CNN/Transformer: 50-54%, shuffle FAIL)
- Orderflow/OBI (52.1% but costs exceed edge)
- Single on-chain metrics (Bonferroni correction fails)

## Architecture Ideas

### Idea A: Regime Smoothing
Keep RACM v1.0 structure but add whipsaw filter:
- ADX > 15 required for regime change (trend confirmation)
- Minimum 6-12H between regime transitions
- Hysteresis: different thresholds for entry vs exit

### Idea B: Volatility-First
Instead of predicting direction, predict VOLATILITY:
- High vol predicted → reduce position (before crash)
- Low vol predicted → increase position (calm period)
- Use GARCH/realized vol for vol forecasting
- Direction from LS momentum (unchanged)

### Idea C: Multi-Timeframe Fusion
- Daily: trend direction (MA, momentum)
- 8H: LS ranking (cross-sectional)
- 1H: entry/exit timing (micro-regime)
- Each layer votes independently, combine by confidence

### Idea D: Ensemble Signals
- Stack v4B (single-asset regime) + RACM LS + vol forecast
- Meta-learner weights signals based on recent performance
- Adaptive: winner signal gets more weight

## v1.0 Reference
- Repo: https://github.com/nanashiNiA/regime-adaptive-crypto-momentum (tag v1.0)
- IS WFS: 367%, OOS: +93%, Win: 39/45
- Paper: paper/racm_paper.pdf
