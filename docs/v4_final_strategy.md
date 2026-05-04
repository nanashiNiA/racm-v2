# V4 Final Strategy: Bear-aware Funding Farm

**確定日**: 2026-05-05
**結論**: BTC perp の funding rate を delta neutral で獲得する戦略が、honest な範囲で最高性能。

## 確定数値 (5年バックテスト, 2021-2025)

| 指標 | 値 | 教授要件 | 判定 |
|------|---:|---------|------|
| 年率リターン | **+54%** | ≥300% | ✗ (300%は honest に到達不可) |
| MaxDD | **-3.0%** | ≤-30% | **✓** |
| Sharpe | **10.80** | - | (極めて高い) |
| Bear 2022 | **+14.4%** | >0% | **✓** |
| 月勝率 | **93% (53/57月)** | - | **優秀** |
| 取引/日 | ~0.29 (週次rebal) | ≥3/日 | △ (rebalあり) |
| Lighter zero fee | ✓ | ✓ | **✓** |

## 戦略構造

```
Capital = $1
  ├─ Spot Long $1 (Binance/Coinbase, 5bps fee)
  └─ Perp Short $1 (Lighter.xyz, 0 fee, lev 1.5x → margin $0.67)

Funding income: 8H毎に short 側が funding rate を受け取る
Delta: spot+perp ≈ 0 (BTC価格に中立)
P&L: funding income - basis variation - rebal cost
```

## Bear-aware Dynamic Leverage

```python
funding_30d_avg_annual:
  > 100%/yr → lev 2.0  (高funding時にmaxレバ)
  >  50%/yr → lev 1.5
  >  20%/yr → lev 1.0
  >   0%/yr → lev 0.5  (低funding時に縮小)
  ≤   0%/yr → exit perp (negative fundingなら退避)
```

## 検証結果 (全PASS)

| テスト | 結果 |
|--------|------|
| Shuffle (200x) | **p=0.000 PASS** (timing has alpha) |
| Leak check | shift全て同一 → **リークなし** |
| Parameter sensitivity | デフォルト最良、頑健 |
| OOS分割 (Train 21-23, Test 24-25) | Train +52%, Test +55% (一致) |
| Real basis cost | 反映済み (Bybit perp - Binance spot) |
| Liquidation handling | 0回 (lev=1.5x で安全) |

## 年別パフォーマンス

| 年 | リターン | Funding率 |
|---|--------:|--------:|
| 2021 | +73% | +161%/yr |
| 2022 | +14% (Bear) | +23%/yr |
| 2023 | +53% | +65%/yr |
| 2024 | +84% | +107%/yr |
| 2025 | +24% | +43%/yr |

**全年プラス。Bear 2022でも+14%。**

## 棄却した代替案 (検証済)

| 戦略 | 結果 | 棄却理由 |
|------|------|---------|
| RACM V2/V3 | WFS 369% (BUG込み) | funding バグで架空利益 |
| RACM 修正後 | WFS 39% | LS momentum alphaなし、funding cost で負ける |
| swin_v25 | 減衰 (window 1: +34% → window 5: -3.5%) | 過学習、funding cost 未計算 |
| LS momentum | シャッフル p=0.866 | alpha 不在 |
| Multi-asset Funding Farm | データ取得失敗 | API limit |
| Spot RACM | 年+27%, MDD-43% | リターン低、MDD大 |
| Lighter native | 年+0.05% gain | Bybit/Binance と差なし |
| 動的レバ + alpha overlay | 全FAIL | 簡易シグナルにalphaなし |
| Multi-Strategy 50/50 | 年+40% | FF 100% より低い |

## 理論的背景

BTC perp funding rate is positive **87% of the time**:
- Most traders are long → exchange charges longs (funding > 0)
- Shorts receive payment as compensation for taking risk
- Average annual funding: **+50-100%/yr** (high in bull, low in bear)
- Structural edge that **doesn't decay** (it's a market mechanism)

Funding Farm captures this:
- Spot Long offsets perp short directional risk
- Net delta = 0 → BTC price doesn't matter
- Income = funding rate (tax for short side)

## 実装メモ

**取引所**: Lighter.xyz (perp, zero fee) + Binance/Coinbase (spot)

**Cross-exchange execution**:
- Initial: open spot on CEX + open perp on Lighter
- Weekly rebalance: adjust perp size to maintain delta=0
- Delta drift: ~0.27%/day (small, weekly rebal sufficient)

**Capital allocation**:
- Spot: 60% of capital ($1.0 spot exposure)
- Perp margin: 40% of capital ($0.67 margin for $1 short, lev=1.5x)

**Risk management**:
- Liquidation buffer: 50% maintenance margin
- Lev=1.5x = no liquidations in 5-year backtest
- Lev=2.0x = 1 liquidation
- Lev=3.0x+ = multiple liquidations, blow up

**Lighter特典 vs CEX**:
- Lighter は perp の手数料0なので rebalance cost が削減
- 実測: +0.05%/年 advantage (small but free)

## なぜ300% WFSは到達不可能か

1. BTC perp funding は構造的に年50-100%が上限 (lev=1x)
2. 高レバ (3x+) は basis variation で清算される
3. 別の独立 alpha 源は存在しない (40+候補検証済み、全FAIL)
4. 過去報告の 300%超 は全て funding rate のバグによる架空利益

**honest な数字: 年+54%, Sharpe 10.8, MDD -3%**

これは伝統的金融商品では到達困難な水準であり、教授の300%要件は recalibrate が必要。

## Bot 実装方針

Phase 1: バックテスト確定 (DONE)
Phase 2: paper trading on Lighter testnet (Lighter API調査必要)
Phase 3: small live capital (~$1000) で動作確認
Phase 4: 本格運用

## 棄却済みコード

以下は funding bug or alpha decay で実用不可:
- src/racm_bot.py (V1)
- src/racm_bot_v2.py (V2)
- src/racm_bot_v3.py (V3)
- src/racm_core.py の funding 計算

新実装が必要:
- src/funding_farm_bot.py (TODO)
