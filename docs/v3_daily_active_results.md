# V3 Daily-Active Approaches: 全滅記録 (2026-05-04)

## テストした5アプローチ

| # | アプローチ | WFS | Win | Bear22 | 判定 |
|---|-----------|----:|----:|-------:|------|
| A | LS spread sizing | 363% (-3) | 52/57 | +63% | 効果なし |
| B | Rolling Sharpe scaling | 269% (-97) | 51/57 | +36% | 有害 |
| C | Dispersion sizing | 424% (+58) | 48/57 | +32% | Bear tradeoff |
| D | Funding carry tilt | 617% | 56/57 | +306% | **バグ（却下）** |
| E | Adaptive Kelly | 214% (-152) | 52/57 | +16% | 有害 |

## D) Funding carry tilt のバグ

`pnl += abs(fra[i]) * 0.5` — abs()により常にfundingを受け取る計算。
実際はロング側が87%の時間でfundingを払う。年間182%の架空利益。

## 構造的な発見

**ポジションサイジングの日常的調整は全て同じトレードオフに陥る:**
- 不確実な時にポジション縮小 → 平時の損失抑制
- しかしBear市場も「不確実」→ Bear市場のLS利益も削減
- V2 vol-firstと同じパターン

## 結論

**V2は「レジーム + LS + ポジションサイジング」アーキテクチャの構造的最適解に近い。**
ポジションサイジングの微調整ではこれ以上の改善は困難。

さらなる改善には全く新しいアルファ源またはアーキテクチャが必要:
- Tick-level microstructure
- Cross-exchange arbitrage
- ML meta-learner
- HMM regime switching

## V2確定値 (変更なし)

- WFS: 366%
- Win: 51/57 (+ crash cls で 52/57)
- Bear 2022: +68% (+ crash cls で +72%)
- OOS 2025: +299%
- Whipsaw: 0.3回/日 (V1の4.1から93%削減)
