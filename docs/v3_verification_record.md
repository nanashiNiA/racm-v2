# V3 (V2 + Crash Classifier) 検証記録 (2026-05-04)

## 変更内容

V2のdanger状態（bp<=0.2）で、一律ポジション縮小ではなく2分岐:
- **LS spread > 0（productive bear）**: bp=0.5（V2の0.2より積極的）
- **BTC-ETH相関 > 0.85 かつ LS spread <= 0（crash）**: bp=0.0（完全撤退）

新パラメータ: 相関閾値0.85（Markowitz文献、事前固定）

## 検証結果

### リークチェック: 全PASS
- corr1h: lag-1 (np.roll)
- lp1h[i-1]: lag-1
- bp<=0.2: lag-1レジーム
- 閾値0.85: 事前固定

### パラメータ感度: 非常に頑健
相関閾値 0.70-0.95: WFS 365-366%（±1pt）

### シャッフルテスト: p=0.000 PASS

### 性能比較

| 指標 | V2 | V3 | 差 |
|------|---:|---:|---:|
| WFS | 366% | 366% | 0 |
| Win | 51/57 | 52/57 | +1 |
| Bear 2022 | +68% | +72% | +4pt |
| OOS 2025 | +299% | +291% | -8pt |
| Alpha decay | -1.62/mo | -1.63/mo | 同じ |

### 評価

効果は微小（Bear +4pt、Win +1）。Crash classifierの発動が稀すぎるため。
リスクは極めて低い（パラメータ感度ゼロ、リークなし）。
OOS 2025は微減だが、2025年はcrash classifierが発動しない環境のため無関係。

## 未解決の課題

1. **Alpha decay (p=0.006)**: V3でも改善なし。LS momentum自体の問題。
2. **Base 1x < B&H**: 変わらず。
3. **日常的に効くアプローチがない**: Crash classifierは稀にしか発動しない。

## 棄却した候補

| 候補 | 結果 | 理由 |
|------|------|------|
| On-chain ensemble | WFS -9pt | 日次データが月次WFの中で変化不足 |
| 1H LS direct | Bear +37%（V2の+68%より悪化） | 1Hノイズが多すぎ |
| GARCH vol forecast | WFS +60pt だがWin 43/57 Bear +3% | 過剰反応 |
| Macro VIX/DXY | WFS -1pt | 影響なし |
| 全部入り | Bear -18% | 過剰制約 |
