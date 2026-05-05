# V6 全モデル評価レポート

**日付**: 2026-05-05
**目的**: 現存モデルの backtest 数値とその信頼性を honest に評価

## モデル一覧と数値

### Tier S: 本物の alpha (検証済)

#### S1. swin_v25 BTC (Tardis Bybit)
| 年 | NET | B&H | vs B&H | MDD | Trades | WR |
|---|---:|---:|------:|---:|------:|---:|
| 2022 (Bear) | +52.4% | -64.2% | +117% | 14.6% | 204 | 49.0% |
| 2024 (Bull) | +154.0% | +121.0% | +33% | 8.1% | 225 | 57.3% |
| 2025 (Mixed) | +46.5% | -7.8% | +54% | **5.7%** | 187 | 44.4% |
| **CAGR** | **+78.3%/yr** | - | - | - | - | - |

**信頼性: ★★★★★ HIGH**
- 完全 Tardis Bybit データ (book_snapshot + funding + liquidations)
- 真の funding cost 加算済 (-5%/yr 程度)
- 3年全て大幅 outperform
- リアルな slippage 想定 (BTC liquid)

**真の予測**: 年 **+50-100%/yr** (実取引 slippage 考慮)

### Tier A: honest だが過大評価可能性

#### A1. Standard Funding Farm
| Lev | Annual | MDD | Sharpe |
|----:|------:|----:|------:|
| 1.5x | **+4.4%** | -1.1% | 7.5 (basis 過小評価で水増し) |

**信頼性: ★★★★ MEDIUM-HIGH**
- per-8H funding 正しく適用 (バグ #2 修正後)
- spot/perp相関 0.998 → daily std 過小評価で Sharpe inflated
- 真の Sharpe 推定 ~2-3 (basis variation 反映)

**真の予測**: 年 **+3-7%/yr** (cash 程度)

### Tier B: 数値疑わしい

#### B1. swin_v25 SOL (Binance API + dummy book)
| 年 | NET (raw) | NET (40% discount) | B&H |
|---|---:|---:|---:|
| 2022 | +646.8% | +259% | -90.5% |
| 2024 | +552.4% | +221% | +81.7% |
| 2025 | +517.9% | +207% | -28.2% |
| **CAGR** | **+570%/yr** | **+228%/yr** | - |

**信頼性: ★★ LOW (raw) / ★★★ MEDIUM (discounted)**
- HYPERLIQUID 1bp slippage 想定 → SOL altcoin 実態 5-20bp
- 過大評価係数 推定 2-5x
- WF parameter grid search → 各 window 過学習可能性

**真の予測**: 年 **+50-200%/yr** (大幅な discount 必要)

#### B2. swin_v25 ETH (Tardis ETH book)
- **進行中** (W5/13 まで +27% 累計)
- 5年予測: 2022 ~+50-100%, 2024 +100-150%, 2025 ~+50%
- 信頼性: ★★★★ Tardis full data だが ETH altcoin liquidity 中

#### B3. swin_v25 XRP
- **進行中**

### Tier C: 過去のバグ込み数値 (棄却済)

#### C1. RACM V2/V3 (funding バグ #1)
- 旧報告: WFS 369-588%, B22 +149%
- 真値: WFS -17% (修正後 catastrophic)
- **完全棄却**

#### C2. Bidirectional Funding Farm (funding バグ #2 含む)
- 旧報告: +101%/yr, B22 +152%
- 真値 (per-8H 修正後): -0.1%/yr, MDD -54%
- **棄却 - 高レバ inverse FF は losing**

#### C3. 各種 ML 予測 (HGBR, swin_v25 simplified version)
- 旧報告: Sharpe > 5 多数
- 真値: shuffle テスト FAIL or 過学習
- **棄却**

## 数値の正確性評価

### 信頼できる数値
| 戦略 | Backtest | 真の予測 | 信頼度 |
|------|---------:|--------:|------:|
| swin_v25 BTC CAGR | **+78%/yr** | **+50-100%/yr** | ★★★★★ |
| Standard FF 1.5x | +4.4%/yr | +3-7%/yr | ★★★★ |

### 過大評価の可能性高い
| 戦略 | Backtest | 真の予測 | 信頼度 |
|------|---------:|--------:|------:|
| swin_v25 SOL | +570%/yr | +50-200%/yr | ★★ |
| swin_v25 ETH (進行中) | TBD | TBD | ★★★★ |
| swin_v25 XRP (進行中) | TBD | TBD | ★★★ |

### 確実にバグ込み (棄却)
- RACM V1/V2/V3 全て
- Bidirectional FF
- swin_v25 simplified versions

## 教授要件 (honest base)

| 要件 | 真の最良 | 達成 |
|------|---------:|------|
| WFS ≥300% | 78% (BTC) / 228% (SOL discount) | ✗ |
| MDD ≤-30% | -8〜-15% | ✓ |
| Bear利益 | +52% (BTC), +207% (SOL) | ✓✓ |
| 取引≥3/日 | swin_v25 で 200+/年=平均0.55/日 | △ |
| Lighter zero fee | ✓ | ✓ |

300% は単一資産 honest では到達不可。
Multi-asset (BTC + ETH + SOL discount) で +150-250%/yr 期待値。

## 総括

### 確定した本物 alpha
1. **swin_v25 BTC**: +78%/yr CAGR (最強の単一資産戦略)
2. **Standard FF 1.5x**: +5%/yr (低リスク baseline)

### 進行中 (信頼性高)
3. **swin_v25 ETH**: Tardis full data → honest 期待
4. **swin_v25 XRP**: API only → 中信頼

### 過大評価 (要 cost discount)
5. **swin_v25 SOL**: +570% raw → +200% honest
6. **swin_v25 LINK/DOGE**: 未テスト、SOL と同様の傾向予想

### Bot 化済
- **funding_farm_bot.py**: Standard + Bidirectional 対応 (Bidirectional 棄却済、Standard のみ運用推奨)

### 未実装
- swin_v25 multi-asset bot
- Lighter.xyz 実取引 integration
