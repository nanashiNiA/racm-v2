# V4 Brainstorm: 200 Variations + 100 Architectures + Top 5

**日付**: 2026-05-05
**前提**: RACM funding バグ発覚により全面再構築。swin_v25 (個別トレード + ATR stop + 65%WR) は有効候補として継続検証中。

## 教訓まとめ

| 教訓 | 内容 |
|------|------|
| 1 | Perpetual の funding は年100%/lev → 連続ポジションは構造的に負ける |
| 2 | 個別トレード + ATR stop は funding 影響軽微で機能する |
| 3 | HGBR ML は板/約定パターンに 65% WR 程度の予測力あり |
| 4 | LS momentum alpha は funding 抜きで WFS 15-37% (B&H と同等) |
| 5 | Volatility 予測は 86% 精度（活用未完） |
| 6 | Bybit/Binance 板データは 2021-2025 取得済み |
| 7 | 0xArchive で Lighter/Hyperliquid データ取得可能 (未検証) |
| 8 | Multi-factor で alpha decay 緩和可能 (未完) |
| 9 | Cross-sectional 6資産データあり (BTC/ETH/SOL/XRP/DOGE/LINK) |
| 10 | Spot は funding 無しだが有意な alpha 確認できず |

---

# Part A: 200 個別変更/バリエーション候補

## A1. 既存戦略の修正・強化 (1-30)

1. RACM funding バグを修正したまま spot で実装
2. RACM を market neutral (long+short equal) で運用 → funding net zero
3. RACM レバを 1.0x に下げて funding コスト削減
4. RACM を 8H bar から 4H bar へ高頻度化
5. RACM を 8H bar から 1D bar へ低頻度化
6. RACM の LS lookback を 60/90 から 30/60 に短縮
7. RACM の LS lookback を 90/180 に拡張
8. RACM の LS lookback を adaptive (vol-based) にする
9. RACM の regime detection を CUSUM ベースに変更
10. RACM の regime detection を HMM (4-state) に変更
11. RACM の DD control を Kelly fraction に統合
12. RACM の skew threshold を rolling percentile based に変更
13. RACM の MA110 を adaptive MA (ALMA) に変更
14. RACM の peak window を 30d/60d/90d で平均化
15. RACM crash detector を Markov regime classifier に置換
16. RACM の vol target を realized vol percentile-based に
17. RACM の position sizing を Risk Parity に変更
18. RACM に exit-only momentum (旧 Plan E) を再評価
19. RACM の 6資産選択を液动性で動的選択
20. RACM の rebalancing を「シグナル変化時のみ」に変更
21. swin_v25 のサイズを 0.15 から 0.30 に拡大
22. swin_v25 のレバ を 4-6x から 2-4x に縮小
23. swin_v25 のホールド期間を 20→48 bar に拡張
24. swin_v25 の min_confidence を 0.35 → 0.50 に厳格化
25. swin_v25 の ATR stop を 1.5 → 1.0 にタイト化
26. swin_v25 の Trail を Chandelier Exit に変更
27. swin_v25 を ETH/SOL にも適用 (multi-asset)
28. swin_v25 の HGBR を XGBoost に置換
29. swin_v25 の Ridge を Lasso に置換
30. swin_v25 を 15min バーで運用

## A2. データソース変更 (31-50)

31. 0xArchive Lighter 板データで MS signal 再構築
32. 0xArchive Hyperliquid 板データで MS signal 検証
33. Lighter funding rate を取得してコスト確認
34. Lighter trades データで volume profile 構築
35. Lighter order lifecycle データで maker/taker 比率
36. Hyperliquid HIP-3 perp データ追加
37. Binance spot orderbook L2 を spread proxy に
38. Tardis bybit derivative_ticker の mark/last spread 利用
39. CoinGecko 全 stablecoin 動向を リスクオン/オフ指標に
40. DefiLlama TVL 変化を flow indicator に
41. Bitfinex margin long/short ratio を sentiment proxy に
42. CME BTC futures basis を term structure 指標に
43. Glassnode UTXO age distribution を holder behavior に
44. Whale alert tx を smart money flow に
45. CryptoQuant exchange inflow を sell pressure に
46. Santiment social volume を retail FOMO 指標に
47. Reddit r/cryptocurrency 投稿数を sentiment proxy に
48. Twitter/X cashtag メンション量を attention 指標に
49. Google Trends "bitcoin" を retail interest に
50. Macro: VIX, DXY, gold, oil の cross-asset signal

## A3. アルファシグナル新規 (51-90)

51. Order book imbalance gradient (top 5 vs top 25)
52. Effective spread vs quoted spread の乖離
53. Trade volume burst detection (z-score > 3)
54. Iceberg order detection (large hidden orders)
55. Spoof/layering pattern detection
56. Trade clustering (intensity peaks)
57. Quote-to-trade ratio anomaly
58. Cancel rate spike (HFT activity)
59. Maker share decline (taker dominance)
60. VPIN (volume-synchronized probability of informed trading)
61. Kyle's lambda (price impact per dollar)
62. Amihud illiquidity ratio
63. Realized vs implied vol spread (using DVOL)
64. Funding rate term structure (8H vs daily avg)
65. Open interest spike + price stagnation = squeeze setup
66. OI/Volume ratio (positioning vs activity)
67. Liquidation imbalance direction (long vs short liq ratio)
68. Liquidation cluster size (single large vs many small)
69. Cross-exchange basis (Binance vs Bybit perp)
70. Spot-perp basis term structure
71. Stablecoin premium/discount (USDT-USDC)
72. Funding rate convexity (kurtosis of recent funding)
73. ETH/BTC ratio momentum as risk indicator
74. SOL/BTC ratio for altseason detection
75. BTC dominance change rate
76. Cross-asset lead-lag: ETH leads BTC by 1-2H
77. Top altcoin first-move detection
78. Sector rotation: L1 vs L2 vs DeFi
79. NFT volume as risk-on proxy
80. Gas price (ETH) as DeFi activity proxy
81. Stablecoin supply growth as new money inflow
82. BTC ATR / ETH ATR ratio (relative vol)
83. 3-asset PCA: first PC = market, second = altcoin
84. Funding rate cross-section dispersion (asset-specific)
85. Open interest cross-asset rotation (capital flow)
86. Spot volume vs perp volume ratio
87. Trader flow bias (whale vs retail by trade size)
88. Aggressor side imbalance (buy aggressor vs sell)
89. CVD (cumulative volume delta) divergence with price
90. Taker buy/sell ratio over rolling window

## A4. ML/AI 手法 (91-120)

91. LightGBM with categorical regime features
92. CatBoost with permutation importance
93. TabNet for orderbook features
94. 1D CNN for price/volume sequences
95. LSTM with attention for orderbook depth
96. Transformer (small) for multi-asset sequences
97. N-BEATS for multi-horizon forecasting
98. TFT (Temporal Fusion Transformer) for crypto
99. Conformal prediction for confidence intervals
100. Ensemble: HGBR + Ridge + LightGBM voting
101. Stacking: meta-learner on base predictions
102. Meta-labeling for entry filtering (de Prado)
103. Triple barrier method for label balancing
104. Sample weighting by uniqueness (de Prado)
105. Walk-forward CV with purged k-fold
106. PCA for feature reduction (top 10 components)
107. Autoencoder denoising for raw orderbook
108. Variational autoencoder for regime embedding
109. Self-supervised contrastive learning on bars
110. Reinforcement Learning: DQN for discrete actions
111. RL: PPO for continuous position sizing
112. RL: SAC for risk-adjusted portfolio
113. Model-based RL with world model
114. Imitation learning from manual rules
115. Bayesian optimization for hyperparameter tuning
116. Multi-task learning: vol + direction + duration
117. Distillation: large transformer → small HGBR
118. Online learning: River-based incremental
119. Active learning: query high-uncertainty samples
120. Federated learning across multiple symbols

## A5. ポジション管理 (121-150)

121. Pyramid entry (3-layer with confirmation)
122. Reverse pyramid (decreasing size)
123. Scale-out at multiple TPs (33%/33%/34%)
124. Time-based partial exit (50% after 12H)
125. Volatility-based dynamic sizing (Kelly with vol haircut)
126. Confidence-weighted sizing (linear in conf)
127. Confidence-weighted sizing (log-linear)
128. Risk parity across multiple signals
129. ATR stop with multi-timeframe confirmation
130. Chandelier exit (highest high - ATR×3)
131. Parabolic SAR for trail
132. Donchian channel break for stops
133. Volume-weighted exit (slow sell into liquidity)
134. Iceberg orders to reduce slippage
135. TWAP entry over 1H window
136. VWAP entry around mid-price
137. POV (% of volume) execution
138. Sniper entry: wait for spread compression
139. Limit order at expected mean reversion price
140. Bracket orders (entry + SL + TP simultaneous)
141. Break-even stop after 1R profit
142. Volatility scale stops (ATR multiple by regime)
143. Hidden stop (mental, not on book)
144. Iceberg + dark pool routing
145. Smart order router (best venue selection)
146. Position rebalancing on signal flip only (not periodic)
147. Capital allocation across N strategies (1/N)
148. Risk budget allocation (vol-based)
149. Hierarchical risk parity (HRP)
150. Mean-variance optimization with shrinkage

## A6. リスク/コスト最適化 (151-180)

151. Funding cost prediction → trade timing
152. Funding-aware position rotation (rotate to negative-funding asset)
153. Spot/perp arbitrage (long spot, short perp = funding capture)
154. Cross-exchange arb (Binance vs Bybit funding diff)
155. Stablecoin yield (USDC/DAI lending) for idle capital
156. Hedge directional exposure with options (long put)
157. Volatility hedge (long VIX equivalent)
158. Tail risk hedge (OTM puts on every position)
159. Correlation hedge (anti-correlated asset)
160. Kelly criterion with fractional (0.25 Kelly)
161. Drawdown-based capital reduction
162. Volatility-targeting (15% annual vol)
163. Pre-event size reduction (FOMC, CPI)
164. Calendar-aware exposure (weekend reduction)
165. Funding rate spike avoidance (> 0.1% per 8H)
166. Liquidity-adjusted execution (depth check)
167. Slippage modeling (square-root impact)
168. Fee tier optimization (volume rebates)
169. Maker rebate maximization (passive orders)
170. Taker fee minimization (aggressive only when necessary)
171. Cross-margin to reduce capital requirement
172. Isolated margin to limit per-position loss
173. Sub-account separation by strategy
174. Leverage cap by recent volatility
175. Position limit by 24h ADV (avg daily volume)
176. Concentration limit (max 20% in single asset)
177. Drawdown circuit breaker (-15% → halt)
178. Daily P&L stop (-5% → close all)
179. Consecutive loss stop (5 losses → cool down)
180. Recovery protocol after drawdown

## A7. インフラ/実装 (181-200)

181. Lighter.xyz API integration
182. Hyperliquid API integration
183. Multi-exchange order routing
184. WebSocket-based real-time signal generation
185. Redis cache for signal state
186. Postgres for trade history
187. Grafana dashboard for live monitoring
188. Telegram alerts for entry/exit
189. Discord webhook for daily summary
190. Email reports for weekly performance
191. Cloud deployment (AWS/GCP)
192. Docker containerization
193. Kubernetes orchestration for multi-strategy
194. CI/CD with backtest validation
195. Blue-green deployment for strategy updates
196. Shadow mode (new strategy runs in parallel without trading)
197. A/B testing framework
198. Logging with structured events (JSON)
199. Distributed tracing for execution latency
200. Disaster recovery: state backup + replay

---

# Part B: 100+ システムアーキテクチャ候補

## B1. 既存パラダイムの変種 (1-25)

1. **RACM-Spot**: spot only, no leverage, no funding
2. **RACM-Neutral**: long winner + short loser equal weight
3. **RACM-Lite**: 1.0x leverage, longer hold (daily rebalancing)
4. **RACM-Adaptive**: Kelly + alpha decay tracker
5. **RACM-Multi-TF**: 1H + 4H + 1D ensemble
6. **swin-v25-Multi-Asset**: BTC + ETH + SOL parallel
7. **swin-v25-Confidence-Tier**: 3 tiers (low/med/high) different sizing
8. **swin-v26-Multi-Channel**: Crypto Daily + Strategic + OBI
9. **Hybrid Continuous + Discrete**: RACM base + swin overlay
10. **Pure HGBR**: ML predicts → HGBR-driven sizing → ATR stop
11. **Pure ML Pipeline**: feature engineering → multiple models → ensemble
12. **Factor Portfolio**: 5 factors equal weight, monthly rebalance
13. **Risk Parity Portfolio**: vol-inverse weight
14. **HRP (Hierarchical Risk Parity)**: cluster + recursive bisection
15. **Black-Litterman**: prior + ML view → posterior weights
16. **Mean-Variance with Shrinkage**: covariance estimation
17. **Markowitz Efficient Frontier**: rebalanced monthly
18. **CPPI (Constant Proportion Portfolio Insurance)**: floor protection
19. **Dynamic Hedging**: Greeks-based delta neutral
20. **Pairs Trading**: cointegrated asset pairs
21. **Statistical Arbitrage**: mean-reverting basket
22. **Index Arbitrage**: BTC vs basket of alts
23. **Calendar Spread**: front month vs back month perp
24. **Triangular Arbitrage**: BTC/ETH/SOL relative pricing
25. **Cross-Exchange Arbitrage**: same asset, different venues

## B2. ML中心アーキテクチャ (26-50)

26. **Single ML Predictor**: HGBR → direction → trade
27. **Multi-Horizon ML**: 1h/4h/24h predictions ensemble
28. **Ensemble Voting**: HGBR + Ridge + LightGBM majority
29. **Stacked Generalization**: meta-learner on base predictions
30. **ML + Rule Hybrid**: ML for entry, rules for exit
31. **Rule + ML Hybrid**: Rules for entry, ML for sizing
32. **Meta-Labeling Pipeline**: primary signal → ML filter
33. **Triple Barrier ML**: classify hit pattern
34. **Direction + Magnitude ML**: two models, gated
35. **Confidence Gated Trading**: trade only if conf > threshold
36. **Online Learning System**: River incremental updates
37. **Active Learning Loop**: human-in-the-loop refinement
38. **Self-Supervised Pretraining**: large unsupervised + fine-tune
39. **Contrastive Learning**: regime embeddings
40. **Variational Autoencoder Regime**: latent space clustering
41. **GAN Synthetic Data**: augment minority regime samples
42. **Transformer Multi-Asset**: cross-attention between coins
43. **Graph Neural Network**: asset network as graph
44. **Reinforcement Learning DQN**: discrete actions
45. **RL PPO**: continuous position sizing
46. **RL SAC**: maximum entropy exploration
47. **Model-Based RL**: learn dynamics + plan
48. **Imitation Learning**: copy expert (manual rules)
49. **Distillation**: large model → small efficient model
50. **Federated Learning**: train on multiple symbols

## B3. マルチストラテジー (51-75)

51. **Strategy of Strategies**: meta-allocator over RACM + swin + others
52. **Capital Bucket Allocation**: 30% RACM + 30% swin + 40% cash
53. **Vol-Inverse Allocation**: rebalance by recent vol
54. **Sharpe-Maximizing Allocation**: optimize portfolio Sharpe
55. **Regime-Switching Allocation**: different allocations per regime
56. **Equal Risk Contribution**: each strategy contributes same risk
57. **Risk Budgeting**: cap each strategy's worst-case loss
58. **Adaptive Allocation**: increase winners, decrease losers
59. **Bandit Allocation**: multi-armed bandit explore/exploit
60. **Bayesian Allocation**: posterior over strategy effectiveness
61. **Strategy Tournament**: rotate based on recent performance
62. **Lifecycle Allocation**: new strategies start small, grow with proof
63. **Veto System**: any strategy can veto position (risk override)
64. **Confirmation System**: require 2+ strategies agree
65. **Diversification Bonus**: prefer uncorrelated strategies
66. **Cluster-Based Allocation**: group similar, diversify across groups
67. **Time-Diversified**: morning vs evening different strategies
68. **Asset-Diversified**: BTC strategies + ETH strategies + ALT
69. **Frequency-Diversified**: HFT + intraday + daily
70. **Style-Diversified**: trend + mean-reversion + carry
71. **Source-Diversified**: technical + ML + fundamental
72. **Coverage Optimizer**: ensure no signal source duplicated
73. **Survivorship Detector**: auto-remove decaying strategies
74. **Replicator Dynamics**: evolutionary strategy selection
75. **Genetic Algorithm**: breed best strategies

## B4. 新パラダイム (76-100)

76. **Event-Driven Trading**: trade only around CPI/FOMC/halving
77. **News-Driven Trading**: NLP on headlines → direction
78. **Sentiment-Driven**: aggregated social/news sentiment
79. **Order Flow Toxicity**: VPIN-based avoidance
80. **Liquidity Provision**: market making with spread capture
81. **Funding Rate Yield Farm**: long spot, short perp net carry
82. **Cross-Exchange Funding Arb**: long high-funding, short low-funding
83. **Statistical Market Making**: provide liquidity at fair price
84. **High-Frequency Stat Arb**: sub-second pattern detection
85. **Latency Arbitrage**: faster data feed exploitation (likely impossible retail)
86. **Information-Driven Bars**: dollar/volume/imbalance bars
87. **Microstructure Alpha**: tick-level patterns
88. **Behavioral Edge**: exploit retail FOMO/panic
89. **Whale Tracking**: follow large wallets
90. **Smart Money Following**: track perpetual whale positions
91. **Carry + Momentum + Value**: classic 3-factor
92. **CTA-Style Trend Following**: long-term momentum across assets
93. **Volatility Risk Premium**: short vol when high, long when low
94. **Convexity Strategy**: structured payoffs (synthetic options)
95. **Tail Risk Strategy**: convex bets on rare events
96. **Income Strategy**: yield from staking + lending + funding
97. **Index Replication**: track BTC dominance / total mcap
98. **Synthetic Cash**: deliver risk-free rate via funding capture
99. **Liquidity Mining**: provide LP, earn fees + rewards
100. **MEV Capture**: arbitrage opportunities on DEX
101. **Token Launch Sniping**: early entry on new listings (high risk)
102. **Airdrop Farming**: position for protocol airdrops
103. **Governance Token Trading**: vote-buying arbitrage
104. **Restaking Yield Stack**: layered staking rewards

---

# Part C: トップ5 候補 (有益度評価)

評価基準:
- **F**: Funding コスト耐性 (perp で動くか)
- **D**: データ可用性 (今すぐ実装可能か)
- **A**: Alpha の証拠 (理論または既存実証)
- **R**: 実装リスク (低いほど良い)
- **U**: 既検証重複度 (低いほど良い = 新しい)

## ★1: swin_v25 Multi-Asset 拡張

**現在検証中、継続候補として残す。**

- 構造: 個別トレード型 + ATR stop + HGBR ML
- 拡張: BTC + ETH + SOL の3資産で並列運用
- 期待: 65% WR x 3資産 = ポジション分散 + サンプル増加
- F:◎ D:◎ A:◎ R:○ U:○
- **理由**: 既に動いていることが確認済み (Window 1: +34%, Window 2: +17%)。3資産化で fold 数が3倍 → 統計的有意性向上。

## ★2: Funding Rate Yield Farm (Spot-Perp Carry)

- 構造: BTC spot 買い + BTC perp 同サイズ売り = delta neutral
- 収益: perp funding rate (年100%) を全額獲得
- リスク: spot/perp basis 変動、ステーブル価格リスク
- F:◎ (funding を稼ぐ側) D:◎ A:◎ (年100% 期待) R:◎ U:◎
- **理由**: BTC perp の年100% funding は **構造的エッジ** (88% 時間ロングが払う)。delta neutral で価格リスクなし。Lighter で資本効率高い (cross margin)。

## ★3: Multi-Strategy Capital Allocator

- 構造: RACM-Spot (15%) + swin_v25 (40%) + Funding Farm (35%) + Cash (10%)
- 動的: 直近 30日 Sharpe で月次再配分
- リスク管理: 全戦略合計 portfolio MDD -15% で halt
- F:◎ D:◎ A:◎ R:○ U:◎
- **理由**: 単一戦略依存を解消。ファクター減衰に強い。各戦略が異なる alpha 源 (timing, ML, carry) で低相関。

## ★4: HGBR Confidence-Gated Trading System

- 構造: HGBR で direction + confidence → 高信頼のみトレード
- サイジング: position = base × confidence (linear)
- ストップ: ATR × 1.5 タイト + 早期 break-even
- F:○ D:◎ A:◎ R:○ U:○
- **理由**: swin_v25 の本質を抽出。signal → confidence → size → stop の純粋な ML pipeline。実装シンプル。

## ★5: Lighter.xyz Native Microstructure Strategy

- 構造: 0xArchive で Lighter 板/約定/funding を取得 → MS signal 構築 → Lighter 上で執行
- アルファ源: Lighter 固有の流動性パターン (DEX で CEX とは異なる)
- 期待: 0手数料 + 取引所固有エッジ
- F:○ (Lighter funding 構造未確認) D:△ (新データソース) A:○ R:△ U:◎
- **理由**: Tardis にない Lighter データを使った独自エッジ。実取引所と検証データが一致。但しデータパイプライン構築が必要。

---

# 採用判断

| Rank | 戦略 | 即実装可? | 期待リターン | リスク |
|------|------|---------|-----------|--------|
| ★1 | swin_v25 Multi-Asset | YES | +30-50% 年 | Low |
| ★2 | Funding Farm | YES | +50-80% 年 | Low (basis risk) |
| ★3 | Multi-Strategy | YES (上3つ統合) | +60-100% 年 | Low-Med |
| ★4 | HGBR Confidence | YES (swin_v25 の subset) | +30-50% 年 | Low |
| ★5 | Lighter Native | NO (要 data pipeline) | 未知 | Med-High |

## 推奨実装順序

1. **★2 Funding Farm** をまず実装 (構造的エッジ、即収益)
2. **★1 swin_v25 Multi-Asset** を完成 (既に動いている)
3. **★3 Multi-Strategy** で 1+2 を統合
4. **★5 Lighter Native** をデータ検証 (Lighter funding 構造確認)
5. **★4 HGBR Confidence** は ★1 の subset として吸収

## 即実行案

**最初の3ヶ月**: Funding Farm + swin_v25 Multi-Asset の2本立て
- 期待: 年 +80-130%, MDD -10% 以内
- funding バグの教訓を活かした「funding を稼ぐ/funding 影響を最小化」の2軸
