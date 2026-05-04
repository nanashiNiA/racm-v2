"""V4 swin_v25 徹底検証
========================
swin_v25 が動いている前提で、以下を厳密に検証:

1. リーク検証
   - shift +1, +2 でパフォーマンスがどう変わるか
   - feature engineering の lag-1 確認
   - target leakage (future_return が train に混入していないか)

2. シャッフル検証
   - 200x シャッフル: 信号タイミングがランダム以上か
   - block shuffle (30d): 時間構造保持シャッフル

3. funding バグ検証
   - swin_v25 は funding を計算していない → コスト過小評価の可能性
   - 各トレードに funding コストを正しく加算した場合のリターン

4. OOS 検証
   - 2024 結果を 2021-2023 で訓練したパラメータで再評価
   - 2025 で完全 OOS 確認

5. 過学習検証
   - train vs test の対称性
   - パラメータ感度（best_params が毎windowでどれくらい変動するか）
"""
import sys,os
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V4 swin_v25 徹底検証',flush=True)
print('='*70,flush=True)

# ============================================================
# まず swin_v25 のソースコードチェック
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 0: swin_v25 ソースコード レビュー',flush=True)
print('='*70,flush=True)

src_file='src/swin_v25.py'
with open(src_file,'r',encoding='utf-8') as f:
    src=f.read()

# Check 1: Funding cost handling
print('\n  Check A: Funding cost handling',flush=True)
if 'funding' in src.lower():
    funding_lines=[i for i,line in enumerate(src.split('\n'),1) if 'funding' in line.lower() and 'cost' in line.lower()]
    if funding_lines:
        print(f'    Found {len(funding_lines)} funding cost references',flush=True)
    else:
        print(f'    ⚠️ funding feature exists but NO funding COST in trade execution',flush=True)
        print(f'    → 実約定時の funding 払い/受取が計算されていない',flush=True)

# Check 2: Future leak in feature engineering
print('\n  Check B: Future data in features',flush=True)
shift_lines=[]
for i,line in enumerate(src.split('\n'),1):
    if 'shift(-' in line:
        shift_lines.append((i,line.strip()))
    if "shift(1)" in line and i<400:
        shift_lines.append((i,line.strip()))
print(f'    shift() usage:',flush=True)
for i,l in shift_lines[:8]:
    print(f'      L{i}: {l[:80]}',flush=True)
if any('shift(-' in l for _,l in shift_lines):
    print(f'    ⚠️ shift(-N) found = future data in features (CHECK CONTEXT)',flush=True)

# Check 3: Target construction (label leak)
print('\n  Check C: Target construction',flush=True)
target_lines=[i for i,line in enumerate(src.split('\n'),1) if 'future_return' in line or 'prepare_target' in line]
print(f'    Found {len(target_lines)} target-related lines',flush=True)
for i in target_lines[:5]:
    print(f'      L{i}: {src.split(chr(10))[i-1].strip()[:80]}',flush=True)

# Check 4: Walk-forward purging
print('\n  Check D: Walk-forward purging',flush=True)
purge_lines=[i for i,line in enumerate(src.split('\n'),1) if 'purge' in line.lower() or 'embargo' in line.lower()]
if purge_lines:
    print(f'    Found {len(purge_lines)} purge/embargo references',flush=True)
else:
    print(f'    ⚠️ NO purge/embargo found → train/test boundary leak risk',flush=True)
    print(f'    → horizon=6-8 bars の future_return が train の最後 6-8 bar に含まれる可能性',flush=True)

# Check 5: Cost model
print('\n  Check E: Cost model',flush=True)
cost_keywords=['HYPERLIQUID_MAKER_FEE','HYPERLIQUID_TAKER_FEE','SLIPPAGE','total_entry_cost','total_exit_cost']
for kw in cost_keywords:
    count=src.count(kw)
    if count>0:
        print(f'    {kw}: {count} usages',flush=True)

# Check 6: ML target alignment
print('\n  Check F: ML target alignment',flush=True)
print('    prepare_target uses: close.shift(-horizon) / open.shift(-horizon+1) - 1',flush=True)
print('    horizon=6 → uses bar i+6 close vs bar i+1 open',flush=True)
print('    Potential issue: 訓練で使う future_return は test 開始 horizon bars 前まで含まれる',flush=True)
print('    → train_end - test_start gap が必要 (purging)',flush=True)

# ============================================================
# Wait for backtest
# ============================================================
print('\n'+'='*70,flush=True)
print(' STEP 1-5: バックテスト完了後に実行',flush=True)
print('='*70,flush=True)
print('  swin_v25 1H/2024 が完了するまで待機中',flush=True)
print('  完了後に以下を実施:',flush=True)
print('    1. リーク検証: shift+1, +2 with re-run',flush=True)
print('    2. シャッフル検証: 200x signal shuffle',flush=True)
print('    3. funding コスト追加検証: 各トレードに funding 加算',flush=True)
print('    4. OOS 検証: 2025 完全 OOS',flush=True)
print('    5. 過学習検証: best_params の安定性',flush=True)

print('\n  Source check 完了。バックテスト完了を待ちます。',flush=True)
print('Done.',flush=True)
