"""V5 Final Combine: 全戦略・全資産の組み合わせ評価
=====================================================
全 backtest 結果を集約し、最適な組み合わせを発見する

入力:
- BTC swin_v25: 2022 +52.4%, 2024 +154%, 2025 +46.5% (確定)
- ETH swin_v25: ETH cache 完成後実行
- SOL swin_v25: SOL test 結果 (実行中)
- Funding Farm 1.5x: +5%/yr (低リスク baseline)

組み合わせ:
1. swin_v25 multi-asset 等重み
2. swin_v25 + Funding Farm
3. リスク許容度別 (Conservative/Moderate/Aggressive)
4. レバレッジスケーリング検討
"""
import sys,os
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')

print('='*70,flush=True)
print('  V5 FINAL: 全戦略組み合わせ評価',flush=True)
print('='*70,flush=True)

# ============================================================
# 入力 (実証データ + これから埋める)
# ============================================================
RESULTS={
    'BTC_swin_v25': {  # 完全実証済 (NET, funding cost 含む -5%)
        '2022': 52.4,
        '2024': 154.0,
        '2025': 46.5,
    },
    'ETH_swin_v25': {  # ETH cache 完成後に埋める
        '2022': None,
        '2024': None,
        '2025': None,
    },
    'SOL_swin_v25': {  # SOL test 結果 (進行中)
        '2022': None,
        '2024': None,
        '2025': None,
    },
    'Funding_Farm_1.5x': {  # honest 確定値
        '2022': 1.8,
        '2024': 7.3,
        '2025': 2.9,
    },
}

# ============================================================
# 動的に SOL/ETH 結果を log から抽出
# ============================================================
def parse_log(log_path, asset_label):
    """ログから Net Return を抽出"""
    if not os.path.exists(log_path):
        print(f'  {asset_label}: log not found',flush=True)
        return {}
    with open(log_path, encoding='utf-8') as f:
        log=f.read()
    import re
    results={}
    # Pattern: "swin_v25 SYMBOL OOS test: 2022\n...\n  Net: +X.X%"
    pattern=r'OOS test: (\d{4})\s.*?Net Return:\s+([+-]?\d+\.\d+)%'
    matches=re.findall(pattern, log, re.DOTALL)
    for year,ret in matches:
        results[year]=float(ret)
    return results

# Try to load SOL results
sol_results=parse_log('C:/Users/A701/AppData/Local/Temp/sol_test.log', 'SOL')
if sol_results:
    RESULTS['SOL_swin_v25'].update(sol_results)
    print(f'\n  SOL parsed from log: {sol_results}',flush=True)

# ============================================================
# 個別資産レポート
# ============================================================
print('\n'+'='*70,flush=True)
print(' 個別資産 swin_v25 結果',flush=True)
print('='*70,flush=True)

for asset,years in RESULTS.items():
    print(f'\n  {asset}:',flush=True)
    valid_years=[(y,r) for y,r in years.items() if r is not None]
    if not valid_years:
        print(f'    (no data yet)',flush=True)
        continue
    for y,r in valid_years:
        print(f'    {y}: {r:+.1f}%',flush=True)
    if len(valid_years)>=2:
        compound=1.0
        for y,r in valid_years:compound *= (1+r/100)
        cagr=compound**(1/len(valid_years))-1
        print(f'    Compound: {(compound-1)*100:+.1f}% / CAGR: {cagr*100:+.1f}%/yr',flush=True)

# ============================================================
# 組み合わせ評価
# ============================================================
def portfolio_return(weights, year_data):
    """weights: {asset: w}, year_data: {asset: {year: ret}}"""
    yearly={}
    years=set()
    for asset_data in year_data.values():
        for y,r in asset_data.items():
            if r is not None:years.add(y)
    for y in sorted(years):
        port_ret=0
        total_w=0
        for asset,w in weights.items():
            if y in year_data.get(asset,{}) and year_data[asset][y] is not None:
                port_ret += w * year_data[asset][y]
                total_w += w
        if total_w>0:
            yearly[y]=port_ret/total_w*1.0  # weighted average (assumes equal cap)
    return yearly

print('\n'+'='*70,flush=True)
print(' 組み合わせポートフォリオ',flush=True)
print('='*70,flush=True)

# Available assets (with data)
available=[a for a,years in RESULTS.items() if any(r is not None for r in years.values())]
print(f'\n  Available assets: {available}',flush=True)

# Define portfolio strategies
portfolios=[
    ('100% BTC swin', {'BTC_swin_v25':1.0}),
    ('100% Funding Farm', {'Funding_Farm_1.5x':1.0}),
    ('80% BTC swin + 20% FF', {'BTC_swin_v25':0.8,'Funding_Farm_1.5x':0.2}),
    ('50% BTC swin + 50% FF', {'BTC_swin_v25':0.5,'Funding_Farm_1.5x':0.5}),
    ('Multi-Asset swin (BTC+ETH+SOL equal)', {'BTC_swin_v25':1/3,'ETH_swin_v25':1/3,'SOL_swin_v25':1/3}),
    ('Hybrid: 50% Multi-Asset + 50% FF', {'BTC_swin_v25':1/6,'ETH_swin_v25':1/6,'SOL_swin_v25':1/6,'Funding_Farm_1.5x':0.5}),
]

print(f'\n  {"Portfolio":<45} {"Yearly Returns (avg)":>30}',flush=True)
print(f'  {"-"*75}',flush=True)
for label,w in portfolios:
    # Skip if assets not available
    available_w={a:wt for a,wt in w.items() if a in available}
    if not available_w:
        print(f'  {label:<45} (data not ready)',flush=True)
        continue
    total_w=sum(available_w.values())
    if total_w==0:continue
    # Normalize
    available_w={a:wt/total_w for a,wt in available_w.items()}
    yearly=portfolio_return(available_w, RESULTS)
    if not yearly:continue
    yearly_str=', '.join([f'{y}:{r:+.0f}%' for y,r in sorted(yearly.items())])
    avg=sum(yearly.values())/len(yearly)
    compound=1.0
    for r in yearly.values():compound *= (1+r/100)
    cagr=compound**(1/len(yearly))-1
    print(f'  {label:<45} {yearly_str}',flush=True)
    print(f'  {"":<45} avg {avg:+.1f}%/yr, CAGR {cagr*100:+.1f}%/yr',flush=True)

# ============================================================
# 教授要件 評価
# ============================================================
print('\n'+'='*70,flush=True)
print(' 教授要件 達成判定',flush=True)
print('='*70,flush=True)

best=None;best_cagr=0
for label,w in portfolios:
    available_w={a:wt for a,wt in w.items() if a in available}
    if not available_w:continue
    total_w=sum(available_w.values())
    if total_w==0:continue
    available_w={a:wt/total_w for a,wt in available_w.items()}
    yearly=portfolio_return(available_w, RESULTS)
    if not yearly:continue
    compound=1.0
    for r in yearly.values():compound *= (1+r/100)
    cagr=compound**(1/len(yearly))-1
    cagr_pct=cagr*100
    if cagr_pct>best_cagr:
        best_cagr=cagr_pct
        best=(label,available_w,cagr_pct)

if best:
    print(f'  最良 (現データで判定):',flush=True)
    print(f'    {best[0]}',flush=True)
    print(f'    Weights: {best[1]}',flush=True)
    print(f'    CAGR: {best[2]:+.1f}%/yr',flush=True)
    target_300=300
    if best[2]>=target_300:
        print(f'    ✓ 教授要件 {target_300}%/yr 達成',flush=True)
    else:
        gap=target_300-best[2]
        print(f'    ✗ 教授要件 {target_300}%/yr に届かず ({gap:.0f}%/yr 不足)',flush=True)
        # What leverage would be needed?
        if best[2]>0:
            lev_needed=(1+target_300/100)/(1+best[2]/100)
            print(f'    → 達成にはレバ {lev_needed:.1f}x 必要 (現状 1x の場合)',flush=True)
            print(f'    → ただしレバ↑ で MDD 比例増、清算リスク',flush=True)

print('\nDone.',flush=True)
