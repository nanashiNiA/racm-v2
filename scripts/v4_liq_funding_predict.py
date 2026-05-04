"""V4: Liquidation → Funding Rate Prediction
=============================================
仮説: 清算カスケード後 → funding rate spike (squeeze)
検証:
1. 清算量 z-score と次8H funding rate の相関
2. 清算カスケード検知時に proactive にFFを leverage up
3. シャッフルテストで真のpredictive power 確認
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from scipy import stats as scistats

print('='*70,flush=True)
print('  V4: Liquidation → Funding Prediction',flush=True)
print('='*70,flush=True)

print('Loading...',flush=True)
cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
ob_frames=[]
for year in range(2021,2026):
    f=os.path.join(cache_dir,f'BTCUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(f):
        with open(f,'rb') as fp:
            ohlc,ticker,liq=pickle.load(fp)
        if ohlc is not None:
            ohlc['timestamp']=pd.to_datetime(ohlc['timestamp'],utc=True)
            if ticker is not None:
                ticker['timestamp']=pd.to_datetime(ticker['timestamp'],utc=True)
                ohlc=pd.merge_asof(ohlc.sort_values('timestamp'),
                    ticker[['timestamp','funding_rate']].sort_values('timestamp'),
                    on='timestamp',direction='backward',tolerance=pd.Timedelta('2H'))
            if liq is not None:
                liq['timestamp']=pd.to_datetime(liq['timestamp'],utc=True)
                ohlc=ohlc.merge(liq,on='timestamp',how='left')
                for c in ['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']:
                    if c in ohlc.columns:ohlc[c]=ohlc[c].fillna(0)
            ob_frames.append(ohlc)
df=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
df=df.drop_duplicates('timestamp').set_index('timestamp')
print(f'  Loaded: {len(df)} bars',flush=True)

# Compute features
df['fra_lag']=df['funding_rate'].fillna(0).shift(1)
df['total_liq']=df['long_liq_usd'].fillna(0)+df['short_liq_usd'].fillna(0)
df['liq_z']=((df['total_liq']-df['total_liq'].rolling(168,min_periods=48).mean())/
             (df['total_liq'].rolling(168,min_periods=48).std()+1e-10)).shift(1)

# Long liquidation = price dropped (longs liquidated)
df['long_liq_z']=((df['long_liq_usd'].fillna(0)-df['long_liq_usd'].rolling(168,min_periods=48).mean())/
                   (df['long_liq_usd'].rolling(168,min_periods=48).std()+1e-10)).shift(1)
df['short_liq_z']=((df['short_liq_usd'].fillna(0)-df['short_liq_usd'].rolling(168,min_periods=48).mean())/
                    (df['short_liq_usd'].rolling(168,min_periods=48).std()+1e-10)).shift(1)

# Future funding (target): mean of next 8 bars
df['future_funding_8h']=df['funding_rate'].rolling(8).mean().shift(-8)

valid=df.dropna(subset=['liq_z','future_funding_8h']).copy()
print(f'  Valid samples: {len(valid)}',flush=True)

# ============================================================
# Correlation analysis
# ============================================================
print('\n'+'='*70,flush=True)
print(' CORRELATION: Liquidation features vs Future Funding',flush=True)
print('='*70,flush=True)

features=['liq_z','long_liq_z','short_liq_z','liq_imbalance']
for feat in features:
    if feat in valid.columns:
        v=valid[[feat,'future_funding_8h']].dropna()
        if len(v)>100:
            r,p=scistats.pearsonr(v[feat],v['future_funding_8h'])
            print(f'  {feat:<20} r={r:+.4f}, p={p:.4f}',flush=True)

# ============================================================
# Spike events analysis
# ============================================================
print('\n'+'='*70,flush=True)
print(' SPIKE EVENTS: high liq → next funding',flush=True)
print('='*70,flush=True)

# When liq_z > 2 (cascade), what's the next 8H avg funding?
high_liq=valid[valid['liq_z']>2.0]
mod_liq=valid[(valid['liq_z']>1.0)&(valid['liq_z']<=2.0)]
normal=valid[valid['liq_z']<=1.0]

print(f'  liq_z>2 (cascade): {len(high_liq)} events',flush=True)
if len(high_liq)>10:
    avg=high_liq['future_funding_8h'].mean()
    print(f'    Next 8H funding avg: {avg*100:.5f}%/bar (annualized: {avg*3*365*100:+.1f}%/yr)',flush=True)

print(f'  liq_z 1-2 (moderate): {len(mod_liq)} events',flush=True)
if len(mod_liq)>10:
    avg=mod_liq['future_funding_8h'].mean()
    print(f'    Next 8H funding avg: {avg*100:.5f}%/bar (annualized: {avg*3*365*100:+.1f}%/yr)',flush=True)

print(f'  liq_z <=1 (normal): {len(normal)} events',flush=True)
if len(normal)>10:
    avg=normal['future_funding_8h'].mean()
    print(f'    Next 8H funding avg: {avg*100:.5f}%/bar (annualized: {avg*3*365*100:+.1f}%/yr)',flush=True)

# Direction-aware: long_liq spike → likely short squeeze setup → funding might rise
print('\n  Long Liquidation spike (longs forced out):',flush=True)
high_long_liq=valid[valid['long_liq_z']>2.0]
print(f'    {len(high_long_liq)} events',flush=True)
if len(high_long_liq)>10:
    avg=high_long_liq['future_funding_8h'].mean()
    print(f'    Next 8H funding: {avg*3*365*100:+.1f}%/yr',flush=True)

print('\n  Short Liquidation spike (shorts forced out):',flush=True)
high_short_liq=valid[valid['short_liq_z']>2.0]
print(f'    {len(high_short_liq)} events',flush=True)
if len(high_short_liq)>10:
    avg=high_short_liq['future_funding_8h'].mean()
    print(f'    Next 8H funding: {avg*3*365*100:+.1f}%/yr',flush=True)

# ============================================================
# Strategy: Liq-aware FF
# ============================================================
print('\n'+'='*70,flush=True)
print(' STRATEGY: Liquidation-aware FF Leverage',flush=True)
print('='*70,flush=True)
print('  Idea: After liquidation cascade, leverage up FF (anticipating funding spike)',flush=True)

# Run modified FF
n=len(df)
S=2760
fra=df['fra_lag'].fillna(0).values
liq_z=df['liq_z'].fillna(0).values
fra_ma_30d=pd.Series(fra).rolling(720,min_periods=168).mean().values
fra_annual=fra_ma_30d*365*24*100
spot_price=df['close'].values

def get_target_lev(i, use_liq_boost=False):
    f30=fra_annual[i] if not np.isnan(fra_annual[i]) else 50
    if f30 > 100: base = 2.0
    elif f30 > 50: base = 1.5
    elif f30 > 20: base = 1.0
    elif f30 > 0: base = 0.5
    else: return 0.001

    if use_liq_boost:
        # Boost if recent liquidation cascade (high funding likely coming)
        lz = liq_z[i] if not np.isnan(liq_z[i]) else 0
        if lz > 2.0:
            base *= 1.5  # 50% boost
        elif lz > 1.0:
            base *= 1.2
    return base

def run(use_liq_boost=False, fee_bps=2, slip_bps=2, rebal_days=7):
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=1.5
    for i in range(S,n):
        target=get_target_lev(i, use_liq_boost)
        if (i-S)%(rebal_days*24)==0 and (i-S)>0 and abs(target-current_lev)>0.1:
            adj=abs(target-current_lev)/max(current_lev,0.1)*(fr+sr)*2
            cap=1+1/max(current_lev,0.1)
            e *= (1-adj/cap)
            current_lev=target
            perp_un=0
        cap=1+1/max(current_lev,0.1)
        if i>0 and spot_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0
        sp=sret*1.0;pp=-sret*1.0  # use spot for both (simplified)
        fi=fra[i]*1.0
        perp_un += pp+fi
        if current_lev>0.01 and perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap)
            perp_un=0
            e *= (1-2*(fr+sr)/cap)
            continue
        bar_ret=(sp+pp+fi)/cap
        e *= (1+bar_ret)
        eq[i]=e
    return eq

def stats(eq):
    years=(n-S)/(365*24)
    annual=((eq[-1])**(1/years)-1)*100 if eq[-1]>0 else -100
    mdd=0;pk=1
    for i in range(S,n):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    daily=pd.Series(eq,index=df.index).resample('1D').last().pct_change().dropna()
    daily=daily[daily.index>=df.index[S]]
    sharpe=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else 0
    return annual,mdd*100,sharpe

eq_base=run(use_liq_boost=False)
eq_liq=run(use_liq_boost=True)
ab,mb,sb=stats(eq_base)
al,ml,sl=stats(eq_liq)

print(f'\n  {"Strategy":<30} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*55}',flush=True)
print(f'  {"FF baseline":<30} {ab:>+9.1f}% {mb:>+7.1f}% {sb:>7.2f}',flush=True)
print(f'  {"FF + liq-aware boost":<30} {al:>+9.1f}% {ml:>+7.1f}% {sl:>7.2f}',flush=True)

# Shuffle test
print('\n  Shuffle test on liq_z timing (200x):',flush=True)
liq_z_orig=liq_z.copy()
shuf_anns=[]
for seed in range(200):
    np.random.seed(seed+777777)
    lz=liq_z.copy()
    np.random.shuffle(lz[S:])
    liq_z[:]=lz
    eq_s=run(use_liq_boost=True)
    liq_z[:]=liq_z_orig
    a,_,_=stats(eq_s)
    shuf_anns.append(a)

p=np.mean([a>=al for a in shuf_anns])
print(f'    Real: {al:+.1f}%, Shuffle: {np.mean(shuf_anns):+.1f}%+/-{np.std(shuf_anns):.1f}%',flush=True)
print(f'    p-value: {p:.3f} {"PASS (liq has predictive value)" if p<0.05 else "FAIL"}',flush=True)

print('\nDone.',flush=True)
