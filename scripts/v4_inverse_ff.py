"""V4: Inverse Funding Farm (negative funding capture)
=======================================================
Funding rate が NEGATIVE な時:
- 通常 FF (spot long + perp short) は perp short が funding 払う → 損
- Inverse FF (spot short + perp long) は perp long が funding 受取 → 益

問題:
- spot を short するのは難しい (借りる必要)
- 代替案: spot 不要 perp only - directional risk 残る

軽量版:
- Negative funding 期間 (現在の市場) に LONG perp のみ保持
- 上昇期待 + funding 受取
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd

print('='*70,flush=True)
print('  V4: Inverse Funding Farm (negative funding periods)',flush=True)
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
            ob_frames.append(ohlc)
df=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
df=df.drop_duplicates('timestamp').set_index('timestamp')

n=len(df)
S=2760
fra=df['funding_rate'].fillna(0).shift(1).values
spot_price=df['close'].values
fra_ma_30d=pd.Series(fra).rolling(720,min_periods=168).mean().values
fra_annual=fra_ma_30d*365*24*100

# Stats
neg_periods=np.sum(fra_annual[S:]<0)
print(f'  Negative funding periods: {neg_periods}/{n-S} ({neg_periods/(n-S)*100:.1f}% of time)',flush=True)
neg_mean=np.mean(fra_annual[S:][fra_annual[S:]<0])
print(f'  Avg funding when negative: {neg_mean:.1f}%/yr',flush=True)

# ============================================================
# Bidirectional Funding Farm
# ============================================================
def run_bidirectional_ff(lev=1.5, fee_bps=2, slip_bps=2, rebal_days=7):
    """
    Standard FF when funding >0
    Inverse FF (perp long, no spot) when funding <0
    Note: spot short is hard, so just perp long with directional risk
    """
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=1.5;current_mode='neutral'  # 'long', 'short', 'neutral'
    cap_unit=1+1/lev
    e *= (1 - 2*(fr+sr)/cap_unit)
    rebalances=0

    for i in range(S,n):
        f30=fra_annual[i] if not np.isnan(fra_annual[i]) else 50

        # Decide mode
        if f30 > 50:
            target_mode='short_perp'  # standard FF
            target_lev=1.5
        elif f30 > 20:
            target_mode='short_perp'
            target_lev=1.0
        elif f30 > 0:
            target_mode='neutral'  # exit
            target_lev=0
        elif f30 > -10:
            target_mode='neutral'
        else:  # very negative funding
            target_mode='long_perp'  # inverse FF
            target_lev=1.0

        # Rebalance if mode/lev change
        if (i-S)%(rebal_days*24)==0 and (i-S)>0:
            if target_mode != current_mode or abs(target_lev-current_lev)>0.1:
                # Close current
                if current_mode != 'neutral':
                    e *= max(0.001, 1+perp_un/cap_unit)
                    e *= (1 - 2*(fr+sr)/cap_unit)
                # Open new
                if target_mode != 'neutral':
                    e *= (1 - 2*(fr+sr)/cap_unit)
                current_mode = target_mode
                current_lev = max(target_lev, 0.001)
                cap_unit = 1+1/current_lev
                perp_un = 0
                rebalances += 1

        if current_mode == 'neutral':
            eq[i] = e
            continue

        if i>0 and spot_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0

        if current_mode == 'short_perp':
            # Standard FF: spot long + perp short
            sp = sret * 1.0
            pp = -sret * 1.0  # perp short
            fi = fra[i] * 1.0  # short receives positive funding
        elif current_mode == 'long_perp':
            # Inverse FF: perp long only (no spot, directional)
            sp = 0
            pp = sret * 1.0  # perp long
            fi = -fra[i] * 1.0  # long pays positive funding (= receives negative)

        perp_un += pp + fi
        if perp_un < -1/current_lev * 0.5:
            e *= max(0.001, 1+perp_un/cap_unit)
            perp_un = 0
            e *= (1 - 2*(fr+sr)/cap_unit)
            continue

        bar_ret = (sp + pp + fi) / cap_unit
        e *= (1+bar_ret)
        eq[i] = e

    return eq, rebalances

def run_standard_ff(lev=1.5, fee_bps=2, slip_bps=2, rebal_days=7):
    """Standard bear-aware FF (exit on negative funding)"""
    eq=np.ones(n);e=1.0
    fr=fee_bps/10000;sr=slip_bps/10000
    perp_un=0.0;current_lev=1.5
    cap_unit=1+1/lev
    e *= (1 - 2*(fr+sr)/cap_unit)
    for i in range(S,n):
        f30=fra_annual[i] if not np.isnan(fra_annual[i]) else 50
        if f30 > 100: target=2.0
        elif f30 > 50: target=1.5
        elif f30 > 20: target=1.0
        elif f30 > 0: target=0.5
        else: target=0.001  # exit

        if (i-S)%(rebal_days*24)==0 and (i-S)>0 and abs(target-current_lev)>0.1:
            adj=abs(target-current_lev)/max(current_lev,0.1)*(fr+sr)*2
            cap_unit=1+1/max(current_lev,0.1)
            e *= (1-adj/cap_unit)
            current_lev=target
            perp_un=0

        cap_unit=1+1/max(current_lev,0.1)
        if i>0 and spot_price[i-1]>0:
            sret=spot_price[i]/spot_price[i-1]-1
        else:sret=0
        sp=sret*1.0;pp=-sret*1.0;fi=fra[i]*1.0
        perp_un += pp+fi
        if current_lev>0.01 and perp_un<-1/current_lev*0.5:
            e *= max(0.001,1+perp_un/cap_unit)
            perp_un=0
            e *= (1-2*(fr+sr)/cap_unit)
            continue
        bar_ret=(sp+pp+fi)/cap_unit
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

eq_std=run_standard_ff()
eq_bi,reb=run_bidirectional_ff()
ass,ms,ss=stats(eq_std)
ab,mb,sb=stats(eq_bi)

print(f'\n  {"Strategy":<35} {"Annual":>10} {"MDD":>8} {"Sharpe":>8}',flush=True)
print(f'  {"-"*60}',flush=True)
print(f'  {"Standard Bear-aware FF":<35} {ass:>+9.1f}% {ms:>+7.1f}% {ss:>7.2f}',flush=True)
print(f'  {f"Bidirectional FF ({reb} mode chgs)":<35} {ab:>+9.1f}% {mb:>+7.1f}% {sb:>7.2f}',flush=True)

# Year by year
print(f'\n  Year-by-year:',flush=True)
print(f'  {"Year":<6} {"Standard":>12} {"Bidirectional":>15}',flush=True)
for yr in range(2021,2026):
    iy=np.where(np.array([d.year==yr for d in df.index]))[0]
    iy_a=iy[iy>=S]
    if len(iy_a)<100:continue
    rs=(eq_std[iy_a[-1]]/eq_std[max(0,iy_a[0]-1)]-1)*100 if eq_std[max(0,iy_a[0]-1)]>0 else 0
    rb=(eq_bi[iy_a[-1]]/eq_bi[max(0,iy_a[0]-1)]-1)*100 if eq_bi[max(0,iy_a[0]-1)]>0 else 0
    print(f'  {yr:<6} {rs:>+11.1f}% {rb:>+14.1f}%',flush=True)

print('\nDone.',flush=True)
