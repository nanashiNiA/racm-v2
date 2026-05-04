"""V3: 資産最大化の全方向探索
================================
MS3+MS4がBear保護を証明した。次は「資産をさらに大きく増やす」方法を探る。

探索方向:
1) MS逆活用: 低スプレッド+低清算 = 安全期間 → ブーストレバレッジ
2) MSショート: 高スプレッド+清算連鎖時にショートポジション
3) LS + Value複合: Momentum + Mean-Reversion の二刀流アルファ
4) 動的レバレッジ: MS信号でレバ1.5-5.0xを切り替え
5) フルモデル: 上記の最良組み合わせ

全てWF + Shuffle + Leak checkで検証。
"""
import sys,os,pickle
sys.path.insert(0,'C:/Users/A701/Documents/nia/racm-v2')
sys.path.insert(0,'C:/Users/A701/Documents/nia/prediction_model_project')
import warnings;warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from v2core.data_loader import build_common_1h,load_6assets_8h,load_6assets_1h
from src.racm_core import RACMLS

print('='*70,flush=True)
print('  V3: MAXIMIZE RETURNS - 全方向探索',flush=True)
print('='*70,flush=True)

# ============================================================
# DATA LOADING
# ============================================================
print('Loading...',flush=True)
h=build_common_1h();nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
S=2760;fra=np.roll(h['funding'].fillna(0).values,1)
adx=h['adx'].values;ma110=h['ma110'].values;ma20=h['ma20'].values
skew_a=h['skew_30d'].values;dd_a=h['dd'].values;vol_30d=h['vol_30d'].values

a8=load_6assets_8h();c8=list(a8.values())[0].index
for df in a8.values():c8=c8.intersection(df.index)
ar8={a:a8[a].loc[c8,'return'].values for a in a8}
lp8,_,_=RACMLS.compute_pnl_8h(ar8,[60,90],len(c8))
lp1h=RACMLS.map_8h_to_1h(lp8,c8,idx_h,nn)

# Value (anti-momentum) LS
lb_val=30*3  # 30d lookback for value
am_val={n:np.roll(pd.Series(r).rolling(lb_val,min_periods=lb_val//3).sum().values,1) for n,r in ar8.items()}
av_val={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in ar8.items()}
val_pnl_8h=np.zeros(len(c8))
for i in range(200,len(c8)):
    ms=[(am_val[n][i]/(av_val[n][i]+1e-10),n) for n in ar8 if not np.isnan(am_val[n][i])]
    if len(ms)<3:continue
    ms.sort(key=lambda x:x[0],reverse=True)
    short_a=ms[0][1];long_a=ms[-1][1]
    val_pnl_8h[i]=(ar8[long_a][i]-ar8[short_a][i])/2
val_pnl_1h=RACMLS.map_8h_to_1h(val_pnl_8h,c8,idx_h,nn)

# V2 regime
def build_bp_v2():
    bp=np.ones(nn);lc=0;pb=1.0
    for i in range(S,nn):
        tv=np.mean(vol_30d[max(S,i-4320):i-1]) if i>S+720 else 0.80
        cv=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else tv
        vs=np.clip(tv/(cv+1e-10),0.3,1.5)
        if dd_a[i-1]<-0.15:vs=min(vs,0.3)
        elif dd_a[i-1]<-0.08:vs=min(vs,0.5)
        if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:vs*=0.7
        dl=ds=0
        if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:dl+=1
        if not np.isnan(ma20[i-1]) and price[i-1]<ma20[i-1]:ds+=1
        if not np.isnan(skew_a[i-1]) and skew_a[i-1]<-0.5:dl+=1;ds+=1
        if dd_a[i-1]<-0.12:dl+=1;ds+=1
        dc=dl*0.3+ds*0.7
        if dc>=1.5:vs=min(vs,0.2)
        if abs(vs-pb)>0.2:
            if dc>=1.5:bp[i]=vs;lc=i;pb=vs
            elif(not np.isnan(adx[i-1]) and adx[i-1]>20)and(i-lc>=6):bp[i]=vs;lc=i;pb=vs
            else:bp[i]=pb
        else:bp[i]=vs;pb=vs
    return bp
bp_v2=build_bp_v2()

# Danger composite
def get_dc(i):
    dl=ds=0
    if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:dl+=1
    if not np.isnan(ma20[i-1]) and price[i-1]<ma20[i-1]:ds+=1
    if not np.isnan(skew_a[i-1]) and skew_a[i-1]<-0.5:dl+=1;ds+=1
    if dd_a[i-1]<-0.12:dl+=1;ds+=1
    return dl*0.3+ds*0.7

# Orderbook data
print('Loading orderbook...',flush=True)
cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/src/data_cache'
ob_frames=[]
for year in range(2021,2026):
    f=os.path.join(cache_dir,f'BTCUSDT_{year}0101_{year}1231_1H.pkl')
    if os.path.exists(f):
        with open(f,'rb') as fp:
            ohlc,ticker,liq=pickle.load(fp)
        if ohlc is not None:
            ohlc['timestamp']=pd.to_datetime(ohlc['timestamp'],utc=True)
            if liq is not None:
                liq['timestamp']=pd.to_datetime(liq['timestamp'],utc=True)
                ohlc=ohlc.merge(liq,on='timestamp',how='left')
                ohlc[['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']]=\
                    ohlc[['long_liq_usd','short_liq_usd','liq_count','liq_imbalance']].fillna(0)
            ob_frames.append(ohlc)

ob_all=pd.concat(ob_frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ob_all=ob_all.drop_duplicates('timestamp').set_index('timestamp')
idx_h_utc=idx_h.tz_localize('UTC') if idx_h.tz is None else idx_h
ob_aligned=ob_all.reindex(idx_h_utc,method='nearest',tolerance=pd.Timedelta('2H'))

spread_pct_raw=ob_aligned['spread_pct'].fillna(0).values
long_liq_raw=ob_aligned['long_liq_usd'].fillna(0).values if 'long_liq_usd' in ob_aligned.columns else np.zeros(nn)
short_liq_raw=ob_aligned['short_liq_usd'].fillna(0).values if 'short_liq_usd' in ob_aligned.columns else np.zeros(nn)
total_liq_raw=long_liq_raw+short_liq_raw
liq_imb_raw=ob_aligned['liq_imbalance'].fillna(0).values if 'liq_imbalance' in ob_aligned.columns else np.zeros(nn)

# MS z-scores
sp_ma=pd.Series(spread_pct_raw).rolling(168,min_periods=48).mean().values
sp_std=pd.Series(spread_pct_raw).rolling(168,min_periods=48).std().values
spread_z=np.zeros(nn)
for i in range(S,nn):
    if not np.isnan(sp_std[i-1]) and sp_std[i-1]>1e-6:
        spread_z[i]=(spread_pct_raw[i-1]-sp_ma[i-1])/(sp_std[i-1]+1e-10)

liq_ma=pd.Series(total_liq_raw).rolling(168,min_periods=48).mean().values
liq_std=pd.Series(total_liq_raw).rolling(168,min_periods=48).std().values
liq_z=np.zeros(nn)
for i in range(S,nn):
    if not np.isnan(liq_std[i-1]) and liq_std[i-1]>1e-6:
        liq_z[i]=(total_liq_raw[i-1]-liq_ma[i-1])/(liq_std[i-1]+1e-10)

print('Data ready.',flush=True)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    ts=cur+pd.DateOffset(months=3);te=ts+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tm=np.array([(d>=ts and d<=te) for d in idx_h])
    if tm.sum()>=20:folds.append(np.where(tm)[0])
    cur+=pd.DateOffset(months=1)

def eval_strat(eq):
    wf=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf)*12 if wf else 0;win=sum(1 for r in wf if r>0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(0,i22[0]-1)]-1)*100 if len(i22)>100 else -999
    i25=np.where(np.array([d.year==2025 for d in idx_h]))[0]
    o25=(eq[i25[-1]]/eq[max(0,i25[0]-1)]-1)*100 if len(i25)>100 else -999
    eq_s=pd.Series(eq,index=idx_h);dr=eq_s.resample('1D').last().pct_change().dropna()*100
    dr=dr[dr.index>=idx_h[S]];dw=int((dr>0).sum());dl=int((dr<0).sum())
    mdd=0;pk=1
    for i in range(S,nn):pk=max(pk,eq[i]);dd=(eq[i]-pk)/pk;mdd=min(mdd,dd)
    yearly={}
    for yr in range(2021,2026):
        iy=np.where(np.array([d.year==yr for d in idx_h]))[0]
        if len(iy)>100:yearly[yr]=(eq[iy[-1]]/eq[max(0,iy[0]-1)]-1)*100
    return wfs,win,len(wf),b22,o25,dw/(dw+dl)*100 if dw+dl>0 else 50,mdd*100,yearly

def print_result(label, eq, tag=''):
    wfs,win,nf,b22,o25,dw,mdd,yr=eval_strat(eq)
    final=eq[-1]
    print(f'  {label:<42} {wfs:>4.0f}% {win:>2}/{nf} {b22:>+5.0f}% {o25:>+5.0f}% {mdd:>+5.1f}% {final:>10.1f}x{tag}',flush=True)
    return wfs,b22,o25,mdd,final,yr

# ============================================================
# BASELINE: V2 + MS3+MS4 (confirmed)
# ============================================================
def run_strategy(use_ms_protect=True, use_ms_boost=False, use_short=False,
                 use_value=False, value_weight=0.0,
                 base_lev=3.0, boost_lev=4.0, short_size=-0.5,
                 dd_levels=(-0.15,-0.22,-0.30)):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp=bp_v2[i]
        dc=get_dc(i)

        if bp<0.5:lw=0.95
        elif bp<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)

        # Base PnL: momentum LS
        pnl_mom=dw*ret[i]*bp+lw*lp1h[i]

        # Value LS blend
        if use_value and value_weight>0:
            pnl_val=dw*ret[i]*bp+lw*val_pnl_1h[i]
            pnl_base=pnl_mom*(1-value_weight)+pnl_val*value_weight
        else:
            pnl_base=pnl_mom

        # MS protection (proven)
        ms_mult=1.0
        if use_ms_protect:
            sz=spread_z[i]
            if sz>2.0: ms_mult*=0.5
            elif sz>1.0: ms_mult*=0.8
            lz=liq_z[i]
            if lz>3.0: ms_mult*=0.3
            elif lz>2.0: ms_mult*=0.6

        # MS boost: safe conditions → increase leverage
        lev=base_lev
        if use_ms_boost:
            sz=spread_z[i];lz=liq_z[i]
            # Low spread + low liquidation = calm market → boost
            if sz<-0.5 and lz<0.5 and dc<0.5:
                lev=boost_lev
            elif sz<0.0 and lz<1.0 and dc<1.0:
                lev=base_lev+0.5

        pnl=pnl_base*ms_mult*lev

        # MS Short: during crisis, take short position
        if use_short:
            sz=spread_z[i];lz=liq_z[i]
            # Conditions for short: high spread + high liq + MA below + negative momentum
            if (sz>2.0 and lz>2.0 and dc>=1.5 and
                not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]):
                # Short replaces long position
                short_pnl=ret[i]*short_size  # short_size is negative
                pnl=short_pnl*abs(lev)
            elif (sz>1.5 and dc>=1.5 and
                  not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]*0.95):
                # Mild short (MA well below)
                short_pnl=ret[i]*(short_size*0.5)
                pnl=short_pnl*abs(lev)

        # Funding
        pnl+=fra[i]*abs(lev*ms_mult)

        # DD control
        dd=(e-pk)/pk if pk>0 else 0
        if dd<dd_levels[2]:pnl*=0.1
        elif dd<dd_levels[1]:pnl*=0.4
        elif dd<dd_levels[0]:pnl*=0.7

        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

# ============================================================
# TESTS
# ============================================================
hdr=f'  {"Model":<42} {"WFS":>5} {"Win":>5} {"B22":>6} {"OOS":>6} {"MDD":>6} {"Final":>10}'
print('\n'+' DIRECTION 1: MS BOOST (安全期間でレバ増) '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*82,flush=True)

print_result('V2 Baseline (lev=3.0)', run_strategy(use_ms_protect=False), ' <--')
print_result('V2+MS34 Protect (confirmed)', run_strategy())
print_result('1a) +MS Boost 4.0x', run_strategy(use_ms_boost=True, boost_lev=4.0))
print_result('1b) +MS Boost 4.5x', run_strategy(use_ms_boost=True, boost_lev=4.5))
print_result('1c) +MS Boost 5.0x', run_strategy(use_ms_boost=True, boost_lev=5.0))
print_result('1d) Base 3.5x + Boost 5.0x', run_strategy(use_ms_boost=True, base_lev=3.5, boost_lev=5.0))
print_result('1e) Base 2.5x + Boost 4.5x', run_strategy(use_ms_boost=True, base_lev=2.5, boost_lev=4.5))

print('\n'+' DIRECTION 2: MS SHORT (危機時にショート) '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*82,flush=True)

print_result('V2+MS34 Protect (base)', run_strategy())
print_result('2a) +Short -0.3x', run_strategy(use_short=True, short_size=-0.3))
print_result('2b) +Short -0.5x', run_strategy(use_short=True, short_size=-0.5))
print_result('2c) +Short -1.0x', run_strategy(use_short=True, short_size=-1.0))
print_result('2d) +Short -0.5x + Boost 4.0x', run_strategy(use_short=True, short_size=-0.5, use_ms_boost=True, boost_lev=4.0))

print('\n'+' DIRECTION 3: VALUE + MOMENTUM 二刀流 '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*82,flush=True)

print_result('V2+MS34 (momentum only)', run_strategy())
print_result('3a) 90% Mom + 10% Value', run_strategy(use_value=True, value_weight=0.10))
print_result('3b) 80% Mom + 20% Value', run_strategy(use_value=True, value_weight=0.20))
print_result('3c) 70% Mom + 30% Value', run_strategy(use_value=True, value_weight=0.30))
print_result('3d) 50% Mom + 50% Value', run_strategy(use_value=True, value_weight=0.50))

print('\n'+' DIRECTION 4: 動的レバレッジ (MS+Vol) '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*82,flush=True)

# Instead of fixed leverage, use MS signals + vol to dynamically set leverage
def run_dynamic_lev(lev_range=(1.5, 5.0), use_short=False, short_size=-0.5):
    eq=np.ones(nn);e=1.0;pk=1.0
    for i in range(S,nn):
        v=vol_30d[i-1] if not np.isnan(vol_30d[i-1]) else 0.80
        bp=bp_v2[i];dc=get_dc(i)

        if bp<0.5:lw=0.95
        elif bp<0.8:lw=0.90
        elif v>1.0:lw=0.90
        elif v>0.50:lw=0.80
        else:lw=0.70
        dw=max(0,1-lw)
        pnl_base=dw*ret[i]*bp+lw*lp1h[i]

        # Dynamic leverage based on safety score
        sz=spread_z[i];lz=liq_z[i]
        safety=1.0
        # Danger signals → reduce
        if sz>2.0: safety-=0.3
        elif sz>1.0: safety-=0.15
        if lz>3.0: safety-=0.3
        elif lz>2.0: safety-=0.15
        if dc>=1.5: safety-=0.2
        elif dc>=1.0: safety-=0.1
        # Safety signals → increase
        if sz<-0.5 and lz<0.5: safety+=0.2
        if v<0.50: safety+=0.1  # low vol → safe
        safety=np.clip(safety, 0.0, 1.5)

        # Map safety score to leverage
        lev_min,lev_max=lev_range
        lev=lev_min+(lev_max-lev_min)*safety

        # Short during extreme danger
        if use_short and sz>2.0 and lz>2.0 and dc>=1.5:
            if not np.isnan(ma110[i-1]) and price[i-1]<ma110[i-1]:
                pnl_base=ret[i]*short_size
                lev=abs(lev)

        pnl=pnl_base*lev
        pnl+=fra[i]*abs(lev)

        dd=(e-pk)/pk if pk>0 else 0
        if dd<-0.30:pnl*=0.1
        elif dd<-0.22:pnl*=0.4
        elif dd<-0.15:pnl*=0.7

        e*=(1+pnl);eq[i]=e;pk=max(pk,e)
    return eq

print_result('V2+MS34 Fixed 3.0x (base)', run_strategy())
print_result('4a) Dynamic 1.5-4.0x', run_dynamic_lev((1.5,4.0)))
print_result('4b) Dynamic 2.0-5.0x', run_dynamic_lev((2.0,5.0)))
print_result('4c) Dynamic 1.5-5.0x', run_dynamic_lev((1.5,5.0)))
print_result('4d) Dynamic 2.0-4.5x', run_dynamic_lev((2.0,4.5)))
print_result('4e) Dynamic 1.5-4.0x + Short', run_dynamic_lev((1.5,4.0), use_short=True))
print_result('4f) Dynamic 2.0-5.0x + Short', run_dynamic_lev((2.0,5.0), use_short=True))

print('\n'+' DIRECTION 5: フルモデル候補 '.center(70,'='),flush=True)
print(hdr,flush=True);print('  '+'-'*82,flush=True)

# Combine best elements from each direction
print_result('V2 Baseline', run_strategy(use_ms_protect=False), ' <--')
print_result('V2+MS34 (Bear保護のみ)', run_strategy())
print_result('5a) MS+Boost4+Short0.5', run_strategy(use_ms_boost=True, boost_lev=4.0, use_short=True, short_size=-0.5))
print_result('5b) MS+Boost4+Value10%', run_strategy(use_ms_boost=True, boost_lev=4.0, use_value=True, value_weight=0.10))
print_result('5c) MS+Boost4+Short+Value10%', run_strategy(use_ms_boost=True, boost_lev=4.0, use_short=True, short_size=-0.5, use_value=True, value_weight=0.10))
print_result('5d) DynLev 2-5x + Short', run_dynamic_lev((2.0,5.0), use_short=True))
print_result('5e) MS+Boost4.5+Short0.3+Val10%', run_strategy(use_ms_boost=True, boost_lev=4.5, use_short=True, short_size=-0.3, use_value=True, value_weight=0.10))

# ============================================================
# YEAR-BY-YEAR for TOP 3 candidates
# ============================================================
print('\n'+' TOP候補 年別分析 '.center(70,'='),flush=True)

top_candidates=[
    ('V2 Baseline', run_strategy(use_ms_protect=False)),
    ('V2+MS34 Protect', run_strategy()),
]
# Add dynamic lev candidates
for name, eq in [
    ('DynLev 2-5x+Short', run_dynamic_lev((2.0,5.0), use_short=True)),
    ('MS+Boost4+Short0.5', run_strategy(use_ms_boost=True, boost_lev=4.0, use_short=True, short_size=-0.5)),
    ('MS+Boost4.5+Short0.3+Val10%', run_strategy(use_ms_boost=True, boost_lev=4.5, use_short=True, short_size=-0.3, use_value=True, value_weight=0.10)),
]:
    top_candidates.append((name, eq))

print(f'  {"Year":<6}', end='',flush=True)
for name,_ in top_candidates:
    print(f' {name[:18]:>18}', end='',flush=True)
print(flush=True)
print('  '+'-'*(6+19*len(top_candidates)),flush=True)

for yr in range(2021,2026):
    print(f'  {yr:<6}', end='',flush=True)
    for _,eq in top_candidates:
        _,_,_,_,_,_,_,yearly=eval_strat(eq)
        v=yearly.get(yr,0)
        if abs(v)>10000:
            print(f' {v/1000:>+16.0f}K%', end='',flush=True)
        else:
            print(f' {v:>+17.1f}%', end='',flush=True)
    print(flush=True)

print('\nDone.',flush=True)
