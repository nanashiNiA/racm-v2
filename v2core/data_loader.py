"""V2 Data Loader: Load all available data sources"""
import sys, os
import numpy as np
import pandas as pd
import json, urllib.request

sys.path.insert(0, 'C:/Users/A701/Documents/nia/prediction_model_project')

def load_btc_1h(cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/data/cache'):
    """BTC 1H from local + API extension. Cached."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'BTC_1h_combined.pkl')

    btc_h = pd.read_pickle('C:/Users/A701/Documents/nia/prediction_model_project/data/external/binance/btcusdt_hourly.pkl')
    btc_h.index = pd.to_datetime(btc_h.index)

    # Load cache if recent (< 2 hours old)
    if os.path.exists(cache_path):
        import time
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 2:
            btc_h = pd.read_pickle(cache_path)
            btc_h['return'] = btc_h['close'].pct_change()
            return btc_h.dropna(subset=['return'])

    # Extend with API
    rows = []; et = int(pd.Timestamp.now().timestamp() * 1000)
    for _ in range(20):
        u = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&endTime={et}'
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'V2'}), timeout=15) as r:
                d = json.loads(r.read())
        except: break
        if not d: break
        for k in d: rows.append({'timestamp': pd.Timestamp(k[0], unit='ms'), 'close': float(k[4]),
                                  'high': float(k[2]), 'low': float(k[3]), 'open': float(k[1]),
                                  'volume': float(k[5])})
        et = int(d[0][0]) - 1
    if rows:
        api = pd.DataFrame(rows).drop_duplicates('timestamp').set_index('timestamp').sort_index()
        btc_h = pd.concat([btc_h, api[~api.index.isin(btc_h.index)]]).sort_index()

    btc_h.to_pickle(cache_path)
    btc_h['return'] = btc_h['close'].pct_change()
    return btc_h.dropna(subset=['return'])

def load_derivatives():
    """Derivatives 1H (funding, OI, liquidation)"""
    deriv = pd.read_pickle('C:/Users/A701/Documents/nia/prediction_model_project/data/processed/derivatives_1h.pkl')
    deriv.index = deriv.index.tz_localize(None)
    return deriv

def load_6assets_8h(assets=['BTC','ETH','SOL','XRP','DOGE','LINK'], cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/data/cache'):
    """6 assets at 8H. Cache locally, only fetch new bars from API."""
    os.makedirs(cache_dir, exist_ok=True)
    data = {}
    for sym in assets:
        cache_path = os.path.join(cache_dir, f'{sym}_8h.pkl')
        # Load cache if exists
        if os.path.exists(cache_path):
            cached = pd.read_pickle(cache_path)
            # Only fetch bars AFTER the last cached bar
            last_ts = int(cached.index[-1].timestamp() * 1000)
            rows = []; et = int(pd.Timestamp.now().timestamp() * 1000)
            for _ in range(5):  # only a few pages for new data
                u = f'https://api.binance.com/api/v3/klines?symbol={sym}USDT&interval=8h&limit=1000&startTime={last_ts}'
                try:
                    with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'V2'}), timeout=15) as r:
                        d = json.loads(r.read())
                except: break
                if not d: break
                for k in d: rows.append({'timestamp': pd.Timestamp(k[0], unit='ms'), 'close': float(k[4])})
                if len(d) < 1000: break  # got all new bars
                et = int(d[-1][0]) + 1
            if rows:
                new_df = pd.DataFrame(rows).drop_duplicates('timestamp').set_index('timestamp').sort_index()
                new_df['close'] = new_df['close'].astype(float)
                combined = pd.concat([cached, new_df[~new_df.index.isin(cached.index)]]).sort_index()
                combined['return'] = combined['close'].pct_change()
                combined = combined.dropna(subset=['return'])
                combined.to_pickle(cache_path)
                data[sym] = combined
            else:
                cached['return'] = cached['close'].pct_change()
                data[sym] = cached.dropna(subset=['return'])
        else:
            # First time: full fetch
            rows = []; et = int(pd.Timestamp.now().timestamp() * 1000)
            for _ in range(25):
                u = f'https://api.binance.com/api/v3/klines?symbol={sym}USDT&interval=8h&limit=1000&endTime={et}'
                try:
                    with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'V2'}), timeout=15) as r:
                        d = json.loads(r.read())
                except: break
                if not d: break
                for k in d: rows.append({'timestamp': pd.Timestamp(k[0], unit='ms'), 'close': float(k[4])})
                et = int(d[0][0]) - 1
            df = pd.DataFrame(rows).drop_duplicates('timestamp').set_index('timestamp').sort_index()
            df['close'] = df['close'].astype(float)
            df['return'] = df['close'].pct_change()
            df = df.dropna(subset=['return'])
            df.to_pickle(cache_path)
            data[sym] = df
    return data

def load_6assets_1h(assets=['BTC','ETH','SOL','XRP','DOGE','LINK'], cache_dir='C:/Users/A701/Documents/nia/prediction_model_project/data/cache'):
    """6 assets at 1H. Cache locally. Full history."""
    os.makedirs(cache_dir, exist_ok=True)
    data = {}
    for sym in assets:
        cache_path = os.path.join(cache_dir, f'{sym}_1h.pkl')
        if os.path.exists(cache_path):
            import time
            age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_hours < 2:
                data[sym] = pd.read_pickle(cache_path)
                continue
        # Full fetch (up to 80 pages)
        print(f'  Fetching {sym} 1H...', end='', flush=True)
        rows = []; et = int(pd.Timestamp.now().timestamp() * 1000)
        for p in range(80):
            u = f'https://api.binance.com/api/v3/klines?symbol={sym}USDT&interval=1h&limit=1000&endTime={et}'
            try:
                with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'V2'}), timeout=15) as r:
                    d = json.loads(r.read())
            except: break
            if not d: break
            for k in d: rows.append({'timestamp': pd.Timestamp(k[0], unit='ms'),
                'close': float(k[4]), 'high': float(k[2]), 'low': float(k[3]),
                'open': float(k[1]), 'volume': float(k[5])})
            et = int(d[0][0]) - 1
            if (p+1) % 20 == 0: print(f' {p+1}p', end='', flush=True)
        df = pd.DataFrame(rows).drop_duplicates('timestamp').set_index('timestamp').sort_index()
        for col in ['close','high','low','open','volume']: df[col] = df[col].astype(float)
        df['return'] = df['close'].pct_change()
        df = df.dropna(subset=['return'])
        df.to_pickle(cache_path)
        data[sym] = df
        print(f' {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})')
    return data

def load_macro():
    """Macro data (if available)"""
    base = 'C:/Users/A701/Documents/nia/prediction_model_project/data/external'
    macro = {}
    for f in ['fear_greed_index.csv', 'mvrv.csv', 'sopr.csv']:
        path = os.path.join(base, f)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=[0], index_col=0)
                macro[f.replace('.csv', '')] = df
            except: pass
    return macro

def build_common_1h():
    """Build aligned 1H dataset with all features"""
    btc = load_btc_1h()
    deriv = load_derivatives()
    common = btc.index.intersection(deriv.index)

    h = pd.DataFrame(index=common)
    h['close'] = btc.loc[common, 'close'].values
    h['high'] = btc.loc[common, 'high'].values if 'high' in btc.columns else h['close']
    h['low'] = btc.loc[common, 'low'].values if 'low' in btc.columns else h['close']
    h['volume'] = btc.loc[common, 'volume'].values if 'volume' in btc.columns else 0
    h['ret'] = h['close'].pct_change()
    h['funding'] = deriv.loc[common, 'funding_rate'].values
    h['oi'] = deriv.loc[common, 'open_interest_last'].values
    h['liq_count'] = deriv.loc[common, 'liq_count'].values

    # Technical indicators
    h['ma20'] = h['close'].rolling(480, min_periods=240).mean()    # 20 days
    h['ma50'] = h['close'].rolling(1200, min_periods=600).mean()   # 50 days
    h['ma110'] = h['close'].rolling(2640, min_periods=1320).mean() # 110 days
    h['ma200'] = h['close'].rolling(4800, min_periods=2400).mean() # 200 days

    # ATR
    tr = pd.concat([h['high']-h['low'],
                     (h['high']-h['close'].shift(1)).abs(),
                     (h['low']-h['close'].shift(1)).abs()], axis=1).max(axis=1)
    h['atr24'] = tr.rolling(24, min_periods=6).mean()
    h['atr_pct'] = h['atr24'] / h['close']

    # ADX (trend strength)
    up = h['high'].diff(); dn = -h['low'].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
    atr72 = tr.rolling(72, min_periods=24).mean()
    plus_di = pd.Series(plus_dm, index=h.index).rolling(72, min_periods=24).mean() / (atr72 + 1e-12)
    minus_di = pd.Series(minus_dm, index=h.index).rolling(72, min_periods=24).mean() / (atr72 + 1e-12)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    h['adx'] = dx.rolling(72, min_periods=24).mean() * 100

    # Volatility
    h['vol_24h'] = h['ret'].rolling(24, min_periods=6).std() * np.sqrt(365*24)
    h['vol_7d'] = h['ret'].rolling(168, min_periods=48).std() * np.sqrt(365*24)
    h['vol_30d'] = h['ret'].rolling(720, min_periods=240).std() * np.sqrt(365*24)
    h['vol_ratio'] = h['vol_24h'] / (h['vol_7d'] + 1e-10)

    # Skew, kurtosis
    h['skew_30d'] = h['ret'].rolling(720, min_periods=360).skew()
    h['kurt_30d'] = h['ret'].rolling(720, min_periods=360).kurt()

    # Drawdown
    h['peak'] = h['close'].rolling(1080, min_periods=100).max()
    h['dd'] = (h['close'] - h['peak']) / h['peak']

    # Donchian channels
    h['donchian_high_20d'] = h['high'].rolling(480, min_periods=240).max()
    h['donchian_low_20d'] = h['low'].rolling(480, min_periods=240).min()
    h['donchian_mid'] = (h['donchian_high_20d'] + h['donchian_low_20d']) / 2
    h['donchian_breakout_up'] = (h['close'] >= h['donchian_high_20d'].shift(1)).astype(float)
    h['donchian_breakout_dn'] = (h['close'] <= h['donchian_low_20d'].shift(1)).astype(float)

    # RSI
    delta = h['close'].diff()
    gain = delta.clip(lower=0).rolling(336, min_periods=168).mean()  # 14 days
    loss = (-delta.clip(upper=0)).rolling(336, min_periods=168).mean()
    h['rsi_14d'] = 100 - 100 / (1 + gain / (loss + 1e-12))

    # Funding / OI features
    h['funding_z'] = ((h['funding'] - h['funding'].rolling(168, min_periods=48).mean())
                       / (h['funding'].rolling(168, min_periods=48).std() + 1e-12))

    h = h.ffill().dropna(subset=['close'])
    return h

if __name__ == '__main__':
    import sys; sys.stdout.reconfigure(encoding='utf-8') if sys.platform == 'win32' else None
    h = build_common_1h()
    print(f'Loaded: {len(h)} bars ({h.index[0]} to {h.index[-1]})')
    print(f'Columns: {list(h.columns)}')
    print(f'Features: {len(h.columns)}')
