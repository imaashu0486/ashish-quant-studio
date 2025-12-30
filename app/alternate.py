# views.py — Ashish Quant Studio (final, cleaned & correct)

from django.shortcuts import render
from django.conf import settings
from plotly.offline import plot
import plotly.graph_objs as go

import os
import logging
import time
import pandas as pd
import numpy as np
import yfinance as yf
import ta

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ----- Optional XGBoost (used if available) -----
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# =========================
# Yahoo helpers (robust)
# =========================
def _yf_flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns to single level (if present)."""
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(0)
        except Exception:
            pass
    return df

def yf_download_single(
    tickers: str,
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
    max_retries: int = 3,
    pause: float = 1.0,
    fallback_periods: tuple = ("2y", "1y", "6mo"),
    **kwargs
) -> pd.DataFrame:
    """
    Robust wrapper: retries + period fallbacks + dual API (download/history).
    Returns single-level columns and avoids over-pruning valid rows.
    """

    # Back-compat for older call sites that pass 'tries'/'pause'/'fallback_periods'
    if "tries" in kwargs:
        try: max_retries = max(1, int(kwargs.pop("tries")))
        except Exception: kwargs.pop("tries", None)
    if "pause" in kwargs:
        try: pause = float(kwargs.pop("pause"))
        except Exception: kwargs.pop("pause", None)
    if "fallback_periods" in kwargs:
        try: fallback_periods = tuple(kwargs.pop("fallback_periods"))
        except Exception: kwargs.pop("fallback_periods", None)

    # Only pass supported args to yfinance.download
    dl_supported = {
        "period","interval","auto_adjust","group_by","prepost","threads",
        "progress","actions","ignore_tz","repair","keepna","raise_errors","show_errors"
    }
    dl_kwargs = dict(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        threads=True,         # allow Yahoo internal parallelization (like your working file)
        progress=False,
        repair=True,          # heal missing rows where possible (>=0.2.x)
    )
    for k, v in kwargs.items():
        if k in dl_supported:
            dl_kwargs[k] = v

    def _flatten(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.droplevel(0)
            except Exception: pass
        return df

    def _try_both() -> pd.DataFrame:
        # Path A: yfinance.download
        try:
            df = yf.download(tickers=tickers, **dl_kwargs)
            df = _flatten(df)
            # don't drop rows too aggressively; keep valid 'Close' even if 'Volume' is NaN
            if isinstance(df, pd.DataFrame) and "Close" in df and not df["Close"].dropna().empty:
                return df
            logger.warning("download() returned empty/invalid for %s (period=%s interval=%s)",
                           tickers, dl_kwargs.get("period"), dl_kwargs.get("interval"))
        except Exception as e:
            logger.warning("download() failed for %s: %s", tickers, e)

        # Path B: Ticker.history (often succeeds when download() doesn't)
        try:
            tk = yf.Ticker(tickers)
            hist = tk.history(period=dl_kwargs["period"], interval=dl_kwargs["interval"],
                              auto_adjust=auto_adjust, actions=False, repair=True)
            hist = _flatten(hist)
            if isinstance(hist, pd.DataFrame) and "Close" in hist and not hist["Close"].dropna().empty:
                return hist
            logger.warning("Ticker.history() returned empty/invalid for %s", tickers)
        except Exception as e:
            logger.warning("Ticker.history() failed for %s: %s", tickers, e)

        return pd.DataFrame()

    # Primary attempts
    for attempt in range(1, max_retries + 1):
        df = _try_both()
        if not df.empty:
            return df
        if attempt < max_retries:
            time.sleep(max(0.0, pause))

    # Period fallbacks
    for p in fallback_periods:
        dl_kwargs["period"] = p
        for attempt in range(1, max_retries + 1):
            df = _try_both()
            if not df.empty:
                return df
            if attempt < max_retries:
                time.sleep(max(0.0, pause))

    return pd.DataFrame()


# =========================
# Theme utilities
# =========================
def _plotly_theme_colors(request):
    """Colors for plotly based on theme cookie set by your base.html toggle."""
    theme = request.COOKIES.get("theme", "dark")
    if theme == "light":
        return {"paper": "#ffffff", "plot": "#ffffff", "font": "#111111", "grid": "#e6e6ee"}
    else:
        return {"paper": "#14151b", "plot": "#14151b", "font": "#ffffff", "grid": "#2a2d36"}


# =========================
# Feature engineering
# =========================
def build_features(ohlcv_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a feature DataFrame with technical indicators (no leakage).
    Expects:
      ohlcv_df: ['Open','High','Low','Close','Volume']
      market_df: ['Close']
    """
    # Defensive: ensure expected columns & 1-D series
    df = ohlcv_df[['Open','High','Low','Close','Volume']].copy()
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].squeeze()

    mkt = market_df.copy()
    if 'Close' in mkt:
        if isinstance(mkt['Close'], pd.DataFrame):
            mkt['Close'] = mkt['Close'].squeeze()
        mkt = mkt[['Close']].reindex(df.index).ffill().bfill()
    else:
        # synthetic market if missing
        mkt = pd.DataFrame(index=df.index, data={'Close': df['Close'].rolling(10).mean().ffill()})

    def s1(x):
        if isinstance(x, pd.DataFrame):  # just in case
            return x.iloc[:, 0]
        return x

    # Basic return
    df['ret'] = s1(df['Close']).pct_change()

    # Shifted inputs (no leakage)
    s_high  = s1(df['High']).shift(1)
    s_low   = s1(df['Low']).shift(1)
    s_close = s1(df['Close']).shift(1)
    s_vol   = s1(df['Volume']).shift(1)

    # Momentum/MA
    df['sma_5']  = ta.trend.SMAIndicator(s_close, window=5).sma_indicator()
    df['sma_10'] = ta.trend.SMAIndicator(s_close, window=10).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(s_close, window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(s_close, window=26).ema_indicator()

    # RSI / MACD / BB width
    df['rsi14'] = ta.momentum.RSIIndicator(s_close, window=14).rsi()
    _macd = ta.trend.MACD(s_close)
    df['macd']     = _macd.macd()
    df['macd_sig'] = _macd.macd_signal()
    _bb = ta.volatility.BollingerBands(s_close, window=20, window_dev=2)
    df['bb_w'] = (_bb.bollinger_hband() - _bb.bollinger_lband()) / s_close

    # ATR / OBV / Stochastic
    df['atr14'] = ta.volatility.AverageTrueRange(high=s_high, low=s_low, close=s_close, window=14).average_true_range()
    # OBV only if volume is valid
    if df['Volume'].dropna().sum() > 0:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=s_close, volume=s_vol).on_balance_volume()
    else:
        df['obv'] = 0  # fallback constant for indices

    # Stochastic only if High and Low vary enough (indices sometimes flat intraday)
    if (s_high != s_low).sum() > 5:
        df['stoch_k'] = ta.momentum.StochasticOscillator(
            high=s_high, low=s_low, close=s_close, window=14, smooth_window=3
        ).stoch()
    else:
        df['stoch_k'] = 0


    # Market & time
    mclose = mkt['Close']
    df['mkt_ret'] = mclose.pct_change().shift(1)
    df['rel_ret'] = df['ret'].shift(1) - df['mkt_ret']
    df['day_of_week']   = df.index.dayofweek
    df['month_of_year'] = df.index.month

    # Lag returns + rolling vol
    for lag in [1, 2, 3, 5, 10]:
        df[f'lag_ret_{lag}'] = df['ret'].shift(lag)
    df['roll_std_5']  = df['ret'].shift(1).rolling(5).std()
    df['roll_std_10'] = df['ret'].shift(1).rolling(10).std()

    # Clean
    df = df.dropna()
    return df


# =========================
# Fundamentals
# =========================
def get_fundamentals(symbol_ns: str, hist_index=None) -> dict:
    """
    Returns dict of fundamentals for an Indian symbol.
    Adds IPO year fallback from first trade date if Yahoo misses it.
    """
    tk = yf.Ticker(symbol_ns)
    out = {}

    # fast fields
    try:
        fi = tk.fast_info
        out.update({
            "cmp": float(fi.get("last_price") or fi.get("lastPrice") or fi.get("last")),
            "day_high": fi.get("day_high"),
            "day_low": fi.get("day_low"),
            "year_high": fi.get("year_high"),
            "year_low": fi.get("year_low"),
            "currency": fi.get("currency") or "INR",
            "previous_close": fi.get("previous_close"),
            "volume": fi.get("last_volume") or fi.get("volume"),
            "market_cap": fi.get("market_cap"),
        })
    except Exception:
        pass

    # slower metadata
    try:
        info = tk.get_info() if hasattr(tk, "get_info") else getattr(tk, "info", None)
        if info:
            out.update({
                "longName": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "ipo_year": info.get("ipoYear"),
                "website": info.get("website"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "eps": info.get("trailingEps"),
                "profit_margins": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
            })
    except Exception:
        pass

    # 52w off-high/low
    try:
        if out.get("year_high") and out.get("year_low") and out.get("cmp"):
            yh, yl, c = float(out["year_high"]), float(out["year_low"]), float(out["cmp"])
            out["off_high_pct"] = round(100*(yh - c)/yh, 2) if yh else None
            out["off_low_pct"]  = round(100*(c - yl)/yl, 2) if yl else None
    except Exception:
        pass

    # IPO fallback from first trade date
    if (not out.get("ipo_year")) and hist_index is not None and len(hist_index) > 0:
        try:
            first_year = pd.to_datetime(hist_index.min()).year
            if first_year and first_year >= 1990:
                out["ipo_year"] = first_year
        except Exception:
            pass

    return {k: v for k, v in out.items() if v is not None}


# =========================
# Misc helpers
# =========================
def normalize_in_symbol(sym: str) -> str:
    s = (sym or "").strip().upper().replace(" ", "")
    # Common index aliases -> Yahoo symbols
    alias = {
        "NIFTY50": "^NSEI", "NIFTY": "^NSEI", "NSEI": "^NSEI",
        "BANKNIFTY": "^NSEBANK", "NIFTYBANK": "^NSEBANK", "NSEBANK": "^NSEBANK",
        "SENSEX": "^BSESN", "BSE": "^BSESN", "BSESENSEX": "^BSESN",
        "FINNIFTY": "^NSEFIN",
    }
    if s in alias:
        return alias[s]
    if s.startswith("^"):              # already an index
        return s
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}.NS"                   # default to NSE stock


def tidy_multiindex(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level=0).reset_index().rename(columns={'level_1': 'Ticker'})
    else:
        df = raw.reset_index()
        df['Ticker'] = tickers[0] if tickers else 'TICKER'
    return df


# =========================
# Pages
# =========================
def index(request):
    nse = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'SBIN.NS']
    raw = yf.download(
        tickers=nse, group_by='ticker', period='1mo', interval='1d',
        auto_adjust=True, threads=False, progress=False
    )

    colors = _plotly_theme_colors(request)
    fig_left = go.Figure()

    if isinstance(raw.columns, pd.MultiIndex):
        for t in nse:
            if t not in raw:
                continue
            ser = raw[t].get('Close', pd.Series(dtype=float)).dropna()
            if ser.empty:
                continue
            is_up = ser.iloc[-1] >= ser.iloc[0]
            color = 'lightgreen' if is_up else 'red'
            fig_left.add_trace(go.Scatter(x=ser.index, y=ser.values,
                                          name=t.replace('.NS',''),
                                          line=dict(color=color, width=2)))
    else:
        if 'Close' in raw:
            s = raw['Close'].dropna()
            if not s.empty:
                color = 'lightgreen' if s.iloc[-1] >= s.iloc[0] else 'red'
                fig_left.add_trace(go.Scatter(x=s.index, y=s.values,
                                              name=nse[0].replace('.NS',''),
                                              line=dict(color=color, width=2)))

    fig_left.update_layout(
        paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
        xaxis=dict(gridcolor=colors["grid"]), yaxis=dict(gridcolor=colors["grid"])
    )
    plot_div_left = plot(fig_left, auto_open=False, output_type='div')

    # recent table
    recent = ['RELIANCE.NS','TCS.NS','INFY.NS','ICICIBANK.NS','HDFCBANK.NS','ITC.NS','LT.NS','SBIN.NS']
    raw_recent = yf.download(
        tickers=recent, period='5d', interval='1d', group_by='ticker',
        auto_adjust=True, threads=False, progress=False
    )
    df = tidy_multiindex(raw_recent, recent)
    df = df[['Date','Ticker','Close','Volume']].dropna(subset=['Close'])
    df['Ticker'] = df['Ticker'].str.replace('.NS','', regex=False).str.replace('.BO','', regex=False)
    df['Date'] = df['Date'].astype(str)
    recent_stocks = df.to_dict(orient='records')

    # quick pills for template
    quick_tickers = ['RELIANCE','TCS','INFY','SBIN','ICICIBANK','HDFCBANK','ITC','LT']

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks,
        'quick_tickers': quick_tickers,
    })


def search(request):
    return render(request, 'search.html')


def predict(request, ticker_value, number_of_days):
    """
    - Next-day prediction only (ignores number_of_days)
    - Robust downloads with retries/fallbacks
    - If INDEX (e.g., ^NSEI / BANKNIFTY), render charts + fundamentals but skip ML
    - Saves CSVs for stocks (not for indexes)
    - Theme-aware charts; IPO year fallback via first trade date
    """
    colors = _plotly_theme_colors(request)

    # --- read POST (ticker only) ---
    if request.method == 'POST':
        ticker_value = request.POST.get('ticker', ticker_value)
    ticker_value = normalize_in_symbol(ticker_value)
    horizon = 1

    is_index = ticker_value.startswith("^")

    # --- intraday candlestick (try 1m, then 5m) ---
    df_intra = yf_download_single(
        ticker_value, period='1d', interval='1m', auto_adjust=True, max_retries=2, pause=0.8
    )
    if df_intra.empty:
        df_intra = yf_download_single(
            ticker_value, period='5d', interval='5m', auto_adjust=True, max_retries=2, pause=0.8
        )

    fig_candle = go.Figure()
    if df_intra.empty or not all(k in df_intra.columns for k in ['Open','High','Low','Close']):
        fig_candle.add_annotation(text="No intraday data (market closed or symbol illiquid).",
                                  x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    else:
        fig_candle.add_trace(go.Candlestick(
            x=df_intra.index,
            open=df_intra['Open'], high=df_intra['High'],
            low=df_intra['Low'],  close=df_intra['Close'],
            increasing_line_color='lightgreen', decreasing_line_color='red',
            increasing_fillcolor='rgba(144,238,144,0.2)', decreasing_fillcolor='rgba(255,0,0,0.2)',
            name='market data'
        ))
    fig_candle.update_layout(
        title=f'{ticker_value} intraday',
        yaxis_title='Price',
        paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
        xaxis=dict(gridcolor=colors["grid"]), yaxis=dict(gridcolor=colors["grid"])
    )
    fig_candle.update_xaxes(rangeslider_visible=True)
    plot_div = plot(fig_candle, auto_open=False, output_type='div')

    # --- daily history (robust) ---
    hist = yf_download_single(ticker_value, period='2y', interval='1d', auto_adjust=True)
    if hist.empty or 'Close' not in hist:
        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
            'error_message': "No historical data available for this symbol (Yahoo timeout/rate-limit).",
            'fundamentals': {}, 'metrics': [], 'best_model': "—",
            'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
            'number_of_days': horizon,
        })

    # keep only needed columns if present
    needed = [c for c in ['Open','High','Low','Close','Volume'] if c in hist.columns]
    hist = hist[needed].dropna().copy()


    # --- market index (for features) ---
    mkt = yf_download_single('^NSEI', period='5y', interval='1d', auto_adjust=True)
    if mkt.empty or 'Close' not in mkt:
        mkt = pd.DataFrame(index=hist.index, data={"Close": hist["Close"].rolling(10).mean().ffill()})
    else:
        mkt = mkt[['Close']].reindex(hist.index).ffill().bfill()

    # --- features & target ---
    df_feat = build_features(hist, mkt).sort_index().copy()
    hist = hist.sort_index().copy()

    # target = next day's close
    close_series = hist['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()
    target_series = close_series.shift(-1).rename('target_price')

    df_feat = df_feat.join(target_series, how='inner').dropna(subset=['target_price']).copy()

    if len(df_feat) < 100:
        try:
            fund = get_fundamentals(ticker_value, hist_index=hist.index)
        except Exception:
            fund = {}
        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
            'error_message': "Not enough clean history to train (need ~100+ rows after indicators).",
            'fundamentals': fund, 'metrics': [], 'best_model': "—",
            'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
            'number_of_days': horizon,
        })

    feature_cols = [c for c in df_feat.columns if c != 'target_price']
    if not feature_cols:
        try:
            fund = get_fundamentals(ticker_value, hist_index=hist.index)
        except Exception:
            fund = {}
        return render(request, 'result.html', {
            'plot_div': plot_div, 'ticker_value': ticker_value,
            'error_message': "Feature creation failed (no columns).",
            'fundamentals': fund,
        })

    X = df_feat[feature_cols].values
    y = df_feat['target_price'].values
    split_idx = int(len(df_feat) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    idx_test = df_feat.index[split_idx:]

    # scale for linear models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    feats_last_s = scaler.transform(df_feat[feature_cols].iloc[[-1]].values)

    # --- models ---
    models = {}
    models['Ridge'] = Ridge(alpha=1.0).fit(X_train_s, y_train)  # Ridge doesn't take random_state
    models['RandomForest'] = RandomForestRegressor(n_estimators=700, random_state=42, n_jobs=-1).fit(X_train, y_train)
    if _HAS_XGB:
        try:
            models['XGBoost'] = XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, objective='reg:squarederror',
                random_state=42, n_jobs=0
            ).fit(X_train, y_train)
        except Exception as e:
            logger.warning("XGBoost training skipped: %s", e)

    # --- evaluate ---
    evals, preds_df = {}, {}
    cur_close_test = hist['Close'].reindex(idx_test).values

    for name, mdl in models.items():
        if name == 'Ridge':
            y_hat = mdl.predict(X_test_s)
            next_pred = float(mdl.predict(feats_last_s)[0])
        else:
            y_hat = mdl.predict(X_test)
            next_pred = float(mdl.predict(df_feat[feature_cols].iloc[[-1]].values)[0])

        true_dir = np.sign(y_test - cur_close_test)
        pred_dir = np.sign(y_hat - cur_close_test)

        evals[name] = {
            "y_pred": y_hat,
            "r2": float(r2_score(y_test, y_hat)) if len(y_test) > 1 else 0.0,
            "mae_pct": float(np.mean(np.abs((y_hat - y_test) / y_test))) * 100.0,
            "dir_acc": float((pred_dir == true_dir).mean()) * 100.0,
            "next_price": next_pred
        }
        preds_df[name] = pd.Series(y_hat, index=idx_test)

    if preds_df:
        ens_series = pd.concat(preds_df, axis=1).mean(axis=1)
        ens_true_dir = np.sign(y_test - cur_close_test)
        ens_pred_dir = np.sign(ens_series.values - cur_close_test)
        evals['Ensemble'] = {
            "y_pred": ens_series.values,
            "r2": float(r2_score(y_test, ens_series.values)) if len(y_test) > 1 else 0.0,
            "mae_pct": float(np.mean(np.abs((ens_series.values - y_test) / y_test))) * 100.0,
            "dir_acc": float((ens_pred_dir == ens_true_dir).mean()) * 100.0,
            "next_price": float(np.mean([v["next_price"] for v in evals.values() if "next_price" in v]))
        }
        preds_df['Ensemble'] = ens_series

    # --- overlay chart ---
    palette = {"Ridge":"#22c55e","RandomForest":"#f59e0b","XGBoost":"#ef4444","Ensemble":"#60a5fa"}
    fig_pred_all = go.Figure()
    fig_pred_all.add_trace(go.Scatter(x=idx_test, y=y_test, name='Actual', mode='lines', line=dict(width=2)))
    for name, series in preds_df.items():
        fig_pred_all.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name, mode='lines',
            line=dict(width=2, dash='dash', color=palette.get(name)),
            hovertemplate="%{x|%b %d, %Y}<br>₹ %{y:,.2f}<extra>"+name+"</extra>"
        ))
    fig_pred_all.update_layout(
        title=f"{ticker_value} — Actual vs Model Predictions (Test)",
        paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
        hovermode="x unified",
        xaxis=dict(type="date", rangeslider=dict(visible=True), gridcolor=colors["grid"]),
        yaxis=dict(title="Price", tickprefix="₹ ", tickformat=",.2f", gridcolor=colors["grid"])
    )
    plot_div_pred = plot(fig_pred_all, auto_open=False, output_type='div')

    # --- metrics & best ---
    metrics = [{
        "Model": name,
        "R2": round(res["r2"], 4),
        "MAE_ret": round(res["mae_pct"], 3),
        "Direction": round(res["dir_acc"], 2),
        "NextDayPrice": round(res["next_price"], 2)
    } for name, res in evals.items()]
    metrics_sorted = sorted(metrics, key=lambda m: (-m["R2"], -m["Direction"], m["MAE_ret"])) if metrics else []
    best_model = metrics_sorted[0]["Model"] if metrics_sorted else "—"
    headline_key = 'Ensemble' if 'Ensemble' in evals else best_model
    headline = next((m for m in metrics if m["Model"] == headline_key), None)
    accuracy_val = headline["R2"] if headline else 0.0
    next_price_val = headline["NextDayPrice"] if headline else float('nan')

    # --- fundamentals ---
    try:
        fund = get_fundamentals(ticker_value, hist_index=hist.index)
    except Exception:
        fund = {}

    # --- save CSVs (stocks only) ---
    base_media = getattr(settings, "MEDIA_ROOT",
                         os.path.join(getattr(settings, "BASE_DIR", os.getcwd()), "media"))
    out_dir = os.path.join(base_media, "training")
    os.makedirs(out_dir, exist_ok=True)
    safe_t = ticker_value.replace(".", "_")
    feat_path = os.path.join(out_dir, f"{safe_t}_features.csv")
    test_path = os.path.join(out_dir, f"{safe_t}_test_actual_vs_preds.csv")

    df_feat_out = df_feat.copy()
    df_feat_out.insert(0, "Ticker", ticker_value)
    df_feat_out.to_csv(feat_path, index=True)

    out_pred = pd.DataFrame(index=idx_test)
    out_pred["Actual"] = y_test
    for name, res in evals.items():
        out_pred[name] = res["y_pred"]
    out_pred.to_csv(test_path, index=True)

    # --- render ---
    return render(request, 'result.html', {
        'plot_div': plot_div,
        'plot_div_pred': plot_div_pred,
        'ticker_value': ticker_value,
        'accuracy': accuracy_val,
        'confidence': accuracy_val,
        'next_day_prediction': next_price_val,
        'number_of_days': horizon,
        'fundamentals': fund,
        'metrics': metrics_sorted,
        'best_model': best_model,
        'csv_features_path': feat_path,
        'csv_test_path': test_path
    })
