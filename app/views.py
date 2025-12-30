# views.py — Stocks: price regression | Index: direction classification
from django.shortcuts import render
from django.conf import settings
from plotly.offline import plot
import plotly.graph_objs as go

import os, logging, time
import numpy as np
import pandas as pd
import yfinance as yf
import ta

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# ===== Indices to show on Home (symbol, display name). All fetched individually & safely.
INDEX_UNIVERSE = [
    ("NIFTY50",        "NIFTY 50"),
    ("BANKNIFTY",     "BANKNIFTY"),
    ("SENSEX",       "SENSEX"),
    ("^NSEFIN",      "FINNIFTY"),           # shows if Yahoo supports; otherwise skipped
    ("^NSEMDCP50",   "NIFTY MIDCAP 50"),    # fallback-skip if unavailable
    ("^NSENEXT50",   "NIFTY NEXT 50"),      # fallback-skip if unavailable
    ("^INDIAVIX",    "INDIA VIX"),          # note: inversed logic for green/red handled below
    # “GIFT NIFTY” proxy (Yahoo doesn’t have a consistent symbol). We proxy to NIFTY50.
    ("^NSEI",        "GIFT NIFTY (proxy)"),
]

# Robust alias list for indices that Yahoo sometimes renames/moves
INDEX_CANDIDATES = {
    "NIFTY50":      ["^NSEI"],                          # NIFTY 50
    "BANKNIFTY":   ["^NSEBANK"],                       # BankNifty
    "SENSEX":     ["^BSESN"],                         # Sensex
    "^NSEFIN":    ["^NSEFIN", "^NIFTYFIN", "^CNXFIN"],# FinNifty (try in order)
    "^NSEMDCP50": ["^NSEMDCP50", "^NIFTY_MIDCAP_50"], # Midcap 50 (aliases if any)
    "^NSENEXT50": ["^NSENEXT50", "^NIFTY_NEXT_50"],   # Next 50
    "^INDIAVIX":  ["^INDIAVIX"],                      # India VIX
}


# --- NIFTY50 universe (simple, robust; update occasionally) ---
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS","INFY.NS","SBIN.NS","ITC.NS",
    "BHARTIARTL.NS","LICI.NS","LT.NS","BAJFINANCE.NS","AXISBANK.NS","ASIANPAINT.NS","KOTAKBANK.NS",
    "MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","ONGC.NS","NTPC.NS","NESTLEIND.NS",
    "WIPRO.NS","POWERGRID.NS","M&M.NS","JSWSTEEL.NS","ADANIENT.NS","TATASTEEL.NS","HCLTECH.NS",
    "HDFCLIFE.NS","ADANIPORTS.NS","BAJAJFINSV.NS","COALINDIA.NS","GRASIM.NS","TATAMOTORS.NS",
    "BRITANNIA.NS","CIPLA.NS","TECHM.NS","TATACONSUM.NS","LTIM.NS","DIVISLAB.NS","SHREECEM.NS",
    "HEROMOTOCO.NS","BPCL.NS","BRITANNIA.NS","EICHERMOT.NS","DRREDDY.NS","SBILIFE.NS","HINDALCO.NS",
    "BAJAJ-AUTO.NS"
]

def _pct_two_day(df_close: pd.Series) -> float:
    """Return 1-day % change using last two closes; 0 if unavailable."""
    s = df_close.dropna()
    if len(s) < 2: return 0.0
    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
    return (last - prev) / prev * 100.0 if prev else 0.0


# ---------- Pretty numbers ----------
def _humanize_number(x):
    try:
        x = float(x)
    except Exception:
        return x
    # global + Indian-style
    if abs(x) >= 1e12:  # trillion
        return f"{x/1e12:.2f} T"
    if abs(x) >= 1e9:
        return f"{x/1e9:.2f} B"
    if abs(x) >= 1e7:
        return f"{x/1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"{x/1e5:.2f} L"
    return f"{x:,.0f}"

# ----- Optional XGBoost -----
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# =========================
# Helpers: symbol, theme
# =========================
def is_index_symbol(sym: str) -> bool:
    return str(sym or "").strip().upper().startswith("^")

def normalize_in_symbol(sym: str) -> str:
    s = (sym or "").strip().upper().replace(" ", "")
    alias = {
        "NIFTY50": "^NSEI", "NIFTY": "^NSEI", "NSEI": "^NSEI",
        "BANKNIFTY": "^NSEBANK", "NIFTYBANK": "^NSEBANK", "NSEBANK": "^NSEBANK",
        "SENSEX": "^BSESN", "BSE": "^BSESN", "BSESENSEX": "^BSESN",
        "FINNIFTY": "^NSEFIN",
    }
    if s in alias: return alias[s]
    if s.startswith("^"): return s
    if s.endswith(".NS") or s.endswith(".BO"): return s
    return f"{s}.NS"

def _plotly_theme_colors(request):
    theme = request.COOKIES.get("theme", "dark")
    if theme == "light":
        return {"paper": "#ffffff", "plot": "#ffffff", "font": "#111111", "grid": "#e6e6ee"}
    else:
        return {"paper": "#14151b", "plot": "#14151b", "font": "#ffffff", "grid": "#2a2d36"}


# =========================
# Robust Yahoo fetch
# =========================
def yf_download_single(
    tickers: str,
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
    max_retries: int = 3,
    pause: float = 1.0,
    fallback_periods: tuple = ("max","10y","8y","5y","2y","1y","6mo"),
    **kwargs
) -> pd.DataFrame:
    """
    Retries + dual API (download/history) + period fallbacks.
    Keeps valid rows even if Volume is NaN. Makes tz-naive.
    Accepts legacy kwargs: tries, pause, fallback_periods (ignored if invalid).
    """
    if "tries" in kwargs:
        try: max_retries = max(1, int(kwargs.pop("tries")))
        except: kwargs.pop("tries", None)
    if "pause" in kwargs:
        try: pause = float(kwargs.pop("pause"))
        except: kwargs.pop("pause", None)
    if "fallback_periods" in kwargs:
        try: fallback_periods = tuple(kwargs.pop("fallback_periods"))
        except: kwargs.pop("fallback_periods", None)

    dl_supported = {
        "period","interval","auto_adjust","group_by","prepost","threads",
        "progress","actions","ignore_tz","repair","keepna","raise_errors","show_errors"
    }
    dl_kwargs = dict(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        threads=True,
        progress=False,
        repair=True,
        show_errors=False,
    )
    for k, v in kwargs.items():
        if k in dl_supported:
            dl_kwargs[k] = v

    def _flatten(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.droplevel(0)
            except: pass
        return df

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        if "Close" not in df or df["Close"].dropna().empty: return pd.DataFrame()
        try:
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
        except: pass
        return df

    def _try_download() -> pd.DataFrame:
        try:
            df = yf.download(tickers=tickers, **dl_kwargs)
            return _clean(_flatten(df))
        except Exception as e:
            logger.info("download() failed for %s: %s", tickers, e)
            return pd.DataFrame()

    def _try_history(p_override: str | None = None) -> pd.DataFrame:
        try:
            tk = yf.Ticker(tickers)
            p = p_override or dl_kwargs["period"]
            hist = tk.history(period=p, interval=dl_kwargs["interval"],
                              auto_adjust=auto_adjust, actions=False, repair=True)
            return _clean(_flatten(hist))
        except Exception as e:
            logger.info("history() failed for %s: %s", tickers, e)
            return pd.DataFrame()

    def _dual_try(p_override: str | None = None) -> pd.DataFrame:
        if is_index_symbol(tickers):
            df = _try_history(p_override)
            if not df.empty: return df
            return _try_download()
        else:
            df = _try_download()
            if not df.empty: return df
            return _try_history(p_override)

    for attempt in range(1, max_retries+1):
        df = _dual_try()
        if not df.empty: return df
        if attempt < max_retries: time.sleep(max(0.0, pause))

    for p in fallback_periods:
        for attempt in range(1, max_retries+1):
            df = _dual_try(p_override=p)
            if not df.empty: return df
            if attempt < max_retries: time.sleep(max(0.0, pause))

    return pd.DataFrame()


# =========================
# Feature engineering
# =========================
def build_features(ohlcv_df: pd.DataFrame, market_df: pd.DataFrame, *, use_volume: bool) -> pd.DataFrame:
    """
    Leak-free features. If use_volume=False (index or missing volume),
    skip volume-based indicators and rely on price features.
    """
    cols_present = [c for c in ['Open','High','Low','Close','Volume'] if c in ohlcv_df.columns]
    df = ohlcv_df[cols_present].copy()

    for c in df.columns:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].squeeze()

    # Market align
    mkt = market_df.copy()
    if 'Close' in mkt:
        if isinstance(mkt['Close'], pd.DataFrame):
            mkt['Close'] = mkt['Close'].squeeze()
        mkt = mkt[['Close']].reindex(df.index).ffill().bfill()
    else:
        mkt = pd.DataFrame(index=df.index, data={'Close': df['Close'].rolling(10).mean().ffill()})

    def s1(x): return x.iloc[:,0] if isinstance(x, pd.DataFrame) else x

    # Basic returns + shifted inputs
    df['ret'] = s1(df['Close']).pct_change()
    s_high  = s1(df.get('High', pd.Series(index=df.index))).shift(1)
    s_low   = s1(df.get('Low', pd.Series(index=df.index))).shift(1)
    s_close = s1(df.get('Close', pd.Series(index=df.index))).shift(1)
    s_vol   = s1(df.get('Volume', pd.Series(index=df.index))).shift(1)

    # Price indicators
    df['sma_5']  = ta.trend.SMAIndicator(s_close, window=5).sma_indicator()
    df['sma_10'] = ta.trend.SMAIndicator(s_close, window=10).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(s_close, window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(s_close, window=26).ema_indicator()
    df['ema_ratio_12_26'] = df['ema_12'] / df['ema_26']
    df['dist_ma10'] = (s_close / df['sma_10']) - 1.0

    df['rsi14'] = ta.momentum.RSIIndicator(s_close, window=14).rsi()
    df['rsi_chg'] = df['rsi14'] - df['rsi14'].shift(3)

    _macd = ta.trend.MACD(s_close)
    df['macd']     = _macd.macd()
    df['macd_sig'] = _macd.macd_signal()
    df['macd_hist']= _macd.macd_diff()

    _bb = ta.volatility.BollingerBands(s_close, window=20, window_dev=2)
    df['bb_w'] = (_bb.bollinger_hband() - _bb.bollinger_lband()) / s_close

    df['atr14'] = ta.volatility.AverageTrueRange(high=s_high, low=s_low, close=s_close, window=14).average_true_range()
    df['adx14'] = ta.trend.ADXIndicator(high=s_high, low=s_low, close=s_close, window=14).adx()
    df['cci20'] = ta.trend.CCIIndicator(high=s_high, low=s_low, close=s_close, window=20).cci()
    df['wpr14'] = ta.momentum.WilliamsRIndicator(high=s_high, low=s_low, close=s_close, lbp=14).williams_r()

    if use_volume and 'Volume' in df:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=s_close, volume=s_vol).on_balance_volume()
    else:
        df['obv'] = 0.0

    if (s_high != s_low).sum() > 5:
        df['stoch_k'] = ta.momentum.StochasticOscillator(
            high=s_high, low=s_low, close=s_close, window=14, smooth_window=3
        ).stoch()
    else:
        df['stoch_k'] = 0.0

    # Market & time
    mclose = mkt['Close']
    df['mkt_ret'] = mclose.pct_change().shift(1)
    df['rel_ret'] = df['ret'].shift(1) - df['mkt_ret']
    df['day_of_week']   = df.index.dayofweek
    df['month_of_year'] = df.index.month

    # Lags & distribution stats
    for lag in [1,2,3,5,10]:
        df[f'lag_ret_{lag}'] = df['ret'].shift(lag)
    df['roll_std_5']   = df['ret'].shift(1).rolling(5).std()
    df['roll_std_10']  = df['ret'].shift(1).rolling(10).std()
    df['ret_z_20']     = (df['ret'] - df['ret'].rolling(20).mean()) / (df['ret'].rolling(20).std() + 1e-9)
    df['roll_skew_20'] = df['ret'].rolling(20).skew()
    df['roll_kurt_20'] = df['ret'].rolling(20).kurt()

    return df.dropna()


# =========================
# Fundamentals (safer IPO, 52w, market-cap fallback)
# =========================
def get_fundamentals(symbol_ns: str, hist_df: pd.DataFrame | None) -> dict:
    """
    IPO priority:
      1) info['ipoYear']
      2) get_history_metadata()['firstTradeDate']
      3) earliest history year if span >= 7 years
    52w: backfill from history if fast_info misses.
    Market cap: fallback to info['marketCap'] and humanize number.
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

    # slower metadata + IPO
    ipo_year = None
    try:
        info = tk.get_info() if hasattr(tk, "get_info") else getattr(tk, "info", None)
        if info:
            out.update({
                "longName": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
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
            ipo_year = info.get("ipoYear")
            if not out.get("market_cap") and info.get("marketCap"):
                out["market_cap"] = info.get("marketCap")
    except Exception:
        pass

    # firstTradeDate metadata
    if not ipo_year:
        try:
            meta = tk.get_history_metadata() if hasattr(tk, "get_history_metadata") else {}
            ftd = meta.get("firstTradeDate")
            if ftd:
                y = pd.to_datetime(ftd, unit='s').year if isinstance(ftd, (int, float)) else pd.to_datetime(ftd).year
                if y and y >= 1980:
                    ipo_year = y
        except Exception:
            pass

    # history fallback iff long span
    if not ipo_year and isinstance(hist_df, pd.DataFrame) and len(hist_df) > 0:
        try:
            years_span = (hist_df.index.max() - hist_df.index.min()).days / 365.25
            if years_span >= 7:
                ipo_year = int(pd.to_datetime(hist_df.index.min()).year)
        except Exception:
            pass

    if ipo_year:
        out["ipo_year"] = ipo_year

    # 52w backfill
    if isinstance(hist_df, pd.DataFrame) and "Close" in hist_df and len(hist_df) > 0:
        try:
            last_252 = hist_df.tail(252)
            yh = float(last_252["Close"].max())
            yl = float(last_252["Close"].min())
            if not out.get("year_high"): out["year_high"] = yh
            if not out.get("year_low"):  out["year_low"]  = yl
            if out.get("cmp"):
                c = float(out["cmp"])
                out["off_high_pct"] = round(100*(out["year_high"] - c)/out["year_high"], 2)
                out["off_low_pct"]  = round(100*(c - out["year_low"])/out["year_low"], 2)
        except Exception:
            pass

    if out.get("market_cap") is not None:
        out["market_cap"] = _humanize_number(out["market_cap"])

    return {k: v for k, v in out.items() if v is not None}


# =========================
# Multi-ticker table helper (homepage)
# =========================
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
    """Home dashboard — ALWAYS returns a response (no early None).

    Changes in this version:
     - Index cards limited to three primary indices to avoid long YFinance loops.
     - Replaced the 'leaders' line chart with a responsive heatmap/tiles HTML (plot_div_left).
     - Safe/fault-tolerant downloads; logs failures and continues.
     - Today's movers (gainers/losers) computed from nifty50 list (fallback).
     - Always returns a render(...) with safe defaults.
    """
    colors = _plotly_theme_colors(request)

    # ---- safe defaults so render always works ----
    index_cards = []
    gainers, losers = [], []
    chips = []
    plot_div_left = ""  # will store heatmap HTML (replaces previous leaders plot)

    # ---- constants (use globals if already defined) ----
    try:
        idx_universe = INDEX_UNIVERSE
    except NameError:
        # keep many in code, but we'll only query a small set (primary) to avoid slowdowns
        idx_universe = [
            ("^NSEI","NIFTY 50"),
            ("^NSEBANK","BANKNIFTY"),
            ("^BSESN","SENSEX"),
            ("^NSEFIN","FINNIFTY"),
            ("^NSEMDCP50","NIFTY MIDCAP 50"),
            ("^NSENEXT50","NIFTY NEXT 50"),
            ("^INDIAVIX","INDIA VIX"),
        ]
    try:
        idx_candidates = INDEX_CANDIDATES
    except NameError:
        idx_candidates = {
            "^NSEI": ["^NSEI"],
            "^NSEBANK": ["^NSEBANK"],
            "^BSESN": ["^BSESN"],
            "^NSEFIN": ["^NSEFIN","^NIFTYFIN","^CNXFIN"],
            "^NSEMDCP50": ["^NSEMDCP50"],
            "^NSENEXT50": ["^NSENEXT50"],
            "^INDIAVIX": ["^INDIAVIX"],
        }
    try:
        nifty50 = NIFTY50
    except NameError:
        # simpler fallback
        nifty50 = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","SBIN.NS","ITC.NS","LT.NS"]

    # ---------- helpers ----------
    def _series_from_df_close(df: pd.DataFrame) -> pd.Series:
        """Return the numeric 'Close' series from df or a generic numeric column if no Close present."""
        if isinstance(df, pd.DataFrame):
            if 'Close' in df:
                return df['Close']
            # fallback: first numeric column
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    return df[c]
        return pd.Series(dtype=float)

    def _card_for_index(symbol: str, label: str):
        """Try candidate aliases for a given index symbol; return a card dict or None."""
        # for index cards we only need last 2 sessions -> use small period for speed
        candidates = idx_candidates.get(symbol, [symbol])
        for sym in candidates:
            try:
                df = yf_download_single(sym, period='5d', interval='1d',
                                        auto_adjust=True, group_by=None, show_errors=False)
                s = _series_from_df_close(df).dropna()
                if len(s) >= 2:
                    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
                    chg_abs = last - prev
                    chg_pct = (chg_abs / prev) * 100.0 if prev else 0.0
                    is_vix = sym.upper().startswith('^INDIAVIX') or 'VIX' in sym.upper()
                    # For vix we prefer red when rises, for indices green when rises
                    if is_vix:
                        cls = "red" if chg_abs >= 0 else "green"
                    else:
                        cls = "green" if chg_abs >= 0 else "red"
                    return {
                        "name": label,
                        "symbol": sym,
                        "last": f"{last:,.2f}" if last < 1000 else f"{last:,.0f}",
                        "chg_abs": f"{chg_abs:+.2f}" if abs(chg_abs) < 1000 else f"{chg_abs:+.0f}",
                        "chg_pct": f"{chg_pct:+.2f}%",
                        "cls": cls,
                    }
            except Exception as e:
                logger.info("index card fetch failed for %s (%s): %s", sym, label, e)
                continue
        return None

    # ---------- index cards (primary only) ----------
    # To avoid many repeated failing yfinance calls, only query 3 primary indices for cards
    primary_indices = [("^NSEI","NIFTY 50"), ("^NSEBANK","BANKNIFTY"), ("^BSESN","SENSEX")]
    try:
        for sym, label in primary_indices:
            card = _card_for_index(sym, label)
            if card:
                index_cards.append(card)
    except Exception as e:
        logger.info("index cards block exception: %s", e)

    # ---------- Heatmap (replaces leaders plot) ----------
    # We'll build a small HTML block (style + tiles + client-side sorting/toggle)
    try:
        # Choose a watchlist for tiles: default to nifty50 subset for speed
        watchlist = nifty50[:]  # list of ".NS" tickers
        # If the user has a larger NIFTY50 constant, the tiles will adapt (client-side grid)
        # We'll fetch last 22 days for 1M approx; shorter for speed.
        raw = {}
        for t in watchlist:
            try:
                # small optimisation: don't ask full 1y for every tile
                raw_df = yf_download_single(t, period='22d', interval='1d', auto_adjust=True, group_by=None)
                s = _series_from_df_close(raw_df).dropna()
                raw[t] = s
            except Exception as e:
                logger.info("heatmap fetch failed %s: %s", t, e)
                raw[t] = pd.Series(dtype=float)

        # helper pct (safe)
        def _pct(a, b):
            try:
                if pd.isna(a) or pd.isna(b) or b == 0:
                    return 0.0
                return float((a - b) / b * 100.0)
            except Exception:
                return 0.0

        heatmap_cells = []
        for full in watchlist:
            s = raw.get(full, pd.Series(dtype=float))
            if s.empty or len(s) < 1:
                last = float('nan')
            else:
                last = float(s.iloc[-1])
            # previous day, ~1w (~5 trading days), ~1m (~21 trading days) using available index-safe positions
            p1 = float(s.iloc[-2]) if len(s) >= 2 else float('nan')
            p5 = float(s.iloc[-6]) if len(s) >= 6 else float('nan')
            p21 = float(s.iloc[-22]) if len(s) >= 22 else float('nan')

            p1d = round(_pct(last, p1), 2) if not pd.isna(last) else 0.0
            p1w = round(_pct(last, p5), 2) if not pd.isna(last) else 0.0
            p1m = round(_pct(last, p21), 2) if not pd.isna(last) else 0.0

            cls = "pos" if p1d >= 0 else "neg"
            bucket = min(4, int(abs(p1d) // 0.5))  # 0..4 intensity buckets

            heatmap_cells.append({
                "t": full.replace('.NS','').replace('.BO',''),
                "full": full,
                "close": f"{last:,.2f}" if last == last else "—",
                "p1d": p1d, "p1w": p1w, "p1m": p1m,
                "cls": cls, "b": bucket,
                "tt": f"{full.replace('.NS','')} · ₹{last:,.2f} | 1D {p1d:+.2f}% · 1W {p1w:+.2f}% · 1M {p1m:+.2f}%"
            })

        # initial sort by 1D desc
        heatmap_cells = sorted(heatmap_cells, key=lambda x: x["p1d"], reverse=True)

        # Build lightweight html for tiles (plot_div_left)
        # We inline styles + script here so the template only needs to safe-render plot_div_left
        heat_html_parts = []
        heat_html_parts.append("<div id='hm_container'>")
        heat_html_parts.append("""
<style>
.hm-toolbar{display:flex;gap:8px;margin:8px 0 14px 0;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid #2a2e40;border-radius:10px;overflow:hidden}
.seg button{background:#1f2230;color:#cfd3dc;border:0;padding:8px 12px;cursor:pointer;font-weight:700}
.seg button.active{background:#2a2e40}
.hm{display:grid;grid-template-columns:repeat(12,1fr);gap:10px}
@media (max-width:1200px){.hm{grid-template-columns:repeat(8,1fr)}}
@media (max-width:900px){.hm{grid-template-columns:repeat(6,1fr)}}
@media (max-width:640px){.hm{grid-template-columns:repeat(3,1fr)}}
.tile{position:relative;padding:10px;border-radius:12px;border:1px solid #2a2e40;background:#1f2230;cursor:pointer;min-height:78px;display:flex;flex-direction:column;justify-content:space-between}
.tile .sym{font-weight:800;letter-spacing:.3px}
.tile .pct{font-weight:800}
.tile .small{color:#9aa0a6;font-size:.85rem}
.pos.b0{background:#122418}.pos.b1{background:#16301e}.pos.b2{background:#184a27}.pos.b3{background:#0f5c2b}.pos.b4{background:#0b6e31}
.neg.b0{background:#2a1316}.neg.b1{background:#3a171c}.neg.b2{background:#571c22}.neg.b3{background:#681c20}.neg.b4{background:#7a181b}
.tile .tip{position:absolute;left:8px;bottom:8px;transform:translateY(110%);background:#0e1118;border:1px solid #2a2e40;border-radius:10px;padding:8px 10px;font-size:.85rem;color:#d8dde7;white-space:nowrap;opacity:0;pointer-events:none;transition:opacity .12s ease}
.tile:hover .tip{opacity:1}
</style>
<div class="hm-toolbar"><div class="seg" id="hmSeg"><button type="button" class="active" data-mode="d">1D</button><button type="button" data-mode="w">1W</button><button type="button" data-mode="m">1M</button></div></div>
<div class="hm" id="hm">
""")
        # Add tiles
        for c in heatmap_cells:
            tile_html = f"""
<div class="tile {c['cls']} b{c['b']}" data-symbol="{c['full']}" data-p1d="{c['p1d']}" data-p1w="{c['p1w']}" data-p1m="{c['p1m']}">
  <div class="sym">{c['t']}</div>
  <div class="pct"><span class="arrow">{'▲' if c['p1d']>=0 else '▼'}</span> <span class="pct-val">{c['p1d']:.2f}</span>%</div>
  <div class="small">₹ {c['close']}</div>
  <div class="tip">{c['tt']}</div>
</div>
"""
            heat_html_parts.append(tile_html)
        heat_html_parts.append("</div>")  # .hm
        # Inline JS for sorting + click handler
        heat_html_parts.append("""
<script>
(function(){
  const container = document.getElementById('hm');
  const seg = document.getElementById('hmSeg');
  let mode = 'd';
  container.querySelectorAll('.tile').forEach(el=>{
    el.addEventListener('click', ()=> {
      const sym = el.dataset.symbol;
      // navigate to predict route (match your urls.py)
      // If your predict route requires number_of_days param, include e.g. /predict/<sym>/1/
      window.location.href = `/predict/${encodeURIComponent(sym)}/1/`;
    });
  });
  function repaint(){
    const tiles = Array.from(container.querySelectorAll('.tile'));
    tiles.forEach(el=>{
      const p = parseFloat(el.dataset['p1'+mode]) || 0.0;
      el.querySelector('.pct-val').textContent = p.toFixed(2);
      el.querySelector('.arrow').textContent = (p >= 0 ? '▲' : '▼');
      el.classList.remove('pos','neg','b0','b1','b2','b3','b4');
      const cls = (p >= 0 ? 'pos' : 'neg');
      const bucket = Math.min(4, Math.floor(Math.abs(p) / 0.5));
      el.classList.add(cls, 'b'+bucket);
    });
    // sort desc
    tiles.sort((a,b)=>{
      const pa = parseFloat(a.dataset['p1'+mode]) || 0.0;
      const pb = parseFloat(b.dataset['p1'+mode]) || 0.0;
      return pb - pa;
    }).forEach(el=>container.appendChild(el));
  }
  seg.querySelectorAll('button').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      seg.querySelectorAll('button').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      mode = btn.dataset.mode;
      repaint();
    });
  });
  repaint();
})();
</script>
""")
        heat_html_parts.append("</div>")  # #hm_container
        plot_div_left = "\n".join(heat_html_parts)
    except Exception as e:
        logger.info("heatmap build failed: %s", e)
        # fallback empty tiny figure (so template renders without crash)
        fig = go.Figure()
        fig.add_annotation(text="Heatmap unavailable", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        try:
            plot_div_left = plot(fig, auto_open=False, output_type='div')
        except Exception:
            plot_div_left = "<div>Chart unavailable</div>"

    # ---------- movers across NIFTY50 (safe) ----------
    try:
        movers = []
        for t in nifty50:
            try:
                df_t = yf_download_single(t, period='10d', interval='1d', auto_adjust=True, group_by=None)
                s = _series_from_df_close(df_t).dropna()
                if len(s) < 2:
                    continue
                last, prev = float(s.iloc[-1]), float(s.iloc[-2])
                pct = (last - prev) / prev * 100.0 if prev else 0.0
                movers.append({"t": t.replace('.NS',''), "pct": pct, "close": f"{last:,.2f}"})
            except Exception as ie:
                logger.info("mover calc fail %s: %s", t, ie)
                continue
        gainers = sorted([m for m in movers if m["pct"] >= 0], key=lambda r: -r["pct"])[:10]
        losers  = sorted([m for m in movers if m["pct"]  < 0], key=lambda r:  r["pct"])[:10]
        for r in gainers + losers:
            r["pct_str"] = f"{r['pct']:.2f}"
        chips = [t.replace('.NS','') for t in nifty50]
    except Exception as e:
        logger.info("movers block failed: %s", e)

    # ---------- ALWAYS render ----------
    return render(request, 'index.html', {
        'index_cards': index_cards,
        'plot_div_left': plot_div_left,
        'gainers': gainers,
        'losers': losers,
        'chips': chips,
    })



def search(request):
    return render(request, 'search.html')


def predict(request, ticker_value, number_of_days):
    """
    Stocks -> regression (price), Index -> classification (direction).
    Saves CSVs. Theme-aware charts. Safer fundamentals (IPO, 52w). Shows best metric value.
    """
    colors = _plotly_theme_colors(request)

    if request.method == 'POST':
        ticker_value = request.POST.get('ticker', ticker_value)
    ticker_value = normalize_in_symbol(ticker_value)
    index_mode = is_index_symbol(ticker_value)
    horizon = 1

    # Intraday
    df_intra = yf_download_single(ticker_value, period='1d', interval='1m', auto_adjust=True, max_retries=2)
    if df_intra.empty:
        df_intra = yf_download_single(ticker_value, period='5d', interval='5m', auto_adjust=True, max_retries=2)

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
        yaxis_title='Price', paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
        xaxis=dict(gridcolor=colors["grid"]), yaxis=dict(gridcolor=colors["grid"])
    )
    fig_candle.update_xaxes(rangeslider_visible=True)
    plot_div = plot(fig_candle, auto_open=False, output_type='div')

    # Daily history
    hist_period = '10y' if index_mode else '5y'
    hist = yf_download_single(ticker_value, period=hist_period, interval='1d', auto_adjust=True)
    if hist.empty or 'Close' not in hist:
        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
            'error_message': "No historical data available (Yahoo empty/ratelimit).",
            'fundamentals': {}, 'metrics': [], 'best_model': "—", 'best_value': "—",
            'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
            'number_of_days': horizon, 'task_mode': 'classification' if index_mode else 'regression',
            'next_day_up_prob': None,
        })

    keep = [c for c in ['Open','High','Low','Close','Volume'] if c in hist.columns]
    hist = hist[keep].dropna(subset=['Close']).copy()

    # Market for features
    mkt = yf_download_single('^NSEI', period=hist_period, interval='1d', auto_adjust=True)
    if mkt.empty or 'Close' not in mkt:
        mkt = pd.DataFrame(index=hist.index, data={'Close': hist['Close'].rolling(10).mean().ffill()})
    else:
        mkt = mkt[['Close']].reindex(hist.index).ffill().bfill()

    vol_ok = 'Volume' in hist and (hist['Volume'].dropna().sum() > 0) and (not index_mode)

    df_feat = build_features(hist, mkt, use_volume=vol_ok).sort_index().copy()
    close_series = hist['Close'].sort_index()
    # Do NOT reindex df_feat to close_series (it introduces NaNs)

    # ---------------- INDEX: classification ----------------
    if index_mode:
        lookahead = 2  # smoother than 1-day noise
        y_dir = (close_series.shift(-lookahead) > close_series).astype(int).rename('target_up')

        df_all = df_feat.join(y_dir, how='inner')
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

        feature_cols = [c for c in df_all.columns if c != 'target_up']
        if len(df_all) < 150 or not feature_cols:
            fund = get_fundamentals(ticker_value, hist_df=hist)
            return render(request, 'result.html', {
                'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
                'error_message': "Not enough clean history to train index classifier.",
                'fundamentals': fund, 'metrics': [], 'best_model': "—", 'best_value': "—",
                'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
                'number_of_days': horizon, 'task_mode': 'classification', 'next_day_up_prob': None,
            })

        X = df_all[feature_cols].values
        y = df_all['target_up'].values.astype(int)

        split_idx = int(len(df_all)*0.75)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        idx_test = df_all.index[split_idx:]

        # validation slice
        val_frac = 0.15
        val_start = max(5, int(len(X_train)*(1 - val_frac)))
        X_tr, X_val = X_train[:val_start], X_train[val_start:]
        y_tr, y_val = y_train[:val_start], y_train[val_start:]

        # impute + scale
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='median')
        X_tr_imp = imp.fit_transform(X_tr)
        X_val_imp = imp.transform(X_val)
        X_test_imp = imp.transform(X_test)
        X_last_imp = imp.transform(df_all[feature_cols].iloc[[-1]].values)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_val_s = scaler.transform(X_val_imp)
        X_test_s = scaler.transform(X_test_imp)
        X_last_s = scaler.transform(X_last_imp)

        def best_threshold(proba, y_true):
            ths = np.linspace(0.35, 0.65, 31)
            accs = [accuracy_score(y_true, (proba >= t).astype(int)) for t in ths]
            i = int(np.argmax(accs))
            return float(ths[i]), float(accs[i]*100.0)

        models = {}
        models['LogReg'] = LogisticRegression(max_iter=200, solver='lbfgs').fit(X_tr_s, y_tr)
        models['RF_cls'] = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1).fit(X_tr_imp, y_tr)
        if _HAS_XGB:
            try:
                models['XGB_cls'] = XGBClassifier(
                    n_estimators=600, max_depth=5, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                    objective='binary:logistic', random_state=42, n_jobs=0
                ).fit(X_tr_imp, y_tr)
            except Exception as e:
                logger.info("XGB_cls skipped: %s", e)

        evals, preds_df, val_prob_store = {}, {}, {}

        for name, mdl in models.items():
            if name == 'LogReg':
                proba_val = mdl.predict_proba(X_val_s)[:,1]
                th, _ = best_threshold(proba_val, y_val)
                proba_test = mdl.predict_proba(X_test_s)[:,1]
                next_p = float(mdl.predict_proba(X_last_s)[0,1])
            else:
                proba_val = mdl.predict_proba(X_val_imp)[:,1]
                th, _ = best_threshold(proba_val, y_val)
                proba_test = mdl.predict_proba(X_test_imp)[:,1]
                next_p = float(mdl.predict_proba(X_last_imp)[0,1])

            val_prob_store[name] = proba_val
            y_hat = (proba_test >= th).astype(int)

            evals[name] = {
                "acc": float(accuracy_score(y_test, y_hat))*100.0,
                "prec": float(precision_score(y_test, y_hat, zero_division=0))*100.0,
                "rec": float(recall_score(y_test, y_hat, zero_division=0))*100.0,
                "next_up_prob": next_p,
                "threshold": th,
                "proba": proba_test
            }
            preds_df[name] = pd.Series(y_hat, index=idx_test)

        # Ensemble with its own threshold
        if evals:
            members = list(val_prob_store.keys())
            if members:
                avg_val = np.mean([val_prob_store[m] for m in members], axis=0)
                ens_th, _ = best_threshold(avg_val, y_val)
            else:
                ens_th = 0.5
            prob_mat = np.column_stack([v["proba"] for v in evals.values()])
            ens_prob = prob_mat.mean(axis=1)
            ens_hat  = (ens_prob >= ens_th).astype(int)
            evals['Ensemble'] = {
                "acc": float(accuracy_score(y_test, ens_hat))*100.0,
                "prec": float(precision_score(y_test, ens_hat, zero_division=0))*100.0,
                "rec": float(recall_score(y_test, ens_hat, zero_division=0))*100.0,
                "next_up_prob": float(np.mean([v["next_up_prob"] for v in evals.values()])),
                "threshold": float(ens_th),
                "proba": ens_prob
            }
            preds_df['Ensemble'] = pd.Series(ens_hat, index=idx_test)

        # Chart
        palette = {"LogReg":"#22c55e","RF_cls":"#f59e0b","XGB_cls":"#ef4444","Ensemble":"#60a5fa"}
        fig_pred_all = go.Figure()
        fig_pred_all.add_trace(go.Scatter(x=close_series.index, y=close_series.values,
                                          name='Actual', mode='lines', line=dict(width=2)))
        for name, series in preds_df.items():
            fig_pred_all.add_trace(go.Scatter(
                x=series.index, y=(series.values*(close_series.reindex(series.index).pct_change().std()*1000)).astype(float),
                name=name, mode='lines', line=dict(width=2, dash='dash', color=palette.get(name))
            ))
        fig_pred_all.update_layout(
            title=f"{ticker_value} — Direction Predictions (Test; lookahead={lookahead}d)",
            paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
            hovermode="x unified",
            xaxis=dict(type="date", rangeslider=dict(visible=True), gridcolor=colors["grid"]),
            yaxis=dict(title="(Scaled) Direction Overlay", gridcolor=colors["grid"])
        )
        plot_div_pred = plot(fig_pred_all, auto_open=False, output_type='div')

        metrics = [{
            "Model": name,
            "ACC": round(res["acc"], 2),
            "Precision": round(res["prec"], 2),
            "Recall": round(res["rec"], 2),
            "NextUpProb": round(res["next_up_prob"]*100.0, 2)
        } for name, res in evals.items()]

        metrics_sorted = sorted(metrics, key=lambda m: (-m["ACC"], -m["Precision"], -m["Recall"])) if metrics else []
        best_model = metrics_sorted[0]["Model"] if metrics_sorted else "—"

        # >>> Always show the TRUE best model; do NOT prefer Ensemble by default
        headline_key = best_model
        headline = next((m for m in metrics if m["Model"] == headline_key), None)

        accuracy_val = headline["ACC"] if headline else 0.0
        next_up_prob = headline["NextUpProb"] if headline else None
        best_value = accuracy_val  # shown on the card


        fund = get_fundamentals(ticker_value, hist_df=hist)

        # Save CSVs
        base_media = getattr(settings, "MEDIA_ROOT", os.path.join(getattr(settings,"BASE_DIR",os.getcwd()), "media"))
        out_dir = os.path.join(base_media, "training"); os.makedirs(out_dir, exist_ok=True)
        safe_t = ticker_value.replace(".", "_")
        feat_path = os.path.join(out_dir, f"{safe_t}_features.csv")
        test_path = os.path.join(out_dir, f"{safe_t}_test_actual_vs_preds.csv")
        df_all_out = df_all.copy(); df_all_out.insert(0, "Ticker", ticker_value); df_all_out.to_csv(feat_path, index=True)
        out_pred = pd.DataFrame(index=idx_test); out_pred["ActualDir"] = y_test
        for name, res in evals.items():
            out_pred[name+"_proba"] = res["proba"]
        out_pred.to_csv(test_path, index=True)

        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': plot_div_pred,
            'ticker_value': ticker_value,
            'accuracy': accuracy_val,
            'confidence': accuracy_val,
            'next_day_prediction': None,
            'next_day_up_prob': next_up_prob,
            'number_of_days': horizon,
            'fundamentals': fund,
            'metrics': metrics_sorted,
            'best_model': best_model,
            'best_value': round(best_value, 2),
            'csv_features_path': feat_path,
            'csv_test_path': test_path,
            'task_mode': 'classification',
            'error_message': None
        })

    # ---------------- STOCKS: regression ----------------
    df_all = df_feat.copy()
    target_series = close_series.shift(-1).rename('target_price')
    df_all = df_all.join(target_series, how='inner')
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

    if len(df_all) < 100:
        fund = get_fundamentals(ticker_value, hist_df=hist)
        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
            'error_message': "Not enough clean history to train (need ~100+ rows after indicators).",
            'fundamentals': fund, 'metrics': [], 'best_model': "—", 'best_value': "—",
            'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
            'number_of_days': horizon, 'task_mode': 'regression', 'next_day_up_prob': None
        })

    feature_cols = [c for c in df_all.columns if c != 'target_price']
    X = df_all[feature_cols].values
    y = df_all['target_price'].values

    split_idx = int(len(df_all) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    idx_test = df_all.index[split_idx:]

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)
    X_last_imp  = imputer.transform(df_all[feature_cols].iloc[[-1]].values)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s  = scaler.transform(X_test_imp)
    feats_last_s = scaler.transform(X_last_imp)

    models = {}
    models['Ridge'] = Ridge(alpha=1.0).fit(X_train_s, y_train)
    models['RandomForest'] = RandomForestRegressor(n_estimators=700, random_state=42, n_jobs=-1).fit(X_train_imp, y_train)
    if _HAS_XGB:
        try:
            models['XGBoost'] = XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective='reg:squarederror', random_state=42, n_jobs=0
            ).fit(X_train_imp, y_train)
        except Exception as e:
            logger.info("XGBRegressor skipped: %s", e)

    evals, preds_df = {}, {}
    cur_close_test = close_series.reindex(idx_test).values
    for name, mdl in models.items():
        if name == 'Ridge':
            y_hat = mdl.predict(X_test_s)
            next_pred = float(mdl.predict(feats_last_s)[0])
        else:
            y_hat = mdl.predict(X_test_imp)
            next_pred = float(mdl.predict(X_last_imp)[0])

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

    metrics = [{
        "Model": name,
        "R2": round(res["r2"], 4),
        "MAE_ret": round(res["mae_pct"], 3),
        "Direction": round(res["dir_acc"], 2),
        "NextDayPrice": round(res["next_price"], 2)
    } for name, res in evals.items()]

    metrics_sorted = sorted(metrics, key=lambda m: (-m["R2"], -m["Direction"], m["MAE_ret"])) if metrics else []
    best_model = metrics_sorted[0]["Model"] if metrics_sorted else "—"

    # >>> Always show the TRUE best model; do NOT prefer Ensemble by default
    headline_key = best_model
    headline = next((m for m in metrics if m["Model"] == headline_key), None)

    accuracy_val = headline["R2"] if headline else 0.0
    next_price_val = headline["NextDayPrice"] if headline else float('nan')
    best_value = accuracy_val  # shown on the card

    fund = get_fundamentals(ticker_value, hist_df=hist)

    base_media = getattr(settings, "MEDIA_ROOT", os.path.join(getattr(settings,"BASE_DIR",os.getcwd()), "media"))
    out_dir = os.path.join(base_media, "training"); os.makedirs(out_dir, exist_ok=True)
    safe_t = ticker_value.replace(".", "_")
    feat_path = os.path.join(out_dir, f"{safe_t}_features.csv")
    test_path = os.path.join(out_dir, f"{safe_t}_test_actual_vs_preds.csv")
    df_all_out = df_all.copy(); df_all_out.insert(0, "Ticker", ticker_value); df_all_out.to_csv(feat_path, index=True)

    out_pred = pd.DataFrame(index=idx_test)
    out_pred["Actual"] = y_test
    for name, res in evals.items(): out_pred[name] = res["y_pred"]
    out_pred.to_csv(test_path, index=True)

    return render(request, 'result.html', {
        'plot_div': plot_div,
        'plot_div_pred': plot_div_pred,
        'ticker_value': ticker_value,
        'accuracy': accuracy_val,
        'confidence': accuracy_val,
        'next_day_prediction': next_price_val,
        'next_day_up_prob': None,
        'number_of_days': horizon,
        'fundamentals': fund,
        'metrics': metrics_sorted,
        'best_model': best_model,
        'best_value': round(best_value, 4),
        'csv_features_path': feat_path,
        'csv_test_path': test_path,
        'task_mode': 'regression',
        'error_message': None
    })


# app/views.py
import os
import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.conf import settings

def autocomplete_tickers(request):
    """
    Simple server-side autocomplete endpoint.
    Query param: q
    Returns JSON array: [{label, value, meta}, ...]
    """
    q = (request.GET.get('q') or '').strip().lower()
    # path to the JSON created by the fetch script
    file_path = os.path.join(settings.BASE_DIR, 'app', 'static', 'app', 'data', 'nse_tickers.json')
    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except FileNotFoundError:
        # helpful JSON response for debugging in dev
        return JsonResponse({"error": "ticker list not found", "path": file_path}, status=500)
    except Exception as e:
        return JsonResponse({"error": "failed to read ticker list", "detail": str(e)}, status=500)

    # if no query, return a small top sample to avoid huge payload
    if not q:
        return JsonResponse(data[:50], safe=False)

    out = []
    # simple substring match with limit
    for item in data:
        label = (item.get('label') or '').lower()
        meta = (item.get('meta') or '').lower()
        val = (item.get('value') or '').lower()
        combined = f"{label} {meta} {val}"
        if q in combined:
            out.append(item)
        if len(out) >= 50:
            break

    return JsonResponse(out, safe=False)
