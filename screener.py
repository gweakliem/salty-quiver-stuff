"""

🧠 What It Does
        •	Filters for MACD bullish cross AND RSI between 50–70
        •	Returns tickers that might support bull call spreads or long calls
        •	Shows first expiration’s call chain for your inspection
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from math import exp, log, sqrt
from pathlib import Path
from pprint import pprint

import click
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from pydantic import BaseModel
from scipy.stats import norm

# fallback static lists (used only if live fetch and cache both fail)
default_lists = {
    "genz": [
        "OPEN",
        "XYZ",
        "CRSP",
        "ASTS",
        "SMR",
        "CRCL",
        "NBIS",
        "QUBT",
        "RGTI",
        "QBTS",
        "NXE",
        "TLN",
        "LEU",
        "VST",
        "GLXY",
        "CIFR",
        "CLSK",
        "RIOT",
        "HUT",
        "BWXT",
        "CCJ",
        "GEV",
        "CEG",
        "OKLO",
        "AMTX",
        "RKT",
        "RIVN",
        "LCID",
    ],
    "nasdaq_100": [
        "ADBE",
        "AMD",
        "ABNB",
        "GOOGL",
        "GOOG",
        "AMZN",
        "AEP",
        "AMGN",
        "ADI",
        "ANSS",
        "AAPL",
        "AMAT",
        "APP",
        "ARM",
        "ASML",
        "AZN",
        "TEAM",
        "ADSK",
        "ADP",
        "AXON",
        "BKR",
        "BIIB",
        "BKNG",
        "AVGO",
        "CDNS",
        "CDW",
        "CHTR",
        "CTAS",
        "CSCO",
        "CCEP",
        "CTSH",
        "CMCSA",
        "CEG",
        "CPRT",
        "CSGP",
        "COST",
        "CRWD",
        "CSX",
        "DDOG",
        "DXCM",
        "FANG",
        "DASH",
        "EA",
        "EXC",
        "FAST",
        "FTNT",
        "GEHC",
        "GILD",
        "GFS",
        "HON",
        "IDXX",
        "INTC",
        "INTU",
        "ISRG",
        "KDP",
        "KLAC",
        "KHC",
        "LRCX",
        "LIN",
        "LULU",
        "MAR",
        "MRVL",
        "MELI",
        "META",
        "MCHP",
        "MU",
        "MSFT",
        "MSTR",
        "MDLZ",
        "MNST",
        "NFLX",
        "NVDA",
        "NXPI",
        "ORLY",
        "ODFL",
        "ON",
        "PCAR",
        "PLTR",
        "PANW",
        "PAYX",
        "PYPL",
        "PDD",
        "PEP",
        "QCOM",
        "REGN",
        "ROP",
        "ROST",
        "SHOP",
        "SBUX",
        "SNPS",
        "TTWO",
        "TMUS",
        "TSLA",
        "TXN",
        "TTD",
        "VRSK",
        "VRTX",
        "WBD",
        "WDAY",
        "XEL",
        "ZS",
    ],
}

INDEX_SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq_100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}

# Alternate CSV mirrors for S&P 500 (useful when Wikipedia blocks scraping)
SP500_CSV_FALLBACKS = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
]

CACHE_DIR = Path(".cache/index_constituents")


def _normalize_ticker(ticker: str) -> str:
    """Normalize tickers (e.g., BRK.B → BRK-B) for consistency across data sources."""

    return ticker.strip().upper().replace(".", "-")


def _save_cache(index_name: str, tickers: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{index_name}.txt"
    cache_path.write_text("\n".join(tickers), encoding="utf-8")


def _load_cache(index_name: str) -> list[str]:
    cache_path = CACHE_DIR / f"{index_name}.txt"
    if cache_path.exists():
        return [_normalize_ticker(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line]
    return []


def _fetch_sp500() -> list[str]:
    # Primary: Wikipedia (allow custom UA to avoid some 403 blocks)
    try:
        tables = pd.read_html(
            INDEX_SOURCES["sp500"],
            storage_options={"User-Agent": "Mozilla/5.0 (compatible; CodexBot/1.0)"},
        )
        tickers = tables[0]["Symbol"].tolist()
        return [_normalize_ticker(t) for t in tickers]
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Wikipedia S&P 500 scrape failed: {exc}")

    # Secondary: CSV mirrors
    for url in SP500_CSV_FALLBACKS:
        try:
            df = pd.read_csv(url)
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            tickers = df[col].tolist()
            return [_normalize_ticker(t) for t in tickers]
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Fallback CSV fetch failed ({url}): {exc}")

    raise RuntimeError("All S&P 500 sources failed")


def _fetch_nasdaq_100() -> list[str]:
    tables = pd.read_html(INDEX_SOURCES["nasdaq_100"])
    table = next(
        tbl for tbl in tables if any(col in tbl.columns for col in ("Ticker", "Symbol"))
    )
    col = "Ticker" if "Ticker" in table.columns else "Symbol"
    return [_normalize_ticker(t) for t in table[col].tolist()]


def get_index_constituents(index_name: str) -> list[str]:
    """Fetch index constituents with caching and fallbacks.

    Order of preference:
    1) Live scrape from Wikipedia (keeps list current).
    2) Cached list from previous successful run.
    3) Hard-coded defaults (only for lists we ship).
    """

    index_name = index_name.lower()
    fetchers = {
        "sp500": _fetch_sp500,
        "nasdaq_100": _fetch_nasdaq_100,
        "genz": lambda: default_lists["genz"],
    }

    if index_name not in fetchers:
        raise ValueError(f"Unknown index '{index_name}'")

    tickers: list[str] = []

    # Try live fetch
    try:
        tickers = fetchers[index_name]()
        if tickers:
            _save_cache(index_name, tickers)
            return tickers
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Could not refresh {index_name} constituents live: {exc}")

    # Fallback to cache
    cached = _load_cache(index_name)
    if cached:
        print(f"Using cached list for {index_name} (live refresh failed).")
        return cached

    # Fallback to hard-coded defaults
    if index_name in default_lists:
        print(f"Using built-in default list for {index_name} (no cache available).")
        return default_lists[index_name]

    raise RuntimeError(f"No tickers available for index '{index_name}'")


class ScreeningResult(BaseModel):
    ticker: str
    price: float
    rsi: float
    macd: float
    signal: float
    sma_50: float
    sma_200: float


def init_database(db_path: str) -> None:
    """Initialize SQLite database with screening results table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screening_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            sma_50 REAL,
            sma_200 REAL
        )
    """)

    conn.commit()
    conn.close()


def black_scholes_call_delta(S, K, T, r, sigma):
    """Annualized inputs. T in years."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def pick_call_strike(
    tk: yf.Ticker,
    expiry: str,
    target_delta: float = 0.60,
    min_volume: float = 100,
    min_oi: float = 100,
    risk_free_rate: float = 0.045,
    delta_window: tuple[float, float] | None = (0.35, 0.75),
):
    """
    Picks the best call option strike for a given ticker and expiry.

    tk: yf.Ticker object for the stock
    expiry: expiration date in "YYYY-MM-DD" format
    target_delta: desired delta for the call option (default 0.60)
    min_volume: minimum volume for the option contract (default 100)
    min_oi: minimum open interest for the option contract (default 100)
    risk_free_rate: annualized risk-free rate (default 0.045)
    """
    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()

    # Convert expiration to year fraction
    exp_date = datetime.strptime(expiry, "%Y-%m-%d")
    today = datetime.today()
    T = max((exp_date - today).days / 365.0, 0)

    # Underlying price
    S = tk.history(period="1d")["Close"].iloc[-1]

    # Use option-implied vol in chain (per-contract); fallback if missing
    # yfinance returns decimal IV (e.g., 0.35)
    calls = calls.dropna(subset=["impliedVolatility"])
    calls = calls[calls["volume"].fillna(0) >= min_volume]
    calls = calls[calls["openInterest"].fillna(0) >= min_oi]

    if calls.empty:
        return None

    # Approximate delta for each strike
    deltas = []
    for _, row in calls.iterrows():
        K = row["strike"]
        sigma = row["impliedVolatility"]
        delta = black_scholes_call_delta(S, K, T, risk_free_rate, sigma)
        deltas.append(delta)
    calls = calls.assign(delta=deltas)

    # Find closest to target_delta
    calls["delta_diff"] = (calls["delta"] - target_delta).abs()
    calls = calls.sort_values("delta_diff")

    # Prefer strikes inside delta window; otherwise accept closest overall
    if delta_window:
        low, high = delta_window
        in_window = calls[(calls["delta"] >= low) & (calls["delta"] <= high)]
        if not in_window.empty:
            best = in_window.iloc[0].copy()
        else:
            return None
    else:
        best = calls.iloc[0].copy()

    # Breakeven if held to expiration
    ask = best.get("ask", np.nan)
    bid = best.get("bid", np.nan)
    last = best.get("lastPrice", np.nan)
    mid = np.nanmean([bid, ask]) if not np.isnan(bid) and not np.isnan(ask) else last

    breakeven = best["strike"] + (mid if pd.notna(mid) else 0)
    breakeven_pct = (breakeven - S) / S * 100

    best["mid"] = mid
    best["breakeven"] = breakeven
    best["breakeven_pct"] = breakeven_pct
    best["underlying"] = S
    best["days_to_exp"] = (exp_date - today).days

    return best[
        [
            "underlying",
            "strike",
            "mid",
            "bid",
            "ask",
            "lastPrice",
            "delta",
            "impliedVolatility",
            "breakeven",
            "breakeven_pct",
            "volume",
            "openInterest",
            "days_to_exp",
        ]
    ]


def get_target_expiry(stock: yf.Ticker, min_days=30, max_days=45) -> str | None:
    try:
        expirations = stock.options
        today = datetime.today()
        target_expiry = None

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            delta_days = (exp_date - today).days

            if min_days <= delta_days <= max_days:
                target_expiry = exp_str
                break  # take the earliest that fits
        return target_expiry
    except Exception as e:
        print(f"Error retrieving expiry: {e}")
        return None


def screen_tickers(
    tickers: str,
    period: str = "1y",
    interval: str = "1d",
    skip_screen: bool = False,
) -> list[ScreeningResult]:
    """
    Screens tickers for bullish momentum based on MACD and RSI indicators.
    Returns a list of ScreeningResult objects with ticker information.
    """
    screened = []

    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        if len(df) < 50:  # TODO: 200?
            continue

        # Indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"].squeeze(), window=14).rsi()
        macd = ta.trend.MACD(df["Close"].squeeze())
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["sma_50"] = ta.trend.SMAIndicator(
            df["Close"].squeeze(), window=50
        ).sma_indicator()
        df["sma_200"] = ta.trend.SMAIndicator(
            df["Close"].squeeze(), window=200
        ).sma_indicator()

        current_price = df["Close"].iloc[-1]
        current_rsi = df["rsi"].iloc[-1]
        current_macd = df["macd"].iloc[-1]
        current_signal = df["macd_signal"].iloc[-1]
        sma_50 = df["sma_50"].iloc[-1]
        sma_200 = df["sma_200"].iloc[-1]

        # Check screening conditions
        macd_crossover = (
            current_macd > current_signal
            and df["macd"].iloc[-2] <= df["macd_signal"].iloc[-2]
        )
        rsi_in_range = 50 < current_rsi < 70
        passed_screen = skip_screen or (macd_crossover and rsi_in_range)

        # Add to screened results if passed
        if passed_screen:
            screened.append(
                ScreeningResult(
                    ticker=ticker,
                    price=current_price,
                    rsi=round(current_rsi, 2),
                    macd=round(current_macd, 3),
                    signal=round(current_signal, 3),
                    sma_50=round(sma_50, 2),
                    sma_200=round(sma_200, 2),
                )
            )
        else:
            print(
                f"Skipping {ticker}: MACD={current_macd}, Signal={current_signal}, RSI={current_rsi}"
            )

    return screened


@click.command()
@click.option(
    "--tickers",
    "-t",
    help="Comma-separated list of tickers to screen (overrides --index list)",
    type=str,
)
@click.option(
    "--index",
    "index_name",
    default="nasdaq_100",
    type=click.Choice(["nasdaq_100", "sp500", "genz"], case_sensitive=False),
    show_default=True,
    help="Use a dynamic list from the given index",
)
@click.option(
    "--target-delta",
    "-d",
    default=0.60,
    type=float,
    help="Target delta for options selection (default: 0.60)",
)
@click.option(
    "--min-delta",
    default=0.35,
    show_default=True,
    type=float,
    help="Lower bound for acceptable option delta (set 0 to disable)",
)
@click.option(
    "--max-delta",
    default=0.75,
    show_default=True,
    type=float,
    help="Upper bound for acceptable option delta (set 1 to disable)",
)
@click.option(
    "--min-expiry",
    default=30,
    type=int,
    help="Minimum days to expiry for options (default: 30)",
)
@click.option(
    "--max-expiry",
    default=45,
    type=int,
    help="Maximum days to expiry for options (default: 45)",
)
@click.option(
    "--skip-screen",
    is_flag=True,
    help="Skip the screening step, just apply options selection",
)
def main(
    tickers: list[str] | None,
    index_name: str,
    target_delta: float,
    min_delta: float,
    max_delta: float,
    min_expiry: int,
    max_expiry: int,
    skip_screen: bool,
):
    """
    Screen stocks for bullish momentum using MACD and RSI indicators.
    Find potential candidates for bull call spreads or long calls.
    """
    if min_expiry >= max_expiry:
        click.echo("min_expiry must be <= max_expiry")
        sys.exit(1)

    db_path = "screener_results.db"

    if tickers:
        ticker_list = [_normalize_ticker(ticker) for ticker in tickers.split(",")]
    else:
        ticker_list = get_index_constituents(index_name)

    screened = screen_tickers(ticker_list, skip_screen=skip_screen)
    # Display screened tickers
    screen_df = pd.DataFrame([result.model_dump() for result in screened])
    print("\n" + "=" * 80)
    print("📈 BULLISH MOMENTUM CANDIDATES")
    print("=" * 80)
    if not screen_df.empty:
        print(screen_df.to_string(index=False))
    else:
        print("No stocks passed the screening criteria.")
    print("=" * 80 + "\n")

    # Options chain for all screened tickers
    if not screen_df.empty:
        timestamp = datetime.now().isoformat()

        # Initialize database
        init_database(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("=" * 80)
        print("🎯 TRADE RECOMMENDATIONS")
        print("=" * 80 + "\n")

        for result in screened:
            ticker = result.ticker
            stock = yf.Ticker(ticker)
            current_price = result.price
            current_rsi = result.rsi
            current_macd = result.macd
            current_signal = result.signal
            sma_50 = result.sma_50
            sma_200 = result.sma_200

            # Insert screening result into database
            cursor.execute(
                """
                INSERT INTO screening_results
                (timestamp, ticker, price, rsi, macd, macd_signal, sma_50, sma_200)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    ticker,
                    current_price,
                    current_rsi,
                    current_macd,
                    current_signal,
                    sma_50,
                    sma_200,
                ),
            )

            print(
                f"Screening for expirations between {min_expiry} and {max_expiry} days out"
            )
            try:
                expiry = get_target_expiry(stock, min_expiry, max_expiry)
                if expiry:
                    delta_window = None
                    if 0 < min_delta <= max_delta < 1:
                        delta_window = (min_delta, max_delta)

                    best_call = pick_call_strike(
                        stock,
                        expiry,
                        target_delta=target_delta,
                        delta_window=delta_window,
                    )
                    if best_call is not None:
                        # Format the trade recommendation
                        print(
                            f"┌─ {ticker} Trade Recommendation "
                            + "─" * (80 - len(ticker) - 26)
                        )
                        print("│")
                        print("│ 📊 STOCK INFO")
                        print(f"│   Current Price: ${current_price:.2f}")
                        print(f"│   RSI: {current_rsi:.1f} (bullish momentum)")
                        print(
                            f"│   MACD: {current_macd:.3f} > Signal: {current_signal:.3f} ✓"
                        )
                        print("│")
                        print("│ 🎯 RECOMMENDED TRADE")
                        print(
                            f"│   BUY TO OPEN: {ticker} ${best_call['strike']:.2f} Call"
                        )
                        print(
                            f"│   Expiration: {expiry} ({best_call['days_to_exp']:.0f} days)"
                        )
                        print("│")
                        print("│ 💰 PRICING")
                        print(f"│   Option Price: ${best_call['mid']:.2f} (mid)")
                        print(
                            f"│   Bid/Ask: ${best_call['bid']:.2f} / ${best_call['ask']:.2f}"
                        )
                        print(f"│   Last Trade: ${best_call['lastPrice']:.2f}")
                        print("│")
                        print("│ 📈 GREEKS & METRICS")
                        print(
                            f"│   Delta: {best_call['delta']:.3f} (~{best_call['delta'] * 100:.0f}% prob ITM)"
                        )
                        print(
                            f"│   Implied Vol: {best_call['impliedVolatility'] * 100:.1f}%"
                        )
                        print(
                            f"│   Breakeven: ${best_call['breakeven']:.2f} ({best_call['breakeven_pct']:+.1f}% from current)"
                        )
                        print("│")
                        print("│ 🔊 LIQUIDITY")
                        print(f"│   Volume: {best_call['volume']:.0f}")
                        print(f"│   Open Interest: {best_call['openInterest']:.0f}")
                        print("└" + "─" * 79)
                        print()
                    else:
                        print(f"⚠️  {ticker}: No suitable call found for {expiry}")
                        print(
                            "   (No options met liquidity requirements: min 100 volume, 100 OI)"
                        )
                        print()
                else:
                    print(f"⚠️  {ticker}: No suitable expiry found")
                    print(
                        f"   (Looking for expiration between {min_expiry}-{max_expiry} days)"
                    )
                    print()
            except Exception as e:
                print(f"❌ {ticker}: Could not fetch options - {e}")
                print()

        conn.commit()
        conn.close()

        print("=" * 80)
        print("✅ Screening complete. Results saved to database.")
        print("=" * 80)


if __name__ == "__main__":
    main()
