"""

🧠 What It Does
        •	Filters for MACD bullish cross AND RSI between 50–70
        •	Returns tickers that might support bull call spreads or long calls
        •	Shows first expiration’s call chain for your inspection
"""

from datetime import datetime, timedelta
import click
from pprint import pprint
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import sqlite3
from pydantic import BaseModel

# default lists
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
        "QQQ",
    ],
}


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


def get_target_expiry(stock: yf.Ticker, min_days=30, max_days=45) -> str:
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
    tickers: list[str],
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
    help="Comma-separated list of tickers to screen (overrides default nasdaq_100)",
    type=str,
)
@click.option(
    "--target-delta",
    "-d",
    default=0.60,
    type=float,
    help="Target delta for options selection (default: 0.60)",
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
    tickers: list[str],
    target_delta: float,
    min_expiry: str,
    max_expiry: str,
    skip_screen: bool,
):
    """
    Screen stocks for bullish momentum using MACD and RSI indicators.
    Find potential candidates for bull call spreads or long calls.
    """
    db_path = "screener_results.db"
    # Use provided tickers or default to nasdaq_100
    if tickers:
        ticker_list = [ticker.strip().upper() for ticker in tickers.split(",")]
    else:
        ticker_list = default_lists["nasdaq_100"]

    screened = screen_tickers(ticker_list, skip_screen=skip_screen)
    # Display screened tickers
    screen_df = pd.DataFrame([result.model_dump() for result in screened])
    print("📈 Bullish Momentum Candidates:")
    print(screen_df)

    # Options chain for all screened tickers
    if not screen_df.empty:
        timestamp = datetime.now().isoformat()

        # Initialize database
        init_database(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for result in screened:
            ticker = result.ticker
            stock = yf.Ticker(ticker)
            current_price = result.price
            current_rsi = result.rsi
            current_macd = result.macd
            current_signal = result.signal
            sma_50 = result.sma_50
            sma_200 = result.sma_200
            print(
                f"Inserting result {ticker}: {current_price}, {current_rsi}, {current_macd}, {current_signal}, {sma_50}, {sma_200}"
            )
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

            print(f"\n🔍 Fetching options for {ticker}")
            try:
                expiry = get_target_expiry(stock, min_expiry, max_expiry)
                if expiry:
                    best_call = pick_call_strike(
                        stock, expiry, target_delta=target_delta
                    )
                    if best_call is not None:
                        print(f"{ticker} {expiry} {best_call.to_dict()}")
                    else:
                        print(f"⚠️ No suitable call found for {ticker} on {expiry}")
                else:
                    print(f"⚠️ No suitable expiry found for {ticker}")
            except Exception as e:
                print(f"❌ Could not fetch options for {ticker}: {e}")
        conn.commit()
        conn.close()


if __name__ == "__main__":
    main()
