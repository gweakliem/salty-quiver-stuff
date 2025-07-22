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

# default lists
genz = [
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
]

nasdaq_100 = [
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
]


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
        print(f"Error retrieving expiry for {ticker}: {e}")
        return None


def screen_tickers(tickers: list[str], period: str = "6mo", interval: str = "1d", skip_screen: bool = False) -> list[dict]:
    """
    Screens tickers for bullish momentum based on MACD and RSI indicators.
    Returns a list of dictionaries with ticker information.
    """
    screened = []

    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        if len(df) < 50:
            continue

        # Indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"].squeeze(), window=14).rsi()
        macd = ta.trend.MACD(df["Close"].squeeze())
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # Signal Conditions
        if skip_screen or (
            df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
            and df["macd"].iloc[-2] <= df["macd_signal"].iloc[-2]  # crossover
            and 50 < df["rsi"].iloc[-1] < 70
        ):
            current_price = df["Close"].iloc[-1]
            screened.append(
                {
                    "Ticker": ticker,
                    "Price": current_price,
                    "RSI": round(df["rsi"].iloc[-1], 2),
                    "MACD": round(df["macd"].iloc[-1], 3),
                    "Signal": round(df["macd_signal"].iloc[-1], 3),
                }
            )
        else:
            print(
                f"Skipping {ticker}: MACD={df['macd'].iloc[-1]}, Signal={df['macd_signal'].iloc[-1]}, RSI={df['rsi'].iloc[-1]}"
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
def main(tickers: list[str], target_delta: float, min_expiry:str, max_expiry: str, skip_screen: bool):
    """
    Screen stocks for bullish momentum using MACD and RSI indicators.
    Find potential candidates for bull call spreads or long calls.
    """
    # Use provided tickers or default to nasdaq_100
    if tickers:
        ticker_list = [ticker.strip().upper() for ticker in tickers.split(",")]
    else:
        ticker_list = nasdaq_100

    screened = screen_tickers(ticker_list, skip_screen=skip_screen)
    # Display screened tickers
    screen_df = pd.DataFrame(screened)
    print("📈 Bullish Momentum Candidates:")
    print(screen_df)

    # Options chain for all screened tickers
    if not screen_df.empty:
        for index, row in screen_df.iterrows():
            ticker = row["Ticker"]
            print(f"\n🔍 Fetching options for {ticker}")
            stock = yf.Ticker(ticker)

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


if __name__ == "__main__":
    main()
