"""

🧠 What It Does
        •	Filters for MACD bullish cross AND RSI between 50–70
        •	Returns tickers that might support bull call spreads or long calls
        •	Shows first expiration’s call chain for your inspection
"""

import sqlite3
import sys
from datetime import datetime
from html import escape
from pathlib import Path

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
    "indexes": [
        "QQQ",
        "SPY",
        "MDY",
        "IJR",
        "VOX",
        "VDC",
        "VCRB",
        "BNDP",
        "VIG",
        "VDE",
        "EDV",
        "VSS",
        "VEU",
        "VEA",
        "VWO",
        "VGK",
        "VUG",
        "VHT",
        "VYM",
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
REPORTS_DIR = Path("reports")


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
        return [
            _normalize_ticker(line)
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
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
        "indexes": lambda: default_lists["indexes"],
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


def fidelity_chart_url(ticker: str) -> str:
    return (
        "https://digital.fidelity.com/prgw/digital/research/quote/dashboard/chart"
        f"?symbol={ticker}"
    )


def _fmt_value(value: float | int | str | None, fmt: str = "", suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float | np.floating | int | np.integer):
        if fmt:
            return f"{value:{fmt}}{suffix}"
        return f"{value}{suffix}"
    return f"{value}{suffix}"


def write_html_review_sheet(
    report_path: Path,
    generated_at: datetime,
    index_name: str,
    min_expiry: int,
    max_expiry: int,
    target_delta: float,
    min_delta: float,
    max_delta: float,
    rows: list[dict[str, str | float | int | None]],
) -> None:
    """Write a single-file HTML review sheet for charting and trade triage."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    table_rows: list[str] = []
    for row in rows:
        trade = row.get("trade")
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['ticker']))}</td>"
            f"<td>{_fmt_value(row.get('price'), '.2f')}</td>"
            f"<td>{_fmt_value(row.get('rsi'), '.1f')}</td>"
            f"<td>{_fmt_value(row.get('macd'), '.3f')}</td>"
            f"<td>{_fmt_value(row.get('signal'), '.3f')}</td>"
            f"<td>{escape(str(row.get('status', '-')))}</td>"
            f"<td>{escape(str(row.get('expiry', '-')))}</td>"
            f"<td>{escape(str(trade if trade else '-'))}</td>"
            f"<td>{_fmt_value(row.get('strike'), '.2f')}</td>"
            f"<td>{_fmt_value(row.get('delta'), '.3f')}</td>"
            f"<td>{_fmt_value(row.get('iv_pct'), '.1f', '%')}</td>"
            f"<td>{_fmt_value(row.get('mid'), '.2f')}</td>"
            f"<td>{_fmt_value(row.get('volume'), '.0f')}</td>"
            f"<td>{_fmt_value(row.get('open_interest'), '.0f')}</td>"
            f"<td>{_fmt_value(row.get('breakeven'), '.2f')}</td>"
            f'<td><a href="{escape(str(row["fidelity_url"]))}"'
            ' target="_blank" rel="noopener noreferrer">Open</a></td>'
            "</tr>"
        )

    if not table_rows:
        table_rows.append(
            '<tr><td colspan="16">No tickers passed the screen in this run.</td></tr>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Screener Review Sheet</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 20px;
      color: #111;
    }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ margin: 0 0 16px 0; color: #444; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 6px 8px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .wrap {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Screener Review Sheet</h1>
  <p class="meta">
    Generated: {generated_at.strftime("%Y-%m-%d %H:%M:%S")}<br>
    Universe: {escape(index_name)}<br>
    Expiry Window: {min_expiry} to {max_expiry} days<br>
    Target Delta: {target_delta:.2f} | Delta Bounds: {min_delta:.2f} to {max_delta:.2f}
  </p>
  <div class="wrap">
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Price</th>
          <th>RSI</th>
          <th>MACD</th>
          <th>Signal</th>
          <th>Status</th>
          <th>Expiry</th>
          <th>Trade</th>
          <th>Strike</th>
          <th>Delta</th>
          <th>IV</th>
          <th>Mid</th>
          <th>Volume</th>
          <th>OI</th>
          <th>Breakeven</th>
          <th>Fidelity</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


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
    type=click.Choice(["nasdaq_100", "sp500", "genz", "indexes"], case_sensitive=False),
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

    run_timestamp = datetime.now()
    db_path = "screener_results.db"
    report_path = (
        REPORTS_DIR / f"screening_review_{run_timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    )

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

    review_rows: list[dict[str, str | float | int | None]] = []

    # Options chain for all screened tickers
    if not screen_df.empty:
        timestamp = run_timestamp.isoformat()

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
            review_row: dict[str, str | float | int | None] = {
                "ticker": ticker,
                "price": current_price,
                "rsi": current_rsi,
                "macd": current_macd,
                "signal": current_signal,
                "status": "Pending options check",
                "expiry": None,
                "trade": None,
                "strike": None,
                "delta": None,
                "iv_pct": None,
                "mid": None,
                "volume": None,
                "open_interest": None,
                "breakeven": None,
                "fidelity_url": fidelity_chart_url(ticker),
            }

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
                    review_row["expiry"] = expiry
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
                        review_row["status"] = "Trade candidate"
                        review_row["trade"] = (
                            f"BUY TO OPEN {ticker} ${best_call['strike']:.2f} Call"
                        )
                        review_row["strike"] = float(best_call["strike"])
                        review_row["delta"] = float(best_call["delta"])
                        review_row["iv_pct"] = float(
                            best_call["impliedVolatility"] * 100
                        )
                        review_row["mid"] = float(best_call["mid"])
                        review_row["volume"] = float(best_call["volume"])
                        review_row["open_interest"] = float(best_call["openInterest"])
                        review_row["breakeven"] = float(best_call["breakeven"])
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
                        review_row["status"] = "No suitable call for filters"
                        print(f"⚠️  {ticker}: No suitable call found for {expiry}")
                        print(
                            "   (No options met liquidity and/or delta-window filters)"
                        )
                        print()
                else:
                    review_row["status"] = "No expiry in selected DTE range"
                    print(f"⚠️  {ticker}: No suitable expiry found")
                    print(
                        f"   (Looking for expiration between {min_expiry}-{max_expiry} days)"
                    )
                    print()
            except Exception as e:
                review_row["status"] = f"Options fetch error: {e}"
                print(f"❌ {ticker}: Could not fetch options - {e}")
                print()
            finally:
                review_rows.append(review_row)

        conn.commit()
        conn.close()

        print("=" * 80)
        print("✅ Screening complete. Results saved to database.")
        print("=" * 80)

    write_html_review_sheet(
        report_path=report_path,
        generated_at=run_timestamp,
        index_name=index_name,
        min_expiry=min_expiry,
        max_expiry=max_expiry,
        target_delta=target_delta,
        min_delta=min_delta,
        max_delta=max_delta,
        rows=review_rows,
    )
    print(f"📝 HTML review sheet written to: {report_path}")


if __name__ == "__main__":
    main()
