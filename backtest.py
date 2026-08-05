import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import ta
import yfinance as yf
from pydantic import BaseModel

from screener import get_index_constituents


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    cost_basis: float
    commission_paid: float


class TradeRecord(BaseModel):
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    holding_days: int
    return_pct: float
    pnl: float
    exit_reason: str


class EquityPoint(BaseModel):
    date: str
    cash: float
    equity: float


class BacktestSummary(BaseModel):
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    trade_count: int
    win_rate_pct: float
    avg_pnl: float
    avg_return_pct: float


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def build_signal_frame(
    ticker: str,
    period: str,
    interval: str,
    rsi_min: float,
    rsi_max: float,
) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if df.empty or len(df) < 220:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    macd_calc = ta.trend.MACD(close)
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd"] = macd_calc.macd()
    df["macd_signal"] = macd_calc.macd_signal()
    df["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()
    df["close"] = close

    macd_cross = (df["macd"] > df["macd_signal"]) & (
        df["macd"].shift(1) <= df["macd_signal"].shift(1)
    )
    rsi_ok = (df["rsi"] > rsi_min) & (df["rsi"] < rsi_max)
    trend_ok = (df["close"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"])

    df["entry_signal"] = (macd_cross & rsi_ok & trend_ok).fillna(False)
    df["exit_signal"] = (df["macd"] < df["macd_signal"]).fillna(False)
    return df[["close", "entry_signal", "exit_signal"]].copy()


def align_market_data(
    raw_data: dict[str, pd.DataFrame],
) -> tuple[list[pd.Timestamp], dict[str, pd.DataFrame]]:
    all_dates = sorted(
        {date for frame in raw_data.values() for date in frame.index.to_list()}
    )
    aligned: dict[str, pd.DataFrame] = {}

    for ticker, frame in raw_data.items():
        re = frame.reindex(all_dates)
        re["close"] = re["close"].ffill()
        re["entry_signal"] = re["entry_signal"].fillna(False)
        re["exit_signal"] = re["exit_signal"].fillna(False)
        aligned[ticker] = re

    return all_dates, aligned


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def annualized_return(total_return: float, periods: int) -> float:
    if periods <= 1:
        return 0.0
    years = periods / 252
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def run_backtest(
    dates: list[pd.Timestamp],
    data: dict[str, pd.DataFrame],
    initial_capital: float,
    risk_pct: float,
    max_allocation_pct: float,
    max_open_positions: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_days: int,
    slippage_bps: float,
    commission_per_order: float,
) -> tuple[pd.DataFrame, pd.DataFrame, BacktestSummary]:
    cash = initial_capital
    positions: dict[str, Position] = {}
    trade_rows: list[TradeRecord] = []
    equity_rows: list[EquityPoint] = []

    for date in dates:
        # Exit checks first so slots are freed before evaluating new entries.
        for ticker in list(positions.keys()):
            frame = data[ticker]
            close = frame.at[date, "close"]
            if pd.isna(close):
                continue

            pos = positions[ticker]
            days_held = (date - pos.entry_date).days
            gross_return = close / pos.entry_price - 1.0
            exit_by_stop = gross_return <= -stop_loss_pct
            exit_by_target = gross_return >= take_profit_pct
            exit_by_time = days_held >= max_hold_days
            exit_by_signal = bool(frame.at[date, "exit_signal"])

            if exit_by_stop or exit_by_target or exit_by_time or exit_by_signal:
                effective_exit = close * (1 - slippage_bps / 10000)
                proceeds = pos.shares * effective_exit
                cash += proceeds - commission_per_order

                pnl = (
                    proceeds
                    - pos.cost_basis
                    - pos.commission_paid
                    - commission_per_order
                )
                trade_rows.append(
                    TradeRecord(
                        ticker=ticker,
                        entry_date=pos.entry_date.date().isoformat(),
                        exit_date=date.date().isoformat(),
                        entry_price=round(pos.entry_price, 4),
                        exit_price=round(effective_exit, 4),
                        shares=round(pos.shares, 6),
                        holding_days=days_held,
                        return_pct=round(
                            (effective_exit / pos.entry_price - 1.0) * 100, 3
                        ),
                        pnl=round(pnl, 2),
                        exit_reason=(
                            "stop_loss"
                            if exit_by_stop
                            else "take_profit"
                            if exit_by_target
                            else "time_stop"
                            if exit_by_time
                            else "signal"
                        ),
                    )
                )
                del positions[ticker]

        for ticker, frame in data.items():
            if ticker in positions:
                continue
            if len(positions) >= max_open_positions:
                break

            if not bool(frame.at[date, "entry_signal"]):
                continue

            close = frame.at[date, "close"]
            if pd.isna(close) or close <= 0:
                continue

            risk_budget = cash * risk_pct
            size_from_risk = risk_budget / stop_loss_pct if stop_loss_pct > 0 else 0
            max_alloc = cash * max_allocation_pct
            planned_notional = min(size_from_risk, max_alloc)
            if planned_notional <= 0:
                continue

            effective_entry = close * (1 + slippage_bps / 10000)
            shares = planned_notional / effective_entry
            gross_cost = shares * effective_entry
            total_cost = gross_cost + commission_per_order

            if total_cost > cash:
                shares = max((cash - commission_per_order) / effective_entry, 0)
                gross_cost = shares * effective_entry
                total_cost = gross_cost + commission_per_order

            if shares <= 0 or total_cost <= 0:
                continue

            cash -= total_cost
            positions[ticker] = Position(
                ticker=ticker,
                entry_date=date,
                entry_price=effective_entry,
                shares=shares,
                cost_basis=gross_cost,
                commission_paid=commission_per_order,
            )

        mtm = 0.0
        for ticker, pos in positions.items():
            close = data[ticker].at[date, "close"]
            if pd.isna(close):
                close = pos.entry_price
            mtm += pos.shares * close

        equity_rows.append(
            EquityPoint(
                date=date.date().isoformat(),
                cash=round(cash, 2),
                equity=round(cash + mtm, 2),
            )
        )

    equity_df = pd.DataFrame([row.model_dump() for row in equity_rows])
    trades_df = pd.DataFrame([row.model_dump() for row in trade_rows])
    final_equity = (
        float(equity_df["equity"].iloc[-1]) if not equity_df.empty else initial_capital
    )
    total_return = final_equity / initial_capital - 1.0
    daily_returns = (
        equity_df["equity"].pct_change().dropna()
        if not equity_df.empty
        else pd.Series(dtype=float)
    )

    sharpe = 0.0
    sortino = 0.0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe = float(math.sqrt(252) * daily_returns.mean() / daily_returns.std())
    downside = daily_returns[daily_returns < 0]
    if not downside.empty and downside.std() > 0:
        sortino = float(math.sqrt(252) * daily_returns.mean() / downside.std())

    wins = int((trades_df["pnl"] > 0).sum()) if not trades_df.empty else 0
    trade_count = len(trades_df)
    summary = BacktestSummary(
        initial_capital=initial_capital,
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return * 100, 2),
        cagr_pct=round(annualized_return(total_return, len(equity_df)) * 100, 2),
        max_drawdown_pct=(
            round(max_drawdown(equity_df["equity"]) * 100, 2)
            if not equity_df.empty
            else 0.0
        ),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        trade_count=trade_count,
        win_rate_pct=round((wins / trade_count) * 100, 2) if trade_count else 0.0,
        avg_pnl=round(float(trades_df["pnl"].mean()), 2) if trade_count else 0.0,
        avg_return_pct=(
            round(float(trades_df["return_pct"].mean()), 3) if trade_count else 0.0
        ),
    )
    return equity_df, trades_df, summary


def summary_to_text(title: str, summary: BacktestSummary) -> str:
    lines = [
        title,
        "-" * len(title),
        f"Initial Capital: ${summary.initial_capital:.2f}",
        f"Final Equity: ${summary.final_equity:.2f}",
        f"Total Return: {summary.total_return_pct:.2f}%",
        f"CAGR: {summary.cagr_pct:.2f}%",
        f"Max Drawdown: {summary.max_drawdown_pct:.2f}%",
        f"Sharpe: {summary.sharpe:.3f}",
        f"Sortino: {summary.sortino:.3f}",
        f"Trades: {summary.trade_count}",
        f"Win Rate: {summary.win_rate_pct:.2f}%",
        f"Avg Trade PnL: ${summary.avg_pnl:.2f}",
        f"Avg Trade Return: {summary.avg_return_pct:.3f}%",
    ]
    return "\n".join(lines)


@click.command()
@click.option(
    "--tickers", "-t", type=str, help="Comma-separated tickers; overrides --index."
)
@click.option(
    "--index",
    "index_name",
    default="nasdaq_100",
    type=click.Choice(["nasdaq_100", "sp500", "genz"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--period",
    default="5y",
    show_default=True,
    help="Yahoo Finance period (e.g., 2y, 5y).",
)
@click.option(
    "--interval", default="1d", show_default=True, help="Bar interval (e.g., 1d)."
)
@click.option("--initial-capital", default=100000.0, show_default=True, type=float)
@click.option(
    "--risk-pct",
    default=0.005,
    show_default=True,
    type=float,
    help="Risk budget per trade.",
)
@click.option("--max-allocation-pct", default=0.20, show_default=True, type=float)
@click.option("--max-open-positions", default=5, show_default=True, type=int)
@click.option("--stop-loss-pct", default=0.08, show_default=True, type=float)
@click.option("--take-profit-pct", default=0.20, show_default=True, type=float)
@click.option("--max-hold-days", default=30, show_default=True, type=int)
@click.option("--slippage-bps", default=5.0, show_default=True, type=float)
@click.option("--commission-per-order", default=0.0, show_default=True, type=float)
@click.option("--rsi-min", default=50.0, show_default=True, type=float)
@click.option("--rsi-max", default=70.0, show_default=True, type=float)
@click.option(
    "--train-ratio",
    default=0.70,
    show_default=True,
    type=float,
    help="Walk-forward split ratio.",
)
@click.option(
    "--report-dir",
    default="reports/backtests",
    show_default=True,
    type=click.Path(path_type=Path),
)
def main(
    tickers: str | None,
    index_name: str,
    period: str,
    interval: str,
    initial_capital: float,
    risk_pct: float,
    max_allocation_pct: float,
    max_open_positions: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_days: int,
    slippage_bps: float,
    commission_per_order: float,
    rsi_min: float,
    rsi_max: float,
    train_ratio: float,
    report_dir: Path,
) -> None:
    if not (0 < train_ratio < 1):
        raise click.ClickException("--train-ratio must be between 0 and 1.")
    if stop_loss_pct <= 0:
        raise click.ClickException("--stop-loss-pct must be > 0.")
    if not (0 < risk_pct <= 1):
        raise click.ClickException("--risk-pct must be in (0, 1].")

    ticker_list = (
        [normalize_ticker(t) for t in tickers.split(",")]
        if tickers
        else get_index_constituents(index_name)
    )
    click.echo(f"Loading data for {len(ticker_list)} tickers...")

    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in ticker_list:
        frame = build_signal_frame(
            ticker=ticker,
            period=period,
            interval=interval,
            rsi_min=rsi_min,
            rsi_max=rsi_max,
        )
        if frame.empty:
            continue
        raw_data[ticker] = frame

    if not raw_data:
        raise click.ClickException("No tickers had enough data to run the backtest.")

    dates, aligned = align_market_data(raw_data)
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * train_ratio)))
    in_dates = dates[:split_idx]
    out_dates = dates[split_idx:]

    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_equity, all_trades, all_summary = run_backtest(
        dates=dates,
        data=aligned,
        initial_capital=initial_capital,
        risk_pct=risk_pct,
        max_allocation_pct=max_allocation_pct,
        max_open_positions=max_open_positions,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_days=max_hold_days,
        slippage_bps=slippage_bps,
        commission_per_order=commission_per_order,
    )

    in_equity, in_trades, in_summary = run_backtest(
        dates=in_dates,
        data=aligned,
        initial_capital=initial_capital,
        risk_pct=risk_pct,
        max_allocation_pct=max_allocation_pct,
        max_open_positions=max_open_positions,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_days=max_hold_days,
        slippage_bps=slippage_bps,
        commission_per_order=commission_per_order,
    )

    out_equity, out_trades, out_summary = run_backtest(
        dates=out_dates,
        data=aligned,
        initial_capital=initial_capital,
        risk_pct=risk_pct,
        max_allocation_pct=max_allocation_pct,
        max_open_positions=max_open_positions,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_days=max_hold_days,
        slippage_bps=slippage_bps,
        commission_per_order=commission_per_order,
    )

    all_trades.to_csv(report_dir / f"trades_all_{stamp}.csv", index=False)
    in_trades.to_csv(report_dir / f"trades_in_sample_{stamp}.csv", index=False)
    out_trades.to_csv(report_dir / f"trades_out_sample_{stamp}.csv", index=False)
    all_equity.to_csv(report_dir / f"equity_all_{stamp}.csv", index=False)
    in_equity.to_csv(report_dir / f"equity_in_sample_{stamp}.csv", index=False)
    out_equity.to_csv(report_dir / f"equity_out_sample_{stamp}.csv", index=False)

    summary_text = "\n\n".join(
        [
            summary_to_text("Full Period", all_summary),
            summary_to_text("In-Sample", in_summary),
            summary_to_text("Out-of-Sample", out_summary),
        ]
    )
    summary_path = report_dir / f"summary_{stamp}.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    click.echo(summary_text)
    click.echo("")
    click.echo(f"Saved summary: {summary_path}")
    click.echo(f"Saved trades/equity CSV files in: {report_dir}")


if __name__ == "__main__":
    main()
