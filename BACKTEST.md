# Backtest Notes

## Argument Semantics

- `--max-allocation-pct`
  - Fraction of available cash allowed for a single new position.
  - Example: `0.20` means at most 20% of current cash can be allocated to one trade.
  - This is a portfolio sizing control, not a price-move threshold.

- `--stop-loss-pct`
  - Exit threshold relative to entry price.
  - The backtest exits when `(current_price / entry_price - 1) <= -stop_loss_pct`.
  - Example: `0.08` means stop at about `-8%` from entry.

- `--take-profit-pct`
  - Exit threshold relative to entry price.
  - The backtest exits when `(current_price / entry_price - 1) >= take_profit_pct`.
  - Example: `0.20` means take profit at about `+20%` from entry.

Important: these values are decimal fractions, not whole percents.  
Use `0.08` for 8%, not `8`.

## Report Sections

- `Full Period`
  - Backtest over the entire date range loaded by `--period` and `--interval`.
  - This is the headline result, combining both development and validation windows.

- `In-Sample`
  - First segment of dates based on `--train-ratio` (default `0.70` = first 70% of bars).
  - Intended as the strategy development/tuning window.

- `Out-of-Sample`
  - Remaining segment after the split (default final 30%).
  - Intended as the validation window for anti-overfitting checks.

Split logic:
- `split_idx = int(len(dates) * train_ratio)`
- `in_dates = dates[:split_idx]`
- `out_dates = dates[split_idx:]`

Implementation caveat:
- In-sample and out-of-sample are run as separate backtests, each starting from the full `--initial-capital`.
- Out-of-sample is therefore a fresh validation run, not a capital carry-forward from in-sample.
