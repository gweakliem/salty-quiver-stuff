# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based financial screening tool that identifies bullish momentum candidates using technical analysis. The main screener filters stocks based on MACD bullish crossovers and RSI levels to find potential opportunities for bull call spreads or long calls.

## Development Environment

- **Python Version**: Requires Python ≥3.12
- **Package Manager**: Uses `uv` for dependency management
- **Dependencies**: pandas, ta (technical analysis), yfinance

## Common Commands

```bash
# Install dependencies
uv sync

# Run the main screener
python screener.py

```

## Architecture

The project consists of two main Python files:

- `screener.py`: Main financial screening logic that:
  - Downloads 6 months of daily stock data for a predefined ticker universe
  - Calculates RSI and MACD technical indicators
  - Filters for bullish MACD crossovers with RSI between 50-70
  - Displays screened candidates and fetches options chain for the top result

The screener uses a hardcoded ticker list (AAPL, MSFT, NVDA, AMD, TSLA) and applies momentum-based filtering criteria to identify potential trading opportunities.

## Coding Style

- This project is managed using uv.
- Use Python type annotations on every method declaration