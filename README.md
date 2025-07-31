
# Salty Quiver Stuff

I got interested in options trading strategies so I [started with this ChatGPT Session](https://chatgpt.com/share/687ee11f-a890-800f-96e7-a93381812d91) to get an idea of how to set up an automated trading screener. This is a simple stock & option screener to find suitable entry points for options.

## Installation

```bash
# Install dependencies
uv sync
```

## Usage

```bash
# Run the main screener
python screener.py

# Run with specific tickers
python screener.py --tickers "AAPL,MSFT,NVDA"

# Skip screening and show all tickers
python screener.py --skip-screen
```

## Development

### Code Formatting

This project uses `ruff` for code formatting and linting. To format your code:

```bash
# Format code with ruff
uv run ruff format .

# Check for linting issues
uv run ruff check .

# Fix auto-fixable linting issues
uv run ruff check --fix .

# Run isort (if installed separately)
uv run isort .
```

### Pre-commit Setup (Optional)

For automatic formatting on commit, you can set up pre-commit hooks:

```bash
# Install pre-commit
uv add --dev pre-commit

# Install the git hooks
uv run pre-commit install
```

## TODO Ideas

* Add dynamic lists using the screener in yfinance.EquityQuery
* Try new strategy, maybe index with 3 days of down volume -> buy puts
* MACD cross signal should happen in negative territory