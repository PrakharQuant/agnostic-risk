"""
data_engine.py
==============
Data Layer — fetches historical price data from Yahoo Finance and
computes the statistical parameters needed by the inequalities module.

Parameters computed
-------------------
Price-level  : μ_price, σ_price  (for Markov on absolute prices)
Log-returns  : μ_ret, σ_ret      (for Chebyshev / Variance bound on returns)

Note: Log returns are preferred in finance because they are additive over
time and approximately normally distributed, making the bounds more
meaningful than applying them to raw price levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssetStats:
    """Immutable snapshot of an asset's statistical summary."""

    ticker: str
    prices: pd.Series          # Adjusted close prices
    log_returns: pd.Series     # Continuously-compounded daily returns
    pct_returns: pd.Series     # Simple daily percentage returns

    # Price-level moments  (used for Markov on prices)
    mu_price: float
    sigma_price: float
    current_price: float

    # Return-level moments  (used for Chebyshev / Variance bound on returns)
    mu_ret: float
    sigma_ret: float

    # Metadata
    period_days: int
    n_obs: int                 # Actual number of trading days returned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_asset_stats(ticker: str, period_days: int) -> AssetStats:
    """
    Download historical OHLCV data for *ticker* and return an AssetStats
    object pre-populated with all statistical moments.

    Parameters
    ----------
    ticker : str
        Valid Yahoo Finance ticker symbol, e.g. "AAPL", "^NSEI", "BTC-USD".
    period_days : int
        Approximate number of calendar days of history to request.
        yfinance will map this to trading days automatically.

    Returns
    -------
    AssetStats

    Raises
    ------
    ValueError
        If the ticker is invalid or the API returns an empty DataFrame.
    """
    raw = yf.download(
        ticker,
        period=f"{period_days}d",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            "Check the ticker symbol and your internet connection."
        )

    prices = raw["Close"].dropna().squeeze()  # Series, not DataFrame

    log_ret  = np.log(prices / prices.shift(1)).dropna()
    pct_ret  = prices.pct_change().dropna()

    return AssetStats(
        ticker=ticker.upper(),
        prices=prices,
        log_returns=log_ret,
        pct_returns=pct_ret,
        mu_price=float(prices.mean()),
        sigma_price=float(prices.std(ddof=1)),
        current_price=float(prices.iloc[-1]),
        mu_ret=float(log_ret.mean()),
        sigma_ret=float(log_ret.std(ddof=1)),
        period_days=period_days,
        n_obs=len(prices),
    )


def empirical_exceedance(series: pd.Series, mu: float, k: float, sigma: float) -> float:
    """
    Compute the *actual* fraction of observations where |X - μ| ≥ k·σ.

    This is the empirical counterpart to the Chebyshev upper bound (1/k²),
    and is used to show how tight (or loose) the theoretical bound is.

    Parameters
    ----------
    series : pd.Series
        The data sample (returns or prices).
    mu, sigma : float
        Mean and standard deviation used in the bound.
    k : float
        Number of standard deviations.

    Returns
    -------
    float in [0, 1]
    """
    deviations = np.abs(series - mu)
    return float((deviations >= k * sigma).mean())
