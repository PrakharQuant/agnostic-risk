# 🛡️ Agnostic Risk — Finance Bound Visualiser

An interactive dashboard that applies **distribution-free probability inequalities** to real market data.
No Gaussian assumption required.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agnostic-risk-prakharquant.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.x-red.svg)](https://streamlit.io)

---

## 🌐 Live App

**[https://agnostic-risk-prakharquant.streamlit.app](https://agnostic-risk-prakharquant.streamlit.app)**

---

## Motivation

Standard financial risk models (VaR, parametric CVaR) assume log-normal returns.
Real asset returns have **fat tails** — the empirical kurtosis of daily equity returns
typically exceeds 3.  Chebyshev and Markov inequalities hold for *any* distribution
with finite first and second moments, making them uniquely robust tools for
model-agnostic worst-case analysis.

---

## Project Architecture
```
agnostic-risk/
├── src/
│   ├── data_engine.py      # Data Layer  — yfinance fetch + statistical moments
│   ├── inequalities.py     # Logic Layer — pure math (Markov, Chebyshev, Variance Bound)
│   └── app.py              # Presentation Layer — Streamlit dashboard
├── requirements.txt
└── README.md
```

The 3-tier separation means the math in `inequalities.py` is fully testable
without network calls, and the UI can be swapped (e.g., Dash or Gradio)
without touching any logic.

---

## Inequalities Covered

| # | Name | Formula | Domain |
|---|------|---------|--------|
| 1 | **Markov** | $P(X \geq a) \leq \mathbb{E}[X]/a$ | Prices ($X \geq 0$) |
| 2 | **Chebyshev** | $P(\|X-\mu\| \geq k\sigma) \leq 1/k^2$ | Returns (any distribution) |
| 3 | **Variance Bound** | $P(\|X-\mu\| \geq c) \leq \sigma^2/c^2$ | Returns in %-point terms |

The dashboard also includes a **Bound Tightness Sweep** (Section 4) that
plots theoretical bounds vs. empirical exceedance rates across k ∈ [1, 5],
with a Normal distribution benchmark overlay.

---

## Quick Start

### Run Locally
```bash
git clone https://github.com/PrakharQuant/agnostic-risk.git
cd agnostic-risk
pip install -r requirements.txt
streamlit run src/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. Enter any valid **Yahoo Finance ticker** in the sidebar (e.g. `AAPL`, `^NSEI`, `BTC-USD`, `GLD`).
2. Set the **history window** (default: 252 trading days ≈ 1 year).
3. Interact with each inequality's slider to explore how the bounds change.
4. Observe Section 4 to see whether the asset's tails are Gaussian or fat-tailed.

---

## Requirements
```
streamlit>=1.28
yfinance>=0.2.28
numpy>=1.24
matplotlib>=3.7
scipy>=1.11
pandas>=2.0
```

---

## Key Design Decisions

- **Log-returns** (not simple returns) are used for Chebyshev/Variance sections
  because they are additive over time and have more tractable moment properties.
- **Markov** is applied to price levels (the non-negativity requirement is satisfied).
- Empirical exceedance rates are computed alongside every theoretical bound to
  immediately show how conservative each inequality is in practice.
- `@st.cache_data` is used with a 5-minute TTL to avoid repeated API calls during
  slider interaction.

---

## References

- Ross, S.M. (2014). *A First Course in Probability*, 9th ed. Pearson. Chapter 8.
- Shreve, S.E. (2004). *Stochastic Calculus for Finance I*. Springer.
- Cont, R. (2001). "Empirical properties of asset returns." *Quantitative Finance*, 1(2).
