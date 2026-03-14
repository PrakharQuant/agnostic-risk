"""
app.py
======
Presentation Layer — Streamlit dashboard for "Agnostic Risk".

Run with:
    streamlit run src/app.py

The dashboard connects the data_engine and inequalities modules and
adds the key quant insight: comparing theoretical probability bounds
against the *empirical* exceedance frequency from real data.
"""

from __future__ import annotations

import sys
import os

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from data_engine import fetch_asset_stats, empirical_exceedance
from inequalities import (
    markov_bound,
    chebyshev_bound,
    variance_bound,
    bound_tightness,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agnostic Risk",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS  — dark-finance aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ---------- global ---------- */
    .block-container { padding-top: 1.5rem; }
    h1 { font-family: 'Georgia', serif; letter-spacing: -0.5px; }
    h2 { font-family: 'Georgia', serif; font-size: 1.25rem; margin-top: 0.3rem; }

    /* ---------- metric cards ---------- */
    div[data-testid="metric-container"] {
        background: #0f1117;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }

    /* ---------- callout boxes ---------- */
    .math-box {
    background: #0f1117;
    border-left: 3px solid #4a9eff;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.88rem;
    margin: 0.5rem 0;
    color: #e0e0e0;
}
    .insight-box {
        background: #0a1f0a;
        color: #c8f5c8;
        border-left: 3px solid #2ecc71;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #1f1500;
        color: #f5e6c8;
        border-left: 3px solid #f39c12;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ticker = st.text_input("Stock / Index Ticker", value="AAPL").upper().strip()
    period_days = st.slider("History Window (calendar days)", 60, 730, 252,
                            help="~252 trading days ≈ 1 year")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "**Agnostic Risk** visualises Markov, Chebyshev and Variance bounds — "
        "distribution-free inequalities that hold for *any* asset regardless of "
        "whether its returns are Gaussian."
    )
    st.markdown(
        "The key insight: compare the **theoretical upper bound** against the "
        "**empirical exceedance rate** to see how conservative each bound is."
    )

# ---------------------------------------------------------------------------
# Data fetch — cached for performance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data(t: str, d: int):
    return fetch_asset_stats(t, d)

try:
    stats = load_data(ticker, period_days)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛡️ Agnostic Risk: Finance Bound Visualiser")
st.markdown(
    "Applying **distribution-free probability inequalities** to real market data. "
    "No Gaussian assumption required."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${stats.current_price:.2f}")
c2.metric("Mean Price (μ)", f"${stats.mu_price:.2f}")
c3.metric("Price Std (σ)", f"${stats.sigma_price:.2f}")
c4.metric("Trading Days", str(stats.n_obs))

st.caption(
    f"Data: {stats.ticker} · {stats.n_obs} trading days · "
    f"Daily log-return μ = {stats.mu_ret*100:.4f}% · σ = {stats.sigma_ret*100:.4f}%"
)

# ============================================================
# SECTION 1 — MARKOV INEQUALITY
# ============================================================

st.divider()
st.header("1. Markov's Inequality — Price Spike Upper Bound")

col_theory, col_controls = st.columns([3, 2])

with col_controls:
    st.markdown("#### Parameters")
    min_thresh = float(stats.mu_price * 1.01)
    max_thresh = float(stats.mu_price * 3.0)
    default_thresh = float(stats.mu_price * 1.5)
    target_price = st.slider(
        "Target Price ($)",
        min_value=min_thresh,
        max_value=max_thresh,
        value=default_thresh,
        step=float(stats.sigma_price * 0.1),
        key="markov_slider",
    )

with col_theory:
    st.markdown("#### Theorem")
    st.markdown(
        r"For any **non-negative** random variable $X$ with finite expectation:"
    )
    st.latex(r"P(X \geq a) \;\leq\; \frac{\mathbb{E}[X]}{a}")
    st.markdown(
        "_Stock prices satisfy non-negativity, so this applies directly to price levels._"
    )

res_m = markov_bound(stats.mu_price, target_price)

# Empirical counterpart
empirical_m = float((stats.prices >= target_price).mean())

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric("Markov Bound", f"{res_m.bound_pct:.2f}%", help="P(price ≥ target) ≤ this")
col_r2.metric("Empirical Rate", f"{empirical_m*100:.2f}%", help="Actual fraction of days above target")
col_r3.metric(
    "Tightness",
    bound_tightness(res_m.bound, empirical_m).capitalize(),
    help="How conservative the bound is relative to observed data",
)

st.markdown(
    f'<div class="math-box">P(Price ≥ ${target_price:.2f}) ≤ '
    f'${stats.mu_price:.2f} / ${target_price:.2f} = <b>{res_m.bound_pct:.2f}%</b></div>',
    unsafe_allow_html=True,
)

if empirical_m <= res_m.bound:
    st.markdown(
        f'<div class="insight-box">✅ Bound is satisfied: empirical rate '
        f'({empirical_m*100:.2f}%) ≤ Markov bound ({res_m.bound_pct:.2f}%). '
        f'The bound is {bound_tightness(res_m.bound, empirical_m)}.</div>',
        unsafe_allow_html=True,
    )

# Plot
fig1, ax1 = plt.subplots(figsize=(10, 3.5))
fig1.patch.set_facecolor("#0e1117")
ax1.set_facecolor("#0e1117")
ax1.plot(stats.prices.values, color="#4a9eff", linewidth=1.2, label="Close Price")
ax1.axhline(stats.mu_price, color="white", linestyle="--", linewidth=1, label=f"Mean μ = ${stats.mu_price:.2f}")
ax1.axhline(target_price, color="#e74c3c", linestyle="-", linewidth=1.5, label=f"Target = ${target_price:.2f}")
ax1.fill_between(
    range(len(stats.prices)), target_price, stats.prices.values,
    where=(stats.prices.values >= target_price),
    color="#e74c3c", alpha=0.25, label="Days ≥ target",
)
ax1.set_title(f"{stats.ticker} — Markov Analysis", color="white", fontsize=11)
ax1.tick_params(colors="gray")
for spine in ax1.spines.values():
    spine.set_edgecolor("#2a2d3e")
ax1.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)
st.pyplot(fig1)
plt.close(fig1)

# ============================================================
# SECTION 2 — CHEBYSHEV INEQUALITY (applied to log-returns)
# ============================================================

st.divider()
st.header("2. Chebyshev's Inequality — Return Deviation Bound")

col_theory2, col_controls2 = st.columns([3, 2])

with col_controls2:
    st.markdown("#### Parameters")
    k = st.slider("k — Standard Deviations", 1.1, 5.0, 2.0, step=0.1, key="cheb_slider")
    st.markdown(
        f"**Interval**: [{stats.mu_ret*100:.4f}% ± {k:.1f}×{stats.sigma_ret*100:.4f}%]"
    )

with col_theory2:
    st.markdown("#### Theorem")
    st.markdown(
        r"For any distribution with mean $\mu$ and variance $\sigma^2$:"
    )
    st.latex(r"P(|X - \mu| \geq k\sigma) \;\leq\; \frac{1}{k^2}")
    st.markdown(
        "_Applied here to **daily log-returns**, which may be fat-tailed — exactly "
        "where the distribution-free nature of Chebyshev is valuable._"
    )

res_c = chebyshev_bound(stats.mu_ret, stats.sigma_ret, k)
empirical_c = empirical_exceedance(stats.log_returns, stats.mu_ret, k, stats.sigma_ret)

col_r4, col_r5, col_r6 = st.columns(3)
col_r4.metric("Chebyshev Bound", f"{res_c.outside_pct:.2f}%", help="P(|ret−μ| ≥ kσ) ≤ this")
col_r5.metric("Empirical Rate", f"{empirical_c*100:.2f}%", help="Actual exceedance frequency")
col_r6.metric(
    "Tightness",
    bound_tightness(res_c.prob_outside, empirical_c).capitalize(),
)

st.markdown(
    f'<div class="math-box">'
    f'P(|ret − μ| ≥ {k:.1f}σ) ≤ 1/{k:.1f}² = <b>{res_c.outside_pct:.2f}%</b><br>'
    f'Safe Zone: [{res_c.lower*100:.4f}%,  {res_c.upper*100:.4f}%]  '
    f'(≥ {res_c.inside_pct:.1f}% confidence)'
    f'</div>',
    unsafe_allow_html=True,
)

if res_c.outside_pct > 5:
    st.markdown(
        f'<div class="warn-box">⚠️ At k = {k:.1f}, the Chebyshev bound ({res_c.outside_pct:.1f}%) '
        f'is still quite wide. Increase k for tighter tail control.</div>',
        unsafe_allow_html=True,
    )

# Plot — return distribution with Chebyshev band
fig2, ax2 = plt.subplots(figsize=(10, 3.8))
fig2.patch.set_facecolor("#0e1117")
ax2.set_facecolor("#0e1117")

ret_vals = stats.log_returns.values * 100  # in percent
ax2.hist(ret_vals, bins=60, color="#4a9eff", alpha=0.5, density=True, label="Return distribution")

# Shade outside the safe zone
lower_ret_pct = res_c.lower * 100
upper_ret_pct = res_c.upper * 100
ax2.axvspan(ret_vals.min(), lower_ret_pct, color="#e74c3c", alpha=0.2, label="Outside kσ band")
ax2.axvspan(upper_ret_pct, ret_vals.max(), color="#e74c3c", alpha=0.2)
ax2.axvspan(lower_ret_pct, upper_ret_pct, color="#2ecc71", alpha=0.1, label=f"≥{res_c.inside_pct:.1f}% zone")
ax2.axvline(stats.mu_ret * 100, color="white", linewidth=1, linestyle="--", label="μ")
ax2.axvline(lower_ret_pct, color="#e74c3c", linewidth=1.2, linestyle=":")
ax2.axvline(upper_ret_pct, color="#e74c3c", linewidth=1.2, linestyle=":", label=f"±{k:.1f}σ")

ax2.set_xlabel("Daily Log-Return (%)", color="gray")
ax2.set_ylabel("Density", color="gray")
ax2.set_title(f"{stats.ticker} — Chebyshev Analysis (k = {k:.1f})", color="white", fontsize=11)
ax2.tick_params(colors="gray")
for spine in ax2.spines.values():
    spine.set_edgecolor("#2a2d3e")
ax2.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)
st.pyplot(fig2)
plt.close(fig2)

# ============================================================
# SECTION 3 — VARIANCE BOUND (dollar-deviation form)
# ============================================================

st.divider()
st.header("3. Variance Bound — Absolute Return Deviation")

col_theory3, col_controls3 = st.columns([3, 2])

with col_controls3:
    st.markdown("#### Parameters")
    sigma_pct = stats.sigma_ret * 100
    c_default = float(sigma_pct * 2)
    c_val = st.slider(
        "Deviation Threshold c (%)",
        min_value=float(sigma_pct * 0.5),
        max_value=float(sigma_pct * 6),
        value=c_default,
        step=float(sigma_pct * 0.1),
        key="var_slider",
        help="Expressed as absolute return percentage points",
    )

with col_theory3:
    st.markdown("#### Theorem")
    st.markdown(
        r"A restatement of Chebyshev in terms of absolute deviation $c$:"
    )
    st.latex(r"P(|X - \mu| \geq c) \;\leq\; \frac{\sigma^2}{c^2}")
    st.markdown(
        "_Useful when risk limits are expressed in return percentage points "
        "(e.g., a fund mandate that caps daily loss at −2%)._"
    )

res_v = variance_bound(stats.mu_ret * 100, sigma_pct, c_val)
empirical_v = empirical_exceedance(stats.log_returns * 100, stats.mu_ret * 100, 1.0, c_val)

col_r7, col_r8, col_r9 = st.columns(3)
col_r7.metric("Variance Bound", f"{res_v.bound_pct:.2f}%")
col_r8.metric("Empirical Rate", f"{empirical_v*100:.2f}%")
col_r9.metric("Equiv. k (c/σ)", f"{res_v.c_in_sigma:.2f}σ")

st.markdown(
    f'<div class="math-box">'
    f'P(|ret − μ| ≥ {c_val:.3f}%) ≤ {sigma_pct:.4f}² / {c_val:.3f}² = <b>{res_v.bound_pct:.2f}%</b>'
    f'</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 4 — THEORETICAL VS EMPIRICAL COMPARISON SWEEP
# ============================================================

st.divider()
st.header("4. Bound Tightness Sweep — Theory vs Empirical")
st.markdown(
    "The most important quant insight: these bounds are **conservative by design**. "
    "The chart below sweeps k from 1 to 5 and compares the Chebyshev theoretical "
    "upper bound (1/k²) against the actual empirical exceedance frequency."
)

k_vals = np.linspace(1.0, 5.0, 80)
cheb_theoretical = 1.0 / k_vals ** 2
cheb_empirical = np.array([
    empirical_exceedance(stats.log_returns, stats.mu_ret, kk, stats.sigma_ret)
    for kk in k_vals
])

# Normal distribution benchmark (for reference)
from scipy.stats import norm
normal_theoretical = 2 * (1 - norm.cdf(k_vals))

fig4, ax4 = plt.subplots(figsize=(10, 4))
fig4.patch.set_facecolor("#0e1117")
ax4.set_facecolor("#0e1117")

ax4.plot(k_vals, cheb_theoretical * 100, color="#e74c3c", linewidth=2,
         label="Chebyshev Bound (1/k²) — distribution-free")
ax4.plot(k_vals, cheb_empirical * 100, color="#4a9eff", linewidth=2,
         linestyle="-", label=f"{stats.ticker} Empirical Exceedance")
ax4.plot(k_vals, normal_theoretical * 100, color="#f39c12", linewidth=1.5,
         linestyle="--", label="Normal Distribution (benchmark)")

ax4.fill_between(k_vals, cheb_empirical * 100, cheb_theoretical * 100,
                 alpha=0.12, color="#e74c3c", label="Bound slack")

ax4.set_xlabel("k (standard deviations from mean)", color="gray")
ax4.set_ylabel("Probability (%)", color="gray")
ax4.set_title(f"{stats.ticker} — Chebyshev Bound vs Empirical vs Normal", color="white", fontsize=11)
ax4.tick_params(colors="gray")
for spine in ax4.spines.values():
    spine.set_edgecolor("#2a2d3e")
ax4.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)
ax4.set_xlim(1, 5)
ax4.set_ylim(bottom=0)

st.pyplot(fig4)
plt.close(fig4)

st.markdown(
    '<div class="insight-box">'
    '📐 <b>Key Insight:</b> The Chebyshev bound (red) is always above the empirical rate (blue). '
    'If the empirical rate tracks <i>closer</i> to the Normal benchmark (orange) than to the '
    'Chebyshev bound, the asset returns are approximately Gaussian. '
    'If it sits <i>above</i> the Normal line, the distribution has fat tails — '
    'the distribution-free bound becomes especially valuable.'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "**Agnostic Risk** · Built with Streamlit, yfinance, NumPy, Matplotlib · "
    "Mathematics: Ross (2014), _A First Course in Probability_, Ch. 8"
)
