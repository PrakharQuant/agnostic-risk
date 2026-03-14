"""
inequalities.py
===============
Logic Layer — pure mathematical functions for probability inequalities.

All three bounds are *distribution-free*: they make no assumption about
the underlying distribution of X (Gaussian, fat-tailed, etc.), which is
exactly why they are so powerful in finance where return distributions
are demonstrably non-normal.

Bounds implemented
------------------
1. Markov's Inequality      P(X ≥ a) ≤ E[X] / a            [X ≥ 0]
2. Chebyshev's Inequality   P(|X−μ| ≥ kσ) ≤ 1/k²           [k > 0]
3. Variance Bound           P(|X−μ| ≥ c) ≤ σ²/c²           [c > 0]
   (Equivalent to Chebyshev with c = kσ; useful when phrasing the
    deviation in dollar / percentage terms rather than multiples of σ.)

Reference: Sheldon Ross, "A First Course in Probability", Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkovResult:
    """
    Result of Markov's Inequality.

    Attributes
    ----------
    bound : float
        Upper bound on P(X ≥ threshold).  Capped at 1.0.
    threshold : float
        The price / value level a in the bound.
    mean : float
        E[X] used in the calculation.
    """
    bound: float
    threshold: float
    mean: float

    @property
    def bound_pct(self) -> float:
        return self.bound * 100


@dataclass(frozen=True)
class ChebyshevResult:
    """
    Result of Chebyshev's Inequality.

    Attributes
    ----------
    prob_outside : float
        Upper bound on P(|X−μ| ≥ kσ).
    prob_inside : float
        Lower bound on P(|X−μ| < kσ)  = 1 − prob_outside.
    k : float
        Number of standard deviations.
    lower, upper : float
        The interval [μ − kσ, μ + kσ].
    mu, sigma : float
        Parameters used.
    """
    prob_outside: float
    prob_inside: float
    k: float
    lower: float
    upper: float
    mu: float
    sigma: float

    @property
    def outside_pct(self) -> float:
        return self.prob_outside * 100

    @property
    def inside_pct(self) -> float:
        return self.prob_inside * 100


@dataclass(frozen=True)
class VarianceBoundResult:
    """
    Result of the Variance (Chebyshev in absolute deviation form).

    Attributes
    ----------
    bound : float
        Upper bound on P(|X−μ| ≥ c).  Capped at 1.0.
    c : float
        The absolute deviation threshold.
    mu, sigma : float
        Parameters used.
    c_in_sigma : float
        Equivalent k = c/σ  (for comparison with Chebyshev slider).
    """
    bound: float
    c: float
    mu: float
    sigma: float

    @property
    def c_in_sigma(self) -> float:
        return self.c / self.sigma if self.sigma > 0 else float("inf")

    @property
    def bound_pct(self) -> float:
        return self.bound * 100


# ---------------------------------------------------------------------------
# Pure mathematical functions
# ---------------------------------------------------------------------------

def markov_bound(mean: float, threshold: float) -> MarkovResult:
    """
    Markov's Inequality: For non-negative random variable X,

        P(X ≥ a) ≤ E[X] / a

    Applies to stock prices (always ≥ 0).

    Parameters
    ----------
    mean : float
        Historical mean price E[X].
    threshold : float
        Target price level a > mean recommended for meaningful results.

    Returns
    -------
    MarkovResult

    Raises
    ------
    ValueError
        If mean < 0 (Markov requires non-negative X) or threshold ≤ 0.
    """
    if mean < 0:
        raise ValueError(
            "Markov's Inequality requires a non-negative random variable (E[X] ≥ 0). "
            "Use on prices, not returns."
        )
    if threshold <= 0:
        raise ValueError("Threshold a must be strictly positive.")
    if threshold < mean:
        # Bound would exceed 1; Markov is trivially satisfied but uninformative
        bound = 1.0
    else:
        bound = min(mean / threshold, 1.0)

    return MarkovResult(bound=bound, threshold=threshold, mean=mean)


def chebyshev_bound(mu: float, sigma: float, k: float) -> ChebyshevResult:
    """
    Chebyshev's Inequality: For *any* distribution with finite variance,

        P(|X − μ| ≥ kσ) ≤ 1/k²

    Equivalently, at least (1 − 1/k²) of the mass lies in [μ−kσ, μ+kσ].

    Parameters
    ----------
    mu : float
        Mean of X (e.g., mean log-return).
    sigma : float
        Standard deviation of X (e.g., daily volatility).
    k : float
        Number of standard deviations.  Must be > 0; meaningful for k > 1.

    Returns
    -------
    ChebyshevResult
    """
    if k <= 0:
        raise ValueError("k must be strictly positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    prob_outside = min(1.0 / k ** 2, 1.0)
    prob_inside  = 1.0 - prob_outside

    return ChebyshevResult(
        prob_outside=prob_outside,
        prob_inside=prob_inside,
        k=k,
        lower=mu - k * sigma,
        upper=mu + k * sigma,
        mu=mu,
        sigma=sigma,
    )


def variance_bound(mu: float, sigma: float, c: float) -> VarianceBoundResult:
    """
    Variance / Absolute-Deviation Form of Chebyshev:

        P(|X − μ| ≥ c) ≤ σ² / c²

    Identical to Chebyshev when c = kσ, but lets users think in
    return percentage points rather than multiples of σ.

    Parameters
    ----------
    mu : float
        Mean of X.
    sigma : float
        Standard deviation of X.
    c : float
        Absolute deviation threshold (same units as X).

    Returns
    -------
    VarianceBoundResult
    """
    if c <= 0:
        raise ValueError("Absolute deviation threshold c must be strictly positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    bound = min(sigma ** 2 / c ** 2, 1.0)

    return VarianceBoundResult(bound=bound, c=c, mu=mu, sigma=sigma)


# ---------------------------------------------------------------------------
# Tightness analysis helper
# ---------------------------------------------------------------------------

def bound_tightness(theoretical: float, empirical: float) -> str:
    """
    Classify how tight the theoretical bound is relative to
    the empirical exceedance rate.

    Returns
    -------
    str : one of 'tight', 'moderate', 'loose'
    """
    if theoretical == 0:
        return "undefined"
    ratio = empirical / theoretical
    if ratio >= 0.75:
        return "tight"
    elif ratio >= 0.30:
        return "moderate"
    else:
        return "loose"
