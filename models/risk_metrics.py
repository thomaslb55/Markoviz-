"""
Mesures de risque de portefeuille (analytique + basées sur simulations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mesures analytiques (à partir de mu, Sigma, w)
# ---------------------------------------------------------------------------
def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(weights, mu))


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def sharpe_ratio(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_rate: float) -> float:
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return float("nan")
    return float((portfolio_return(weights, mu) - risk_free_rate) / vol)


def diversification_ratio(weights: np.ndarray, vols: np.ndarray, cov: np.ndarray) -> float:
    """Ratio de diversification = (somme pondérée des vols individuelles) /
    (volatilité du portefeuille). Un ratio > 1 indique un bénéfice de
    diversification (plus il est élevé, plus la diversification est forte).
    """
    weighted_avg_vol = float(np.dot(weights, vols))
    port_vol = portfolio_volatility(weights, cov)
    if port_vol == 0:
        return float("nan")
    return weighted_avg_vol / port_vol


def risk_contribution(weights: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contribution marginale et contribution au risque (% de la volatilité
    totale du portefeuille) de chaque actif.

    Contribution_i = w_i * (Sigma w)_i / sigma_p
    Somme des contributions = sigma_p (donc % contributions somment à 100%).
    """
    port_vol = portfolio_volatility(weights, cov)
    if port_vol == 0:
        zeros = np.zeros_like(weights)
        return zeros, zeros
    marginal = cov @ weights / port_vol  # dSigma_p / dw_i
    contrib = weights * marginal
    pct_contrib = contrib / port_vol
    return contrib, pct_contrib


# ---------------------------------------------------------------------------
# Mesures basées sur des rendements simulés / historiques (arrays 1D)
# ---------------------------------------------------------------------------
def downside_volatility(returns: np.ndarray, target: float = 0.0) -> float:
    """Volatilité des rendements inférieurs à la cible (semi-déviation)."""
    downside = np.minimum(returns - target, 0.0)
    return float(np.sqrt(np.mean(downside**2)))


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """VaR historique/simulée, exprimée en perte positive (ex. 0.12 = -12%)."""
    alpha = 1 - confidence
    q = np.quantile(returns, alpha)
    return float(-q)


def conditional_value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) : perte moyenne au-delà de la VaR."""
    alpha = 1 - confidence
    q = np.quantile(returns, alpha)
    tail = returns[returns <= q]
    if len(tail) == 0:
        return value_at_risk(returns, confidence)
    return float(-tail.mean())


def max_drawdown_from_paths(nav_paths: np.ndarray) -> np.ndarray:
    """Maximum drawdown par trajectoire simulée.

    Parameters
    ----------
    nav_paths : np.ndarray, shape (n_paths, n_periods)
        Trajectoires de valeur liquidative (base 1.0 en t=0).

    Returns
    -------
    np.ndarray, shape (n_paths,) : max drawdown (positif) par trajectoire.
    """
    running_max = np.maximum.accumulate(nav_paths, axis=1)
    drawdowns = (running_max - nav_paths) / running_max
    return drawdowns.max(axis=1)


def probability_of_capital_loss(final_tvpi: np.ndarray, threshold: float = 1.0) -> float:
    """Probabilité que le TVPI final soit inférieur à `threshold` (perte de
    capital nominal si threshold=1.0)."""
    return float(np.mean(final_tvpi < threshold))


def risk_summary_table(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    vols: np.ndarray,
    asset_names: list[str],
    risk_free_rate: float = 0.0,
    simulated_returns: np.ndarray | None = None,
    simulated_tvpi: np.ndarray | None = None,
    nav_paths: np.ndarray | None = None,
) -> pd.DataFrame:
    """Construit un tableau récapitulatif des mesures de risque du
    portefeuille, combinant les mesures analytiques et, si fournies, les
    mesures issues de la simulation Monte Carlo.
    """
    contrib, pct_contrib = risk_contribution(weights, cov)

    rows = {
        "Rendement attendu": portfolio_return(weights, mu),
        "Volatilité": portfolio_volatility(weights, cov),
        "Ratio de Sharpe": sharpe_ratio(weights, mu, cov, risk_free_rate),
        "Ratio de diversification": diversification_ratio(weights, vols, cov),
    }

    if simulated_returns is not None:
        rows["Downside volatility (simulée)"] = downside_volatility(simulated_returns)
        rows["VaR 95% (simulée)"] = value_at_risk(simulated_returns, 0.95)
        rows["CVaR 95% (simulée)"] = conditional_value_at_risk(simulated_returns, 0.95)

    if nav_paths is not None:
        mdd = max_drawdown_from_paths(nav_paths)
        rows["Max drawdown moyen (simulé)"] = float(mdd.mean())
        rows["Max drawdown 95e percentile (simulé)"] = float(np.quantile(mdd, 0.95))

    if simulated_tvpi is not None:
        rows["Probabilité de perte de capital (TVPI<1x)"] = probability_of_capital_loss(simulated_tvpi)

    summary = pd.Series(rows, name="Portefeuille").to_frame()

    contrib_df = pd.DataFrame(
        {"Contribution au risque (abs.)": contrib, "Contribution au risque (%)": pct_contrib},
        index=asset_names,
    )
    return summary, contrib_df
