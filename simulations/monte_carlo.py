"""
Moteur de simulation Monte Carlo pour portefeuilles d'actifs privés.

Fonctionnalités
---------------
- Simulation de trajectoires de rendements multivariés corrélés (Cholesky),
  avec choix entre distribution normale et distribution de Student-t
  (queues épaisses, plus réaliste pour les actifs privés qui présentent des
  chocs de valorisation discontinus - ex. write-downs, marks to model).
- Agrégation au niveau portefeuille selon les poids optimisés.
- Distribution de TVPI simulée (croissance cumulée de la NAV).
- Modèle de cash-flows simplifié (J-curve stylisée) par classe d'actifs,
  calibré sur `capital_duration` (phase d'appels de fonds) et
  `holding_horizon` (durée totale jusqu'à la distribution complète).

Limite méthodologique explicite
--------------------------------
Le modèle de cash-flows est une approximation stylisée (courbe en J
paramétrique), PAS un moteur contractuel de type PME/cash-flow engine issu
des données réelles des fonds sous-jacents. Il est adapté à une analyse de
risque de liquidité au niveau stratégique (probabilité de shortfall de
liquidité agrégée), mais ne doit pas être utilisé pour un budgeting de
trésorerie opérationnel fin (capital calls réels par fonds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    asset_return_paths: np.ndarray  # (n_paths, n_years, n_assets)
    portfolio_return_paths: np.ndarray  # (n_paths, n_years)
    nav_paths: np.ndarray  # (n_paths, n_years+1), base 1.0 en t=0
    final_tvpi: np.ndarray  # (n_paths,)
    cashflow_paths: np.ndarray | None = None  # (n_paths, n_years), net cashflow (>0 = distribution nette)


def _cholesky_psd(cov: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Décomposition de Cholesky robuste (ajoute un jitter numérique si la
    matrice n'est pas strictement définie positive, ce qui peut arriver avec
    des matrices de corrélation "stressées" ou mal calibrées)."""
    n = cov.shape[0]
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(cov + jitter * np.eye(n))


def simulate_asset_returns(
    mu: np.ndarray,
    cov: np.ndarray,
    n_paths: int = 100_000,
    n_years: int = 10,
    distribution: str = "normal",
    student_t_dof: float = 5.0,
    random_seed: int | None = 42,
) -> np.ndarray:
    """Simule des rendements annuels multivariés corrélés.

    Parameters
    ----------
    distribution : {"normal", "student_t"}
        "student_t" produit des queues plus épaisses (risque extrême plus
        fréquent), pertinent pour les actifs privés (write-downs abrupts).

    Returns
    -------
    np.ndarray, shape (n_paths, n_years, n_assets)
    """
    rng = np.random.default_rng(random_seed)
    n_assets = len(mu)
    L = _cholesky_psd(cov)

    if distribution == "normal":
        z = rng.standard_normal(size=(n_paths, n_years, n_assets))
        correlated = z @ L.T
        returns = mu[None, None, :] + correlated
    elif distribution == "student_t":
        dof = student_t_dof
        z = rng.standard_normal(size=(n_paths, n_years, n_assets))
        correlated_normal = z @ L.T
        # Mélange chi2 pour obtenir une t-multivariée avec la même matrice de
        # covariance cible (mise à l'échelle par sqrt(dof/(dof-2)) pour que
        # la variance corresponde à `cov`).
        chi2 = rng.chisquare(dof, size=(n_paths, n_years, 1))
        t_scale = np.sqrt(dof / chi2) / np.sqrt(dof / (dof - 2))
        returns = mu[None, None, :] + correlated_normal * t_scale
    else:
        raise ValueError("distribution doit être 'normal' ou 'student_t'.")

    return returns


def aggregate_portfolio_paths(asset_return_paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Agrège les rendements par actif en rendement de portefeuille par
    trajectoire et par année (rebalancement implicite annuel aux poids
    cibles — hypothèse simplificatrice standard pour une allocation
    stratégique)."""
    return asset_return_paths @ weights  # (n_paths, n_years)


def build_nav_paths(portfolio_return_paths: np.ndarray) -> np.ndarray:
    """Construit les trajectoires de NAV cumulées (base 1.0 en t=0) à partir
    des rendements annuels du portefeuille."""
    n_paths, n_years = portfolio_return_paths.shape
    nav = np.ones((n_paths, n_years + 1))
    nav[:, 1:] = np.cumprod(1 + portfolio_return_paths, axis=1)
    return nav


def simplified_jcurve_cashflows(
    weights: np.ndarray,
    capital_duration: np.ndarray,
    holding_horizon: np.ndarray,
    n_years: int,
    portfolio_return_paths: np.ndarray,
) -> np.ndarray:
    """Modèle de cash-flows stylisé (courbe en J) au niveau portefeuille.

    Hypothèses simplificatrices :
    - Les appels de capital sont linéaires sur `capital_duration` années
      (mise à l'échelle par le poids de l'actif) : 100% du capital "engagé"
      (normalisé à 1 par unité de poids) est appelé de façon linéaire.
    - Les distributions démarrent après `capital_duration` et se terminent à
      `holding_horizon`, avec un profil croissant puis dégressif (forme en
      cloche), modulé par la performance simulée de l'année (les bonnes
      années de marché accélèrent les sorties/distributions, hypothèse
      simplificatrice mais directionnellement réaliste).
    - Le cash-flow net par an = distributions - appels, pondéré par le poids
      de chaque classe d'actifs.

    Returns
    -------
    np.ndarray, shape (n_paths, n_years) : cash-flow net (positif = flux
    entrant net pour l'investisseur) par trajectoire/année, en % du capital
    total engagé (=1).
    """
    n_paths = portfolio_return_paths.shape[0]
    years = np.arange(1, n_years + 1)
    cashflows = np.zeros((n_paths, n_years))

    for i, w in enumerate(weights):
        if w <= 0:
            continue
        dur = max(capital_duration[i], 1e-6)
        horizon = max(holding_horizon[i], dur + 1e-6)

        # Profil d'appels : linéaire décroissant sur [0, dur]
        calls = np.where(years <= dur, 1.0 / dur, 0.0)
        calls = calls / calls.sum() if calls.sum() > 0 else calls  # normalise à 1 sur la durée d'appel

        # Profil de distributions : forme en cloche entre dur et horizon
        dist_window = (years > dur) & (years <= horizon)
        dist_years = years[dist_window]
        if len(dist_years) > 0:
            # pondération triangulaire (monte puis descend) sur la fenêtre de distribution
            mid = dur + (horizon - dur) / 2
            weights_dist = 1 - np.abs(dist_years - mid) / ((horizon - dur) / 2 + 1e-6)
            weights_dist = np.clip(weights_dist, 0.01, None)
            weights_dist = weights_dist / weights_dist.sum()
            dist_profile = np.zeros(n_years)
            dist_profile[dist_window] = weights_dist * 1.0  # distribue 100% du capital appelé (TVPI=1 de base)
        else:
            dist_profile = np.zeros(n_years)

        # modulation par la performance simulée (accélère/ralentit les distributions)
        perf_factor = np.clip(1 + portfolio_return_paths, 0.2, None)  # (n_paths, n_years)

        asset_cashflow = w * (dist_profile[None, :] * perf_factor - calls[None, :])
        cashflows += asset_cashflow

    return cashflows


def run_monte_carlo(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    n_years: int = 10,
    distribution: str = "normal",
    student_t_dof: float = 5.0,
    random_seed: int | None = 42,
    capital_duration: np.ndarray | None = None,
    holding_horizon: np.ndarray | None = None,
) -> MonteCarloResult:
    """Point d'entrée principal du moteur Monte Carlo."""
    asset_paths = simulate_asset_returns(
        mu, cov, n_paths=n_paths, n_years=n_years, distribution=distribution,
        student_t_dof=student_t_dof, random_seed=random_seed,
    )
    portfolio_paths = aggregate_portfolio_paths(asset_paths, weights)
    nav_paths = build_nav_paths(portfolio_paths)
    final_tvpi = nav_paths[:, -1]

    cashflow_paths = None
    if capital_duration is not None and holding_horizon is not None:
        cashflow_paths = simplified_jcurve_cashflows(
            weights, capital_duration, holding_horizon, n_years, portfolio_paths
        )

    return MonteCarloResult(
        asset_return_paths=asset_paths,
        portfolio_return_paths=portfolio_paths,
        nav_paths=nav_paths,
        final_tvpi=final_tvpi,
        cashflow_paths=cashflow_paths,
    )


def summarize_distribution(values: np.ndarray, label: str) -> pd.Series:
    """Statistiques descriptives standard d'une distribution simulée."""
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stats = {f"P{p}": float(np.percentile(values, p)) for p in percentiles}
    stats["Moyenne"] = float(np.mean(values))
    stats["Écart-type"] = float(np.std(values))
    return pd.Series(stats, name=label)
