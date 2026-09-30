"""
Modèle Black-Litterman adapté aux actifs privés.

Adaptation méthodologique
-------------------------
Le modèle Black-Litterman standard part des rendements d'équilibre implicites
d'un portefeuille de marché observable (pondérations de capitalisation).
Pour les actifs privés, il n'existe pas de "portefeuille de marché" au sens
strict. On utilise donc, comme point de départ (rendements d'équilibre "pi"),
l'allocation STRATÉGIQUE cible définie par l'investisseur (ou une allocation
equal-risk-contribution par défaut), ce qui est une pratique courante en
allocation stratégique d'actifs privés (faute de mieux). Cette hypothèse
doit être explicitée dans toute documentation destinée à un comité
d'investissement.

Étapes
------
1. Rendements d'équilibre implicites : pi = δ * Σ * w_eq
   où δ (aversion au risque implicite) = (mu_eq - rf) / sigma_eq^2, calculé à
   partir du portefeuille d'équilibre choisi (ou fourni directement par
   l'utilisateur).
2. Vues de l'utilisateur : matrice P (sélection des actifs concernés par
   chaque vue), vecteur Q (rendement de la vue), et Omega (incertitude de la
   vue, dérivée d'une pondération de confiance 0-100%).
3. Rendements postérieurs (mu_BL) et covariance postérieure (Sigma_BL) par la
   formule standard de Black & Litterman (1992).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class View:
    """Vue stratégique de l'utilisateur.

    Exemple : "Je pense que le Venture Capital va surperformer l'Infrastructure
    de 4% par an, avec une confiance de 60%" ->
        assets=["Venture Capital", "Infrastructure Equity"], weights=[1, -1],
        expected_value=0.04, confidence=0.60

    Exemple de vue absolue : "Je pense que Natural Capital va délivrer 9%"->
        assets=["Natural Capital"], weights=[1], expected_value=0.09, confidence=0.5
    """

    assets: list[str]
    weights: list[float]  # ex. [1] pour une vue absolue, [1, -1] pour une vue relative
    expected_value: float
    confidence: float  # 0 (aucune confiance) à 1 (certitude totale)


def implied_risk_aversion(eq_return: float, eq_variance: float, risk_free_rate: float = 0.0) -> float:
    if eq_variance <= 0:
        raise ValueError("La variance du portefeuille d'équilibre doit être positive.")
    return (eq_return - risk_free_rate) / eq_variance


def implied_equilibrium_returns(cov: np.ndarray, w_eq: np.ndarray, delta: float) -> np.ndarray:
    """pi = delta * Sigma * w_eq (delta calculé au préalable via `implied_risk_aversion`)."""
    return delta * (cov @ w_eq)


def build_views_matrices(
    views: list[View], asset_names: list[str], tau: float, cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les matrices P (K x N), Q (K,) et Omega (K x K, diagonale)
    à partir d'une liste de vues utilisateur.

    Omega_kk = tau * (P_k . Sigma . P_k^T) / confidence_k
    (plus la confiance est élevée, plus Omega_kk est petit, i.e. la vue est
    considérée comme précise). Une confiance -> 0 rend la vue quasi ignorée ;
    une confiance -> 1 fait fortement converger mu_BL vers la vue.
    """
    n = len(asset_names)
    k = len(views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)
    idx = {name: i for i, name in enumerate(asset_names)}

    for i, view in enumerate(views):
        for asset, w in zip(view.assets, view.weights):
            P[i, idx[asset]] = w
        Q[i] = view.expected_value
        confidence = min(max(view.confidence, 1e-4), 1.0)  # évite division par zéro
        p_row = P[i, :]
        variance_of_view = float(p_row @ cov @ p_row)
        omega_diag[i] = tau * variance_of_view / confidence

    Omega = np.diag(omega_diag)
    return P, Q, Omega


def black_litterman_posterior(
    pi: np.ndarray,
    cov: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les rendements et la covariance postérieurs de Black-Litterman.

    mu_BL = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 [(tau*Sigma)^-1 pi + P' Omega^-1 Q]
    Sigma_BL = Sigma + [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
    """
    tau_sigma_inv = np.linalg.inv(tau * cov)
    omega_inv = np.linalg.inv(Omega)

    middle_inv = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    mu_bl = middle_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    sigma_bl = cov + middle_inv

    return mu_bl, sigma_bl


def run_black_litterman(
    asset_names: list[str],
    cov: pd.DataFrame,
    strategic_weights: np.ndarray,
    strategic_expected_return: float,
    views: list[View],
    risk_free_rate: float = 0.0,
    tau: float = 0.05,
) -> dict:
    """Point d'entrée complet : calcule pi, delta, construit les matrices de
    vues, puis retourne mu_BL et Sigma_BL sous forme de Series/DataFrame.
    """
    cov_arr = cov.to_numpy()
    eq_variance = float(strategic_weights @ cov_arr @ strategic_weights)
    delta = implied_risk_aversion(strategic_expected_return, eq_variance, risk_free_rate)
    pi = delta * (cov_arr @ strategic_weights)

    if len(views) == 0:
        mu_bl, sigma_bl = pi, cov_arr
    else:
        P, Q, Omega = build_views_matrices(views, asset_names, tau, cov_arr)
        mu_bl, sigma_bl = black_litterman_posterior(pi, cov_arr, P, Q, Omega, tau)

    return {
        "pi_equilibrium": pd.Series(pi, index=asset_names, name="Rendement d'équilibre implicite"),
        "mu_bl": pd.Series(mu_bl, index=asset_names, name="Rendement postérieur (BL)"),
        "sigma_bl": pd.DataFrame(sigma_bl, index=asset_names, columns=asset_names),
        "delta": delta,
    }
