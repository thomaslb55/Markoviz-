"""
Optimisation robuste : gestion de l'incertitude sur les rendements attendus
et sur la matrice de corrélation, et pénalisation de la concentration.

Approche retenue
-----------------
1. Uncertainty sets sur les rendements (approche "box" / robuste worst-case,
   Ben-Tal & Nemirovski) :
   Pour chaque actif i, le rendement attendu est supposé appartenir à
   l'intervalle [mu_i - delta_i, mu_i + delta_i]. Le contrepartie robuste du
   problème de maximisation de rendement s'écrit alors comme la
   maximisation du pire cas :

        max_w   min_{|e_i|<=delta_i}  w'(mu + e)
              = w'mu - kappa * sqrt(w' D w)

   où D = diag(delta_i^2) et kappa est un paramètre d'aversion à
   l'incertitude (kappa=0 -> pas de robustesse ; kappa élevé -> très
   conservateur). Ceci équivaut à pénaliser les rendements par un terme
   proportionnel à l'incertitude pondérée par les poids.

2. Stress sur les corrélations : on "tord" la matrice de corrélation vers un
   scénario de crise (toutes les corrélations tendent vers une valeur élevée
   commune, ex. 0.9) via un paramètre de mélange stress_level ∈ [0, 1] :

        Corr_stressed = (1 - stress_level) * Corr + stress_level * Corr_crisis

   Cela permet de reconstruire une frontière efficiente "en scénario de
   crise" et de comparer les allocations robustes vs non-robustes.

3. Pénalisation de la concentration : ajout d'un terme de régularisation
   L2 (lambda_conc * sum(w_i^2)) à l'objectif, qui pousse les solutions vers
   une allocation plus diversifiée (équivalent à un shrinkage vers
   l'équipondération). C'est une heuristique standard (proche du
   Ridge/Ledoit-Wolf) pour éviter les solutions extrêmes/dégénérées propres
   au MVO classique, particulièrement instable avec des matrices de
   covariance bruitées (cas typique des actifs privés, peu d'historique).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from optimization.mean_variance import InstitutionalConstraints, _equal_weights


def stress_correlation(
    corr: pd.DataFrame, stress_level: float, crisis_correlation: float = 0.90
) -> pd.DataFrame:
    """Mélange la matrice de corrélation vers un scénario de crise où toutes
    les corrélations hors-diagonale convergent vers `crisis_correlation`.

    stress_level = 0   -> matrice de corrélation inchangée
    stress_level = 1   -> toutes les corrélations hors diagonale = crisis_correlation
    """
    if not (0.0 <= stress_level <= 1.0):
        raise ValueError("stress_level doit être dans [0, 1].")
    n = corr.shape[0]
    crisis = np.full((n, n), crisis_correlation)
    np.fill_diagonal(crisis, 1.0)
    stressed = (1 - stress_level) * corr.to_numpy() + stress_level * crisis
    np.fill_diagonal(stressed, 1.0)
    return pd.DataFrame(stressed, index=corr.index, columns=corr.columns)


def robust_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    constraints: InstitutionalConstraints,
    return_uncertainty: np.ndarray,
    kappa: float = 1.0,
    concentration_penalty: float = 0.0,
    target_vol: float | None = None,
) -> np.ndarray:
    """Optimisation robuste : maximise le rendement "worst-case" ajusté de
    l'incertitude, moins une pénalité de concentration, sous les contraintes
    institutionnelles, avec une contrainte optionnelle de volatilité maximale.

    Parameters
    ----------
    return_uncertainty : np.ndarray
        Demi-largeur de l'intervalle d'incertitude par actif (delta_i).
        Peut être dérivée, par exemple, d'un écart-type d'estimation du
        rendement attendu (erreur d'échantillonnage) ou d'un pourcentage du
        rendement attendu (ex. 20% * mu_i).
    kappa : float
        Paramètre d'aversion à l'incertitude (0 = pas de robustesse).
    concentration_penalty : float
        Poids de la pénalité de concentration (lambda_conc, cf. module docstring).
    target_vol : float, optional
        Contrainte de volatilité maximale (sur la matrice de covariance
        NON stressée fournie en argument — utiliser `stress_correlation` en
        amont pour tester un scénario de crise).
    """
    n = len(mu)
    D = return_uncertainty**2

    def objective(w):
        worst_case_return = w @ mu - kappa * np.sqrt(np.sum(D * w**2))
        concentration = concentration_penalty * np.sum(w**2)
        return -(worst_case_return) + concentration

    cons = constraints.all_constraints()
    if target_vol is not None:
        cons = cons + [{"type": "ineq", "fun": lambda w: target_vol**2 - w @ cov @ w}]

    x0 = _equal_weights(n)
    result = minimize(
        objective, x0, method="SLSQP", bounds=constraints.bounds(), constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation robuste échouée : {result.message}")
    return result.x


def robust_efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    constraints: InstitutionalConstraints,
    return_uncertainty: np.ndarray,
    kappa: float = 1.0,
    concentration_penalty: float = 0.0,
    n_points: int = 30,
    vol_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Balaye une grille de contraintes de volatilité maximale et résout
    l'optimisation robuste pour chacune, afin de construire une "frontière
    efficiente robuste" comparable à la frontière MVO classique.
    """
    if vol_range is None:
        vols_diag = np.sqrt(np.diag(cov))
        vol_range = (float(vols_diag.min()) * 0.8, float(vols_diag.max()) * 1.05)

    targets = np.linspace(vol_range[0], vol_range[1], n_points)
    rows = []
    for tv in targets:
        try:
            w = robust_optimize(
                mu, cov, constraints, return_uncertainty, kappa, concentration_penalty, target_vol=tv
            )
            rows.append(
                {
                    "target_vol": tv,
                    "return": float(w @ mu),
                    "volatility": float(np.sqrt(w @ cov @ w)),
                    "weights": w,
                }
            )
        except RuntimeError:
            continue
    return pd.DataFrame(rows)
