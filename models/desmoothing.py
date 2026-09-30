"""
Désmoothing des rendements d'actifs non cotés (méthode Geltner, 1991).

Motivation
----------
Les rendements des actifs non cotés sont généralement dérivés de valorisations
(NAV) trimestrielles ou annuelles basées sur des appraisals ou des modèles
(DCF, multiples de transactions comparables), mises à jour de façon
partielle et décalée dans le temps. Il en résulte une autocorrélation
positive des rendements observés ("smoothing"), qui :
    - sous-estime fortement la volatilité réelle du sous-jacent ;
    - sous-estime les corrélations avec les marchés cotés (au moment des
      chocs) ;
    - biaise à la baisse le risque calculé dans une optimisation MVO.

Le modèle de Geltner (1991) part de l'hypothèse que le rendement observé au
temps t est une moyenne pondérée du rendement "vrai" (non observable) et du
rendement observé à t-1 :

    r_t^obs = (1 - λ) * r_t^true + λ * r_{t-1}^obs

où λ ∈ [0, 1) est un paramètre de lissage (proche de l'autocorrélation
d'ordre 1 des rendements observés). En inversant cette relation, on peut
reconstituer la série de rendements "vrais" :

    r_t^true = (r_t^obs - λ * r_{t-1}^obs) / (1 - λ)

Si l'on ne dispose pas d'une série temporelle mais seulement d'une
volatilité observée et d'une hypothèse de lissage λ, on peut directement
recalculer la variance "vraie" implicite via la relation de variance :

    Var(r_t^obs) = Var(r_t^true) * [(1 - λ)^2 + λ^2]
    =>  Var(r_t^true) = Var(r_t^obs) / [(1 - λ)^2 + λ^2]

Limite méthodologique : le modèle de Geltner est une approximation d'ordre 1
(AR(1)). Il ne capture pas des structures de lissage plus complexes
(moyennes mobiles d'ordre supérieur, changements de méthode de valorisation
dans le temps, gestion "mark-to-model" discontinue). Il doit être calibré
avec prudence, idéalement en croisant plusieurs sources (autocorrélation
observée, littérature académique par classe d'actifs, dispersion entre
gérants).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_autocorrelation(returns: pd.Series, lag: int = 1) -> float:
    """Autocorrélation empirique d'ordre `lag` d'une série de rendements."""
    r = returns.dropna()
    if len(r) <= lag:
        raise ValueError("Série de rendements trop courte pour estimer l'autocorrélation.")
    return float(r.autocorr(lag=lag))


def geltner_desmooth_series(
    returns: pd.Series, lam: float | None = None, clip_lambda: tuple[float, float] = (0.0, 0.95)
) -> pd.Series:
    """Désmoothing (Geltner AR(1)) d'une série de rendements observés.

    Parameters
    ----------
    returns : pd.Series
        Série de rendements périodiques observés (NAV-based).
    lam : float, optional
        Paramètre de lissage λ. Si None, estimé comme l'autocorrélation
        d'ordre 1 de la série, bornée par `clip_lambda`.
    clip_lambda : tuple
        Bornes de sécurité appliquées à λ (évite une division quasi par zéro
        ou un λ négatif non interprétable).

    Returns
    -------
    pd.Series : rendements désmoothés ("vrais"), même index que `returns`.
    """
    r = returns.copy()
    if lam is None:
        lam = estimate_autocorrelation(r, lag=1)
    lam = float(np.clip(lam, clip_lambda[0], clip_lambda[1]))

    true_returns = (r - lam * r.shift(1)) / (1 - lam)
    true_returns.iloc[0] = r.iloc[0]
    true_returns.name = f"{returns.name}_desmoothed" if returns.name else "desmoothed"
    return true_returns


def desmoothed_volatility_from_series(
    returns: pd.Series, lam: float | None = None, periods_per_year: int = 4
) -> float:
    """Volatilité annualisée désmoothée à partir d'une série de rendements
    périodiques observés (ex. rendements trimestriels de NAV).
    """
    true_r = geltner_desmooth_series(returns, lam=lam).dropna()
    vol_period = true_r.std(ddof=1)
    return float(vol_period * np.sqrt(periods_per_year))


def desmooth_volatility_only(observed_vol: float, lam: float) -> float:
    """Inflation de variance directe lorsqu'on ne dispose que d'une
    volatilité observée (pas de série temporelle) et d'une hypothèse de
    lissage λ.

        Var(vraie) = Var(observée) / [(1-λ)^2 + λ^2]

    Utile en pratique lorsque les hypothèses de volatilité proviennent
    d'une base de données de NAV trimestrielles agrégées (ex. vendor data)
    sans accès à la série brute.
    """
    if not (0.0 <= lam < 1.0):
        raise ValueError("λ doit être dans [0, 1).")
    denom = (1 - lam) ** 2 + lam**2
    var_obs = observed_vol**2
    var_true = var_obs / denom
    return float(np.sqrt(var_true))


# Paramètres de lissage λ indicatifs par classe d'actif, tirés de la
# littérature (Geltner 1991 ; Getmansky, Lo & Makarov 2004 pour les hedge
# funds/private assets illiquides). Ces valeurs sont des points de départ
# raisonnables et DOIVENT être recalibrées si des séries de NAV historiques
# sont disponibles (cf. `geltner_desmooth_series`).
DEFAULT_LAMBDA_BY_ASSET_CLASS: dict[str, float] = {
    "Private Equity Buyout": 0.55,
    "Growth Equity": 0.55,
    "Venture Capital": 0.60,
    "Infrastructure Equity": 0.45,
    "Renewable Energy Infrastructure": 0.40,
    "Natural Capital": 0.50,
    "Private Credit": 0.35,
}


def apply_default_desmoothing(naive_vol_by_asset: pd.Series) -> pd.Series:
    """Applique l'inflation de variance par défaut (DEFAULT_LAMBDA_BY_ASSET_CLASS)
    à une série de volatilités "naïves" (calculées sur NAV trimestrielles non
    désmoothées), pour obtenir des volatilités désmoothées prêtes à être
    utilisées dans l'optimisation.
    """
    out = {}
    for name, vol in naive_vol_by_asset.items():
        lam = DEFAULT_LAMBDA_BY_ASSET_CLASS.get(name, 0.5)
        out[name] = desmooth_volatility_only(vol, lam)
    return pd.Series(out, name="Volatility (désmoothée)")
