"""
Optimisation Mean-Variance (Markowitz) sous contraintes institutionnelles,
adaptée aux portefeuilles d'actifs privés.

Moteur principal : scipy.optimize (SLSQP), qui permet d'exprimer nativement
les contraintes institutionnelles demandées :
    - bornes min/max par actif ;
    - bornes min/max par regroupement (ex. "Private Equity" = Buyout + Growth
      + VC, entre 20% et 60%) ;
    - budget d'illiquidité maximum (duration de capital pondérée) ;
    - tracking error maximum vs une allocation stratégique cible.

Un wrapper optionnel basé sur PyPortfolioOpt est fourni pour les cas simples
(bornes par actif uniquement, sans contraintes de groupe), à titre de
validation croisée / convenance (cf. `max_sharpe_pypfopt`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Contraintes institutionnelles
# ---------------------------------------------------------------------------
@dataclass
class InstitutionalConstraints:
    asset_names: list[str]
    min_weights: np.ndarray  # bornes par actif
    max_weights: np.ndarray
    group_membership: dict[str, list[str]] = field(default_factory=dict)  # groupe -> liste d'actifs
    group_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)  # groupe -> (min, max)
    max_single_weight: float | None = None  # concentration max par actif (override max_weights)
    illiquidity_metric: np.ndarray | None = None  # ex. capital_duration ou holding_horizon par actif
    illiquidity_budget_max: float | None = None  # plafond de la moyenne pondérée de illiquidity_metric
    strategic_weights: np.ndarray | None = None  # allocation stratégique cible (pour tracking error)
    tracking_error_max: float | None = None  # écart type max des écarts de poids vs cible (approx. simple)

    def bounds(self) -> list[tuple[float, float]]:
        mins = self.min_weights.copy()
        maxs = self.max_weights.copy()
        if self.max_single_weight is not None:
            maxs = np.minimum(maxs, self.max_single_weight)
        return list(zip(mins, maxs))

    def group_constraint_dicts(self) -> list[dict]:
        cons = []
        n = len(self.asset_names)
        for group, members in self.group_membership.items():
            if group not in self.group_bounds:
                continue
            g_min, g_max = self.group_bounds[group]
            mask = np.array([1.0 if a in members else 0.0 for a in self.asset_names])

            cons.append({"type": "ineq", "fun": (lambda w, mask=mask, g_min=g_min: mask @ w - g_min)})
            cons.append({"type": "ineq", "fun": (lambda w, mask=mask, g_max=g_max: g_max - mask @ w)})
        return cons

    def illiquidity_constraint_dicts(self) -> list[dict]:
        if self.illiquidity_metric is None or self.illiquidity_budget_max is None:
            return []
        metric = self.illiquidity_metric
        budget = self.illiquidity_budget_max
        return [{"type": "ineq", "fun": lambda w, metric=metric, budget=budget: budget - metric @ w}]

    def tracking_error_constraint_dicts(self) -> list[dict]:
        """Approximation simple de la tracking error : distance euclidienne
        entre le vecteur de poids et l'allocation stratégique cible, bornée
        par `tracking_error_max`. (Une définition plus fine utiliserait la
        covariance des écarts de rendement ; cette approximation "distance
        de poids" est un choix pragmatique standard en allocation stratégique
        d'actifs privés, où les rendements des écarts eux-mêmes sont peu
        observables.)
        """
        if self.strategic_weights is None or self.tracking_error_max is None:
            return []
        target = self.strategic_weights
        max_te = self.tracking_error_max
        return [
            {
                "type": "ineq",
                "fun": lambda w, target=target, max_te=max_te: max_te - np.sqrt(np.sum((w - target) ** 2)),
            }
        ]

    def all_constraints(self) -> list[dict]:
        sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        return (
            [sum_to_one]
            + self.group_constraint_dicts()
            + self.illiquidity_constraint_dicts()
            + self.tracking_error_constraint_dicts()
        )


def build_constraints_from_assumptions(
    assumptions_df: pd.DataFrame,
    group_bounds: dict[str, tuple[float, float]] | None = None,
    illiquidity_metric_col: str = "Capital Duration",
    illiquidity_budget_max: float | None = None,
    strategic_weights: np.ndarray | None = None,
    tracking_error_max: float | None = None,
    max_single_weight: float | None = None,
) -> InstitutionalConstraints:
    """Construit un objet `InstitutionalConstraints` à partir du DataFrame
    d'hypothèses (issu de `models.asset_classes.assumptions_to_dataframe` ou
    de `utils.excel_io.read_inputs`), qui doit contenir les colonnes
    'Min Weight', 'Max Weight', 'Group' et la colonne d'illiquidité choisie.
    """
    names = assumptions_df.index.tolist()
    group_membership: dict[str, list[str]] = {}
    for asset, group in assumptions_df["Group"].items():
        group_membership.setdefault(group, []).append(asset)

    return InstitutionalConstraints(
        asset_names=names,
        min_weights=assumptions_df["Min Weight"].to_numpy(dtype=float),
        max_weights=assumptions_df["Max Weight"].to_numpy(dtype=float),
        group_membership=group_membership,
        group_bounds=group_bounds or {},
        max_single_weight=max_single_weight,
        illiquidity_metric=(
            assumptions_df[illiquidity_metric_col].to_numpy(dtype=float)
            if illiquidity_metric_col in assumptions_df.columns
            else None
        ),
        illiquidity_budget_max=illiquidity_budget_max,
        strategic_weights=strategic_weights,
        tracking_error_max=tracking_error_max,
    )


# ---------------------------------------------------------------------------
# Optimisation scipy (moteur principal)
# ---------------------------------------------------------------------------
def _equal_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


def min_volatility(
    mu: np.ndarray,
    cov: np.ndarray,
    constraints: InstitutionalConstraints,
    target_return: float | None = None,
) -> np.ndarray:
    """Minimise la volatilité du portefeuille, avec une contrainte optionnelle
    de rendement cible (frontière efficiente si `target_return` est balayé).
    """
    n = len(mu)
    cons = constraints.all_constraints()
    if target_return is not None:
        cons = cons + [{"type": "eq", "fun": lambda w, mu=mu, tr=target_return: w @ mu - tr}]

    def objective(w):
        return w @ cov @ w

    x0 = _equal_weights(n)
    result = minimize(
        objective, x0, method="SLSQP", bounds=constraints.bounds(), constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation min_volatility échouée : {result.message}")
    return result.x


def max_return_for_target_risk(
    mu: np.ndarray,
    cov: np.ndarray,
    constraints: InstitutionalConstraints,
    target_vol: float,
) -> np.ndarray:
    """Maximise le rendement attendu sous une contrainte de volatilité
    maximale (target_vol)."""
    n = len(mu)
    cons = constraints.all_constraints() + [
        {"type": "ineq", "fun": lambda w, cov=cov, tv=target_vol: tv**2 - w @ cov @ w}
    ]

    def objective(w):
        return -(w @ mu)

    x0 = _equal_weights(n)
    result = minimize(
        objective, x0, method="SLSQP", bounds=constraints.bounds(), constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation max_return_for_target_risk échouée : {result.message}")
    return result.x


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    constraints: InstitutionalConstraints,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Maximise le ratio de Sharpe (rendement en excès / volatilité)."""
    n = len(mu)
    cons = constraints.all_constraints()

    def neg_sharpe(w):
        ret = w @ mu - risk_free_rate
        vol = np.sqrt(w @ cov @ w)
        if vol <= 1e-12:
            return 1e6
        return -ret / vol

    x0 = _equal_weights(n)
    result = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=constraints.bounds(), constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation max_sharpe échouée : {result.message}")
    return result.x


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    constraints: InstitutionalConstraints,
    n_points: int = 50,
) -> pd.DataFrame:
    """Calcule la frontière efficiente en balayant une grille de rendements
    cibles entre le rendement du portefeuille de volatilité minimale et le
    rendement maximum atteignable sous contraintes.
    """
    n = len(mu)

    min_vol_weights = min_volatility(mu, cov, constraints, target_return=None)
    min_ret = float(min_vol_weights @ mu)

    # rendement max atteignable : on maximise directement w @ mu sous contraintes
    cons = constraints.all_constraints()
    result = minimize(
        lambda w: -(w @ mu), _equal_weights(n), method="SLSQP",
        bounds=constraints.bounds(), constraints=cons, options={"maxiter": 500},
    )
    max_ret = float(result.x @ mu) if result.success else float(np.max(mu))

    targets = np.linspace(min_ret, max_ret * 0.999, n_points)
    rows = []
    for t in targets:
        try:
            w = min_volatility(mu, cov, constraints, target_return=t)
            rows.append(
                {
                    "target_return": t,
                    "return": float(w @ mu),
                    "volatility": float(np.sqrt(w @ cov @ w)),
                    "weights": w,
                }
            )
        except RuntimeError:
            continue
    return pd.DataFrame(rows)


def strategic_allocation_stats(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray
) -> dict:
    """Statistiques simples pour une allocation stratégique fixée par
    l'utilisateur (objectif 4 : "Allocation stratégique long terme"),
    par opposition à une allocation ré-optimisée."""
    return {
        "return": float(weights @ mu),
        "volatility": float(np.sqrt(weights @ cov @ weights)),
    }


# ---------------------------------------------------------------------------
# Wrapper optionnel PyPortfolioOpt (cas simples, bornes par actif uniquement)
# ---------------------------------------------------------------------------
def max_sharpe_pypfopt(
    mu: pd.Series, cov: pd.DataFrame, weight_bounds: tuple[float, float] = (0.0, 1.0), risk_free_rate: float = 0.0
) -> pd.Series:
    """Calcule l'allocation Max Sharpe via PyPortfolioOpt (sans contraintes
    de groupe / illiquidité / tracking error — pour cela, utiliser le moteur
    scipy ci-dessus). Utile comme validation croisée rapide.
    """
    from pypfopt import EfficientFrontier

    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned = ef.clean_weights()
    return pd.Series(cleaned)
