"""
Visualisations Plotly pour l'analyse de portefeuille d'actifs privés.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TEMPLATE = "plotly_white"


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    asset_points: pd.DataFrame | None = None,
    highlighted_portfolios: dict[str, tuple[float, float]] | None = None,
) -> go.Figure:
    """Frontière efficiente (volatilité en x, rendement en y).

    Parameters
    ----------
    frontier_df : colonnes ['volatility', 'return'] (une ligne par point de
        la frontière).
    asset_points : colonnes ['Volatility', 'Expected Return'], indexé par nom
        d'actif (affiche chaque classe d'actifs en points individuels).
    highlighted_portfolios : dict nom -> (vol, ret), ex. {"Max Sharpe": (.18, .12)}.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"], y=frontier_df["return"], mode="lines",
            name="Frontière efficiente", line=dict(color="#0091C7", width=3),
        )
    )

    if asset_points is not None:
        fig.add_trace(
            go.Scatter(
                x=asset_points["Volatility"], y=asset_points["Expected Return"], mode="markers+text",
                text=asset_points.index, textposition="top center", name="Classes d'actifs",
                marker=dict(size=10, color="#003781"),
            )
        )

    if highlighted_portfolios:
        for name, (vol, ret) in highlighted_portfolios.items():
            fig.add_trace(
                go.Scatter(
                    x=[vol], y=[ret], mode="markers", name=name,
                    marker=dict(size=14, symbol="star"),
                )
            )

    fig.update_layout(
        title="Frontière efficiente — Portefeuille d'actifs privés",
        xaxis_title="Volatilité annualisée (désmoothée)",
        yaxis_title="Rendement annuel attendu",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template=TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_capital_market_line(
    frontier_df: pd.DataFrame, risk_free_rate: float, tangency_point: tuple[float, float]
) -> go.Figure:
    """Frontière efficiente + Capital Market Line (droite du taux sans risque
    au portefeuille tangent / Max Sharpe)."""
    fig = plot_efficient_frontier(frontier_df, highlighted_portfolios={"Portefeuille tangent (Max Sharpe)": tangency_point})

    max_vol = float(frontier_df["volatility"].max()) * 1.2
    slope = (tangency_point[1] - risk_free_rate) / tangency_point[0]
    cml_x = np.linspace(0, max_vol, 50)
    cml_y = risk_free_rate + slope * cml_x

    fig.add_trace(
        go.Scatter(
            x=cml_x, y=cml_y, mode="lines", name="Capital Market Line",
            line=dict(color="#E8871E", width=2, dash="dash"),
        )
    )
    fig.update_layout(title="Frontière efficiente & Capital Market Line")
    return fig


def plot_allocation_bar(weights: pd.Series, title: str = "Allocation optimale") -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=weights.index, y=weights.values, text=[f"{w:.1%}" for w in weights.values],
            textposition="outside", marker_color="#0091C7",
        )
    )
    fig.update_layout(
        title=title, yaxis_title="Poids", yaxis_tickformat=".0%", template=TEMPLATE, xaxis_tickangle=-30,
    )
    return fig


def plot_risk_contribution(pct_contrib: pd.Series, title: str = "Contribution au risque") -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=pct_contrib.index, y=pct_contrib.values, text=[f"{c:.1%}" for c in pct_contrib.values],
            textposition="outside", marker_color="#003781",
        )
    )
    fig.update_layout(
        title=title, yaxis_title="Contribution au risque (%)", yaxis_tickformat=".0%",
        template=TEMPLATE, xaxis_tickangle=-30,
    )
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Matrice de corrélation", template=TEMPLATE)
    return fig


def plot_sensitivity_returns(
    base_weights: pd.Series, shocked_weights_by_asset: dict[str, pd.Series]
) -> go.Figure:
    """Compare l'allocation de base à des allocations recalculées après choc
    (+/- x%) sur le rendement attendu de chaque actif, un à la fois."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=base_weights.index, y=base_weights.values, name="Allocation de base"))
    for shock_name, w in shocked_weights_by_asset.items():
        fig.add_trace(go.Bar(x=w.index, y=w.values, name=shock_name))
    fig.update_layout(
        title="Sensibilité de l'allocation aux rendements attendus", barmode="group",
        yaxis_tickformat=".0%", template=TEMPLATE, xaxis_tickangle=-30,
    )
    return fig


def plot_sensitivity_correlations(stress_levels: list[float], metrics_by_stress: pd.DataFrame) -> go.Figure:
    """metrics_by_stress : DataFrame indexé par stress_level, colonnes ex.
    ['volatility', 'return', 'sharpe']."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=stress_levels, y=metrics_by_stress["volatility"], name="Volatilité", line=dict(color="#0091C7")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=stress_levels, y=metrics_by_stress["return"], name="Rendement", line=dict(color="#003781")),
        secondary_y=True,
    )
    fig.update_layout(title="Sensibilité au stress de corrélation", template=TEMPLATE, xaxis_title="Niveau de stress")
    fig.update_yaxes(title_text="Volatilité", tickformat=".1%", secondary_y=False)
    fig.update_yaxes(title_text="Rendement", tickformat=".1%", secondary_y=True)
    return fig


def plot_stress_scenarios(scenario_metrics: pd.DataFrame) -> go.Figure:
    """scenario_metrics : DataFrame indexé par nom de scénario, colonnes
    ['return', 'volatility']."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=scenario_metrics.index, y=scenario_metrics["return"], name="Rendement attendu", marker_color="#0091C7")
    )
    fig.add_trace(
        go.Bar(x=scenario_metrics.index, y=scenario_metrics["volatility"], name="Volatilité", marker_color="#E8871E")
    )
    fig.update_layout(
        title="Scénarios de stress", barmode="group", yaxis_tickformat=".0%", template=TEMPLATE,
    )
    return fig


def plot_tvpi_distribution(final_tvpi: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Histogram(x=final_tvpi, nbinsx=100, marker_color="#0091C7"))
    fig.update_layout(
        title="Distribution simulée du TVPI (horizon complet)", xaxis_title="TVPI", yaxis_title="Fréquence",
        template=TEMPLATE,
    )
    return fig


def plot_nav_fanchart(nav_paths: np.ndarray) -> go.Figure:
    """Fan chart des trajectoires de NAV simulées (percentiles 5/25/50/75/95)."""
    years = np.arange(nav_paths.shape[1])
    p5, p25, p50, p75, p95 = np.percentile(nav_paths, [5, 25, 50, 75, 95], axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=p95, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(x=years, y=p5, mode="lines", fill="tonexty", line=dict(width=0),
                    name="Intervalle P5-P95", fillcolor="rgba(0,145,199,0.15)")
    )
    fig.add_trace(go.Scatter(x=years, y=p75, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(x=years, y=p25, mode="lines", fill="tonexty", line=dict(width=0),
                    name="Intervalle P25-P75", fillcolor="rgba(0,145,199,0.35)")
    )
    fig.add_trace(go.Scatter(x=years, y=p50, mode="lines", name="Médiane", line=dict(color="#003781", width=3)))

    fig.update_layout(
        title="Trajectoires simulées de la valeur du portefeuille (base 1.0)",
        xaxis_title="Année", yaxis_title="Valeur (TVPI cumulé)", template=TEMPLATE,
    )
    return fig
