"""
Application Streamlit — Frontière efficiente pour portefeuille d'actifs privés.

Lancement :
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from models.asset_classes import (
    DEFAULT_ASSET_CLASSES,
    DEFAULT_CORRELATION,
    assumptions_to_dataframe,
    build_covariance_matrix,
)
from models.desmoothing import DEFAULT_LAMBDA_BY_ASSET_CLASS, desmooth_volatility_only
from models.risk_metrics import risk_summary_table
from optimization.mean_variance import (
    build_constraints_from_assumptions,
    efficient_frontier,
    max_sharpe,
    min_volatility,
    max_return_for_target_risk,
    strategic_allocation_stats,
)
from optimization.black_litterman import View, run_black_litterman
from optimization.robust import robust_optimize, robust_efficient_frontier, stress_correlation
from simulations.monte_carlo import run_monte_carlo, summarize_distribution
from utils.excel_io import read_inputs, write_outputs
from visualization.plots import (
    plot_efficient_frontier, plot_capital_market_line, plot_allocation_bar,
    plot_risk_contribution, plot_correlation_heatmap, plot_sensitivity_correlations,
    plot_tvpi_distribution, plot_nav_fanchart,
)
from reporting.pdf_report import generate_report, format_metric_cell

st.set_page_config(page_title="Frontière efficiente — Actifs privés", layout="wide")

DEFAULT_GROUP_BOUNDS = {
    "Private Equity": (0.20, 0.60),
    "Infrastructure": (0.10, 0.50),
    "Renewables": (0.00, 0.30),
    "Natural Capital": (0.00, 0.20),
    "Private Credit": (0.00, 0.40),
}

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SCENARIOS_DIR = os.path.join(OUTPUTS_DIR, "scenarios")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
os.makedirs(SCENARIOS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Chargement des inputs
# ---------------------------------------------------------------------------
@st.cache_data
def load_default_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    assumptions_df = assumptions_to_dataframe(DEFAULT_ASSET_CLASSES)
    return assumptions_df, DEFAULT_CORRELATION


def load_inputs_from_excel(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    tmp_path = os.path.join(OUTPUTS_DIR, "_uploaded_inputs.xlsx")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    data = read_inputs(tmp_path)
    return data.assumptions, data.correlation, data.group_bounds, data.risk_budget


st.sidebar.title("Paramètres")

source = st.sidebar.radio("Source des hypothèses", ["Exemple par défaut", "Importer un fichier Excel"])

group_bounds = DEFAULT_GROUP_BOUNDS.copy()
risk_budget = {}

if source == "Importer un fichier Excel":
    uploaded = st.sidebar.file_uploader("Fichier d'inputs (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        assumptions_df, correlation_df, group_bounds, risk_budget = load_inputs_from_excel(uploaded)
    else:
        st.sidebar.info("Aucun fichier importé — utilisation des hypothèses par défaut.")
        assumptions_df, correlation_df = load_default_inputs()
else:
    assumptions_df, correlation_df = load_default_inputs()

asset_names = assumptions_df.index.tolist()
n_assets = len(asset_names)

st.sidebar.markdown("---")
risk_free_rate = st.sidebar.slider("Taux sans risque", 0.0, 0.06, 0.02, 0.0025, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Contraintes de groupe")
for group in group_bounds:
    g_min, g_max = group_bounds[group]
    new_min, new_max = st.sidebar.slider(
        f"{group}", 0.0, 1.0, (float(g_min), float(g_max)), 0.01, key=f"group_{group}"
    )
    group_bounds[group] = (new_min, new_max)

st.sidebar.markdown("---")
st.sidebar.subheader("Budget d'illiquidité & tracking error")
illiquidity_budget_max = st.sidebar.slider("Durée de capital moyenne pondérée max (années)", 1.0, 10.0, 5.0, 0.5)
use_tracking_error = st.sidebar.checkbox("Contraindre la tracking error vs allocation stratégique équipondérée")
tracking_error_max = st.sidebar.slider("Tracking error max", 0.0, 0.5, 0.10, 0.01) if use_tracking_error else None

st.sidebar.markdown("---")
st.sidebar.subheader("Sauvegarde de scénario")
scenario_name = st.sidebar.text_input("Nom du scénario", value=f"scenario_{datetime.now():%Y%m%d_%H%M}")
if st.sidebar.button("Sauvegarder le scénario courant"):
    scenario = {
        "assumptions": assumptions_df.to_dict(), "correlation": correlation_df.to_dict(),
        "group_bounds": group_bounds, "risk_free_rate": risk_free_rate,
    }
    with open(os.path.join(SCENARIOS_DIR, f"{scenario_name}.pkl"), "wb") as f:
        pickle.dump(scenario, f)
    st.sidebar.success(f"Scénario sauvegardé : {scenario_name}.pkl")


# ---------------------------------------------------------------------------
# Construction mu / cov / contraintes
# ---------------------------------------------------------------------------
cov_df = build_covariance_matrix(assumptions_df, correlation_df)
mu = assumptions_df["Expected Return"].to_numpy()
cov = cov_df.to_numpy()
vols = assumptions_df["Volatility"].to_numpy()

equal_weights = np.ones(n_assets) / n_assets

constraints = build_constraints_from_assumptions(
    assumptions_df,
    group_bounds=group_bounds,
    illiquidity_metric_col="Capital Duration",
    illiquidity_budget_max=illiquidity_budget_max,
    strategic_weights=equal_weights if use_tracking_error else None,
    tracking_error_max=tracking_error_max,
)

st.title("Frontière efficiente — Portefeuille d'actifs non cotés")

tabs = st.tabs(
    [
        "Hypothèses", "Optimisation", "Frontière efficiente", "Black-Litterman",
        "Optimisation robuste", "Simulation Monte Carlo", "Rapport",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 : Hypothèses
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Hypothèses par classe d'actifs")
    st.dataframe(assumptions_df.style.format("{:.2%}", subset=[
        "Expected Return", "Volatility", "Illiquidity Premium", "Cash Yield",
    ]))

    st.subheader("Matrice de corrélation")
    st.plotly_chart(plot_correlation_heatmap(correlation_df), use_container_width=True)

    with st.expander("Désmoothing des volatilités (méthode Geltner)"):
        st.write(
            "Si vos volatilités proviennent de NAV trimestrielles non désmoothées, "
            "appliquez un facteur d'inflation de variance par classe d'actifs (paramètre λ)."
        )
        lambdas = {}
        cols = st.columns(4)
        for i, name in enumerate(asset_names):
            default_lam = DEFAULT_LAMBDA_BY_ASSET_CLASS.get(name, 0.5)
            lambdas[name] = cols[i % 4].slider(f"λ — {name}", 0.0, 0.9, default_lam, 0.05, key=f"lam_{name}")
        if st.button("Appliquer le désmoothing aux volatilités affichées"):
            desmoothed = {
                name: desmooth_volatility_only(assumptions_df.loc[name, "Volatility"], lambdas[name])
                for name in asset_names
            }
            st.write(pd.Series(desmoothed, name="Volatilité désmoothée").to_frame().style.format("{:.2%}"))
            st.caption(
                "Ces volatilités désmoothées ne sont pas automatiquement réinjectées dans le "
                "fichier d'hypothèses — mettez à jour la colonne 'Volatility' du fichier Excel "
                "d'inputs si vous souhaitez les utiliser dans l'optimisation."
            )


# ---------------------------------------------------------------------------
# Tab 2 : Optimisation (objectifs 1 à 4)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Objectif d'optimisation")
    objective = st.selectbox(
        "Choisir un objectif",
        [
            "Maximum de rendement pour un niveau de risque donné",
            "Minimum de risque pour un rendement cible",
            "Maximum ratio de Sharpe",
            "Allocation stratégique équipondérée (référence)",
        ],
    )

    weights = None
    if objective == "Maximum de rendement pour un niveau de risque donné":
        target_vol = st.slider("Volatilité cible max", float(vols.min()) * 0.8, float(vols.max()), float(vols.mean()), 0.01)
        if st.button("Optimiser", key="opt1"):
            weights = max_return_for_target_risk(mu, cov, constraints, target_vol)

    elif objective == "Minimum de risque pour un rendement cible":
        target_return = st.slider("Rendement cible", float(mu.min()), float(mu.max()), float(mu.mean()), 0.005)
        if st.button("Optimiser", key="opt2"):
            weights = min_volatility(mu, cov, constraints, target_return=target_return)

    elif objective == "Maximum ratio de Sharpe":
        if st.button("Optimiser", key="opt3"):
            weights = max_sharpe(mu, cov, constraints, risk_free_rate=risk_free_rate)

    else:
        weights = equal_weights

    if weights is not None:
        w_series = pd.Series(weights, index=asset_names, name="Poids")
        st.session_state["last_weights"] = w_series

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(plot_allocation_bar(w_series), use_container_width=True)
        with col2:
            summary, contrib_df = risk_summary_table(weights, mu, cov, vols, asset_names, risk_free_rate)
            formatted_summary = summary.copy()
            formatted_summary["Portefeuille"] = [
                format_metric_cell(idx, v) for idx, v in summary["Portefeuille"].items()
            ]
            st.dataframe(formatted_summary)

        st.plotly_chart(plot_risk_contribution(contrib_df["Contribution au risque (%)"]), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 : Frontière efficiente + CML
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Frontière efficiente")
    n_points = st.slider("Nombre de points de la frontière", 10, 100, 40, 5)
    if st.button("Calculer la frontière efficiente"):
        with st.spinner("Optimisation en cours..."):
            frontier_df = efficient_frontier(mu, cov, constraints, n_points=n_points)
            st.session_state["frontier_df"] = frontier_df

    if "frontier_df" in st.session_state:
        frontier_df = st.session_state["frontier_df"]
        asset_points = assumptions_df[["Expected Return", "Volatility"]]

        try:
            sharpe_weights = max_sharpe(mu, cov, constraints, risk_free_rate=risk_free_rate)
            tangency = (
                float(np.sqrt(sharpe_weights @ cov @ sharpe_weights)),
                float(sharpe_weights @ mu),
            )
            fig = plot_capital_market_line(frontier_df, risk_free_rate, tangency)
        except RuntimeError:
            fig = plot_efficient_frontier(frontier_df, asset_points)

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(frontier_df.drop(columns=["weights"]).style.format("{:.2%}"))


# ---------------------------------------------------------------------------
# Tab 4 : Black-Litterman
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Vues stratégiques (Black-Litterman)")
    st.caption(
        "Les rendements d'équilibre implicites sont calculés à partir d'une allocation "
        "stratégique de référence (équipondérée par défaut), en l'absence de portefeuille "
        "de marché observable pour les actifs privés."
    )

    n_views = st.number_input("Nombre de vues", 0, 5, 1)
    views = []
    for i in range(int(n_views)):
        st.markdown(f"**Vue {i+1}**")
        c1, c2, c3, c4 = st.columns(4)
        asset_a = c1.selectbox(f"Actif A ({i})", asset_names, key=f"va_{i}")
        is_relative = c2.checkbox(f"Vue relative ({i})", key=f"rel_{i}")
        asset_b = c2.selectbox(f"vs Actif B ({i})", asset_names, key=f"vb_{i}", disabled=not is_relative)
        expected_value = c3.slider(f"Écart / niveau attendu ({i})", -0.10, 0.20, 0.02, 0.005, key=f"ev_{i}")
        confidence = c4.slider(f"Confiance ({i})", 0.05, 1.0, 0.5, 0.05, key=f"cf_{i}")

        if is_relative:
            views.append(View(assets=[asset_a, asset_b], weights=[1, -1], expected_value=expected_value, confidence=confidence))
        else:
            views.append(View(assets=[asset_a], weights=[1], expected_value=expected_value, confidence=confidence))

    if st.button("Calculer les rendements postérieurs (Black-Litterman)"):
        strategic_expected_return = float(equal_weights @ mu)
        bl_result = run_black_litterman(
            asset_names, cov_df, equal_weights, strategic_expected_return, views, risk_free_rate,
        )
        comparison = pd.DataFrame(
            {"Rendement (hypothèses)": assumptions_df["Expected Return"],
             "Rendement d'équilibre implicite": bl_result["pi_equilibrium"],
             "Rendement postérieur (BL)": bl_result["mu_bl"]}
        )
        st.session_state["bl_comparison"] = comparison
        st.session_state["bl_mu"] = bl_result["mu_bl"].to_numpy()

    if "bl_comparison" in st.session_state:
        st.dataframe(st.session_state["bl_comparison"].style.format("{:.2%}"))

    if "bl_mu" in st.session_state and st.button("Optimiser avec les rendements Black-Litterman"):
        w_bl = max_sharpe(st.session_state["bl_mu"], cov, constraints, risk_free_rate)
        st.plotly_chart(
            plot_allocation_bar(pd.Series(w_bl, index=asset_names), "Allocation Max Sharpe (rendements BL)"),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 5 : Optimisation robuste
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Optimisation robuste")

    col1, col2, col3 = st.columns(3)
    uncertainty_pct = col1.slider("Incertitude sur les rendements (% du rendement attendu)", 0.0, 0.5, 0.20, 0.05)
    kappa = col2.slider("Aversion à l'incertitude (kappa)", 0.0, 3.0, 1.0, 0.1)
    concentration_penalty = col3.slider("Pénalité de concentration", 0.0, 0.5, 0.05, 0.01)

    return_uncertainty = np.abs(mu) * uncertainty_pct

    stress_level = st.slider("Stress de corrélation (0 = aucun, 1 = crise totale)", 0.0, 1.0, 0.0, 0.05)
    stressed_corr = stress_correlation(correlation_df, stress_level)
    stressed_cov = build_covariance_matrix(assumptions_df, stressed_corr).to_numpy()

    target_vol_robust = st.slider(
        "Volatilité cible (optimisation robuste)", float(vols.min()) * 0.8, float(vols.max()), float(vols.mean()), 0.01,
        key="robust_vol",
    )

    if st.button("Optimiser (robuste)"):
        w_robust = robust_optimize(
            mu, stressed_cov, constraints, return_uncertainty, kappa, concentration_penalty, target_vol_robust,
        )
        w_classic = min_volatility(mu, cov, constraints, target_return=None)

        comp = pd.DataFrame(
            {"Allocation robuste": w_robust, "Allocation MVO classique (vol. min)": w_classic}, index=asset_names,
        )
        st.plotly_chart(
            plot_allocation_bar(comp["Allocation robuste"], "Allocation robuste vs classique"),
            use_container_width=True,
        )
        st.dataframe(comp.style.format("{:.1%}"))

    if st.button("Sensibilité au stress de corrélation"):
        levels = np.linspace(0, 1, 6)
        rows = []
        for lvl in levels:
            sc = stress_correlation(correlation_df, lvl)
            scov = build_covariance_matrix(assumptions_df, sc).to_numpy()
            w = min_volatility(mu, scov, constraints, target_return=None)
            rows.append({"volatility": float(np.sqrt(w @ scov @ w)), "return": float(w @ mu)})
        metrics_df = pd.DataFrame(rows, index=levels)
        st.plotly_chart(plot_sensitivity_correlations(list(levels), metrics_df), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 6 : Monte Carlo
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Simulation Monte Carlo")

    col1, col2, col3 = st.columns(3)
    n_paths = col1.number_input("Nombre de trajectoires", 1_000, 200_000, 100_000, 1_000)
    n_years = col2.number_input("Horizon (années)", 1, 20, 10, 1)
    distribution = col3.selectbox("Distribution", ["normal", "student_t"])

    weights_for_mc = st.session_state.get("last_weights", pd.Series(equal_weights, index=asset_names))
    st.write("Allocation utilisée pour la simulation :")
    st.dataframe(weights_for_mc.to_frame().T.style.format("{:.1%}"))

    if st.button("Lancer la simulation Monte Carlo"):
        with st.spinner(f"Simulation de {int(n_paths):,} trajectoires..."):
            mc_result = run_monte_carlo(
                mu, cov, weights_for_mc.to_numpy(), n_paths=int(n_paths), n_years=int(n_years),
                distribution=distribution,
                capital_duration=assumptions_df["Capital Duration"].to_numpy(),
                holding_horizon=assumptions_df["Holding Horizon"].to_numpy(),
            )
        st.session_state["mc_result"] = mc_result

    if "mc_result" in st.session_state:
        mc_result = st.session_state["mc_result"]
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_nav_fanchart(mc_result.nav_paths), use_container_width=True)
        with col2:
            st.plotly_chart(plot_tvpi_distribution(mc_result.final_tvpi), use_container_width=True)

        st.write("Statistiques de la distribution du TVPI final :")
        st.dataframe(summarize_distribution(mc_result.final_tvpi, "TVPI final").to_frame().T)

        annual_returns_flat = mc_result.portfolio_return_paths[:, -1]
        st.write("Statistiques de la distribution des rendements annuels (dernière année simulée) :")
        st.dataframe(summarize_distribution(annual_returns_flat, "Rendement annuel").to_frame().T.style.format("{:.2%}"))


# ---------------------------------------------------------------------------
# Tab 7 : Rapport
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Génération du rapport PDF")

    if "last_weights" not in st.session_state:
        st.warning("Calculez d'abord une allocation dans l'onglet 'Optimisation'.")
    else:
        w_series = st.session_state["last_weights"]
        summary, contrib_df = risk_summary_table(w_series.to_numpy(), mu, cov, vols, asset_names, risk_free_rate)

        methodology_notes = [
            "Les volatilités doivent être désmoothées (méthode Geltner) avant utilisation dans ce modèle "
            "si elles proviennent de NAV trimestrielles non ajustées.",
            "Le modèle Black-Litterman utilise une allocation stratégique de référence comme proxy du "
            "portefeuille de marché, faute de portefeuille de marché observable pour les actifs privés.",
            "Le modèle de cash-flows Monte Carlo est une approximation stylisée (courbe en J paramétrique), "
            "pas un moteur contractuel fondé sur les données réelles des fonds sous-jacents.",
            "Les performances passées et les hypothèses forward-looking ne préjugent pas des performances futures.",
        ]

        report_name = st.text_input("Nom du fichier de rapport", value=f"rapport_{datetime.now():%Y%m%d_%H%M}.pdf")
        if st.button("Générer le rapport PDF"):
            frontier_fig = None
            if "frontier_df" in st.session_state:
                frontier_fig = plot_efficient_frontier(st.session_state["frontier_df"], assumptions_df[["Expected Return", "Volatility"]])
            allocation_fig = plot_allocation_bar(w_series)
            correlation_fig = plot_correlation_heatmap(correlation_df)

            output_path = os.path.join(REPORTS_DIR, report_name)
            generate_report(
                output_path, "Allianz — Allocation stratégique actifs privés", w_series, summary, contrib_df,
                efficient_frontier_fig=frontier_fig, allocation_fig=allocation_fig,
                correlation_fig=correlation_fig, methodology_notes=methodology_notes,
            )
            st.success(f"Rapport généré : {output_path}")
            with open(output_path, "rb") as f:
                st.download_button("Télécharger le rapport PDF", f, file_name=report_name)

        if st.button("Exporter les résultats en Excel"):
            frontier_df = st.session_state.get("frontier_df")
            xlsx_path = os.path.join(OUTPUTS_DIR, f"resultats_{datetime.now():%Y%m%d_%H%M}.xlsx")
            write_outputs(xlsx_path, w_series, summary, contrib_df, frontier_df)
            st.success(f"Résultats exportés : {xlsx_path}")
