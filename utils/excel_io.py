"""
Lecture / écriture des fichiers Excel d'entrée et de sortie.

Le fichier d'inputs (voir inputs/inputs_template.xlsx) contient 4 feuilles :
    - "Assumptions"   : hypothèses par classe d'actifs
    - "Correlation"   : matrice de corrélation (modifiable)
    - "Constraints"   : contraintes de groupe (min/max par regroupement)
    - "RiskBudget"     : budget de risque / illiquidité / tracking error
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class InputsData:
    assumptions: pd.DataFrame       # indexé par 'Asset Class'
    correlation: pd.DataFrame       # matrice carrée, mêmes index/colonnes que assumptions
    group_bounds: dict[str, tuple[float, float]]
    risk_budget: dict[str, float]


def read_inputs(path: str) -> InputsData:
    """Charge le classeur Excel d'inputs et retourne un objet `InputsData`
    structuré, prêt à être consommé par les modules d'optimisation.
    """
    assumptions = pd.read_excel(path, sheet_name="Assumptions", index_col="Asset Class")

    correlation = pd.read_excel(path, sheet_name="Correlation", index_col=0)
    correlation = correlation.loc[assumptions.index, assumptions.index]

    constraints_df = pd.read_excel(path, sheet_name="Constraints")
    group_bounds = {
        row["Group"]: (float(row["Min"]), float(row["Max"])) for _, row in constraints_df.iterrows()
    }

    risk_budget_df = pd.read_excel(path, sheet_name="RiskBudget")
    risk_budget = {row["Paramètre"]: float(row["Valeur"]) for _, row in risk_budget_df.iterrows()}

    return InputsData(
        assumptions=assumptions, correlation=correlation, group_bounds=group_bounds, risk_budget=risk_budget
    )


def write_outputs(
    path: str,
    weights: pd.Series,
    risk_summary: pd.DataFrame,
    risk_contribution: pd.DataFrame,
    frontier_df: pd.DataFrame | None = None,
) -> None:
    """Exporte les résultats d'optimisation dans un classeur Excel."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        weights.to_frame("Poids").to_excel(writer, sheet_name="Allocation")
        risk_summary.to_excel(writer, sheet_name="Résumé du risque")
        risk_contribution.to_excel(writer, sheet_name="Contribution au risque")
        if frontier_df is not None:
            export_df = frontier_df.drop(columns=["weights"], errors="ignore")
            export_df.to_excel(writer, sheet_name="Frontière efficiente", index=False)
