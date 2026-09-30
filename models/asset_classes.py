"""
Définitions des classes d'actifs non cotés et de leurs hypothèses par défaut.

Ces valeurs sont des hypothèses de marché indicatives (forward-looking),
cohérentes avec les fourchettes communiquées par l'utilisateur :
    - Private Equity (Buyout/Growth/VC) : 14% - 18%
    - Infrastructure Equity             : 9% - 11%
    - Renewable Energy Infrastructure   : 8% - 10%
    - Natural Capital                   : 7% - 9%

Elles sont fournies à titre d'exemple réaliste et doivent être remplacées par
les hypothèses propres à l'utilisateur via le fichier Excel d'inputs
(voir inputs/inputs_template.xlsx et utils/excel_io.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class AssetClassAssumption:
    """Hypothèses de marché pour une classe d'actifs non cotée."""

    name: str
    expected_return: float          # rendement annuel attendu (arithmétique), ex. 0.15
    volatility: float                # volatilité annualisée DÉSMOOTHÉE, ex. 0.22
    illiquidity_premium: float       # prime d'illiquidité incluse dans le rendement attendu
    capital_duration: float          # durée moyenne d'appel du capital (années)
    holding_horizon: float           # horizon de détention total (années)
    cash_yield: float                # distribution de cash récurrente (% NAV / an)
    expected_tvpi: float              # TVPI attendu à maturité
    expected_dpi: float               # DPI attendu à maturité
    min_weight: float = 0.0          # borne min d'allocation (contrainte institutionnelle)
    max_weight: float = 1.0          # borne max d'allocation
    group: str = ""                  # regroupement (PE, Infrastructure, Renewables, Natural Capital, Credit)

    def to_dict(self) -> dict:
        return {
            "Asset Class": self.name,
            "Expected Return": self.expected_return,
            "Volatility": self.volatility,
            "Illiquidity Premium": self.illiquidity_premium,
            "Capital Duration": self.capital_duration,
            "Holding Horizon": self.holding_horizon,
            "Cash Yield": self.cash_yield,
            "Expected TVPI": self.expected_tvpi,
            "Expected DPI": self.expected_dpi,
            "Min Weight": self.min_weight,
            "Max Weight": self.max_weight,
            "Group": self.group,
        }


# ---------------------------------------------------------------------------
# Jeu d'hypothèses par défaut (exemple réaliste, à titre indicatif uniquement)
# ---------------------------------------------------------------------------
DEFAULT_ASSET_CLASSES: list[AssetClassAssumption] = [
    AssetClassAssumption(
        name="Private Equity Buyout",
        expected_return=0.15,
        volatility=0.22,
        illiquidity_premium=0.03,
        capital_duration=4.0,
        holding_horizon=10.0,
        cash_yield=0.00,
        expected_tvpi=1.80,
        expected_dpi=1.60,
        min_weight=0.00,
        max_weight=0.60,
        group="Private Equity",
    ),
    AssetClassAssumption(
        name="Growth Equity",
        expected_return=0.155,
        volatility=0.26,
        illiquidity_premium=0.03,
        capital_duration=3.5,
        holding_horizon=8.0,
        cash_yield=0.00,
        expected_tvpi=1.90,
        expected_dpi=1.50,
        min_weight=0.00,
        max_weight=0.60,
        group="Private Equity",
    ),
    AssetClassAssumption(
        name="Venture Capital",
        expected_return=0.18,
        volatility=0.35,
        illiquidity_premium=0.035,
        capital_duration=4.0,
        holding_horizon=10.0,
        cash_yield=0.00,
        expected_tvpi=2.20,
        expected_dpi=1.40,
        min_weight=0.00,
        max_weight=0.60,
        group="Private Equity",
    ),
    AssetClassAssumption(
        name="Infrastructure Equity",
        expected_return=0.10,
        volatility=0.15,
        illiquidity_premium=0.02,
        capital_duration=3.0,
        holding_horizon=12.0,
        cash_yield=0.05,
        expected_tvpi=1.55,
        expected_dpi=1.45,
        min_weight=0.10,
        max_weight=0.50,
        group="Infrastructure",
    ),
    AssetClassAssumption(
        name="Renewable Energy Infrastructure",
        expected_return=0.09,
        volatility=0.14,
        illiquidity_premium=0.02,
        capital_duration=3.0,
        holding_horizon=15.0,
        cash_yield=0.06,
        expected_tvpi=1.50,
        expected_dpi=1.50,
        min_weight=0.00,
        max_weight=0.30,
        group="Renewables",
    ),
    AssetClassAssumption(
        name="Natural Capital",
        expected_return=0.08,
        volatility=0.16,
        illiquidity_premium=0.025,
        capital_duration=5.0,
        holding_horizon=15.0,
        cash_yield=0.02,
        expected_tvpi=1.60,
        expected_dpi=1.20,
        min_weight=0.00,
        max_weight=0.20,
        group="Natural Capital",
    ),
    AssetClassAssumption(
        name="Private Credit",
        expected_return=0.09,
        volatility=0.10,
        illiquidity_premium=0.015,
        capital_duration=1.5,
        holding_horizon=6.0,
        cash_yield=0.08,
        expected_tvpi=1.35,
        expected_dpi=1.30,
        min_weight=0.00,
        max_weight=0.40,
        group="Private Credit",
    ),
]


# ---------------------------------------------------------------------------
# Matrice de corrélation par défaut (post-désmoothing).
# Les actifs privés partagent une exposition commune au cycle
# macro-économique et aux taux d'actualisation ; Natural Capital est plus
# idiosyncratique (rendement lié aux matières premières / carbone / biologie) ;
# Private Credit est moins corrélé aux actifs "growth" (VC) et davantage
# corrélé à l'Infrastructure (sensibilité taux/crédit).
# ---------------------------------------------------------------------------
DEFAULT_ASSET_NAMES = [a.name for a in DEFAULT_ASSET_CLASSES]

_DEFAULT_CORR_VALUES = [
    # PE Buyout, Growth, VC,  Infra, Renew, NatCap, PrivCredit
    [1.00, 0.75, 0.60, 0.45, 0.40, 0.30, 0.45],
    [0.75, 1.00, 0.65, 0.40, 0.35, 0.30, 0.40],
    [0.60, 0.65, 1.00, 0.30, 0.25, 0.25, 0.30],
    [0.45, 0.40, 0.30, 1.00, 0.65, 0.35, 0.55],
    [0.40, 0.35, 0.25, 0.65, 1.00, 0.40, 0.50],
    [0.30, 0.30, 0.25, 0.35, 0.40, 1.00, 0.30],
    [0.45, 0.40, 0.30, 0.55, 0.50, 0.30, 1.00],
]

DEFAULT_CORRELATION = pd.DataFrame(
    _DEFAULT_CORR_VALUES, index=DEFAULT_ASSET_NAMES, columns=DEFAULT_ASSET_NAMES
)


def assumptions_to_dataframe(
    assumptions: list[AssetClassAssumption] | None = None,
) -> pd.DataFrame:
    """Convertit une liste d'hypothèses en DataFrame indexé par nom d'actif."""
    assumptions = assumptions or DEFAULT_ASSET_CLASSES
    df = pd.DataFrame([a.to_dict() for a in assumptions])
    return df.set_index("Asset Class")


def build_covariance_matrix(
    assumptions_df: pd.DataFrame, correlation: pd.DataFrame
) -> pd.DataFrame:
    """Construit la matrice de covariance annualisée à partir des volatilités
    (colonne 'Volatility' du DataFrame d'hypothèses) et de la matrice de
    corrélation, dans le même ordre d'actifs.
    """
    names = assumptions_df.index.tolist()
    corr = correlation.loc[names, names].to_numpy()
    vols = assumptions_df.loc[names, "Volatility"].to_numpy()
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame(cov, index=names, columns=names)
