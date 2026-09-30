# Frontière efficiente — Portefeuille d'actifs privés

Application Python (Streamlit) permettant de construire des allocations
optimales pour un portefeuille composé exclusivement d'actifs non cotés :
Private Equity Buyout, Growth Equity, Venture Capital, Infrastructure Equity,
Renewable Energy Infrastructure, Natural Capital, et Private Credit
(optionnel).

## ⚠️ Prérequis d'exécution

**Ce code n'a pas pu être exécuté ni testé dans l'environnement où il a été
généré (aucun interpréteur Python réel n'y était disponible).** Il a été
écrit avec la plus grande attention à la cohérence des signatures de
fonctions et des imports, mais vous devez le faire tourner dans un
environnement Python 3.10+ standard avant toute mise en production, et
corriger toute erreur résiduelle qui n'aurait pas pu être détectée sans
exécution réelle (erreurs de type, edge cases numériques, versions de
librairies).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Structure du projet

```
efficient_frontier_private_assets/
├── data/                          # données d'exemple (séries illustratives)
│   └── example_quarterly_returns.csv
├── inputs/
│   └── inputs_template.xlsx       # template d'hypothèses (4 feuilles)
├── outputs/
│   ├── scenarios/                 # scénarios sauvegardés (.pkl)
│   └── reports/                   # rapports PDF générés
├── models/
│   ├── asset_classes.py           # hypothèses par défaut, construction Sigma
│   ├── desmoothing.py             # désmoothing Geltner
│   └── risk_metrics.py            # VaR, CVaR, diversification, contributions
├── optimization/
│   ├── mean_variance.py           # MVO scipy + contraintes institutionnelles
│   ├── black_litterman.py         # vues stratégiques
│   └── robust.py                  # optimisation robuste, stress corrélation
├── simulations/
│   └── monte_carlo.py             # simulation multivariée, TVPI, cash-flows
├── visualization/
│   └── plots.py                   # graphiques Plotly
├── reporting/
│   └── pdf_report.py              # rapport PDF (ReportLab + Kaleido)
├── utils/
│   └── excel_io.py                # lecture/écriture Excel
├── app.py                         # application Streamlit
├── requirements.txt
└── README.md
```

## Fichier d'inputs Excel (`inputs/inputs_template.xlsx`)

Quatre feuilles :

1. **Assumptions** — une ligne par classe d'actifs : rendement attendu,
   volatilité (déjà désmoothée — voir section méthodologie), prime
   d'illiquidité, durée de capital, horizon de détention, cash yield, TVPI
   et DPI attendus, bornes min/max d'allocation, et regroupement (`Group`)
   utilisé pour les contraintes institutionnelles.
2. **Correlation** — matrice de corrélation carrée (mêmes noms d'actifs que
   la feuille Assumptions), entièrement modifiable.
3. **Constraints** — bornes min/max par regroupement (`Group`), ex. :
   Private Equity 20%-60%, Infrastructure 10%-50%, Renewables 0%-30%,
   Natural Capital 0%-20%, Private Credit 0%-40%.
4. **RiskBudget** — paramètres globaux : durée de capital moyenne pondérée
   maximale (budget d'illiquidité), tracking error maximum vs allocation
   stratégique, taux sans risque par défaut.

Les valeurs par défaut de ce template correspondent exactement aux
hypothèses codées en dur dans `models/asset_classes.py` (utilisées si aucun
fichier n'est importé dans l'application).

## Méthodologie

### 1. Désmoothing des rendements (`models/desmoothing.py`)

Les valorisations d'actifs non cotés sont mises à jour de façon partielle et
décalée (appraisals, DCF, comparables), ce qui lisse artificiellement la
série de rendements observés et **sous-estime fortement la volatilité et les
corrélations réelles**. Le modèle de Geltner (1991) est utilisé pour
inverser ce lissage :

- à partir d'une série historique de rendements (ex. NAV trimestrielles) :
  `geltner_desmooth_series()` reconstitue les rendements "vrais" par
  inversion d'un processus autorégressif d'ordre 1 ;
- à partir d'une simple hypothèse de volatilité observée + paramètre de
  lissage λ (cas le plus fréquent en pratique, faute d'historique
  granulaire) : `desmooth_volatility_only()` applique l'inflation de
  variance correspondante.

Des valeurs de λ par défaut, indicatives, sont fournies par classe d'actifs
(`DEFAULT_LAMBDA_BY_ASSET_CLASS`) — **à recalibrer** si des séries réelles
sont disponibles.

### 2. Optimisation Mean-Variance (`optimization/mean_variance.py`)

Moteur `scipy.optimize` (SLSQP), choisi plutôt que PyPortfolioOpt seul afin
de pouvoir exprimer nativement :
- les bornes par actif ;
- les contraintes de groupe (ex. Private Equity agrégé 20%-60%) ;
- le budget d'illiquidité (durée de capital pondérée maximale) ;
- la tracking error vs une allocation stratégique cible.

Un wrapper PyPortfolioOpt (`max_sharpe_pypfopt`) est fourni pour les cas
simples (bornes par actif uniquement), à titre de validation croisée.

### 3. Black-Litterman (`optimization/black_litterman.py`)

**Adaptation nécessaire pour les actifs privés** : en l'absence de
portefeuille de marché observable (pas de capitalisation de marché pour du
Private Equity ou de l'Infrastructure), les rendements d'équilibre
implicites (`pi`) sont calculés à partir d'une allocation stratégique de
référence choisie par l'utilisateur (équipondérée par défaut dans
l'application). **Cette hypothèse doit être explicitée à tout comité
d'investissement** utilisant ces résultats.

### 4. Optimisation robuste (`optimization/robust.py`)

Trois mécanismes de robustesse, combinables :
- **Uncertainty sets sur les rendements** (approche worst-case de type
  Ben-Tal/Nemirovski) : le rendement de chaque actif est supposé incertain
  dans un intervalle ± delta_i, pénalisant l'objectif par un terme
  proportionnel à cette incertitude pondérée.
- **Stress de corrélation** (`stress_correlation`) : mélange paramétrique
  de la matrice de corrélation vers un scénario de crise (toutes les
  corrélations convergent vers une valeur élevée commune).
- **Pénalisation de la concentration** : terme de régularisation L2 sur les
  poids, qui décourage les solutions extrêmes typiques du MVO classique
  lorsque la matrice de covariance est bruitée (peu d'historique, cas
  fréquent pour les actifs privés).

### 5. Simulation Monte Carlo (`simulations/monte_carlo.py`)

- Simulation multivariée corrélée (décomposition de Cholesky), avec choix
  entre distribution normale et Student-t (queues épaisses).
- Distribution du TVPI final et trajectoires de NAV (fan chart de
  percentiles).
- **Modèle de cash-flows simplifié** (courbe en J paramétrique, calibrée sur
  `capital_duration` et `holding_horizon`) : approximation stylisée à
  vocation stratégique, **PAS un moteur contractuel** fondé sur les données
  réelles des fonds sous-jacents. Ne pas utiliser pour du budgeting de
  trésorerie opérationnel fin.

### 6. Mesures de risque (`models/risk_metrics.py`)

Volatilité, downside volatility, VaR 95%, CVaR 95%, maximum drawdown simulé,
probabilité de perte de capital (TVPI < 1x), ratio de diversification,
contribution au risque par actif (marginale et en %).

## Hypothèses de marché par défaut (exemples, non contractuelles)

| Classe d'actifs | Rendement attendu | Volatilité (désmoothée) | Cash yield |
|---|---|---|---|
| Private Equity Buyout | 15,0% | 22% | 0% |
| Growth Equity | 15,5% | 26% | 0% |
| Venture Capital | 18,0% | 35% | 0% |
| Infrastructure Equity | 10,0% | 15% | 5% |
| Renewable Energy Infrastructure | 9,0% | 14% | 6% |
| Natural Capital | 8,0% | 16% | 2% |
| Private Credit | 9,0% | 10% | 8% |

Ces valeurs sont des **hypothèses indicatives à titre d'exemple**, cohérentes
avec les fourchettes demandées (PE 14-16%, Infrastructure 9-11%, Renewables
8-10%, Natural Capital 7-9%), et doivent être remplacées par les hypothèses
propres de l'utilisateur via `inputs/inputs_template.xlsx`.

## Limites du modèle

- **Désmoothing** : le modèle de Geltner est une approximation AR(1). Il ne
  capture pas les changements de méthode de valorisation dans le temps ni
  les structures de lissage plus complexes. Les paramètres λ par défaut
  sont indicatifs et doivent être recalibrés avec des données réelles.
- **Corrélations** : les corrélations entre actifs privés sont
  structurellement difficiles à estimer (peu d'historique désmoothé
  disponible). La matrice par défaut est une hypothèse de travail
  raisonnable, pas une estimation empirique rigoureuse.
- **Black-Litterman** : utilise une allocation stratégique comme proxy du
  portefeuille de marché, faute d'alternative pour les actifs privés.
- **Monte Carlo / cash-flows** : le modèle de J-curve est une approximation
  stylisée à but stratégique, pas un moteur de cash-flows contractuel.
- **Rebalancement** : le moteur Monte Carlo suppose un rebalancement
  implicite annuel aux poids cibles, ce qui n'est pas réaliste pour des
  engagements en fonds fermés (l'allocation réelle dérive entre les
  vintages) — à interpréter comme une approximation stratégique de long
  terme, pas un modèle de portefeuille au jour le jour.
- **Absence de test d'exécution réel** : voir avertissement en tête de ce
  document.

## Licence / usage

Ce projet est un outil d'aide à la décision interne. Il ne constitue pas un
conseil en investissement et les résultats produits ne préjugent pas des
performances futures.
