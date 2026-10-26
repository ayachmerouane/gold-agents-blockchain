# Gold Agents Blockchain

Investissement **simulé** sur l'or (XAU/USD) à l'aide de **Deep Learning** (LSTM/GRU), d'un **système multi-agents** (prévision, stratégie, risque, exécution) et d'un **registre de type blockchain** (chaîne de hachage, arbre de Merkle, signatures Ed25519) qui rend chaque décision vérifiable.

> Projet pédagogique : aucun argent réel n'est engagé.

- Le sujet expliqué : [docs/explication_projet.md](docs/explication_projet.md)
- L'architecture technique (outils, modèles, tests, planning) : [docs/architecture_technique.md](docs/architecture_technique.md)

## État d'avancement

Phase 0 (cadrage et socle) : structure du dépôt, configuration validée, CI, téléchargement des données.
Phase 1 (données et EDA) : `gold-agents eda` produit le rapport d'anomalies et les graphiques. Les autres modules de `src/gold_agents/` sont encore des squelettes.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) (il installe lui-même Python 3.11).

```bash
make install        # dépendances de base + outils de dev (pytest, ruff, mypy) + EDA (matplotlib, statsmodels)
make install-all    # tout, y compris PyTorch, scikit-learn, MLflow, Streamlit
uv run pre-commit install   # optionnel : lint automatique avant chaque commit
```

## Données

**Historique (figé)** : Kaggle « [XAUUSD Gold Price Tracker: 2004–Present](https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004present) », fichier quotidien `XAU_1d_data.csv` (export MetaTrader 4). Si ce dataset ne contient pas le fichier quotidien, `download` le prend dans la version multi-unités du même auteur ([2004-2026](https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024)) ; seul ce fichier est téléchargé, pas l'archive complète. Train, validation et test sont construits **uniquement** sur ce fichier.

**Mise à jour** : [Twelve Data](https://twelvedata.com) `XAU/USD` (même instrument, au comptant). Les jours postérieurs au CSV Kaggle sont écrits dans un fichier séparé, `data/raw/XAU_1d_update.csv`, après un contrôle de cohérence sur ~60 jours communs (écart des clôtures et corrélation des rendements). Secours sans clé : yfinance `GC=F` (contrat à terme, donc prix légèrement différents).

Les données ne sont pas versionnées et **aucune clé n'est stockée dans le dépôt** :

| Besoin | Où le mettre |
|---|---|
| Jeton Kaggle (`KGAT_…`) | Kaggle > Settings > API > *Generate New Token*, puis soit la variable `KAGGLE_API_TOKEN`, soit le fichier `~/.kaggle/access_token` (`chmod 600`). L'ancien `~/.kaggle/kaggle.json` (ou `KAGGLE_USERNAME` + `KAGGLE_KEY`) reste accepté. |
| Clé Twelve Data (gratuite) | variable d'environnement `TWELVEDATA_API_KEY` (par exemple dans un `.env` local, ignoré par Git) |

```bash
make download                    # télécharge XAU_1d_data.csv et note son SHA-256 dans CHECKSUMS.txt
export TWELVEDATA_API_KEY=...    # clé personnelle, jamais commitée
make update-data                 # ajoute les jours manquants (Twelve Data)
uv sync --extra dev --extra fetch && uv run gold-agents update-data --source yfinance   # secours
```

**Sans `make` (Windows, Git Bash ou PowerShell)** : chaque cible appelle la CLI, on peut la lancer directement :

```bash
uv sync --extra dev --extra eda  # une fois (équivalent de make install)
uv run gold-agents download      # = make download
uv run gold-agents update-data   # = make update-data
uv run gold-agents eda           # = make eda
uv run pytest -m "not slow"      # = make test
```

En PowerShell, la clé se définit avec `$env:TWELVEDATA_API_KEY = "..."` au lieu de `export`.

`make download` refuse d'écraser un CSV existant (option `--force`) et signale si son empreinte ne correspond plus à `CHECKSUMS.txt`. Les chemins se règlent dans `config/config.yaml` (`data.raw_path`, `data.update_path`). Comparaison des sources : voir `docs/sources_donnees_or.md`.

## Analyse exploratoire (phase 1)

`make eda` (ou `uv run gold-agents eda`) lit le CSV Kaggle et, s'il existe, le fichier de mise à jour, puis :

- affiche un résumé d'une quarantaine de lignes dans le terminal ;
- écrit `outputs/eda/rapport_eda.md`, `outputs/eda/summary.json` et 5 figures dans `outputs/eda/figures/`.

Contrôles : trous de dates et barres de week-end, cohérence OHLC, distribution des rendements (queues épaisses, Jarque-Bera), valeurs aberrantes (z-score robuste, allers-retours suspects), régimes de volatilité, autocorrélation (Ljung-Box) et stationnarité (ADF, KPSS), aperçu du découpage train/val/test et raccord avec la mise à jour API. Tous les seuils et tests sont calculés **sur le train seulement**. Le notebook `notebooks/01_eda.ipynb` appelle les mêmes fonctions.

## Commandes

| Commande | Rôle |
|---|---|
| `make check` | Lint + typage + tests (ce que fait la CI) |
| `make test` | Tests rapides (`pytest -m "not slow"`) |
| `make cov` | Tests avec couverture |
| `make format` | Formatage automatique (ruff) |
| `make download` | Téléchargement du CSV Kaggle + empreinte |
| `make update-data` | Ajout des jours récents (Twelve Data) |
| `make eda` | Analyse exploratoire : résumé dans le terminal, rapport et figures dans `outputs/eda/` |
| `make data` | Préparation des données |
| `make train` | Entraînement des modèles |
| `make backtest` | Simulation multi-agents + écriture du registre |
| `make evaluate` | Métriques et figures |
| `make verify` | Vérification du registre |
| `make dashboard` | Tableau de bord Streamlit |
| `make all` | Chaîne complète |

La CLI est aussi accessible directement : `uv run gold-agents --help`.

## Structure

```
config/              config.yaml (tous les paramètres)
data/                raw/ (CSV + CHECKSUMS.txt) et processed/ (hors Git)
notebooks/           EDA, baselines, analyse des modèles
src/gold_agents/
  config.py          validation pydantic de la configuration
  cli.py             commandes Typer
  data/              chargement, nettoyage, features, splits, fenêtres
  models/            baselines, LSTM, GRU, entraînement, incertitude
  agents/            prévision, stratégie, risque, exécution, orchestrateur
  simulation/        portefeuille, coûts, moteur de backtest
  blockchain/        blocs, Merkle, chaîne, signatures, stockage
  evaluation/        métriques de prévision et financières, rapport
dashboard/           application Streamlit
scripts/             verify_ledger.py (script autonome pour le professeur)
tests/               unit, leakage, ledger, property, integration, golden
outputs/             modèles, résultats, registre
docs/                documentation et rapport
```

## Conventions

- Code et noms de variables en anglais, commentaires et rapport en français.
- Une branche par fonctionnalité, *pull request* et CI verte avant fusion.
- Messages de commit : `feat:`, `fix:`, `test:`, `docs:`, `chore:`…
