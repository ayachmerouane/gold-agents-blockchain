# Gold Agents Blockchain

Investissement **simulé** sur l'or (XAU/USD) à l'aide de **Deep Learning** (LSTM/GRU), d'un **système multi-agents** (prévision, stratégie, risque, exécution) et d'un **registre de type blockchain** (chaîne de hachage, arbre de Merkle, signatures Ed25519) qui rend chaque décision vérifiable.

> Projet pédagogique : aucun argent réel n'est engagé.

- Le sujet expliqué : [docs/explication_projet.md](docs/explication_projet.md)
- L'architecture technique (outils, modèles, tests, planning) : [docs/architecture_technique.md](docs/architecture_technique.md)

## État d'avancement

Phase 0 (cadrage et socle) : structure du dépôt, configuration validée, CI, téléchargement des données.
Phase 1 (données et EDA) : `gold-agents eda` produit le rapport d'anomalies et les graphiques.
Phase 2 (features et découpage) : `gold-agents prepare` calcule les indicateurs, la cible et le découpage train/val/test, avec des tests anti-fuite du futur. 
Phase 3 (moteur, Buy & Hold, blockchain v1) : `gold-agents backtest` fait passer Buy & Hold et le croisement SMA 20/50 par le moteur jour par jour (exécution à l'ouverture du lendemain, frais et slippage) sur la période de test, enregistre chaque jour dans un registre de blocs chaînés par SHA-256 (`outputs/ledger/*.jsonl`), montre qu'une falsification est détectée et dessine le schéma `outputs/results/backtest/backtest_test.png`. `gold-agents verify` revérifie les registres. Les modules des phases suivantes sont encore des squelettes.

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
uv run gold-agents fill-gaps     # = make fill-gaps
uv run gold-agents clean         # = make clean
uv run gold-agents prepare       # = make prepare
uv run pytest -m "not slow"      # = make test
```

En PowerShell, la clé se définit avec `$env:TWELVEDATA_API_KEY = "..."` au lieu de `export`.

`make download` refuse d'écraser un CSV existant (option `--force`) et signale si son empreinte ne correspond plus à `CHECKSUMS.txt`. Les chemins se règlent dans `config/config.yaml` (`data.raw_path`, `data.update_path`). Comparaison des sources : voir `docs/sources_donnees_or.md`.

## Analyse exploratoire (phase 1)

`make eda` (ou `uv run gold-agents eda`) lit le CSV Kaggle et, s'il existe, le fichier de mise à jour, puis :

- affiche un résumé d'une quarantaine de lignes dans le terminal ;
- écrit `outputs/eda/rapport_eda.md`, `outputs/eda/summary.json` et 5 figures dans `outputs/eda/figures/`.

Contrôles : trous de dates et barres de week-end, cohérence OHLC, distribution des rendements (queues épaisses, Jarque-Bera), valeurs aberrantes (z-score robuste, allers-retours suspects), régimes de volatilité, autocorrélation (Ljung-Box) et stationnarité (ADF, KPSS), aperçu du découpage train/val/test et raccord avec la mise à jour API. Tous les seuils et tests sont calculés **sur le train seulement**. **Notebook** : `notebooks/01_eda_nettoyage.ipynb` parcourt les données pas à pas (graphes, relations entre variables, nettoyage) et écrit `data/processed/xau_1d_clean.parquet`. Pour l'ouvrir : `uv run --with jupyterlab jupyter lab notebooks/01_eda_nettoyage.ipynb`.

## Nettoyage

Le CSV Kaggle a deux trous anormaux, tous les deux dans le test : 22 jours ouvrés entre le 2025-09-12 et le 2025-10-15, et 6 jours entre le 2026-01-13 et le 2026-01-22. Ils sont **comblés avec Twelve Data**, le même instrument XAU/USD :

```bash
make fill-gaps   # télécharge les jours manquants, contrôle leur cohérence avec Kaggle, fige data/raw/XAU_1d_gapfill.csv
make clean       # écrit data/processed/xau_1d_clean.parquet (+ xau_1d_clean.json)
```

Les lignes comblées gardent `source = twelvedata` et `filled = True`. Le dataset ajoute aussi les colonnes `ohlc_fixed`, `missing_bdays_before`, `valid_return`, `log_return` et `split`. Les rendements extrêmes, qui sont de vrais événements, sont conservés. Le détail des choix est dans `docs/constats_eda.md`.

## Features, cible et découpage

```bash
make prepare     # écrit data/processed/xau_1d_features.parquet (+ scaler.json et xau_1d_features.json)
```

- **Features** (toutes calculées avec le passé seulement) : rendements des 1, 5, 10 et 20 derniers jours ; écart du prix à ses moyennes mobiles SMA/EMA 10, 20, 50 ; RSI 14 ; MACD (rapporté au prix) ; ATR 14 en % du prix ; amplitude du jour ; largeur de Bollinger ; volatilité 20 j ; jour de la semaine (sin, cos). Pas de volume (volume de ticks MT4).
- **Cible** : `target_return`, le rendement log du lendemain, et `target_up` (1 si hausse).
- **Découpage** : dates de `config.yaml`. Un jour est un exemple (`sample = True`) si sa fenêtre de W jours et sa cible restent dans son ensemble ; les W premiers jours de validation et de test servent seulement d'historique.
- **Scaler** : médiane et écart interquartile ajustés sur le train seulement (`scaler.json`).
- **Tests anti-fuite** (`tests/leakage/`) : features identiques si l'on coupe la série à t ou si l'on remplace le futur par du bruit ; scaler inchangé si val/test changent ; ensembles ordonnés et séparés.

## Backtest de référence et registre (phase 3)

```bash
make backtest    # Buy & Hold + SMA 20/50 sur le test, registres et schéma
make verify      # revérifie tous les registres de outputs/ledger/
```

- **Moteur** (`simulation/`) : chaque jour, l'ordre décidé la veille est exécuté à l'**ouverture** (frais 0,1 % + slippage 0,02 % de `config.yaml`), le portefeuille est valorisé à la clôture, puis la stratégie décide en ne voyant que l'historique jusqu'à ce jour. Pas de levier ni de vente à découvert.
- **Période** : les jours d'exemple du test (les mêmes que pour les futurs modèles), données Kaggle seulement.
- **Stratégies** : Buy & Hold (achat au premier jour, référence exigée) et croisement SMA 20/50 (règle technique exigée, montre des achats et des ventes).
- **Registre v1** (`blockchain/`) : un bloc genèse (config complète + empreinte SHA-256 du fichier de features) puis un bloc par jour (trade, décision, portefeuille). Chaque bloc porte le hash du précédent ; la vérification recalcule tout et donne le premier bloc fautif. La commande falsifie aussi des copies du registre pour montrer que c'est détecté. Merkle, signature Ed25519 et `verify_ledger.py` autonome : phase 7.
- **Sorties** : `outputs/results/backtest/` (`metrics.json`, courbes et trades en CSV, schéma `backtest_test.png`) et `outputs/ledger/<stratégie>.jsonl`. `--costs 2` double frais et slippage (sensibilité).

## Commandes

| Commande | Rôle |
|---|---|
| `make check` | Lint + typage + tests (ce que fait la CI) |
| `make test` | Tests rapides (`pytest -m "not slow"`) |
| `make cov` | Tests avec couverture |
| `make format` | Formatage automatique (ruff) |
| `make download` | Téléchargement du CSV Kaggle + empreinte |
| `make update-data` | Ajout des jours récents (Twelve Data) |
| `make fill-gaps` | Comble les trous du CSV Kaggle avec Twelve Data (fichier figé `data/raw/XAU_1d_gapfill.csv`) |
| `make clean` | Nettoyage : `data/processed/xau_1d_clean.parquet` et sa fiche JSON |
| `make eda` | Analyse exploratoire : résumé dans le terminal, rapport et figures dans `outputs/eda/` |
| `make prepare` | Features, cible et découpage : `data/processed/xau_1d_features.parquet` |
| `make data` | Nettoyage puis préparation (`clean` + `prepare`) |
| `make train` | Entraînement des modèles |
| `make backtest` | Buy & Hold et SMA 20/50 sur le test : métriques, registres, schéma |
| `make evaluate` | Métriques et figures |
| `make verify` | Vérification des registres de `outputs/ledger/` |
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
