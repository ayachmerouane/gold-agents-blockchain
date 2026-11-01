# Gold Agents Blockchain

Investissement **simulé** sur l'or (XAU/USD) à l'aide de **Deep Learning** (LSTM/GRU), d'un **système multi-agents** (prévision, stratégie, risque, exécution) et d'un **registre de type blockchain** (chaîne de hachage, arbre de Merkle, signatures Ed25519) qui rend chaque décision vérifiable.

> Projet pédagogique : aucun argent réel n'est engagé.

- Le sujet expliqué : [docs/explication_projet.md](docs/explication_projet.md)
- L'architecture technique (outils, modèles, tests, planning) : [docs/architecture_technique.md](docs/architecture_technique.md)

## État d'avancement

Phase 0 (cadrage et socle) : structure du dépôt, configuration validée, CI, téléchargement des données.
Phase 1 (données et EDA) : `gold-agents eda` produit le rapport d'anomalies et les graphiques.
Phase 2 (features et découpage) : `gold-agents prepare` calcule les indicateurs, la cible et le découpage train/val/test, avec des tests anti-fuite du futur. 
Phase 3 (moteur, Buy & Hold, blockchain v1) : `gold-agents backtest` fait passer Buy & Hold et le croisement SMA 20/50 par le moteur jour par jour (exécution à l'ouverture du lendemain, frais et slippage) sur la période de test, enregistre chaque jour dans un registre de blocs chaînés par SHA-256 (`outputs/ledger/*.jsonl`), montre qu'une falsification est détectée et dessine le schéma `outputs/results/backtest/backtest_test.png`. `gold-agents verify` revérifie les registres.
Phase 4 (baselines puis LSTM/GRU) : `gold-agents train` compare sur la validation les baselines B0 à B4 aux réseaux LSTM et GRU (Optuna, MLflow, early stopping, 5 graines) et dessine `outputs/results/forecast/forecast_val.png`.
Phase 5 (agents) : `gold-agents simulate` fait trader les quatre agents (prévision, stratégie, risque, exécution) dans le moteur de la phase 3 avec les modèles figés de la phase 4, sur la validation, contre Buy & Hold et SMA 20/50, avec un registre par stratégie et le schéma `outputs/results/agents_val/agents_val.png`.

Phase 6 (réglage) : `gold-agents tune` règle les seuils et le risque des agents sur une grille grossière, en walk-forward sur 2011-2021 (modèles réentraînés sur le passé de chaque période), avec la sensibilité aux frais et les références hasard et oracle ; `gold-agents freeze` scelle le réglage retenu dans un bloc genèse.

Phase 7 (blockchain v2) : chaque bloc porte la racine de Merkle de ses événements, une petite preuve de travail et une signature Ed25519 (`gold-agents keygen` crée la clé ; la clé privée reste dans `keys/`, jamais versionnée). `scripts/verify_ledger.py` vérifie les registres sans le projet (Python + `cryptography`).

Phase 8 (évaluation finale) : `gold-agents evaluate` fait **un seul run** sur le test (2022 à la fin des données) avec les réglages figés : toutes les stratégies, métriques financières et de prévision, sensibilité aux frais, alerte « trop beau », registres signés et ancrés sur Sepolia avant et après le run.

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
- **Registre v1** (`blockchain/`) : un bloc genèse (config complète + empreinte SHA-256 du fichier de features) puis un bloc par jour (trade, décision, portefeuille). Chaque bloc porte le hash du précédent ; la vérification recalcule tout et donne le premier bloc fautif. La commande falsifie aussi des copies du registre pour montrer que c'est détecté. Depuis la phase 7, les registres sont en v2 (voir plus bas).
- **Sorties** : `outputs/results/backtest/` (`metrics.json`, courbes et trades en CSV, schéma `backtest_test.png`) et `outputs/ledger/<stratégie>.jsonl`. `--costs 2` double frais et slippage (sensibilité).

## Modèles de prévision (phase 4)

```bash
uv sync --extra dev --extra eda --extra ml   # PyTorch, scikit-learn, LightGBM, Optuna, MLflow
uv run gold-agents train --quick             # essai de la chaîne en 1 à 2 minutes
uv run gold-agents train                     # run complet (environ 30 à 60 min sur CPU)
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # essais Optuna dans le navigateur
```

- **Tâche** : à partir des 30 derniers jours (20 features), prévoir le rendement log du lendemain (régression).
- **Baselines** : B0 rendement nul, B1 rendement d'hier (et moyenne des 20 derniers jours), B2 ARIMA (ordre au meilleur AIC sur le train, paramètres figés ensuite), B3 Ridge et B4 LightGBM sur les features du jour.
- **Réseaux** : M1 LSTM et M2 GRU petits (1-2 couches, 32-64 unités, dropout), AdamW, gradient clipping, early stopping (patience 15) ; E = moyenne LSTM + GRU.
- **Réglage** : Optuna (20 essais par réseau, 30 pour LightGBM, abandon tôt des essais sans avenir) sur la validation ; chaque essai est suivi dans MLflow (`mlruns/`, hors Git). Puis 5 graines : moyenne ± écart-type.
- **Sur- et sous-apprentissage** : early stopping, dropout et weight decay ; courbes train/validation des réseaux dans le schéma ; colonnes « vs B0 train » et « vs B0 » (validation) : un gain en train qui disparaît en validation signale un surapprentissage, aucun gain même en train signale un modèle qui n'a rien appris.
- **Mesures** : MAE, RMSE, précision directionnelle, gain sur B0, test de Diebold-Mariano contre B0 ; incertitude par MC Dropout (50 passages). **Alerte** au-delà de 60 % de bonnes directions (fuite probable).
- **Le test n'est pas utilisé** : il est réservé à l'évaluation finale (phase 8).
- **Sorties** : `outputs/results/forecast/` (`summary_val.csv`, métriques par graine, prévisions de validation, schéma) et `outputs/models/<lstm|gru>/` (poids de chaque graine + `model.json`). Options : `--models lstm,gru`, `--trials 10`, `--no-mlflow`.

## Agents de trading (phase 5)

```bash
uv run gold-agents train      # une fois : poids des réseaux dans outputs/models/
uv run gold-agents simulate   # agents contre Buy & Hold sur la validation (quelques minutes)
uv run gold-agents simulate --costs 2   # frais et slippage doublés (sensibilité)
```

Chaque jour t, après la clôture, quatre agents se passent des messages typés (`agents/messages.py`) :

1. **Prévision** : `r̂(t+1)` du modèle figé, lu sur la fenêtre des 30 jours qui finissent à t (jamais au-delà) ; pour l'ensemble LSTM + GRU, `σ̂` par MC Dropout (50 passages par graine).
2. **Stratégie** : BUY si `r̂ > θ_buy`, SELL si `r̂ < -θ_sell`, sinon HOLD. Variante « confiance » : n'acheter que si `r̂ / σ̂ > k` (`strategy.min_confidence`, 0,5 = probabilité de hausse d'au moins 69 %) ; la vente n'exige aucune confiance.
3. **Risque** : coupe-circuit au drawdown max (-20 % : tout vendre et rester en cash), stop-loss (-5 % sous le prix de revient), refus des ordres impossibles (pas de cash, rien à vendre, pas de vente à découvert).
4. **Exécution** : l'ordre passe à l'ouverture de t+1 par le moteur de la phase 3, avec les mêmes frais et le même slippage que Buy & Hold.

- **Systèmes comparés** : Agents + LSTM, + GRU, + ensemble LSTM/GRU, + ensemble avec le filtre de confiance, + LightGBM (le concurrent sans réseau, réentraîné avec ses réglages Optuna de la phase 4), contre Buy & Hold et SMA 20/50, sur la même période.
- **Traçabilité** : chaque jour, un bloc contient la prévision, le signal, l'ordre (avec la règle qui a décidé) et le trade ; les paramètres des agents sont dans le bloc genèse.
- **Sorties** : `outputs/results/agents_val/` (`metrics.json`, courbes, trades, décisions jour par jour et prévisions en CSV, schéma `agents_val.png`) et `outputs/ledger/agents_val/*.jsonl`. `--period test` existe mais le test reste réservé à la phase 8.
- **Limite** : les seuils ne sont pas encore réglés (phase 6) et la validation a déjà servi à régler les modèles ; ces résultats sont donc optimistes pour les modèles.

## Réglage des agents (phase 6)

```bash
uv run gold-agents tune --quick   # essai rapide de la chaîne (1 graine, 5 époques)
uv run gold-agents tune           # le vrai réglage (réentraîne LSTM, GRU, LightGBM 5 fois)
# recopier le bloc affiché dans config/config.yaml, puis :
uv run gold-agents freeze         # scelle les réglages dans outputs/ledger/reglages_figes.jsonl
```

- **Prévisions walk-forward** (`models/walk_forward.py`) : 5 périodes (2011-12, 2013-14, 2015-16, 2017-18, 2019-21 = la validation). Chacune est prédite par LSTM, GRU, l'ensemble et LightGBM **réentraînés sur tout ce qui la précède** (hyperparamètres de la phase 4, pas de nouvel Optuna ; la dernière année du passé sert à l'arrêt anticipé ; scaler réajusté). Les lignes du test sont retirées avant tout calcul. Les prévisions sont gardées en cache (`--retrain` pour refaire).
- **Grille grossière** (R4, `tuning.grid` dans la config) : modèle (LSTM, ensemble, LightGBM) × θ_buy × θ_sell × k × stop-loss × coupe-circuit, soit 1 296 réglages par période. La reprise après le coupe-circuit (`risk.cooldown_days`, 20 jours en cash) est fixe.
- **Règle de choix** : perte max pas plus profonde que Buy & Hold dans le plus de périodes possible, puis meilleur score = le plus petit entre le Sharpe moyen du réglage et celui de ses voisins dans la grille (un plateau plutôt qu'un pic).
- **Walk-forward** : la période k est jugée avec le réglage choisi sur les périodes 1 à k-1. Le réglage figé est celui choisi sur les 5 périodes.
- **Robustesse** : frais × 0, 1, 2, 5 ; stratégie au hasard avec le même nombre de trades (200 tirages par période) ; oracle qui connaît le lendemain (plafond, pour illustration).
- **Contrôle** : un simulateur rapide (`simulation/fast.py`) fait tourner la grille ; il est vérifié contre le vrai système d'agents (tests, et à chaque `tune` sur la dernière période).
- **Sorties** : `outputs/results/tuning/` (`grid.csv`, `grid_summary.csv`, `walk_forward.csv`, `costs.csv`, `references.csv`, `chosen.json`, schéma `tuning.png`). `freeze` refuse si la config ne correspond pas à `chosen.json`.

## Registre signé, blockchain v2 (phase 7)

```bash
uv run gold-agents keygen        # une seule fois : keys/ledger_ed25519.key (privée) + ledger/public_key.pem (publique)
uv run gold-agents simulate      # les registres écrits ensuite sont signés
uv run gold-agents verify        # liens, hash, Merkle, preuve de travail, signatures
uv run scripts/verify_ledger.py outputs/ledger   # script autonome du professeur
```

- **Bloc v2** (`blockchain/block.py`) : le hash est le SHA-256 de l'en-tête `version, index, sim_date, prev_hash, merkle_root, difficulty, nonce` (JSON canonique). Les événements y sont couverts par la **racine de Merkle** (`merkle.py`, préfixes 0x00/0x01, preuves d'inclusion). Le hash doit commencer par `difficulty` = 3 zéros hexadécimaux (**preuve de travail**, environ 4 000 essais, quelques millisecondes par bloc). La **signature Ed25519** (`signing.py`) porte sur le hash.
- **Pourquoi la signature** : en v1, un fraudeur pouvait modifier un bloc puis recalculer tous les hash suivants. En v2, il lui faudrait aussi resigner chaque bloc, impossible sans la clé privée. La preuve de travail est pédagogique (elle ralentit, n'empêche pas).
- **Pas d'heure réelle dans les blocs** : Ed25519 est déterministe, donc rejouer un backtest redonne un registre identique octet pour octet. L'antériorité vient de Git et, en bonus, de l'ancrage du dernier hash sur Sepolia.
- **Sans clé privée**, les commandes écrivent des registres v2 non signés et le disent. Avec la clé publique, `verify` exige une signature valide sur chaque bloc.
- **Registres v1** (phases 3 à 6, dont `reglages_figes.jsonl`, bloc genèse `ca1c3fd2…60c9`) : toujours lus et vérifiés, signalés « v1 sans signature ». `--strict` les refuse.
- **`scripts/verify_ledger.py`** : réécrit toutes les règles sans importer le projet (deux implémentations, même verdict, testé). Options `--public-key`, `--head <hash attendu>`, `--strict`. Un job de la CI le lance sur une machine vierge avec le registre de démonstration `tests/golden/ledger/demo_v2.jsonl`.
- **Tests de falsification** (`tests/ledger/`) : événement modifié (racine de Merkle), en-tête modifié (hash), bloc re-haché (signature), chaîne réécrite avec une autre clé (signature), preuve de travail absente, bloc supprimé, inséré, inversé, versions mélangées, rejeu identique.

## Ancrage public sur Sepolia (bonus)

Publie dans une transaction Sepolia (réseau de test d'Ethereum, gratuit) une empreinte des derniers hash des registres. La transaction est datée et publique : elle prouve que les registres existaient déjà, tels quels, à cette date.

1. MetaMask : un compte **de test** (jamais de vrai argent), réseau Sepolia (Paramètres > Avancé > « Afficher les réseaux de test »).
2. Un peu d'ETH de test gratuit, par exemple sur le faucet Google Cloud (Ethereum Sepolia), avec l'adresse du compte.
3. `cp .env.example .env`, puis y coller la clé privée de ce compte de test (`.env` n'est jamais versionné).

```bash
uv sync --all-extras                          # installe web3 (extra anchor)
uv run gold-agents anchor outputs/ledger/agents_val/*.jsonl
uv run scripts/verify_ledger.py outputs/ledger/agents_val --anchors ledger/anchors.json
```

- `anchor` vérifie chaque registre (signatures comprises) avant d'ancrer, refuse tout réseau autre que Sepolia, envoie 0 ETH de l'adresse vers elle-même avec le texte `gold-agents ledger sha256:<empreinte>` en données, et ajoute la transaction à `ledger/anchors.json` (à versionner). Le lien Etherscan est affiché.
- `verify_ledger.py --anchors` relit la transaction avec un nœud public (sans portefeuille ni web3) et vérifie que chaque registre ancré a le même dernier hash qu'au moment de l'ancrage.
- Pour le professeur : sur Etherscan, « View input as UTF-8 » montre l'empreinte ; la date est celle du bloc.

## Évaluation finale sur le test (phase 8)

```bash
uv sync --all-extras                                   # PyTorch, LightGBM, web3
uv run gold-agents evaluate --rehearsal --quick        # répétition rapide sur la validation (le test n'est pas lu)
uv run gold-agents evaluate                            # LE run sur le test (une seule fois)
uv run scripts/verify_ledger.py outputs/ledger/test --anchors ledger/anchors.json
```

- **Rien n'est réglé** : avant de lire le moindre prix du test, `evaluate` relit `outputs/ledger/reglages_figes.jsonl` et refuse si son bloc genèse n'est pas `ca1c3fd2…60c9`, si un réglage ou un paramètre de `config.yaml` a changé depuis le gel, si le fichier de features n'est plus le même (SHA-256), ou si les hyperparamètres des modèles diffèrent de ceux de `tune`.
- **Prévisions** : exactement la procédure du walk-forward de la phase 6, avec une période de plus. LSTM, GRU (3 graines, MC Dropout) et LightGBM sont réentraînés sur 2004-2021 (la dernière année sert à l'arrêt anticipé), puis prédisent chaque jour du test avec sa seule fenêtre de 30 jours.
- **Stratégies**, toutes dans le même moteur, du 1er jour du test au dernier : Buy & Hold, SMA 20/50, Agents + LSTM, + GRU, + LSTM/GRU, + LSTM/GRU avec confiance (le système retenu en phase 6), + LightGBM, une stratégie au hasard (même nombre de trades que le système retenu) et l'oracle (connaît le futur, plafond pour illustration).
- **Mesures** : profit net, rendement, CAGR, Sharpe, Sortino, perte max, trades, taux de réussite, frais, exposition ; MAE, RMSE, bonne direction, gain sur B0 et Diebold-Mariano des prévisions ; rendement par année ; 200 stratégies au hasard par système d'agents ; frais × 0, 1, 2, 5 pour toutes les stratégies.
- **Alerte « trop beau »** : plus de 60 % de bonnes directions, un Sharpe au-dessus de 3, ou des agents meilleurs que l'oracle. C'est presque toujours une fuite du futur.
- **Un seul run, prouvé** : `evaluate` écrit d'abord `outputs/ledger/test/plan.jsonl` (réglages figés, config, empreinte des données, commit Git, procédure) et l'**ancre sur Sepolia avant de lire le test**. Puis chaque stratégie a son registre signé, et `resultats.jsonl` scelle les tableaux, les alertes et le SHA-256 de chaque fichier de résultats. Le tout est ancré à nouveau. Un second `evaluate` est refusé ; `--again "raison"` le permet, mais garde l'ancien plan et inscrit la raison et le numéro de tentative dans le nouveau.
- **Prérequis** : la clé privée du registre (`keys/`, sinon refus), `.env` avec le portefeuille Sepolia (sinon `--no-anchor`, inscrit dans le plan).
- **Sorties** : `outputs/results/final_test/` (`final_table.md` et `.csv` pour le rapport, `metrics.json`, `costs.csv`, `forecast_metrics.csv`, `yearly_returns.csv`, `random_references.csv`, courbes, trades et décisions, figures `final_test.png` et `robustesse_test.png`) et `outputs/ledger/test/*.jsonl`. À versionner après le run : `ledger/anchors.json` et `outputs/ledger/test/`.

### Tentative 2 : seuil d'achat relatif

Résultat de la tentative 1 (30/10) : le système principal (ensemble LSTM + GRU avec confiance) n'a fait **aucun trade**. Son seuil figé, θ_buy = 0,10 %, venait du walk-forward ; mais les modèles réentraînés pour le test prévoient des rendements plus petits (au plus 0,09 %, 0,02 % au milieu). Un seuil en % dépend de l'échelle des prévisions ; il ne se transpose pas d'un réentraînement à l'autre.

Correction, décidée sans regarder le test : **θ_buy devient un quantile** (`buy_rule: quantile`). On achète si r̂ dépasse le quantile θ_buy des prévisions des 250 jours précédents du même modèle (jamais sous 0, et pas d'achat avant 60 jours d'historique). Le quantile est choisi par le même walk-forward 2011-2021, avec une contrainte de plus : **entre 4 et 60 trades par an dans chaque période** (« ni 0, ni des milliers »).

```bash
uv run gold-agents tune --apply                       # grille de quantiles (prévisions en cache), écrit le réglage dans config.yaml
uv run gold-agents freeze --again "raison"            # nouveau gel ; l'ancien devient reglages_figes_v1.jsonl
uv run gold-agents evaluate --frozen-head <hash> --again "raison"   # commande exacte affichée par freeze
```

Rien n'est effacé : la tentative 1 est archivée dans `outputs/ledger/test_tentative1/` et `outputs/results/final_test_tentative1/`, citée dans le nouveau plan avec la raison. `ledger/moves.json` note chaque déplacement : l'ancrage Sepolia de la tentative 1 reste vérifiable (tableau de bord, `verify_ledger.py --anchors`). Le tableau de bord montre les deux tentatives (sélecteur « Période »).

## Simulation en direct (paper trading)

Pour juger le système sur des prix **qui n'existaient pas encore** quand il a décidé, on le
fait tourner jour après jour sur les nouvelles cotations, avec **la même règle gelée que le
test** (celle du dernier `freeze`, par exemple le seuil relatif de la tentative 2) :
`live-init` refuse de démarrer si la config ne correspond pas à ce gel.

```bash
uv run gold-agents live-init        # 1 fois : données à jour, modèles figés, plan signé et ancré
uv run gold-agents live             # chaque jour de cotation, après minuit UTC
```

- `live-init` réentraîne LSTM et GRU (3 graines) sur tout l'historique, scelle dans le
  bloc genèse la règle gelée, le SHA-256 de chaque poids, les données et le commit, puis
  inscrit et ancre la décision du dernier jour connu. Les 250 dernières prévisions de
  l'arrêt anticipé servent d'historique au seuil relatif dès le premier jour.
- `live` met à jour les données (Twelve Data), refuse de continuer si un poids a changé,
  puis, pour chaque nouveau jour : exécute l'ordre de la veille à l'ouverture, valorise à
  la clôture, décide pour le lendemain, signe le bloc et ancre la tête sur Sepolia.
  Buy & Hold (achat le premier jour) est suivi dans le même bloc, comme référence.
- Une décision est **à temps** si la barre du jour suivant n'existait pas encore quand
  elle a été écrite. Un jour manqué est rattrapé (même résultat, tout est figé) mais
  marqué **en retard**.
- Registre : `outputs/ledger/live/agents_live.jsonl` ; page « Simulation en direct » du
  tableau de bord. Options : `--no-update`, `--no-anchor`.

### Automatiser chaque nuit, PC éteint (GitHub Actions)

`.github/workflows/live.yml` lance `gold-agents live` chaque nuit à 1 h 30 UTC sur un
serveur de GitHub, puis enregistre le nouveau bloc (signé, ancré sur Sepolia) dans le
dépôt. Les modèles figés et les données sont versionnés pour cela : tout le monde peut
vérifier que les poids sont ceux dont le SHA-256 est scellé dans le genèse.

Une seule fois, depuis le PC où `live-init` a tourné (Git Bash) :

```bash
git pull --no-edit origin main
git add outputs/models/live data/raw/XAU_1d_data.csv data/raw/XAU_1d_gapfill.csv data/raw/XAU_1d_update.csv
git commit -m "Simulation en direct : modèles figés et données pour GitHub Actions"
git push origin main
cat keys/ledger_ed25519.key    # à copier dans le secret LEDGER_PRIVATE_KEY
```

Puis, sur GitHub, **Settings → Secrets and variables → Actions → New repository secret**,
quatre secrets (les trois premiers sont dans `.env`) :

| Secret | Valeur |
|---|---|
| `TWELVEDATA_API_KEY` | la clé Twelve Data |
| `SEPOLIA_RPC_URL` | l'adresse du nœud Sepolia |
| `SEPOLIA_PRIVATE_KEY` | la clé du portefeuille de test |
| `LEDGER_PRIVATE_KEY` | tout le contenu de `keys/ledger_ed25519.key`, lignes `BEGIN`/`END` comprises |

Essai : onglet **Actions → Simulation en direct → Run workflow**. Un lancement raté
envoie un e-mail ; le jour est rattrapé le lendemain, marqué « en retard ». Les poids
sont les mêmes, mais un calcul sur Linux peut différer de Windows au dernier chiffre
près. Si GitHub met la planification en pause après une longue inactivité du dépôt, la
réactiver dans l'onglet Actions.

### Ou depuis votre PC (Windows)

`scripts/live_daily.sh` fait la même routine sur votre PC : `git pull`, `gold-agents live`,
puis commit et push (journal : `outputs/live_daily.log`). N'utilisez **qu'une** des deux
méthodes. Pour le lancer chaque nuit à 2 h, une fois dans **PowerShell** (pas Git Bash) :

```powershell
$a = New-ScheduledTaskAction -Execute "C:\Program Files\Git\bin\bash.exe" -Argument '-lc "bash /c/gold-agents-blockchain/scripts/live_daily.sh"' -WorkingDirectory "C:\gold-agents-blockchain"
$t = New-ScheduledTaskTrigger -Daily -At 2am
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "GoldAgentsLive" -Action $a -Trigger $t -Settings $s -Force
Start-ScheduledTask -TaskName "GoldAgentsLive"   # essai immédiat, puis lire outputs/live_daily.log
```

PC éteint à 2 h : la tâche part dès qu'il se rallume (`-StartWhenAvailable`). La clé
Twelve Data est lue dans `.env`.
Arrêter : `Unregister-ScheduledTask -TaskName "GoldAgentsLive" -Confirm:$false`.

## Tableau de bord (phase 9)

```bash
uv sync --extra dev --extra dashboard     # Streamlit et Plotly (une fois)
make dashboard                            # puis ouvrir http://localhost:8501
```

- **Ne calcule rien** : il lit les registres de `outputs/ledger/<période>/` (versionnés pour le test : un simple clone suffit), `ledger/` (clé publique, ancrages) et, s'ils existent, `outputs/results/` et `mlruns/mlflow.db`. La période (test final, validation…) se choisit dans la barre latérale.
- **Six pages** : *Vue d'ensemble* (indicateurs du système choisi, tableau comparatif, robustesse aux frais, comparaison au hasard), *Prévisions* (prix et rendement réels contre prévus, bande d'incertitude, MAE/RMSE contre B0), *Portefeuille* (valeur, drawdown, exposition), *Décisions* (achats et ventes sur le prix, règle de l'agent risque, journal filtrable et exportable), *Blockchain* (explorateur de blocs, vérification de tous les registres, démo de falsification sur une copie en mémoire, état des ancrages Sepolia avec lien Etherscan), *Expériences* (réglages figés, hyperparamètres retenus, runs MLflow, figures).
- La logique de lecture est dans `src/gold_agents/dashboard/data.py` (testée sans Streamlit) ; `dashboard/` ne fait qu'afficher.

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
| `make train` | Baselines B0-B4, LSTM et GRU : tableau comparatif sur la validation, schéma |
| `make backtest` | Buy & Hold et SMA 20/50 sur le test : métriques, registres, schéma |
| `make simulate` | Agents (prévision, stratégie, risque, exécution) contre Buy & Hold sur la validation |
| `make tune` | Réglage des agents en walk-forward : grille, frais, hasard, oracle, schéma |
| `make freeze` | Scelle les réglages de la config dans un bloc genèse (après `tune`) |
| `make evaluate` | Phase 8 : le run unique sur le test, tableau final, figures, registres ancrés |
| `make keygen` | Paire de clés Ed25519 du registre (une seule fois) |
| `make verify` | Vérification des registres de `outputs/ledger/` (signatures avec `ledger/public_key.pem`) |
| `make verify-ledger` | Script autonome `scripts/verify_ledger.py` sur `ledger/` |
| `make anchor` | Bonus : ancre le dernier hash des registres sur Sepolia (`.env` requis) |
| `make dashboard` | Tableau de bord Streamlit (6 pages, http://localhost:8501) |
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
