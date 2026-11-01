# Architecture technique du projet
## Investissement simulé sur l'or (XAU/USD) : Deep Learning, agents autonomes et blockchain

> **Rôle de ce document.** C'est le document de référence du « tech lead » : il fixe les choix d'outils, de modèles, de tests, l'organisation du code et le planning. Il complète [explication_projet.md](explication_projet.md), qui explique le sujet. Ici on décide **comment** on le construit.
>
> **Document vivant.** Chaque décision est justifiée et peut être révisée. Quand une décision change, on met à jour la section concernée et on ajoute une ligne dans le [journal des révisions](#16-journal-des-révisions).
>
> Aucune implémentation dans ce document.

---

## Sommaire

1. [Principes directeurs](#1-principes-directeurs)
2. [Vue d'ensemble de l'architecture](#2-vue-densemble-de-larchitecture)
3. [Stack technique (outils et frameworks)](#3-stack-technique-outils-et-frameworks)
4. [Données et pipeline de features](#4-données-et-pipeline-de-features)
5. [Modèles d'IA : choix, justification, baselines](#5-modèles-dia--choix-justification-baselines)
6. [Système multi-agents](#6-système-multi-agents)
7. [Moteur de simulation (backtest)](#7-moteur-de-simulation-backtest)
8. [Blockchain / registre partagé](#8-blockchain--registre-partagé)
9. [Tableau de bord](#9-tableau-de-bord)
10. [Stratégie de tests](#10-stratégie-de-tests)
11. [Structure du dépôt](#11-structure-du-dépôt)
12. [Phases, jalons et planning](#12-phases-jalons-et-planning)
13. [Risques et parades](#13-risques-et-parades)
14. [Extension future : facteurs externes au prix](#14-extension-future--facteurs-externes-au-prix)
15. [Décisions ouvertes](#15-décisions-ouvertes)
16. [Journal des révisions](#16-journal-des-révisions)

---

## 1. Principes directeurs

Ces cinq règles tranchent les débats quand deux options se valent.

| # | Principe | Conséquence concrète |
|---|---|---|
| P1 | **Zéro fuite du futur** | Toute donnée utilisée à la date *t* doit être connue à *t*. Testé automatiquement (§10.3). |
| P2 | **Reproductibilité** | Graines aléatoires fixées, versions figées, config unique (`config.yaml`), données brutes jamais modifiées. Un `make all` reproduit tous les résultats. |
| P3 | **Simple d'abord, complexe ensuite** | Chaque brique démarre avec la version la plus simple qui marche (Buy & Hold, chaîne de hachage locale). On complexifie seulement si le temps le permet. |
| P4 | **Modules découplés** | Chaque brique a une entrée et une sortie typées. On peut remplacer le LSTM par un GRU, ou la blockchain locale par Ethereum, sans toucher au reste. |
| P5 | **Tout est vérifiable** | Chaque résultat chiffré du rapport est produit par un script, jamais recopié à la main. Le professeur peut tout rejouer et vérifier la chaîne seul. |

---

## 2. Vue d'ensemble de l'architecture

### 2.1 Vue en couches

```
┌───────────────────────────────────────────────────────────────────────┐
│  PRÉSENTATION        dashboard/ (Streamlit + Plotly)                  │
│                      verify_ledger.py (CLI pour le professeur)        │
├───────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION       main.py / CLI (Typer) : prepare, train,          │
│                      backtest, evaluate, verify                       │
├───────────────────────────────────────────────────────────────────────┤
│  DÉCISION            agents/ : Forecast → Strategy → Risk → Execution │
│                      (bus de messages synchrone, un tour par jour)    │
├───────────────────────────────────────────────────────────────────────┤
│  MÉTIER              models/ (PyTorch)   simulation/ (portefeuille)   │
│                      evaluation/ (métriques)                          │
├───────────────────────────────────────────────────────────────────────┤
│  PERSISTANCE         data/ (Parquet)   ledger/ (blockchain JSON+SQLite)│
│                      outputs/ (modèles, résultats)  mlruns/ (MLflow)  │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.2 Flux d'un jour simulé

```
Jour t (clôture connue)
   │
   ▼
[FeatureStore] ── fenêtre [t-W+1 … t] ──► [Agent Prévision] ── r̂(t+1), σ̂ ──►
[Agent Stratégie] ── signal BUY/SELL/HOLD + confiance ──►
[Agent Risque] ── ordre validé / réduit / rejeté + motif ──►
[Agent Exécution] ── exécute au prix d'OUVERTURE de t+1 (+ frais, slippage)
   │                         │
   ▼                         ▼
[Portefeuille mis à jour]  [Événements → Blockchain : 1 bloc par jour]
```

**Règle d'or d'exécution :** une décision prise avec la clôture de *t* est exécutée à l'ouverture de *t+1*. Jamais au prix de clôture qui a servi à la décider.

---

## 3. Stack technique (outils et frameworks)

### 3.1 Tableau récapitulatif

| Besoin | Choix retenu | Alternatives écartées (et pourquoi) |
|---|---|---|
| Langage | **Python 3.11** | Imposé de fait par l'écosystème DL et le livrable « code Python ». |
| Environnement | **uv** (ou `venv` + `pip-tools`) avec versions figées | conda : plus lourd, inutile ici. |
| Manipulation de données | **pandas** + **numpy** ; stockage **Parquet** (pyarrow) | polars : plus rapide mais moins connu des correcteurs ; le volume (≈5 000 lignes/jour) ne le justifie pas. |
| Indicateurs techniques | **pandas** pur (ou `ta`) | TA-Lib : installation C pénible. Les indicateurs utiles (SMA, EMA, RSI, MACD, ATR, Bollinger) s'écrivent en quelques lignes et se testent mieux. |
| Deep Learning | **PyTorch 2.x** + **PyTorch Lightning** (optionnel) | Keras/TensorFlow : tout aussi valable. PyTorch retenu pour le contrôle fin de la boucle d'entraînement, le MC Dropout et la lisibilité du code. *Décision révisable si l'équipe maîtrise mieux Keras.* |
| Baselines ML | **scikit-learn**, **statsmodels** (ARIMA), **LightGBM** | — |
| Recherche d'hyperparamètres | **Optuna** | Grid search manuelle : plus lente, moins traçable. |
| Suivi d'expériences | **MLflow** (local, `mlruns/`) | Weights & Biases : compte externe ; TensorBoard : moins adapté aux comparaisons de runs. |
| Configuration | **YAML** + **pydantic** (validation) | Hydra : puissant mais courbe d'apprentissage. |
| Agents | **Classes Python maison** + bus de messages simple | Mesa, SPADE, LangGraph : voir §6.4. |
| Blockchain | **Chaîne de hachage maison** (SHA-256, `hashlib`) + **signatures Ed25519** (`cryptography`) | Option bonus : contrat Solidity sur **Hardhat/Anvil** via **web3.py**. Voir §8. |
| Stockage du registre | **JSON Lines** (une ligne = un bloc) + index **SQLite** | Base distribuée : surdimensionné pour un projet pédagogique. |
| Tableau de bord | **Streamlit** + **Plotly** | Dash : plus verbeux ; Grafana : pensé pour du monitoring live, pas du backtest. |
| CLI | **Typer** | argparse : plus verbeux. |
| Tests | **pytest**, **hypothesis** (tests par propriétés), **pytest-cov** | unittest : plus verbeux. |
| Qualité de code | **ruff** (lint + format), **mypy** (typage), **pre-commit** | black + flake8 + isort : remplacés par ruff seul. |
| CI | **GitHub Actions** (lint + tests à chaque push) | — |
| Versionnement | **Git** + GitHub ; données brutes hors Git (lien Kaggle + checksum) ou **DVC** si besoin | — |
| Rapport | Markdown / LaTeX, figures générées par script | — |

### 3.2 Dépendances principales (`pyproject.toml`, indicatif)

```
pandas, numpy, pyarrow
torch, lightning (optionnel)
scikit-learn, statsmodels, lightgbm
optuna, mlflow
pydantic, pyyaml, typer
cryptography
streamlit, plotly
pytest, hypothesis, pytest-cov, ruff, mypy, pre-commit
# bonus blockchain publique : web3, eth-account
```

---

## 4. Données et pipeline de features

### 4.1 Source
- **Kaggle « XAUUSD Gold Price Tracker: 2004–Present »**, fréquence **quotidienne**.
- Le CSV brut est stocké dans `data/raw/` avec son **empreinte SHA-256** notée dans `data/raw/CHECKSUMS.txt`. Il n'est jamais modifié.

### 4.2 Pipeline (chaque étape = une fonction pure, testée)

```
raw CSV ─► load() ─► clean() ─► add_returns() ─► add_indicators()
        ─► split_chrono() ─► fit_scaler(train) ─► transform(all)
        ─► make_windows(W) ─► Parquet + metadata.json
```

| Étape | Détail |
|---|---|
| `load` | Parsing des dates, fuseau horaire unique (UTC), tri chronologique. |
| `clean` | Doublons, dates manquantes (week-ends normaux, trous anormaux signalés), valeurs aberrantes (High < Low, prix ≤ 0). Aucune interpolation sur la cible. |
| `add_returns` | Rendement logarithmique `r_t = ln(C_t / C_{t-1})`. **La cible est `r_{t+1}`.** |
| `add_indicators` | SMA/EMA (10, 20, 50), RSI(14), MACD, ATR(14), largeur de Bollinger, volatilité réalisée 20 j, jour de la semaine. Tous calculés **uniquement avec le passé** (pas de `center=True`, pas de `shift(-k)` hors cible). |
| `split_chrono` | Train ≈ 2004-2018, Validation ≈ 2019-2021, Test ≈ 2022-fin. Dates exactes dans `config.yaml`. **Gap** de W jours entre ensembles pour qu'aucune fenêtre ne chevauche deux ensembles. |
| `fit_scaler` | `StandardScaler` (ou `RobustScaler`) **ajusté sur le train seulement**, sauvegardé avec le modèle. |
| `make_windows` | Fenêtres de W = 30 (à régler : 20, 30, 60) jours → tenseur `(N, W, n_features)`. |

### 4.3 Validation glissante (walk-forward)
En plus du découpage fixe, une **validation walk-forward** (5 plis à fenêtre croissante) est utilisée pour choisir les hyperparamètres. Elle montre que le modèle ne marche pas « par chance » sur une seule période. Le **test n'est utilisé qu'une seule fois**, à la fin.

---

## 5. Modèles d'IA : choix, justification, baselines

### 5.1 Formulation du problème

| Option | Sortie | Usage |
|---|---|---|
| **A. Régression (retenue)** | Rendement prévu `r̂(t+1)` | Permet un seuil d'action et un dimensionnement proportionnel. |
| B. Classification | P(hausse) | Variante à tester ; plus robuste au bruit, mais perd l'amplitude. |
| C. Probabiliste (bonus) | `μ̂(t+1)` et `σ̂(t+1)` | Donne l'incertitude exigée « éventuellement » par le sujet. |

### 5.2 Échelle des modèles (du plus simple au plus complexe)

Un modèle complexe n'est retenu que s'il **bat les plus simples sur la validation**. C'est la question n°1 du jury : « votre LSTM fait-il mieux qu'une règle naïve ? »

| Niveau | Modèle | Rôle | Pourquoi |
|---|---|---|---|
| B0 | **Naïf « rendement nul »** (marche aléatoire) | Plancher absolu | Sur les marchés, c'est souvent très dur à battre en RMSE. |
| B1 | **Naïf « même rendement qu'hier »** / moyenne mobile du rendement | Plancher | — |
| B2 | **ARIMA** (statsmodels) | Baseline statistique classique | Référence académique des séries temporelles. |
| B3 | **Régression linéaire / Ridge** sur les features | Baseline ML linéaire | Montre ce que les features apportent sans DL. |
| B4 | **LightGBM** sur features tabulaires (avec retards) | Baseline ML forte | Très souvent compétitif en finance ; si le LSTM ne le bat pas, il faut le dire. |
| **M1** | **LSTM** (1-2 couches, 32-64 unités, dropout 0,2-0,3) | Modèle principal exigé | Mémoire à long terme, adapté aux séquences. |
| **M2** | **GRU** (même gabarit) | Modèle principal exigé | Moins de paramètres que le LSTM, souvent aussi bon et plus rapide ; peu de données → moins de surapprentissage. |
| M3 (bonus) | **TCN / CNN-1D** | Comparaison | Convolutions causales, entraînement parallèle. |
| M4 (bonus) | **Petit Transformer** (encodeur, 2 têtes) ou modèle pré-entraîné type séries temporelles | Ouverture | À présenter comme perspective : avec ~4 000 exemples, un Transformer risque de surapprendre. |
| E (bonus) | **Ensemble** LSTM + GRU (moyenne) | Robustesse | Réduit la variance des prévisions. |

### 5.3 Détails d'entraînement (M1/M2)
- **Perte :** MSE ou Huber (plus robuste aux chocs de prix). Variante probabiliste : Gaussian NLL avec deux sorties (μ, log σ²).
- **Optimiseur :** Adam / AdamW, lr ≈ 1e-3, *gradient clipping* (norme 1,0).
- **Régularisation :** dropout, weight decay, **early stopping** sur la perte de validation (patience 10-20).
- **Hyperparamètres réglés avec Optuna** (sur walk-forward) : W, unités, couches, dropout, lr, batch size.
- **Graines multiples :** chaque configuration finale entraînée avec 5 graines ; on rapporte moyenne ± écart-type.
- **Incertitude :** **MC Dropout** (dropout actif à l'inférence, 50 passages) → `σ̂`. Simple et compatible avec le LSTM existant.

### 5.4 Métriques du modèle
- **MAE, RMSE** (exigées).
- **Précision directionnelle** (% de bons signes) : la plus parlante pour le trading.
- **Test de Diebold-Mariano** (bonus) pour dire si l'écart avec le modèle naïf est statistiquement significatif.
- Toujours présentées **à côté des baselines B0 à B4**.

---

## 6. Système multi-agents

### 6.1 Choix d'architecture
Les agents sont des **agents logiciels réactifs à rôle unique**, pas des LLM. Chacun :
- reçoit un **message typé** (dataclass / pydantic),
- applique sa logique,
- émet un message typé au suivant,
- **trace** son raisonnement (entrée, sortie, motif) dans le registre.

Coordination par **pipeline séquentiel** orchestré par un `SimulationClock` : à chaque jour, chaque agent agit une fois, dans l'ordre. C'est le modèle le plus simple à tester et à auditer.

### 6.2 Contrats entre agents

| Agent | Entrée | Sortie | Logique principale |
|---|---|---|---|
| **1. Prévision** | Fenêtre de features à *t* | `Forecast(date, r_hat, sigma_hat, model_id)` | Charge le modèle figé, prédit. Ne voit **jamais** de données > t. |
| **2. Stratégie** | `Forecast` + position actuelle | `Signal(action ∈ {BUY, SELL, HOLD}, strength, reason)` | Seuil : BUY si `r_hat > θ_buy`, SELL si `r_hat < −θ_sell`, sinon HOLD. Variante : seuil relatif à l'incertitude (`r_hat / σ_hat > k`) et au coût de transaction (n'agir que si le gain attendu > frais). |
| **3. Risque** | `Signal` + état du portefeuille | `Order(qty, approved, reason)` ou rejet | Trésorerie dispo, taille max de position (ex. 100 % ou 50 % du capital), pas de vente à découvert (v1), **stop-loss** (ex. −5 % sur position), **drawdown max** (ex. −20 % → passage en cash), *volatility targeting* optionnel. |
| **4. Exécution** | `Order` | `Trade(date_exec, price, qty, fees, slippage)` + événement ledger | Exécute à l'ouverture de t+1, applique frais (ex. 0,1 %) et slippage (ex. 0,02 %), met à jour le portefeuille, écrit le bloc. |

### 6.3 Paramètres
Tous les seuils (θ, tailles, stops, frais) sont dans `config.yaml`, **réglés sur la validation, figés avant le test**, et leur valeur est elle-même inscrite dans le **bloc genèse** de la blockchain (preuve qu'on ne les a pas changés après avoir vu le test).

### 6.4 Pourquoi pas un framework multi-agents ?

| Framework | Verdict |
|---|---|
| **Mesa** | Pensé pour des simulations à base d'agents spatiales/nombreuses. Utilisable pour l'horloge, mais apporte peu ici. *Option si le professeur exige un framework reconnu.* |
| **SPADE** (agents XMPP) | Vrai système multi-agents asynchrone, mais complexité réseau inutile pour un backtest déterministe. |
| **LangGraph / agents LLM** | Hors sujet : non déterministe, coûteux, difficile à auditer. Peut être une **perspective** (agent « analyste » qui explique les décisions en langage naturel). |
| **Maison (retenu)** | 4 classes + une interface `Agent.act(message) -> message`. Déterministe, testable, lisible. |

### 6.5 Extension possible (bonus)
Plusieurs **agents stratégie concurrents** (LSTM, GRU, croisement de moyennes) avec un **agent arbitre** qui pondère leurs signaux selon leur performance récente. Cela renforce l'aspect « multi-agents » sans complexifier le reste.

---

## 7. Moteur de simulation (backtest)

### 7.1 Composants
- `Portfolio` : cash, quantité d'or (en onces, fractionnable), historique des valeurs.
- `CostModel` : frais proportionnels + slippage + (optionnel) spread fixe.
- `BacktestEngine` : boucle jour par jour, branche n'importe quelle stratégie qui respecte l'interface `Strategy`.

### 7.2 Règles
- Capital initial fixé (ex. 10 000 USD).
- Exécution à l'**ouverture de t+1**.
- Pas d'effet de levier, pas de vente à découvert en v1 (activables plus tard).
- Valorisation quotidienne à la clôture.
- **Profit net = V_T − V_0**, avec les coûts déjà déduits pendant la simulation ; on rapporte aussi le total des coûts séparément.

### 7.3 Stratégies comparées (toutes passent par le même moteur)
1. **Buy & Hold** (exigée).
2. **Croisement de moyennes mobiles** SMA 20/50 (règle technique exigée).
3. **RSI** sur-achat / sur-vente (optionnelle).
4. **Système multi-agents + LSTM**.
5. **Système multi-agents + GRU**.
6. **Stratégie aléatoire** (même nombre de trades) : montre si le modèle bat le hasard.
7. **Oracle** (connaît le futur) : borne supérieure théorique, **pour illustration uniquement**.

### 7.4 Métriques financières
Profit net, rendement total, rendement annualisé (CAGR), volatilité annualisée, **Sharpe** (taux sans risque explicite dans la config), Sortino, **Maximum Drawdown**, Calmar, nombre de trades, **taux de réussite**, gain moyen / perte moyenne, total des frais, exposition moyenne au marché.

**Analyse de sensibilité :** refaire le test avec frais × 0, × 1, × 2, × 5. Une stratégie qui ne survit pas à des frais doublés n'est pas robuste.

---

## 8. Blockchain / registre partagé

### 8.1 Options étudiées

| Option | Description | Avantages | Inconvénients | Verdict |
|---|---|---|---|---|
| **A. Chaîne de hachage maison** | Blocs JSON chaînés par SHA-256, Merkle root des événements, preuve de travail légère, signature Ed25519 de l'agent exécution | Totalement maîtrisée, explicable, vérifiable hors ligne, zéro coût | Pas « distribuée » au sens strict | **Retenue (socle)** |
| **B. Ethereum local** (Hardhat / Anvil) + contrat Solidity, via web3.py | Chaque bloc du registre = une transaction | Vraie blockchain, vraie EVM | Le professeur doit lancer le nœud ; plus de travail | Bonus |
| **C. Ancrage sur testnet public** (Sepolia) | On publie seulement le **hash de tête** de la chaîne A (1 transaction par jour simulé ou par run) | Horodatage public infalsifiable, très peu coûteux | Nécessite un portefeuille testnet et un faucet | **Bonus recommandé** : meilleur rapport preuve/effort |
| D. Hyperledger Fabric | Registre permissionné d'entreprise | Très « pro » | Énorme complexité de déploiement | Écartée |

### 8.2 Structure d'un bloc (option A)

```json
{
  "index": 128,
  "timestamp_utc": "2026-10-26T16:00:00Z",
  "sim_date": "2023-03-14",
  "prev_hash": "a3f…",
  "merkle_root": "9c1…",
  "nonce": 48211,
  "difficulty": 4,
  "events": [
    {"type": "FORECAST", "agent_id": "forecast-v1", "model_id": "gru_2026-10-20_seed42", "r_hat": 0.0031, "sigma_hat": 0.009},
    {"type": "SIGNAL",   "agent_id": "strategy-v1", "action": "BUY", "strength": 0.34, "reason": "r_hat > θ_buy"},
    {"type": "RISK",     "agent_id": "risk-v1", "approved": true, "qty": 2.5, "reason": "within limits"},
    {"type": "TRADE",    "agent_id": "exec-v1", "price": 1912.40, "qty": 2.5, "fees": 4.78,
     "portfolio_before": 10012.3, "portfolio_after": 10007.5}
  ],
  "signature": "ed25519:…",
  "hash": "0000b7e…"
}
```

- **Bloc genèse :** contient la config complète (seuils, frais, dates de split), le hash du jeu de données brut et le hash des poids du modèle. Cela prouve que **rien n'a été réglé après coup**.
- **Hash du bloc** = SHA-256 de la sérialisation canonique (JSON trié, sans espaces) de tous les champs sauf `hash`.
- **Validation :** chaque `prev_hash` correspond, chaque hash est recalculé, chaque Merkle root est recalculée, chaque signature vérifiée avec la clé publique publiée, la difficulté est respectée.
- **Réalisé en phase 7 (v2) :** le hash couvre l'en-tête (`version`, `index`, `sim_date`, `prev_hash`, `merkle_root`, `difficulty`, `nonce`) ; les événements y entrent par la racine de Merkle (comme Bitcoin), et la signature Ed25519 porte sur le hash, donc n'est pas dans l'en-tête. Difficulté 3 (zéros hexadécimaux). Pas de `timestamp_utc` réel dans les blocs : il rendrait les registres non reproductibles ; l'antériorité vient de Git et de l'ancrage (bonus C). Les registres v1 des phases 3 à 6 restent vérifiables.

### 8.3 Partage avec le professeur
1. Dépôt GitHub (accès en lecture pour le professeur) contenant `ledger/chain.jsonl`, la **clé publique** et le script `verify_ledger.py` (dépendances minimales).
2. Onglet « Vérifier la chaîne » dans le tableau de bord.
3. (Bonus C) Hash de tête ancré sur Sepolia : le professeur peut comparer avec Etherscan.

### 8.4 Ce que la blockchain garantit (à écrire dans le rapport)
Elle garantit l'**intégrité** et l'**antériorité** des décisions enregistrées. Elle **ne garantit pas** que les prévisions sont bonnes, ni que la stratégie est rentable, ni que le code qui a produit les décisions était correct.

---

## 9. Tableau de bord

**Streamlit + Plotly**, lancé par `streamlit run dashboard/app.py`, lit uniquement les fichiers de `outputs/` et `ledger/` (aucun calcul lourd).

| Page | Contenu |
|---|---|
| **Vue d'ensemble** | KPI : profit net, rendement, Sharpe, MDD, nb de trades, comparaison aux baselines. |
| **Prévisions** | Prix réel vs prévu, rendement réel vs prévu, bande d'incertitude, MAE/RMSE par modèle. |
| **Portefeuille** | Courbe de valeur de chaque stratégie, courbe de drawdown, exposition. |
| **Décisions** | Prix avec marqueurs BUY/SELL, tableau filtrable des décisions et de leur motif (quel agent a bloqué quoi). |
| **Blockchain** | Explorateur de blocs, bouton « Vérifier l'intégrité », démo de falsification (modifier une copie → la vérification échoue). |
| **Expériences** | Tableau des runs MLflow, hyperparamètres retenus. |

**Réalisé en phase 9 :** navigation Streamlit (`st.navigation`) avec une page par onglet dans `dashboard/pages/`, et toute la lecture des fichiers dans `src/gold_agents/dashboard/data.py` (sans Streamlit, testée). Les registres du test contiennent chaque jour la prévision, le signal, l'ordre, le trade et le portefeuille : le tableau de bord se construit à partir d'eux seuls, sur un simple clone. Les CSV de `outputs/results/` et la base MLflow, non versionnés, ne servent qu'en complément. La démo de falsification modifie une copie en mémoire selon trois niveaux (valeur modifiée, bloc re-haché, chaîne réécrite avec une autre clé) ; l'onglet d'ancrage recalcule l'empreinte publiée sur Sepolia à partir des registres présents.

---

## 10. Stratégie de tests

### 10.1 Pyramide

```
             ┌───────────────┐
             │  Backtests de │  quelques runs complets, résultats figés
             │  bout en bout │
           ┌─┴───────────────┴─┐
           │   Intégration     │  pipeline données→modèle→agents→ledger
         ┌─┴───────────────────┴─┐
         │   Unitaires + propriétés │  la majorité des tests
         └─────────────────────────┘
```

Objectif de couverture : **≥ 80 %** sur `src/` (hors notebooks et dashboard). Tous les tests tournent dans la CI en < 2 min (le modèle est remplacé par un petit modèle ou un faux modèle).

### 10.2 Tests unitaires (exemples)

| Module | Tests |
|---|---|
| `data` | Le chargement trie les dates ; pas de doublons ; High ≥ Low ; rendements corrects sur une petite série connue à la main. |
| `features` | RSI, SMA, ATR comparés à des valeurs calculées à la main ; forme des fenêtres `(N, W, F)`. |
| `models` | Le modèle accepte la bonne forme et renvoie la bonne forme ; il **surapprend volontairement** un mini-lot (preuve que la boucle d'entraînement fonctionne) ; même graine → même résultat. |
| `agents` | Stratégie : `r_hat` au-dessus du seuil → BUY ; Risque : refuse un achat sans trésorerie, refuse au-delà de la taille max, déclenche le stop-loss ; Exécution : frais bien déduits. |
| `simulation` | Valeur du portefeuille = cash + qté × prix à chaque pas ; Buy & Hold sans frais = rendement exact du prix ; aucun trade → valeur constante. |
| `blockchain` | Hash déterministe ; chaîne valide après N ajouts ; Merkle root correcte ; signature valide/invalide. |
| `evaluation` | MDD, Sharpe, CAGR sur des courbes synthétiques dont on connaît la réponse. |

### 10.3 Tests anti-fuite du futur (*look-ahead*) — **critiques**

| Test | Méthode |
|---|---|
| **Test de troncature** | Calculer les features sur toute la série, puis sur la série coupée à *t*. Les features jusqu'à *t* doivent être **identiques**. Si une valeur change, une feature regarde le futur. |
| **Test de permutation du futur** | Remplacer toutes les données après *t* par du bruit : les décisions jusqu'à *t* ne doivent **pas changer**. |
| **Scaler** | Vérifier que les statistiques du scaler ne dépendent que des dates du train. |
| **Splits** | `max(date_train) + gap < min(date_val)` et idem val/test ; aucune fenêtre ne mélange deux ensembles. |
| **Exécution** | Chaque trade a `date_exec > date_decision` et utilise le prix d'ouverture de `date_exec`. |
| **Test « modèle trop beau »** | Si la précision directionnelle dépasse ~60 % ou si le Sharpe dépasse ~3 sur le test, alerte automatique : c'est presque toujours une fuite. |

### 10.4 Tests d'intégrité du registre

| Test | Résultat attendu |
|---|---|
| Modifier un champ d'un événement | Vérification échoue au bloc concerné (Merkle root ≠). |
| Modifier un bloc et recalculer son hash | Échec au bloc suivant (`prev_hash` ≠). |
| Supprimer / insérer / réordonner un bloc | Échec. |
| Recalculer toute la chaîne après modification | Échec de **signature** (le fraudeur n'a pas la clé privée) et, en bonus, écart avec le hash ancré sur testnet. |
| Rejouer le backtest avec les mêmes paramètres | Chaîne **identique** octet pour octet (hors horodatage réel, isolé dans un champ à part). |

### 10.5 Tests par propriétés (hypothesis)
Pour n'importe quelle suite de prix et d'ordres générée aléatoirement : le cash n'est jamais négatif, la quantité n'est jamais négative (sans vente à découvert), la valeur du portefeuille est toujours ≥ 0, la somme des frais est ≥ 0.

### 10.6 Backtests de validation
- **Golden test :** un backtest de référence sur une petite période dont les résultats sont figés dans un fichier ; toute modification du code qui change ces résultats fait échouer le test (on doit alors justifier le changement).
- **Walk-forward** sur plusieurs périodes ; **analyse de sensibilité** aux frais et aux seuils ; **plusieurs graines**.
- Comparaison systématique avec Buy & Hold, SMA 20/50 et stratégie aléatoire.

---

## 11. Structure du dépôt

```
gold-agents-chain/
├── README.md                     # Installation, commandes, résultats clés
├── pyproject.toml                # Dépendances figées (uv / pip-tools)
├── Makefile                      # make data | train | backtest | test | dashboard | all
├── config/
│   ├── config.yaml               # Dates, fenêtre, seuils, frais, capital, graines
│   └── schema.py                 # Validation pydantic de la config
├── data/
│   ├── raw/                      # CSV Kaggle + CHECKSUMS.txt (hors Git ou DVC)
│   └── processed/                # Parquet train/val/test + metadata.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_model_analysis.ipynb
├── src/gold_agents/
│   ├── cli.py                    # Typer : prepare, train, backtest, evaluate, verify
│   ├── data/          load.py, clean.py, features.py, splits.py, windows.py
│   ├── models/        baselines.py, lstm.py, gru.py, train.py, predict.py, uncertainty.py
│   ├── agents/        base.py, messages.py, forecast.py, strategy.py, risk.py, execution.py, orchestrator.py
│   ├── simulation/    portfolio.py, costs.py, engine.py, strategies.py
│   ├── blockchain/    block.py, merkle.py, chain.py, signing.py, storage.py, anchor.py (bonus)
│   └── evaluation/    forecast_metrics.py, financial_metrics.py, report.py
├── dashboard/
│   ├── app.py
│   └── pages/                    # Une page Streamlit par onglet (§9)
├── scripts/
│   └── verify_ledger.py          # Script autonome pour le professeur
├── contracts/ (bonus)            # Solidity + config Hardhat
├── tests/
│   ├── unit/
│   ├── leakage/                  # Tests anti look-ahead (§10.3)
│   ├── ledger/                   # Tests d'intégrité (§10.4)
│   ├── property/                 # hypothesis
│   ├── integration/
│   └── golden/                   # Backtest de référence figé
├── outputs/
│   ├── models/                   # Poids + scaler + hash
│   ├── results/                  # Métriques JSON/CSV, figures
│   └── ledger/                   # chain.jsonl, public_key.pem
├── mlruns/                       # MLflow (ignoré par Git)
├── docs/
│   ├── explication_projet.md
│   ├── architecture_technique.md # Ce document
│   └── rapport/
└── .github/workflows/ci.yml      # ruff + mypy + pytest
```

**Conventions :** code et noms de variables en anglais, commentaires et rapport en français ; une branche par fonctionnalité ; *pull request* + CI verte avant fusion ; messages de commit clairs (`feat:`, `fix:`, `test:`…).

---

## 12. Phases, jalons et planning

Planning indicatif sur **10 semaines** (à ajuster selon la date de rendu). Chaque phase se termine par un **jalon vérifiable** (« Definition of Done »).

| Semaine | Phase | Livrables | Jalon / critère de fin |
|---|---|---|---|
| S1 | **0. Cadrage et socle** | Dépôt, structure, CI, config, téléchargement des données + checksum | `make test` passe dans la CI ; le CSV se charge. |
| S1-S2 | **1. Données et EDA** | Notebook EDA, `clean.py`, constats écrits | Rapport d'anomalies ; graphiques prix/rendements/volatilité. |
| S2-S3 | **2. Features et splits** | `features.py`, `splits.py`, Parquet | **Tests anti-fuite verts** (§10.3). |
| S3 | **3. Moteur + Buy & Hold + blockchain v1** | `simulation/`, `blockchain/` (chaîne simple) | Buy & Hold cohérent avec le prix réel ; falsification détectée. *La « plomberie » marche sans IA.* |
| S4-S5 | **4. Baselines puis LSTM/GRU** | B0-B4, M1, M2, Optuna, MLflow | Tableau MAE/RMSE/précision directionnelle vs baselines (5 graines). |
| S6 | **5. Agents** | 4 agents, orchestrateur, messages typés | Simulation complète sur la validation, chaque décision tracée. |
| S7 | **6. Réglage et robustesse** | Seuils/risque réglés sur la validation, walk-forward, sensibilité aux frais | Paramètres **figés** dans `config.yaml` et dans le bloc genèse. |
| S7 | **7. Blockchain v2** | Merkle, signature Ed25519, `verify_ledger.py`, (bonus) ancrage Sepolia | Suite de tests d'intégrité verte ; le script tourne sur une machine vierge. |
| S8 | **8. Évaluation finale (test)** | **Un seul** run sur le test pour toutes les stratégies | Tableau comparatif final, figures générées par script. |
| S8-S9 | **9. Tableau de bord** | 6 pages Streamlit | Démo de bout en bout fluide. |
| S9-S10 | **10. Rapport et soutenance** | Rapport, slides, démo répétée | Relecture ; réponses préparées aux questions du jury. |

**Chemin critique :** données → features (sans fuite) → moteur → modèle → agents → test final. La blockchain et le dashboard peuvent avancer en parallèle dès S3.

**Répartition possible si travail en équipe :** une personne « données + modèles », une personne « agents + simulation », une personne « blockchain + dashboard + CI ». Points de synchronisation : les **contrats de messages** (§6.2) et le **format des fichiers** de `outputs/`.

---

## 13. Risques et parades

| # | Risque | Probabilité | Impact | Parade |
|---|---|---|---|---|
| R1 | **Fuite du futur** (features, scaler, exécution) qui gonfle les résultats | Élevée | Critique | Tests §10.3 dans la CI, exécution à t+1, alerte « modèle trop beau ». |
| R2 | **Le LSTM ne bat pas le modèle naïf** | Très élevée (c'est normal en finance) | Moyen | L'annoncer honnêtement ; mettre en avant la précision directionnelle, l'incertitude, le filtrage par l'agent risque ; c'est un **résultat scientifique** valable. |
| R3 | **Surapprentissage** (peu de données, ~4 000 jours) | Élevée | Élevé | Petits modèles, dropout, early stopping, walk-forward, plusieurs graines. |
| R4 | **Sur-optimisation des seuils** sur la validation | Moyenne | Élevé | Peu de paramètres libres, grilles grossières, test utilisé une seule fois, paramètres scellés dans le bloc genèse. |
| R5 | **Qualité des données Kaggle** (trous, décalages horaires, prix MT4 propres au courtier) | Moyenne | Moyen | EDA, contrôles automatiques, comparaison ponctuelle avec une autre source (ex. LBMA) mentionnée dans le rapport. |
| R6 | **Frais sous-estimés** → profits irréalistes | Moyenne | Élevé | Frais + slippage + sensibilité × 2 / × 5. |
| R7 | **Blockchain jugée « pas assez blockchain »** | Moyenne | Moyen | Merkle + signature + PoW légère ; bonus ancrage Sepolia ou contrat Hardhat. |
| R8 | **Non-reproductibilité** (versions, graines, GPU) | Moyenne | Moyen | Versions figées, graines, `torch.use_deterministic_algorithms`, CPU suffisant pour ce volume. |
| R9 | **Dérive de périmètre** (Transformer, RL, LLM, facteurs externes…) | Élevée | Moyen | Les bonus ne démarrent qu'après le jalon S8 ; liste de « perspectives » pour le rapport. |
| R10 | **Retard sur le planning** | Moyenne | Élevé | Ordre « plomberie d'abord », jalons hebdomadaires, version minimale livrable dès S6. |

---

## 14. Extension future : facteurs externes au prix

> **Statut : à faire plus tard.** Le sujet demande une prévision à partir des prix historiques. On livre d'abord cette version « prix seuls » (phases S1 à S8). Cette extension ne démarre qu'une fois la version de base terminée et évaluée.

### 14.1 Pourquoi
Les bougies (Open, High, Low, Close) ne sont que la **trace finale** de ce qui fait bouger l'or : décisions et discours de la Fed, chiffres de l'inflation et de l'emploi, guerres, droits de douane, déclarations politiques. Le prix réagit en quelques minutes, et la bougie enregistre seulement le résultat.

Le modèle « prix seuls » suppose que ces informations sont déjà dans le prix. C'est en partie vrai, mais il **ne peut pas anticiper** une annonce qu'il ne connaît pas. Ajouter des facteurs externes permet de tester s'ils améliorent vraiment la prévision et le profit.

### 14.2 Familles de variables envisagées

| Famille | Exemples | Source possible | Fréquence |
|---|---|---|---|
| **Macro-économie** | Taux américains (10 ans, taux réel), inflation (CPI), indice du dollar **DXY** | **FRED** (Réserve fédérale de Saint-Louis), API gratuite | Quotidienne à mensuelle |
| **Peur du marché** | **VIX** (volatilité implicite du S&P 500) | FRED / Yahoo Finance | Quotidienne |
| **Calendrier économique** | Dates des réunions de la Fed (**FOMC**), chiffres de l'emploi (**NFP**), CPI | Calendriers officiels (Fed, BLS) | Événements datés, **connus à l'avance** |
| **Sentiment des actualités** | Score positif / négatif par jour à partir des titres de presse ou des discours | Modèle NLP **FinBERT** (Hugging Face `transformers`) appliqué à un jeu de titres datés (ex. GDELT, jeux Kaggle de news financières) | Quotidienne (agrégée) |
| **Risque géopolitique** | Indice **GPR** (Caldara & Iacoviello) | matteoiacoviello.com/gpr.htm | Quotidienne |

Les variables du calendrier sont encodées en indicateurs simples : « réunion Fed demain », « jours avant le prochain NFP », etc.

### 14.3 Les deux pièges

1. **Fuite du futur (le plus grave).** On ne doit utiliser que l'information **réellement connue avant la décision** :
   - **Date de publication, pas date de référence** : le CPI de septembre est publié mi-octobre. Il n'est utilisable qu'à partir de sa date de publication.
   - **Révisions** : certaines séries macro sont révisées après coup. Il faut idéalement la valeur d'origine (données *point-in-time*, par exemple **ALFRED**, l'archive des versions de FRED).
   - **Fuseaux horaires** : une nouvelle publiée après la clôture de l'or au jour *t* ne compte qu'au jour *t+1*.
   - Les tests anti-fuite du §10.3 (troncature, permutation du futur) sont étendus à ces nouvelles variables.
2. **Réactions instables.** Le marché ne réagit pas toujours de la même façon à une même nouvelle (une hausse des taux peut faire baisser l'or une fois et le laisser stable une autre). Ces effets sont donc difficiles à apprendre avec peu de données, et le risque de surapprentissage augmente.

### 14.4 Intégration dans l'architecture
- Nouveau module `src/gold_agents/data/external/` : un chargeur par source (`fred.py`, `calendar.py`, `gpr.py`, `sentiment.py`), chacun renvoyant une série **alignée sur la date de disponibilité**.
- Le sentiment FinBERT est **pré-calculé hors ligne** et sauvegardé en Parquet, pour que le backtest reste rapide et déterministe.
- Les nouvelles colonnes s'ajoutent aux features existantes ; le reste de la chaîne (modèle, agents, simulation, blockchain) ne change pas, grâce au découplage (principe P4).
- Option : un **agent « Contexte »** (ou « Macro ») qui fournit un score de régime de marché à l'agent Stratégie ou à l'agent Risque (par exemple réduire l'exposition la veille d'une réunion de la Fed).
- Le hash des données externes est inscrit dans le bloc genèse, comme pour les prix.

### 14.5 Méthode d'évaluation : étude d'ablation
On compare, sur le **même** protocole (mêmes splits, mêmes graines, mêmes frais) :

| Variante | Features |
|---|---|
| V0 | Prix seuls (version de base) |
| V1 | Prix + macro (taux, DXY, inflation, VIX) |
| V2 | V1 + calendrier économique |
| V3 | V2 + sentiment des actualités |
| V4 | V3 + indice GPR |

Pour chaque variante : MAE, RMSE, précision directionnelle, puis profit net, Sharpe et drawdown après frais. On conclut seulement si le gain est **stable sur plusieurs graines et plusieurs périodes** (walk-forward). Si les facteurs n'apportent rien, c'est aussi un résultat à présenter au professeur.

### 14.6 Place dans le planning
Après le jalon S8 (évaluation finale prix seuls), dans le temps restant avant le rapport, ou comme **perspective** dans le rapport si le temps manque. Ordre conseillé : macro → calendrier → GPR → sentiment (le sentiment est le plus coûteux à mettre en place).

---

## 15. Décisions ouvertes

À trancher (et à reporter dans le journal des révisions) :

1. **PyTorch ou Keras ?** Recommandation : PyTorch, sauf si l'équipe ou le cours utilise déjà Keras.
2. **Régression ou classification ?** Recommandation : régression d'abord, classification comme variante comparée.
3. **Vente à découvert autorisée ?** Recommandation : non en v1 (plus simple, plus prudent), oui en variante.
4. **Niveau de blockchain attendu par le professeur** : chaîne locale suffit-elle, ou faut-il une vraie blockchain (Ethereum local / testnet) ? *À demander au professeur.*
5. **Comment le professeur accède au registre** : dépôt GitHub, fichier envoyé, ou URL du dashboard déployé (Streamlit Community Cloud) ?
6. **Valeurs par défaut des frais et du capital** : à confirmer avec le cahier des charges du professeur.
7. **Date de rendu exacte** pour caler le planning.

---

## 16. Journal des révisions

| Date | Version | Auteur | Changement |
|---|---|---|---|
| 2026-10-26 | 1.0 | ayach (avec Claude) | Première version de l'architecture technique. |
| 2026-10-26 | 1.1 | ayach (avec Claude) | Ajout de la section 14 : extension future avec facteurs externes au prix (macro, calendrier, sentiment FinBERT, GPR), pièges et étude d'ablation. |
| 2026-10-28 | 1.2 | ayach (avec Claude) | Phase 4 réalisée : réglage Optuna sur la validation fixe (le walk-forward du §4.3 reste pour la phase 6, à cause du temps de calcul sur CPU), fenêtre W = 30 non réglée, modèles tabulaires sur les features du jour t. |
| 2026-10-29 | 1.3 | ayach (avec Claude) | Phase 5 réalisée : 4 agents + orchestrateur (`gold-agents simulate`) sur la validation. Seuils θ mis à 0 (signe de la prévision) en attendant leur réglage en phase 6 ; filtre de confiance MC Dropout `r̂/σ̂ > k` (k = 0,5) à l'achat seulement ; le drawdown max coupe le trading jusqu'à la fin de la période ; LightGBM, non sauvegardé en phase 4, est réentraîné avec ses réglages Optuna. |
| 2026-10-30 | 1.4 | ayach (avec Claude) | Phase 7 réalisée : blocs v2 (racine de Merkle, preuve de travail de difficulté 3, signature Ed25519 sur le hash d'en-tête), clé privée locale (`gold-agents keygen`), `scripts/verify_ledger.py` autonome, tests de falsification. Pas d'heure réelle dans les blocs (rejeu identique) ; registres v1 toujours vérifiables. |
| 2026-10-31 | 1.5 | ayach (avec Claude) | Bonus C de la phase 7 : `gold-agents anchor` publie sur Sepolia (0 ETH vers soi-même) l'empreinte des derniers hash des registres, journal `ledger/anchors.json` ; `verify_ledger.py --anchors` relit la transaction avec un nœud public, sans web3. |
| 2026-10-31 | 1.6 | ayach (avec Claude) | Phase 9 réalisée : tableau de bord Streamlit + Plotly à 6 pages, lu uniquement depuis les registres (et les résultats facultatifs), vérification, falsification et contrôle des ancrages dans l'interface. |
| 2026-11-01 | 1.7 | ayach (avec Claude) | Tentative 2 du test : le seuil d'achat absolu (θ_buy = 0,10 %) n'a donné aucun trade au système principal, les prévisions du test étant plus petites que celles du walk-forward. θ_buy devient un quantile des 250 prévisions précédentes (`buy_rule: quantile`), choisi par le même walk-forward avec 4 à 60 trades par an exigés dans chaque période. `freeze` et `evaluate --again` archivent la version précédente au lieu de l'écraser ; `ledger/moves.json` garde les ancrages vérifiables. |
