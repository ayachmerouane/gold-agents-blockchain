# Gold Agents Blockchain

Investissement **simulé** sur l'or (XAU/USD) à l'aide de **Deep Learning** (LSTM/GRU), d'un **système multi-agents** (prévision, stratégie, risque, exécution) et d'un **registre de type blockchain** (chaîne de hachage, arbre de Merkle, signatures Ed25519) qui rend chaque décision vérifiable.

> Projet pédagogique : aucun argent réel n'est engagé.

- Le sujet expliqué : [docs/explication_projet.md](docs/explication_projet.md)
- L'architecture technique (outils, modèles, tests, planning) : [docs/architecture_technique.md](docs/architecture_technique.md)

## État d'avancement

Phase 0 (cadrage et socle) : structure du dépôt, configuration validée, CI. Les modules de `src/gold_agents/` sont encore des squelettes.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) (il installe lui-même Python 3.11).

```bash
make install        # dépendances de base + outils de dev (pytest, ruff, mypy)
make install-all    # tout, y compris PyTorch, scikit-learn, MLflow, Streamlit
uv run pre-commit install   # optionnel : lint automatique avant chaque commit
```

## Données

Jeu de données Kaggle « XAUUSD Gold Price Tracker: 2004–Present » (quotidien). Le CSV n'est pas versionné : le placer dans `data/raw/`, puis noter son empreinte :

```bash
sha256sum data/raw/*.csv >> data/raw/CHECKSUMS.txt
```

Le chemin du fichier se règle dans `config/config.yaml` (`data.raw_path`).

## Commandes

| Commande | Rôle |
|---|---|
| `make check` | Lint + typage + tests (ce que fait la CI) |
| `make test` | Tests rapides (`pytest -m "not slow"`) |
| `make cov` | Tests avec couverture |
| `make format` | Formatage automatique (ruff) |
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
