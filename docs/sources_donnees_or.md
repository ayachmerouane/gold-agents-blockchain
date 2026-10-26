# Sources de données XAU/USD : comparaison et recommandation

*Recherche du 26/10/2026. Les accès réseau depuis l'environnement de travail étant bloqués, les sources n'ont pas pu être testées en direct ; les points marqués (à vérifier) seront confirmés lors de l'implémentation.*

## 1. Le dataset brut de la spec

**Kaggle « XAUUSD Gold Price Tracker: 2004–Present »** — auteur Novandra Anugrah  
- Page : https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004present  
- Une variante du même auteur existe : https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024 (titre actuel « 2004-2026 », plusieurs unités de temps : 1 min, 5 min, 15 min, 1 h, 1 j…).  
- Fichier quotidien : `XAU_1d_data.csv` (c'est déjà le nom attendu dans `config/config.yaml`).  
- Colonnes : `Date, Open, High, Low, Close, Volume`, dates au format `YYYY.MM.DD HH:MM` (format MetaTrader). Séparateur probablement `;` (à vérifier).  
- Provenance : export MetaTrader 4 via des scripts MQL4 de l'auteur (https://github.com/nvn01/MQL4-Script-fetching-XAUUSD-price-history). Mise à jour **manuelle et irrégulière** par l'auteur. `Volume` = tick volume du courtier, pas un vrai volume.  
- Un miroir Hugging Face existe (https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025), arrêté à octobre 2025 : utile en secours seulement.

Téléchargement reproductible : `make download` (jeton `KGAT_…` dans la variable `KAGGLE_API_TOKEN` ou le fichier `~/.kaggle/access_token`, jamais dans le dépôt), qui note aussi l'empreinte SHA-256 dans `data/raw/CHECKSUMS.txt`.

## 2. Comparaison des sources pour la mise à jour

| Source | Instrument | Historique | OHLC quotidien | Clé / coût | Fiabilité | Cohérence avec Kaggle |
|---|---|---|---|---|---|---|
| **Kaggle API** (même dataset) | XAU/USD spot (courtier MT4) | 2004 → dernière mise à jour de l'auteur | Oui + tick volume | Jeton Kaggle gratuit | Officielle, mais rythme de mise à jour dépendant de l'auteur | **Parfaite** (même série) |
| **Twelve Data** `XAU/USD` | Spot agrégé | Plusieurs décennies, 5 000 barres par appel | Oui (volume souvent absent) | Clé gratuite, ~800 crédits/jour, 8 req/min | API officielle, stable, documentée | Bonne : même instrument, écarts de quelques $ dus au courtier |
| **yfinance** `GC=F` | **Contrat à terme** COMEX (front month) | ~2000 → aujourd'hui | Oui + vrai volume | Aucune clé | Non officiel (scraping Yahoo), casse régulièrement | Moyenne : prime de portage (contango) et sauts au roulement des contrats |
| **Alpha Vantage** | Spot via FX / endpoints métaux | Variable | Partiel (à vérifier pour XAU) | Clé gratuite, **25 req/jour** | Officielle mais offre gratuite très limitée | Bonne si XAU disponible |
| **Stooq** `xauusd` | Spot | Long (années 1990+) | Oui | **Clé requise depuis mars 2026** | Gratuit mais conditions changeantes | Bonne |
| **FRED / LBMA** | Fixing LBMA (1 prix/jour) | — | **Non** (pas d'OHLC) | — | Séries LBMA **retirées de FRED en 2022** (licence ICE) | Faible (fixing ≠ clôture) |

## 3. Recommandation

1. **Base historique = Kaggle**, conformément à la spec, téléchargée via l'API Kaggle, figée et signée par son SHA-256. Les jeux **train / validation / test** sont construits **uniquement** sur ce fichier figé : c'est ce qui garantit la reproductibilité et l'absence de mélange de sources dans l'évaluation.
2. **Mise à jour = Twelve Data `XAU/USD`** (clé gratuite lue dans la variable d'environnement `TWELVEDATA_API_KEY`, jamais commitée). Le script ajoute seulement les jours postérieurs à la dernière date Kaggle, dans un fichier séparé (`data/raw/XAU_1d_update.csv`), avec une colonne `source`.
3. **Contrôle de cohérence automatique** avant toute fusion : sur la période commune, comparer clôtures et rendements (écart moyen absolu, corrélation des rendements). Si l'écart dépasse un seuil, le script refuse de fusionner et le signale.
4. **Secours sans clé : yfinance `GC=F`**, uniquement pour le tableau de bord ou une démo, en signalant qu'il s'agit de futures (prix légèrement différents du spot).

Pourquoi pas yfinance en principal : non officiel et ce n'est pas le même instrument (futures vs spot), ce qui biaise les rendements autour des roulements de contrats. Pourquoi pas Stooq/Alpha Vantage : clé désormais nécessaire pour Stooq, quotas trop faibles pour Alpha Vantage, sans avantage sur Twelve Data.

## 4. Ce qu'il faudra implémenter (si validé)

- `gold_agents.data.load` : lecture du CSV Kaggle (format MT4, UTC, tri, typage).
- `gold_agents.data.sources` : `fetch_kaggle()`, `fetch_twelvedata(start, end)`, `fetch_yfinance(start, end)` (optionnel), toutes renvoyant le même schéma `Date, Open, High, Low, Close, Volume, source`.
- `gold_agents.data.update` : ajout incrémental + contrôle de cohérence + mise à jour de `CHECKSUMS.txt`.
- Commande CLI `gold-agents data download` / `data update` et cibles `make data` / `make update-data`.
- Tests pytest sans réseau (réponses API simulées, CSV d'exemple au format MT4).
