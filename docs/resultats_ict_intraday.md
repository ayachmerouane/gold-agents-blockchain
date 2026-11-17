# Agent ICT intraday (barres d'une heure) : protocole et résultats

Commande : `uv run gold-agents ict-intraday` (ou `make ict-intraday`). Code : `src/gold_agents/data/intraday.py`, `src/gold_agents/simulation/ict_intraday.py`, `src/gold_agents/evaluation/ict_intraday_report.py`.

## Données

`data/raw/intraday/XAU_1h_data.csv` : même dataset Kaggle que le fichier journalier, 125 206 barres d'une heure, du 11/06/2004 au 30/01/2026, SHA-256 `f143104b6ecd5634d82dcf2805711254c4d2e771cae84d742a21543791bca954`.
Les heures du fichier sont celles du serveur MT4 : **New York + 7 h**. Cela a été vérifié de deux façons : chaque semaine rouvre à 18 h (New York), et 98,8 % des clôtures journalières reconstruites sont égales à celles de `XAU_1d_data.csv`. Le jour de cotation commence à 18 h (New York).

## L'agent

Il suit le marché barre par barre et décide pendant la séance (heures de New York, achat seulement) :

1. Range asiatique : plus haut et plus bas de 20 h à minuit.
2. Sweep : après minuit, le prix passe sous le bas asiatique. Le creux devient le stop.
3. MSS : une clôture au-dessus du dernier sommet confirmé avant le sweep.
4. Entrée seulement si la barre suivante est en kill zone (Londres 2 h-5 h, New York 7 h-10 h), par un ordre valable une barre, écrit à la clôture de la barre :
   - `limit_fvg` : achat à cours limité dans le dernier FVG ;
   - `stop_pdh` : achat si la barre suivante casse le plus haut de la barre (réaction confirmée).
5. Objectif : haut asiatique, plus haut de la veille ou R fixe (`fixed_r`).
6. Sortie au stop, à l'objectif ou à 16 h. Si le stop et l'objectif tombent dans la même barre, on prend le pire cas. Jamais de position la nuit.

## Protocole

Grille de 24 réglages : entrée × swing (1, 2) × objectif (3) × R/R (1, 2). Walk-forward sur les périodes de la phase 6 : la période k est jugée avec le réglage choisi sur 1..k-1. Les données de 2022 et après ne sont jamais lues.
Hasard comparable : même nombre d'allers-retours, achat à l'ouverture d'une barre de kill zone tirée au hasard, vente à 16 h.

**Écart au plan, déclaré** : la grille a été élargie une fois, après un premier run (16 réglages, sans `fixed_r`), parce que l'objectif asiatique était presque toujours déjà atteint au moment du MSS (2 à 7 trades par période). Le walk-forward retenu est resté identique d'un run à l'autre.

## Résultats (run du 17/11/2026)

### Walk-forward (hors échantillon)

| Période | Réglage | Sharpe ICT | Sharpe B&H | Rendement ICT | Rendement B&H | Perte max ICT | Perte max B&H | Trades | Gagnants |
|---|---|---|---|---|---|---|---|---|---|
| 2013-2014 | limit_fvg, swing 2, objectif prev_day_high, R/R ≥ 2,00 | -0,64 | -0,86 | -0,7 % | -29,7 % | -0,8 % | -32,5 % | 6 | 33 % |
| 2015-2016 | limit_fvg, swing 1, objectif prev_day_high, R/R ≥ 2,00 | -1,48 | -0,03 | -3,3 % | -3,2 % | -3,3 % | -19,2 % | 14 | 14 % |
| 2017-2018 | stop_pdh, swing 2, objectif prev_day_high, R/R ≥ 2,00 | -0,58 | 0,54 | -0,8 % | +10,3 % | -1,0 % | -13,6 % | 10 | 40 % |
| 2019-2021 | stop_pdh, swing 2, objectif prev_day_high, R/R ≥ 2,00 | -1,10 | 0,83 | -2,4 % | +41,8 % | -2,4 % | -18,5 % | 14 | 14 % |

Moyenne : Sharpe ICT -0,95 contre 0,12 pour Buy & Hold et 0,21 pour SMA 20/50.

### Hasard (même nombre d'allers-retours, entrée en kill zone, sortie en fin de séance)

| Période | Trades | Sharpe ICT | Hasard médian | Hasard 95e centile | Hasard battu |
|---|---|---|---|---|---|
| 2 | 6 | -0,64 | -0,69 | 0,69 | 52 % |
| 3 | 14 | -1,48 | -0,78 | 0,48 | 7 % |
| 4 | 10 | -0,58 | -0,90 | 0,35 | 72 % |
| 5 | 14 | -1,10 | -0,59 | 0,41 | 12 % |

### Sensibilité aux frais (Sharpe moyen sur le walk-forward)

| Frais | ICT | Buy & Hold |
|---|---|---|
| × 0 | -0,32 | 0,12 |
| × 1 | -0,95 | 0,12 |
| × 2 | -1,21 | 0,12 |
| × 5 | -1,36 | 0,10 |

### Réglage choisi sur les 5 périodes (candidat au gel)

stop_pdh, swing 2, objectif prev_day_high, R/R ≥ 2,00

Attention : 24 réglages essayés. Le meilleur Sharpe moyen de la grille (0,03) est optimiste ; seul le walk-forward est hors échantillon.

### Ce que montre toute la grille (moyenne des 5 périodes, déjà vues)

| Réglages | Trades par an | Sharpe moyen | Taux de gain |
|---|---|---|---|
| Objectif à R fixe (8 réglages) | 26 à 85 | -1,85 à -3,16 | 21 à 30 % |
| Plus haut de la veille (8 réglages) | 5 à 29 | -0,97 à -2,36 | 16 à 21 % |
| Haut asiatique (8 réglages) | 0 à 2 | -0,84 à +0,03 | peu de trades |

Borne sur l'hypothèse du pire cas, pour `limit_fvg`, swing 2, objectif 1 R, sur 2011-2021 (292 trades) :

| Hypothèse dans la barre | Sans frais | Avec frais |
|---|---|---|
| Pire cas (stop d'abord) | -4,6 %, Sharpe -0,25, 45 % de gains | -32,8 %, Sharpe -1,84 |
| Meilleur cas (objectif d'abord) | +1,7 %, Sharpe 0,09, 51 % de gains | -28,5 %, Sharpe -1,57 |

## Lecture

- **Sans frais, les setups ICT en 1 h n'ont pas d'avantage** : avec un objectif de 1 R, ils gagnent 45 à 51 % du temps, soit le niveau d'une pièce de monnaie.
- **Avec des frais réalistes, ils perdent régulièrement.** Un risque de quelques dixièmes de pour cent par trade ne couvre pas 0,24 % de frais et de glissement par aller-retour. Plus l'agent trade, plus il perd.
- **Hors échantillon, le réglage choisi a un Sharpe moyen de -0,95**, contre 0,12 pour Buy & Hold. Il ne bat jamais le hasard au seuil de 95 %.
- Le seul avantage est le risque : l'agent n'est jamais investi la nuit, et sa perte maximale reste sous 3,5 %.

**Conclusion** : sur 2011-2021, les règles ICT intraday (range asiatique, sweep, MSS, FVG, kill zones) ne montrent aucun avantage mesurable sur l'or, même en données horaires et avec une exécution favorable. Le test 2022-2026 n'a pas été lu.
