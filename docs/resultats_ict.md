# Stratégie ICT sur l'or : règles, protocole et résultats

Commande : `uv run gold-agents ict` (ou `make ict`). Code : `src/gold_agents/simulation/ict.py`, `src/gold_agents/evaluation/ict_report.py`.

## La stratégie

Achat seulement, sur données journalières. Chaque soir, avec les bougies jusqu'au jour même :

1. **Sweep de liquidité** : dans les derniers jours, une bougie est passée sous le dernier creux confirmé puis a clôturé au-dessus.
2. **Changement de structure (MSS)** : depuis, une clôture au-dessus du dernier sommet confirmé.
3. Selon l'entrée :
   - `limit_fvg` : un **Fair Value Gap** haussier, né après le sweep et pas comblé. Ordre d'achat à cours limité au haut du FVG.
   - `stop_pdh` : clôture en zone discount. Ordre d'achat si le lendemain **casse le plus haut du jour** (la réaction après l'ouverture confirme).
4. **Stop** sous le creux du sweep, **objectif** au plus haut du range. Pas d'ordre si l'objectif est inférieur à `min_reward_risk` × risque. Sortie au plus tard après 20 jours.

Un sommet (creux) n'est connu que `swing` jours après : la stratégie ne le voit pas avant. L'ordre est écrit **avant l'ouverture** et vaut un jour. Le lendemain, le plus haut et le plus bas disent s'il est exécuté, puis si le stop ou l'objectif est touché. Quand les deux le sont le même jour, on prend le **pire cas** (le stop). Frais 0,1 % et slippage 0,02 % par ordre.

## Protocole

Même protocole que la phase 6. Grille de 72 réglages (2 entrées × swing 1-3 × sweep ≤ 10 ou 20 jours × discount 0,5 ou sans filtre × R/R 0,5-2). La période k est jugée avec le réglage choisi sur les périodes 1 à k-1 (meilleur Sharpe moyen). Le test 2022-2026 n'est jamais lu : les données sont coupées avant 2022.

## Résultats (run du 17/11/2026)

### Walk-forward (hors échantillon)

| Période | Réglage | Sharpe ICT | Sharpe B&H | Rendement ICT | Rendement B&H | Perte max ICT | Perte max B&H | Trades | Investi |
|---|---|---|---|---|---|---|---|---|---|
| 2013-2014 | stop_pdh, swing 1, sweep ≤ 10 j, discount 1,00, R/R ≥ 0,50 | -0,57 | -0,86 | -3,7 % | -29,7 % | -7,3 % | -32,5 % | 8 | 4 % |
| 2015-2016 | stop_pdh, swing 1, sweep ≤ 10 j, discount 1,00, R/R ≥ 0,50 | 0,14 | -0,03 | +0,9 % | -3,2 % | -4,4 % | -19,2 % | 10 | 9 % |
| 2017-2018 | stop_pdh, swing 1, sweep ≤ 10 j, discount 1,00, R/R ≥ 0,50 | -0,52 | 0,54 | -2,5 % | +10,3 % | -2,8 % | -13,6 % | 8 | 7 % |
| 2019-2021 | stop_pdh, swing 1, sweep ≤ 10 j, discount 1,00, R/R ≥ 0,50 | 0,10 | 0,84 | +0,6 % | +41,8 % | -3,5 % | -18,5 % | 10 | 2 % |

Moyenne : Sharpe ICT -0,21 contre 0,12 pour Buy & Hold et 0,21 pour SMA 20/50.

### Hasard (même nombre de trades, exécution à l'ouverture)

| Période | Trades | Sharpe ICT | Hasard médian | Hasard 95e centile | Hasard battu |
|---|---|---|---|---|---|
| 2 | 8 | -0,57 | -0,54 | 0,27 | 48 % |
| 3 | 10 | 0,14 | -0,10 | 0,80 | 68 % |
| 4 | 8 | -0,52 | 0,12 | 0,86 | 14 % |
| 5 | 10 | 0,10 | 0,53 | 1,04 | 16 % |

### Sensibilité aux frais (Sharpe moyen sur le walk-forward)

| Frais | ICT | Buy & Hold |
|---|---|---|
| × 0 | -0,04 | 0,12 |
| × 1 | -0,21 | 0,12 |
| × 2 | -0,38 | 0,12 |
| × 5 | -0,81 | 0,10 |

### Réglage choisi sur les 5 périodes (pour un futur gel)

stop_pdh, swing 1, sweep ≤ 20 j, discount 1,00, R/R ≥ 0,50

Attention : 72 réglages essayés. Le meilleur Sharpe moyen de la grille (0,35) est optimiste ; seul le walk-forward ci-dessus est hors échantillon.

## Lecture

- **Les setups ICT sont rares sur l'or en journalier** : 8 à 10 trades (achats et ventes) par période de 2-3 ans, investi 2 à 9 % du temps. Les réglages retenus atteignent tout juste le minimum de 4 trades par an de la phase 6, soit 2 allers-retours.
- **Hors échantillon, l'ICT ne bat ni Buy & Hold ni le hasard** : Sharpe moyen -0,21, contre 0,12 pour Buy & Hold et 0,21 pour SMA 20/50. Il bat le hasard dans 14 à 68 % des tirages, jamais au seuil de 95 %.
- **Il protège surtout en restant dehors** : -3,7 % en 2013-2014 quand l'or perd 30 %, mais il rate toute la hausse de 2019-2021 (+0,6 % contre +41,8 %).
- **Les frais pèsent lourd** : le Sharpe passe de -0,04 sans frais à -0,81 avec des frais × 5.
- Le meilleur réglage de la grille, sur les 5 périodes déjà vues, a un Sharpe moyen de 0,35. Il est optimiste (72 essais) : seul le walk-forward est hors échantillon.

**Conclusion** : sur l'or en données journalières, les règles ICT traduites objectivement n'apportent pas d'avantage mesurable. Les concepts de séance (kill zones, ouverture de Londres ou New York) demandent des données intraday ; c'est la suite logique si l'on veut tester l'ICT dans sa forme d'origine. Le test 2022-2026 n'a pas été utilisé et reste réservé ; la simulation en direct ne trade pas cette stratégie.
