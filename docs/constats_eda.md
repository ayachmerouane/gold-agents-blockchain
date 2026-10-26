# Constats de l'analyse exploratoire (EDA)

Résultats de `gold-agents eda` sur le CSV Kaggle `XAU_1d_data.csv` (5 531 jours, du 2004-06-11 au 2026-01-30) et la mise à jour Twelve Data (190 jours, du 2026-02-02 au 2026-10-23), lancé le 26 octobre 2026. Les chiffres de distribution, d'autocorrélation et de stationnarité portent sur le **train** (2004-2018).

## 1. Qualité des données

**Bonne dans l'ensemble.**

- **Aucune barre OHLC incohérente** (pas de plus haut sous la clôture, pas de prix nul ou manquant) et **aucune barre de week-end**.
- **Jours fériés normaux** : 3 à 4 jours ouvrés absents par an (Noël, Nouvel An, Vendredi saint), un peu plus en 2004-2006 (Thanksgiving, 4 juillet). C'est le comportement attendu d'un marché qui ferme ces jours-là.
- **Volume** toujours renseigné. Avec MetaTrader 4, c'est un volume de *ticks* (nombre de cotations) propre au courtier, pas un volume échangé : on ne s'en servira pas comme feature fiable.

**Deux trous anormaux, tous les deux dans le test :**

| Trou | Jours ouvrés absents | Écart d'ouverture au retour |
|---|---|---|
| 2025-09-12 → 2025-10-15 | 22 (un mois entier) | +13,7 % |
| 2026-01-13 → 2026-01-22 | 6 | +6,3 % |

Ce ne sont pas des jours fériés : c'est l'export Kaggle qui n'a pas enregistré ces jours. Pendant ces trous, l'or a beaucoup monté, si bien que la ligne qui suit chaque trou affiche un « rendement d'un jour » de +13,7 % ou +6,3 % qui résume en réalité plusieurs semaines. Conséquences si on ne fait rien :

- le modèle serait évalué sur une « journée » impossible à prévoir ;
- la volatilité glissante explose artificiellement (le pic de 60,7 % du 2025-11-10 est un artefact de ce trou, pas un vrai krach) ;
- le backtest ne pourrait pas trader pendant un mois.

## 2. Valeurs aberrantes

- **Rendements (18 jours)** : ce sont de vrais chocs de marché, pas des erreurs. Par exemple le 2013-04-15 (-9,5 %, krach de l'or) ou le 2008-09-17 (+10,4 %, faillite de Lehman). **Aucun aller-retour suspect**, c'est-à-dire aucun gros mouvement annulé le lendemain, qui serait la signature d'une erreur de cotation. Décision : **on les garde**. Le modèle doit apprendre que ces jours existent.
- **Écarts d'ouverture (152)** : faux positifs pour l'essentiel. Dans MT4, l'ouverture est presque toujours égale à la clôture de la veille, donc le moindre écart (un lundi après un week-end agité, par exemple) paraît énorme en z-score. Seuls les deux premiers (les trous ci-dessus) comptent.
- **Amplitudes (59)** : ce sont des journées très agitées (2008, 2013, 2020), réelles.

## 3. Comportement des rendements

- **Queues épaisses** : kurtosis en excès de 6,3 (0 pour une loi normale). Les jours au-delà de 4 écarts-types sont **70 fois plus fréquents** que ce que prévoit une loi normale (0,46 % contre 0,006 %). Le test de Jarque-Bera rejette nettement la normalité. → Cela justifie la **perte de Huber** et le **RobustScaler** prévus dans l'architecture : la MSE et le StandardScaler sont trop sensibles à ces jours extrêmes.
- **Asymétrie négative (-0,30)** : les fortes baisses sont un peu plus violentes que les fortes hausses.
- Train : +8,1 % par an en moyenne, volatilité de 18 %, 52 % de jours positifs.

## 4. Prévisibilité

- **Direction peu prévisible** : l'autocorrélation des rendements est quasi nulle (-0,005 au retard 1, dans la bande de bruit ±0,032), et le test de Ljung-Box ne rejette pas l'absence de dépendance (p = 0,13). Le rendement d'hier ne dit presque rien de celui d'aujourd'hui. → La baseline naïve « rendement nul » (B0) sera difficile à battre en RMSE. C'est normal sur un marché, et il faudra le dire honnêtement dans le rapport.
- **Volatilité prévisible** : |r| est fortement autocorrélé (0,12 à 0,17, p < 0,001). Une journée agitée est suivie d'autres journées agitées (*volatility clustering*). → Les features de volatilité (ATR, volatilité 20 j) et la prévision d'incertitude σ̂ ont de bonnes chances d'être utiles, notamment pour l'agent Risque.

## 5. Stationnarité

- **Le prix n'est pas stationnaire** (ADF p = 0,13, KPSS p = 0,01) : son niveau dérive sans revenir vers une moyenne.
- **Les rendements le sont** (ADF p < 0,001, KPSS p = 0,06).
- → Cela confirme le choix de **prédire le rendement et non le prix**. Un modèle entraîné sur des prix de 400 à 1 800 USD ne saurait pas quoi faire à 4 000 USD.

## 6. Découpage et régimes

| Ensemble | Période | Lignes | Volatilité | Buy & Hold |
|---|---|---|---|---|
| Train | 2004-06-11 → 2018-12-31 | 3 731 (67 %) | 18,0 % | +233 % |
| Validation | 2019-01-02 → 2021-12-31 | 775 (14 %), 745 après purge | 15,0 % | +42 % |
| Test | 2022-01-03 → 2026-01-30 | 1 025 (19 %), 995 après purge | 18,4 % | +171 % |

- Le **découpage est sain** : les ensembles sont disjoints et ordonnés, la purge de W = 30 jours est respectée et tous les contrôles anti-fuite passent.
- La **validation est calme** : seulement 15 % de jours en volatilité haute (seuils du train : 12,2 % et 17,4 %), contre 24 % pour le test. Les seuils de stratégie et de risque, réglés sur la validation, seront donc testés sur un marché plus agité. Il faudra le mentionner dans le rapport.
- Le **test est un très fort marché haussier** (+171 % en Buy & Hold). Une stratégie qui sort souvent du marché aura du mal à faire mieux que Buy & Hold en profit brut. La comparaison doit aussi porter sur le risque (Sharpe, drawdown maximal).

## 7. Raccord avec la mise à jour Twelve Data

Entre la dernière clôture Kaggle (2026-01-30) et la première clôture Twelve Data (2026-02-02), on observe -4,7 % (z robuste de -5,6), sans jour manquant. C'est un mouvement fort mais probablement réel : l'or a chuté brutalement fin janvier 2026. Cela n'a pas été vérifié jour par jour. Le contrôle de cohérence de `update-data` (corrélation des rendements sur les jours communs) était passé. Ces jours ne servent pas à l'évaluation.

## 8. Décisions pour `clean.py`

1. Garder les valeurs extrêmes réelles (pas d'écrêtage des rendements). Leur effet est géré par la perte de Huber et le RobustScaler.
2. Ne pas utiliser le volume comme feature (volume de ticks MT4).
3. **Trous 2025-09 et 2026-01** : décision en cours (voir la PR et le fil du projet). Options :
   - (a) les combler avec Twelve Data (même instrument), en marquant ces lignes ;
   - (b) les laisser et invalider les rendements et fenêtres qui les traversent ;
   - (c) arrêter le test au 2025-09-12.
4. Règle générale dans `clean.py` : tout trou de plus de 5 jours ouvrés est signalé, et le rendement qui le traverse n'est jamais traité comme un rendement d'un jour.
