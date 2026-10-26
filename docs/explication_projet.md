# Investissement financier simulé avec agents autonomes et blockchain
## Explication complète du projet, de A à Z

> **But de ce document.** Comprendre parfaitement le sujet *avant* d'écrire la moindre ligne de code : quoi construire, pourquoi, dans quel ordre, et quels pièges éviter.
> Aucun code n'est implémenté ici : c'est une feuille de route explicative.

---

## Sommaire

1. [Le projet en une phrase](#1-le-projet-en-une-phrase)
2. [Contexte et problématique](#2-contexte-et-problématique)
3. [Vue d'ensemble : le pipeline complet](#3-vue-densemble--le-pipeline-complet)
4. [Brique 1 : les données (XAU/USD)](#4-brique-1--les-données-xauusd)
5. [Brique 2 : prétraitement et ingénierie des caractéristiques](#5-brique-2--prétraitement-et-ingénierie-des-caractéristiques)
6. [Brique 3 : le modèle de Deep Learning (LSTM / GRU)](#6-brique-3--le-modèle-de-deep-learning-lstm--gru)
7. [Brique 4 : les quatre agents autonomes](#7-brique-4--les-quatre-agents-autonomes)
8. [Brique 5 : le moteur de simulation financière](#8-brique-5--le-moteur-de-simulation-financière)
9. [Brique 6 : la blockchain partagée avec le professeur](#9-brique-6--la-blockchain-partagée-avec-le-professeur)
10. [Brique 7 : évaluation et tableau de bord](#10-brique-7--évaluation-et-tableau-de-bord)
11. [Le piège n°1 : la fuite d'information du futur](#11-le-piège-n1--la-fuite-dinformation-du-futur)
12. [Livrables attendus](#12-livrables-attendus)
13. [Structure de projet Python proposée](#13-structure-de-projet-python-proposée)
14. [Plan de travail étape par étape](#14-plan-de-travail-étape-par-étape)
15. [Questions que le professeur risque de poser](#15-questions-que-le-professeur-risque-de-poser)
16. [Glossaire](#16-glossaire)

---

## 1. Le projet en une phrase

On construit un **investisseur virtuel automatique sur l'or** : un réseau de neurones prédit l'évolution du prix, quatre agents logiciels transforment cette prédiction en décisions (acheter, vendre, attendre), un moteur simule le portefeuille, et chaque décision est inscrite dans une **blockchain** que le professeur peut vérifier.

**Aucun argent réel n'est engagé.** Tout se passe sur des données historiques.

---

## 2. Contexte et problématique

### Pourquoi ce projet ?
Il combine quatre domaines en un seul système :

| Domaine | Rôle dans le projet |
|---|---|
| **Deep Learning** | Prévoir le rendement futur de l'or |
| **Systèmes multi-agents** | Découper la prise de décision en rôles spécialisés |
| **Finance quantitative** | Simuler un portefeuille, gérer le risque, mesurer la performance |
| **Blockchain** | Garantir que l'historique des décisions n'a pas été modifié |

### La problématique officielle
> *Comment concevoir un système multi-agents intelligent capable d'exploiter les prévisions du Deep Learning pour maximiser le profit d'un portefeuille simulé, tout en assurant la traçabilité et la vérifiabilité des décisions ?*

On peut la découper en trois sous-questions :
1. **Prévoir** : le modèle prédit-il mieux que le hasard ou qu'une règle naïve ?
2. **Décider** : ces prévisions, une fois transformées en actions, rapportent-elles plus qu'une stratégie simple, *après frais* ?
3. **Prouver** : peut-on démontrer que les décisions enregistrées n'ont pas été retouchées après coup ?

---

## 3. Vue d'ensemble : le pipeline complet

```
┌──────────────────────┐
│ 1. Données XAU/USD   │  Prix quotidiens Open/High/Low/Close (Kaggle, 2004 → aujourd'hui)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Prétraitement     │  Nettoyage, rendements, indicateurs, normalisation, fenêtres
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Modèle LSTM/GRU   │  Entrée : 30 derniers jours → Sortie : rendement prévu de demain
└──────────┬───────────┘
           ▼
┌───────────────────────────────────────────────────────────────┐
│ 4. Agents (à chaque jour simulé)                              │
│   Prévision ──► Stratégie ──► Risque ──► Exécution            │
└──────────┬──────────────────────────────────────┬─────────────┘
           ▼                                      ▼
┌──────────────────────┐              ┌──────────────────────┐
│ 5. Moteur simulation │              │ 6. Blockchain        │
│ cash, positions,     │              │ journal horodaté,    │
│ frais, valeur        │              │ hachages chaînés     │
└──────────┬───────────┘              └──────────┬───────────┘
           └──────────────┬──────────────────────┘
                          ▼
              ┌──────────────────────┐
              │ 7. Tableau de bord   │  Profit, drawdown, Sharpe, décisions, vérif. chaîne
              └──────────────────────┘
```

**Idée clé :** chaque brique a une entrée et une sortie claires. On peut donc les développer et les tester une par une.

---

## 4. Brique 1 : les données (XAU/USD)

### L'actif : l'or (XAU/USD)
- **XAU** = code ISO de l'once d'or ; **USD** = dollar américain.
- XAU/USD = **prix d'une once d'or en dollars**.
- Fréquence choisie : **quotidienne** (une ligne par jour de cotation).

### La source : Kaggle « XAUUSD Gold Price Tracker: 2004–Present »
- Données historiques quotidiennes (et horaires) depuis juin 2004.
- Provenance annoncée : MetaTrader 4 (plateforme de trading).
- Colonnes attendues : `Date`, `Open`, `High`, `Low`, `Close`, et éventuellement `Volume`.

| Colonne | Signification |
|---|---|
| **Open** | Prix à l'ouverture de la journée |
| **High** | Prix le plus haut de la journée |
| **Low** | Prix le plus bas de la journée |
| **Close** | Prix à la clôture : **c'est la colonne principale** |
| **Volume** | Quantité échangée. Attention : sur le Forex/l'or via MT4, c'est souvent un *tick volume* (nombre de variations de prix), pas un vrai volume. À utiliser avec prudence. |

### Ce qu'il faut vérifier dans l'analyse exploratoire (EDA)
- Dates manquantes (week-ends normaux, mais trous anormaux ?).
- Doublons, valeurs nulles, valeurs aberrantes (prix à 0, sauts impossibles).
- Cohérence : `Low ≤ Open, Close ≤ High`.
- Graphiques : courbe du prix, histogramme des rendements, volatilité dans le temps.
- Grandes périodes à repérer : crise 2008, pic 2011, baisse 2013, COVID 2020, forte hausse 2023–2025.

---

## 5. Brique 2 : prétraitement et ingénierie des caractéristiques

### 5.1 Pourquoi prédire le **rendement** plutôt que le **prix** ?
Le prix de l'or passe d'environ 400 $ (2004) à plusieurs milliers de dollars aujourd'hui. Un modèle entraîné sur des prix bas ne sait pas gérer des prix jamais vus. Le **rendement** (variation relative) reste dans une plage stable, quel que soit le niveau du prix.

- **Rendement simple** : `r_t = (P_t − P_{t−1}) / P_{t−1}`
- **Log-rendement** : `r_t = ln(P_t / P_{t−1})` (souvent préféré : additif dans le temps)

Exemple : l'or passe de 2000 $ à 2020 $ → rendement = +1 %.

### 5.2 Caractéristiques (features) possibles
| Famille | Exemples | Ce que ça capture |
|---|---|---|
| Rendements passés | r(t−1), r(t−2), … r(t−5) | Mémoire à court terme |
| Moyennes mobiles | SMA 10/20/50, EMA | Tendance |
| Momentum | RSI (14), MACD | Sur-achat / sur-vente, accélération |
| Volatilité | écart-type glissant sur 20 jours, ATR, bandes de Bollinger | Niveau de risque du marché |
| Forme de la bougie | (High−Low)/Close, (Close−Open)/Open | Amplitude intra-journalière |
| Calendrier (optionnel) | jour de la semaine, mois | Effets saisonniers |

Conseil : commencer **simple** (rendements + 3 ou 4 indicateurs), puis ajouter si le gain est prouvé.

### 5.3 Normalisation
Les réseaux de neurones apprennent mal si les variables ont des échelles très différentes (RSI entre 0 et 100, rendement autour de 0,01).
- On utilise un `StandardScaler` (moyenne 0, écart-type 1) ou un `MinMaxScaler`.
- **Règle d'or : le scaler est « appris » (fit) uniquement sur l'ensemble d'entraînement**, puis appliqué tel quel à la validation et au test. Sinon, on injecte de l'information du futur.

### 5.4 Fenêtres temporelles (sliding windows)
Un LSTM a besoin de séquences. On construit des échantillons comme ceci :

```
Entrée X : features des jours [t−29, …, t]      (fenêtre de 30 jours)
Cible  y : rendement du jour t+1                (horizon = 1 jour)
```

Forme du tenseur d'entrée : `(nombre_échantillons, taille_fenêtre, nombre_features)`, par exemple `(4000, 30, 8)`.

### 5.5 Découpage chronologique
**Jamais de mélange aléatoire** (pas de `train_test_split(shuffle=True)`). On coupe dans l'ordre du temps :

```
2004 ─────────────── 2018 │ 2019 ──── 2022 │ 2023 ──── aujourd'hui
      ENTRAÎNEMENT        │   VALIDATION   │       TEST
  (apprendre les poids)   │ (régler les    │ (évaluation finale,
                          │ hyperparamètres)│  touché une seule fois)
```

Les dates exactes sont un choix à justifier. Option avancée : *walk-forward validation* (on ré-entraîne périodiquement en avançant la fenêtre dans le temps).

---

## 6. Brique 3 : le modèle de Deep Learning (LSTM / GRU)

### 6.1 Pourquoi un réseau récurrent ?
Un réseau classique (dense) voit ses entrées comme un sac de nombres sans ordre. Un **réseau récurrent** lit la séquence jour après jour et garde une **mémoire** (état caché) de ce qu'il a vu.

- **LSTM** (*Long Short-Term Memory*) : possède des « portes » (oubli, entrée, sortie) qui décident quoi retenir ou oublier. Gère bien les dépendances longues.
- **GRU** (*Gated Recurrent Unit*) : version simplifiée du LSTM (2 portes au lieu de 3). Moins de paramètres, s'entraîne plus vite, performances souvent comparables.

Bonne pratique pour le rapport : **entraîner les deux et comparer**.

### 6.2 Architecture type (à titre indicatif)
```
Entrée (30 jours × N features)
   → LSTM/GRU (64 unités)
   → Dropout (0,2)
   → LSTM/GRU (32 unités)
   → Dense (1)  → rendement prévu de demain
```

### 6.3 Entraînement
- **Fonction de perte** : MSE (erreur quadratique moyenne) ou Huber (plus robuste aux valeurs extrêmes).
- **Optimiseur** : Adam.
- **Early stopping** sur la perte de validation pour éviter le sur-apprentissage.
- Fixer la **graine aléatoire** (seed) pour que les résultats soient reproductibles.

### 6.4 Variante : classification
Au lieu de prédire la valeur du rendement (régression), on peut prédire la **direction** (hausse / baisse / neutre). Plus simple à transformer en décision, mais on perd l'information d'amplitude. Le sujet parle de « prévision du rendement » : la régression est le choix principal, la classification peut être une comparaison.

### 6.5 L'incertitude (optionnel mais valorisant)
Le sujet mentionne « éventuellement son incertitude ». Méthode simple : **Monte Carlo Dropout**. On garde le dropout actif lors de la prédiction, on prédit 50 fois, et on obtient une moyenne (la prévision) et un écart-type (l'incertitude). L'agent stratégie peut alors ignorer les prévisions trop incertaines.

### 6.6 Évaluer le modèle seul
| Métrique | Formule (idée) | Lecture |
|---|---|---|
| **MAE** | moyenne de \|prévu − réel\| | Erreur moyenne, en unités de rendement |
| **RMSE** | racine de la moyenne de (prévu − réel)² | Pénalise davantage les grosses erreurs |
| **Précision directionnelle** | % de jours où le signe est bon | La métrique qui compte vraiment pour trader |

**Point honnête à mettre dans le rapport :** les rendements financiers sont très bruités. Un modèle qui prédit toujours « 0 » obtient souvent un MAE très proche de celui d'un LSTM. Il faut donc **toujours comparer à une prévision naïve** (rendement nul, ou rendement d'hier). Une précision directionnelle de 52–55 % est déjà un résultat sérieux.

---

## 7. Brique 4 : les quatre agents autonomes

### 7.1 C'est quoi un « agent » ici ?
Un agent est un **module logiciel autonome avec un rôle unique**, qui reçoit une information, applique sa logique, et transmet un résultat au suivant. Concrètement en Python : une classe avec une méthode du type `agir(état) → résultat`.

Il ne s'agit **pas** d'agents conversationnels type ChatGPT : ce sont des composants déterministes qui s'enchaînent automatiquement.

### 7.2 Le relais quotidien

```
Jour t (après la clôture)
  │
  ├─► Agent 1 – PRÉVISION
  │     Entrée  : fenêtre des 30 derniers jours (jusqu'à t inclus)
  │     Sortie  : rendement prévu pour t+1 (+ incertitude éventuelle)
  │
  ├─► Agent 2 – STRATÉGIE
  │     Entrée  : prévision
  │     Sortie  : signal ACHAT / VENTE / CONSERVER (+ force du signal)
  │
  ├─► Agent 3 – RISQUE
  │     Entrée  : signal + état du portefeuille
  │     Sortie  : ordre AUTORISÉ (éventuellement réduit) ou REFUSÉ, avec raison
  │
  └─► Agent 4 – EXÉCUTION & TRAÇABILITÉ
        Entrée  : ordre autorisé
        Action  : exécute virtuellement, applique les frais, met à jour le portefeuille
        Sortie  : événement envoyé à la blockchain
```

### 7.3 Agent 1 – Prévision
- Charge le modèle entraîné (il ne s'entraîne **pas** pendant la simulation, sauf en walk-forward).
- Ne reçoit **que** les données disponibles au jour t.

### 7.4 Agent 2 – Stratégie
Règles de décision prédéfinies, par exemple avec un **seuil** :

| Condition | Signal |
|---|---|
| rendement prévu > +0,3 % | ACHAT |
| rendement prévu < −0,3 % | VENTE |
| entre les deux | CONSERVER |

Pourquoi un seuil ? Parce que chaque transaction coûte des frais. Si la hausse prévue est plus petite que les frais, acheter fait perdre de l'argent. Le seuil se règle **sur la validation**, jamais sur le test.

Variantes possibles : seuil dépendant de l'incertitude, taille de position proportionnelle à la confiance.

### 7.5 Agent 3 – Gestion du risque
C'est le « garde-fou ». Exemples de règles :
- **Trésorerie** : impossible d'acheter plus que le cash disponible.
- **Taille de position** : pas plus de X % du portefeuille par opération.
- **Exposition maximale** : pas plus de Y % du portefeuille investi en or.
- **Stop-loss** : si une position perd plus de Z %, on vend.
- **Perte maximale globale** : si le portefeuille a perdu plus de W % depuis son sommet, on arrête de trader (mode sécurité).
- **Vente à découvert** : autorisée ou non ? (Plus simple : **non**, on ne peut vendre que ce qu'on possède.)

Chaque refus doit être **motivé** et enregistré (c'est une décision aussi !).

### 7.6 Agent 4 – Exécution et traçabilité
- Exécute l'ordre à un prix réaliste (voir section 8).
- Calcule et déduit les frais.
- Met à jour cash, quantité d'or détenue, valeur totale.
- Construit l'événement et l'ajoute à la blockchain.

---

## 8. Brique 5 : le moteur de simulation financière

### 8.1 L'état du portefeuille
```
cash            : argent disponible (ex. 10 000 $ au départ = V_0)
quantité_or     : onces détenues
valeur_totale   = cash + quantité_or × prix_actuel
```

### 8.2 Les règles d'exécution (très important)
- **Décision au jour t, exécution au jour t+1** (par exemple au prix d'ouverture de t+1). On ne peut pas décider avec la clôture de t *et* acheter à cette même clôture : ce serait légèrement optimiste.
- **Frais de transaction** : par exemple 0,1 % du montant échangé à chaque achat ou vente.
- **Spread / slippage** (optionnel) : on achète un peu plus cher et on vend un peu moins cher que le prix affiché.

### 8.3 Exemple chiffré
- V_0 = 10 000 $, prix de l'or = 2 000 $/once, frais 0,1 %.
- Signal ACHAT, l'agent risque autorise 50 % du portefeuille → 5 000 $.
- Frais = 5 $ ; or acheté = 4 995 / 2 000 = 2,4975 onces.
- Cash restant = 5 000 $.
- Le lendemain l'or vaut 2 040 $ → valeur = 5 000 + 2,4975 × 2 040 = 10 094,90 $.

### 8.4 Fonction objectif
```
Profit net = V_T − V_0 − Coûts
```
Remarque : si les frais sont déjà retirés du cash au fil de la simulation, V_T est déjà « net » ; il faut simplement ne pas les compter deux fois. L'important est de **rapporter séparément** le total des frais payés, pour montrer leur impact.

### 8.5 Les stratégies de référence (baselines)
Le système doit être comparé à :
1. **Buy and Hold** : on achète tout le premier jour du test et on ne touche plus à rien. Sur l'or récent (forte hausse), c'est une référence **très difficile à battre**.
2. **Stratégie technique simple** : par exemple croisement de moyennes mobiles (acheter quand SMA 20 > SMA 50, vendre sinon).
3. (Optionnel) **Stratégie aléatoire** : pour montrer que le modèle fait mieux que le hasard.

Toutes les stratégies doivent tourner dans **le même moteur**, avec **les mêmes frais** et **la même période**.

---

## 9. Brique 6 : la blockchain partagée avec le professeur

### 9.1 Le principe en une image
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Bloc 0      │     │ Bloc 1      │     │ Bloc 2      │
│ (genèse)    │     │ données     │     │ données     │
│ hash: a1b2… │◄────│ prev: a1b2… │◄────│ prev: 9f3c… │
│             │     │ hash: 9f3c… │     │ hash: 7e0d… │
└─────────────┘     └─────────────┘     └─────────────┘
```
Chaque bloc contient l'**empreinte (hash) du bloc précédent**. Si on modifie une donnée dans le bloc 1, son hash change, il ne correspond plus au `prev` du bloc 2 : **la chaîne est cassée et la fraude est détectée**.

### 9.2 Contenu d'un bloc (exigé par le sujet)
| Champ | Exemple |
|---|---|
| Identifiant du bloc (index) | 152 |
| Horodatage | 2024-03-14T22:00:00Z |
| Date simulée | 2024-03-14 |
| Identifiant de l'agent | `agent_execution` |
| Prévision utilisée | +0,42 % |
| Signal généré | ACHAT |
| Action simulée et quantité | ACHAT 1,25 once |
| Valeur portefeuille avant / après | 10 512,30 $ / 10 507,10 $ |
| Hash du bloc précédent | `9f3c…` |
| Hash de ce bloc (SHA-256) | `7e0d…` |

### 9.3 Mécanisme de validation
- **Hachage** : SHA-256 (module `hashlib` de Python) sur le contenu du bloc sérialisé de façon stable (JSON avec clés triées).
- **Vérification** : une fonction parcourt toute la chaîne et contrôle pour chaque bloc que (1) son hash recalculé est correct et (2) son `prev_hash` égale le hash du bloc précédent.
- **Preuve de travail** (optionnelle, pédagogique) : exiger que le hash commence par quelques zéros. Ça rend la falsification coûteuse, mais ce n'est pas indispensable pour ce projet.

### 9.4 Partage avec le professeur
Options, de la plus simple à la plus avancée :
1. **Fichier JSON / base SQLite** de la chaîne + script `verifier_chaine.py` que le professeur exécute lui-même.
2. **Petite API ou page web en lecture seule** (Flask/FastAPI ou dans le tableau de bord Streamlit) qui affiche les blocs et le résultat de la vérification.
3. **Ancrage** (bonus) : publier régulièrement le hash du dernier bloc à un endroit horodaté externe (email au professeur, commit Git, ou testnet Ethereum). Le professeur peut alors prouver que la chaîne n'a pas été entièrement recalculée après coup.
4. **Blockchain réelle de test** (Ganache, testnet Ethereum + smart contract) : impressionnant mais plus long. À envisager seulement si le temps le permet.

### 9.5 Ce que la blockchain garantit… et ne garantit pas
- ✅ **Intégrité** : détection de toute modification d'un bloc passé.
- ✅ **Traçabilité / auditabilité** : on sait qui a décidé quoi, quand, sur quelle base.
- ❌ Elle ne garantit **pas** que les prévisions sont justes.
- ❌ Elle ne garantit **pas** que la stratégie est rentable.
- ⚠️ Une chaîne purement locale peut être entièrement recalculée par son propriétaire : d'où l'intérêt de l'ancrage externe ou de la remise régulière du dernier hash au professeur.

---

## 10. Brique 7 : évaluation et tableau de bord

### 10.1 Indicateurs financiers
| Indicateur | Définition simple | Bon signe |
|---|---|---|
| **Profit net** | V_T − V_0 (après frais) | > 0 et > baselines |
| **Rendement total** | (V_T / V_0) − 1 | Élevé |
| **Rendement annualisé** | (V_T / V_0)^(252/nb_jours) − 1 | Comparable entre périodes |
| **Maximum Drawdown** | Pire chute depuis un sommet (ex. −18 %) | Le plus petit possible |
| **Ratio de Sharpe** | moyenne(rendements quotidiens) / écart-type × √252 | > 1 est bon, > 2 excellent |
| **Nombre de transactions** | Total des achats et ventes | Ni 0, ni des milliers |
| **Taux de réussite** | % d'opérations clôturées gagnantes | Plutôt > 50 % (mais à lire avec le gain moyen) |
| **Total des frais** | Somme payée | Montre l'impact des coûts |

*252 = nombre approximatif de jours de cotation par an.*

### 10.2 Indicateurs du modèle
MAE, RMSE, précision directionnelle, comparés à la prévision naïve.

### 10.3 Le tableau de bord (ex. Streamlit ou Dash)
Pages ou sections suggérées :
1. **Prix et prévisions** : courbe réelle vs prévue sur la période de test.
2. **Opérations** : prix avec marqueurs ▲ achat / ▼ vente.
3. **Portefeuille** : courbe de valeur du système vs Buy and Hold vs stratégie technique.
4. **Drawdown** : courbe sous l'eau.
5. **Tableau des métriques** : une ligne par stratégie.
6. **Blockchain** : liste des blocs + bouton « Vérifier l'intégrité » (✅ chaîne valide / ❌ bloc n°X altéré).

---

## 11. Le piège n°1 : la fuite d'information du futur

En anglais : *look-ahead bias* ou *data leakage*. C'est l'erreur la plus fréquente et celle que le professeur va chercher. Si elle existe, les profits affichés sont faux.

| Erreur | Pourquoi c'est une fuite | Correction |
|---|---|---|
| Mélanger aléatoirement les données | Le modèle apprend sur 2024 pour prédire 2019 | Découpage chronologique strict |
| Normaliser sur tout le jeu de données | La moyenne/écart-type contiennent le futur | `fit` du scaler sur l'entraînement seulement |
| Indicateur centré (ex. moyenne mobile qui inclut t+1) | Utilise des prix pas encore connus | N'utiliser que des données ≤ t |
| Décider avec la clôture de t et exécuter à cette clôture | Optimiste : on ne peut pas agir sur un prix déjà fixé | Exécuter à l'ouverture de t+1 |
| Régler le seuil ou les hyperparamètres en regardant le test | Le test n'est plus « hors échantillon » | Tout régler sur la validation, toucher le test une seule fois |
| Remplir les trous avec des valeurs futures (`bfill`) | Copie une valeur du futur dans le passé | Utiliser `ffill` (valeur passée) |
| Oublier les frais | Profit artificiellement gonflé | Frais à chaque opération |

**Test de bon sens :** si le système fait +300 % par an avec un Sharpe de 5, il y a presque certainement une fuite.

---

## 12. Livrables attendus

| # | Livrable | Contenu |
|---|---|---|
| 1 | **Code source Python** | Prétraitement, entraînement DL, agents, moteur de simulation, blockchain |
| 2 | **Base de données préparée** | Données nettoyées, features, ensembles train/validation/test |
| 3 | **Registre blockchain** | Chaîne des décisions et transactions, vérifiable par le professeur |
| 4 | **Tableau de bord** | Prévisions, opérations, évolution du portefeuille, métriques, vérification |
| 5 | **Rapport + démonstration** | Méthodologie, résultats, comparaison aux baselines, limites, démo |

---

## 13. Structure de projet Python proposée

```
projet_or_agents/
├── README.md                  # Comment installer et lancer
├── requirements.txt           # pandas, numpy, scikit-learn, tensorflow ou torch, streamlit, plotly…
├── config.yaml                # Paramètres : dates de découpage, fenêtre, seuils, frais, capital…
│
├── data/
│   ├── raw/                   # CSV Kaggle tel quel (ne jamais le modifier)
│   └── processed/             # Données nettoyées + features + splits
│
├── notebooks/
│   ├── 01_exploration.ipynb   # EDA et graphiques
│   └── 02_modele.ipynb        # Expérimentations du modèle
│
├── src/
│   ├── data/
│   │   ├── load.py            # Chargement et nettoyage
│   │   └── features.py        # Rendements, indicateurs, normalisation, fenêtres
│   ├── models/
│   │   ├── lstm_gru.py        # Définition des architectures
│   │   └── train.py           # Entraînement + sauvegarde du modèle
│   ├── agents/
│   │   ├── forecast_agent.py  # Agent 1
│   │   ├── strategy_agent.py  # Agent 2
│   │   ├── risk_agent.py      # Agent 3
│   │   └── execution_agent.py # Agent 4
│   ├── simulation/
│   │   ├── portfolio.py       # État du portefeuille, frais
│   │   ├── engine.py          # Boucle jour par jour
│   │   └── baselines.py       # Buy & Hold, croisement de moyennes
│   ├── blockchain/
│   │   ├── block.py           # Structure d'un bloc + hash
│   │   ├── chain.py           # Ajout, sauvegarde, vérification
│   │   └── verify.py          # Script autonome pour le professeur
│   └── evaluation/
│       └── metrics.py         # Profit, drawdown, Sharpe, MAE, RMSE…
│
├── dashboard/
│   └── app.py                 # Tableau de bord Streamlit
│
├── outputs/
│   ├── models/                # Modèles entraînés (.keras / .pt)
│   ├── ledger/                # blockchain.json
│   └── results/               # Métriques, graphiques
│
└── main.py                    # Lance tout le pipeline de bout en bout
```

---

## 14. Plan de travail étape par étape

Chaque étape produit un résultat **vérifiable** avant de passer à la suivante.

| Étape | Travail | Résultat attendu (« c'est fini quand… ») |
|---|---|---|
| **0. Mise en place** | Créer l'arborescence, environnement Python, télécharger le CSV Kaggle | Le CSV se charge dans pandas |
| **1. Analyse des données** | Nettoyage, contrôles, graphiques, calcul des rendements | Notebook d'EDA avec constats écrits |
| **2. Features & splits** | Indicateurs, normalisation (fit sur train), fenêtres, découpage chronologique | Fichiers train/val/test sauvegardés, formes vérifiées |
| **3. Modèle** | Baseline naïve, puis LSTM et GRU, early stopping | Tableau MAE/RMSE/précision directionnelle vs naïf |
| **4. Moteur de simulation** | Portefeuille, frais, exécution à t+1, **d'abord testé avec Buy & Hold** | Buy & Hold donne un résultat cohérent avec la hausse réelle du prix |
| **5. Agents** | Les 4 classes et leur enchaînement quotidien | Une simulation complète tourne sur la validation |
| **6. Réglage** | Seuils et règles de risque réglés sur la **validation** | Paramètres figés dans `config.yaml` |
| **7. Blockchain** | Blocs, hachage, vérification, test de falsification | Modifier un bloc à la main → la vérification le détecte |
| **8. Évaluation finale** | Une seule exécution sur le **test** pour toutes les stratégies | Tableau comparatif final |
| **9. Tableau de bord** | Visualisations + vérification de la chaîne | Démo fluide de bout en bout |
| **10. Rapport** | Méthodologie, résultats, limites, perspectives | Rapport rendu + démo répétée |

**Conseil d'ordre :** construire le moteur de simulation et la blockchain **avec une stratégie bête** (Buy & Hold) avant d'y brancher le LSTM. Ainsi, si quelque chose ne marche pas, on sait que le problème vient du modèle, pas de la plomberie.

---

## 15. Questions que le professeur risque de poser

- *Comment garantissez-vous l'absence de fuite d'information ?* → Section 11, découpage chronologique, scaler fit sur train, exécution à t+1.
- *Votre modèle fait-il mieux qu'une prévision naïve ?* → Comparaison MAE/RMSE/précision directionnelle.
- *Battez-vous le Buy and Hold ?* → Si non, l'expliquer honnêtement : peut-être un meilleur Sharpe ou un plus petit drawdown, ce qui est aussi une réussite.
- *Que se passe-t-il si les frais doublent ?* → Analyse de sensibilité (relancer avec 0,05 %, 0,1 %, 0,2 %).
- *Qu'est-ce qui empêche de recalculer toute la blockchain ?* → Ancrage externe du dernier hash / remise régulière au professeur.
- *Pourquoi LSTM plutôt que GRU (ou l'inverse) ?* → Comparaison expérimentale.
- *Pourquoi des agents séparés plutôt qu'un seul script ?* → Séparation des responsabilités, testabilité, chaque décision traçable par agent.

---

## 16. Glossaire

| Terme | Définition |
|---|---|
| **Actif** | Ce dans quoi on investit (ici, l'or). |
| **XAU/USD** | Prix d'une once d'or en dollars américains. |
| **OHLC** | Open, High, Low, Close : prix d'ouverture, plus haut, plus bas, clôture. |
| **Rendement** | Variation relative du prix entre deux dates. |
| **Série temporelle** | Données ordonnées dans le temps. |
| **Feature (caractéristique)** | Variable d'entrée du modèle. |
| **Fenêtre glissante** | Séquence de N jours consécutifs utilisée comme entrée. |
| **Horizon** | Distance de la prévision (ici 1 jour). |
| **LSTM / GRU** | Réseaux de neurones récurrents avec mémoire, adaptés aux séquences. |
| **Dropout** | Désactivation aléatoire de neurones pendant l'entraînement, contre le sur-apprentissage. |
| **Early stopping** | Arrêt de l'entraînement quand la validation cesse de s'améliorer. |
| **Sur-apprentissage (overfitting)** | Le modèle apprend le bruit de l'entraînement et généralise mal. |
| **MAE / RMSE** | Erreurs moyennes absolue / quadratique entre prévision et réalité. |
| **Agent** | Module autonome avec un rôle précis dans la chaîne de décision. |
| **Signal** | Recommandation : achat, vente ou conservation. |
| **Position** | Quantité d'actif détenue. |
| **Vente à découvert (short)** | Vendre un actif qu'on ne possède pas pour profiter d'une baisse. |
| **Frais / coûts de transaction** | Ce que coûte chaque achat ou vente. |
| **Slippage** | Écart entre le prix attendu et le prix réellement obtenu. |
| **Stop-loss** | Vente automatique si la perte dépasse un seuil. |
| **Buy and Hold** | Acheter au début et garder jusqu'à la fin. |
| **Backtest** | Simulation d'une stratégie sur des données historiques. |
| **Hors échantillon** | Données jamais vues pendant l'entraînement ni le réglage. |
| **Look-ahead bias** | Utilisation, volontaire ou non, d'informations du futur. |
| **Drawdown** | Baisse de la valeur du portefeuille depuis son dernier sommet. |
| **Ratio de Sharpe** | Rendement moyen divisé par la volatilité : gain par unité de risque. |
| **Blockchain** | Registre de blocs chaînés par leurs empreintes cryptographiques. |
| **Hash (empreinte)** | Résumé unique d'une donnée ; change complètement au moindre changement. |
| **SHA-256** | Algorithme de hachage standard utilisé en blockchain. |
| **Bloc de genèse** | Premier bloc de la chaîne, sans prédécesseur. |
| **Preuve de travail** | Calcul coûteux exigé pour valider un bloc (optionnel ici). |
| **Ancrage** | Publication externe d'un hash pour prouver l'état de la chaîne à une date. |

---

### En résumé
1. **Données** propres et découpées dans l'ordre du temps.
2. **Modèle** LSTM/GRU qui prédit le rendement, comparé à une prévision naïve.
3. **Quatre agents** qui transforment la prévision en ordre contrôlé.
4. **Moteur** réaliste : frais, exécution le lendemain, pas de triche avec le futur.
5. **Blockchain** qui rend chaque décision vérifiable par le professeur.
6. **Évaluation honnête** contre Buy and Hold et une stratégie technique, sur une période jamais vue.

**Prochaine étape pratique : Étape 0 et Étape 1 (mise en place et analyse des données).**
