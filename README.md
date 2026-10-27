# Prédiction du risque de défaut de crédit

### Projet de Data Science — Home Credit Default Risk

## Présentation

Ce projet de machine learning porte sur la prédiction du risque de défaut de crédit à partir du jeu de données **Home Credit Default Risk**, disponible sur Kaggle.

L'objectif est de comparer deux algorithmes de classification supervisée, la **régression logistique** et **LightGBM**, afin d'étudier l'influence de l'enrichissement progressif des données sur leurs performances prédictives.

L'analyse repose sur **307 511 observations** et sur plusieurs sources d'informations relatives aux demandeurs de crédit.

## Problématique

**Dans quelle mesure l'enrichissement progressif des données améliore-t-il la prédiction du risque de défaut de crédit, et quel algorithme présente les meilleures performances ?**

## Méthodologie

Le projet suit plusieurs étapes :

1. Exploration et analyse descriptive des données.
2. Préparation des données et construction de nouvelles variables.
3. Séparation des observations en ensembles d'entraînement, de validation et de test.
4. Entraînement de la régression logistique et de LightGBM.
5. Optimisation des hyperparamètres par validation croisée stratifiée.
6. Comparaison progressive des performances.
7. Analyse de l'importance des variables et évaluation du modèle retenu.

La **ROC-AUC** est utilisée comme métrique principale de sélection.

L'**Average Precision** constitue une métrique complémentaire adaptée au déséquilibre de la variable cible.

## Enrichissement progressif des données

Cinq configurations ont été étudiées.

| Étape | Sources de données             | Variables |
| ----- | ------------------------------ | --------: |
| M0    | Application                    |         3 |
| M1    | Application + Bureau           |         9 |
| M2    | Ajout de Previous Application  |        13 |
| M3    | Ajout de Installments Payments |        17 |
| M4    | Ajout des variables EXT_SOURCE |        20 |

Cette approche permet d'évaluer l'apport de chaque nouvelle configuration de données.

## Résultats

### Évolution de la ROC-AUC sur validation

| Configuration       | Régression logistique |   LightGBM |
| ------------------- | --------------------: | ---------: |
| M0 — Application    |                0,5888 |     0,5975 |
| M1 — Bureau         |                0,6185 |     0,6290 |
| M2 — Previous       |                0,6391 |     0,6503 |
| M3 — Installments   |                0,6518 |     0,6638 |
| **M4 — EXT_SOURCE** |            **0,7338** | **0,7443** |

### Meilleur modèle

**LightGBM — Configuration M4**

* ROC-AUC sur validation : **0,7443**
* Average Precision : **0,2257**
* Nombre de variables : **20**

LightGBM présente les meilleures performances sur les cinq configurations étudiées.

### Principaux enseignements

**1. L'enrichissement des données améliore la prédiction.**

Les performances des deux algorithmes progressent à mesure que de nouvelles sources de données sont intégrées.

**2. LightGBM surpasse la régression logistique.**

L'algorithme obtient une ROC-AUC systématiquement supérieure sur les configurations étudiées.

**3. Les variables EXT_SOURCE jouent un rôle majeur.**

Leur introduction entraîne la plus forte progression de la ROC-AUC et elles occupent les premières positions dans l'analyse de l'importance des variables.

## Technologies utilisées

**Langage :** Python

**Bibliothèques principales :**

* Pandas et NumPy — manipulation des données.
* Matplotlib et Seaborn — visualisation.
* Scikit-learn — préparation, modélisation et évaluation.
* LightGBM — classification par gradient boosting.

**Environnement :** Jupyter Notebook.

## Limites

Les résultats présentés correspondent principalement à l'ensemble de validation.

La comparaison des configurations sur ce même ensemble peut introduire un biais de sélection.

Par ailleurs, avant toute utilisation opérationnelle dans un contexte bancaire, il serait nécessaire d'approfondir la calibration des probabilités, la stabilité des performances et la vérification de la disponibilité temporelle des informations utilisées.

## Conclusion

Ce projet met en évidence l'importance du **feature engineering** et de l'enrichissement des données dans les problèmes de classification.

Il démontre également qu'un algorithme plus complexe n'est pas, à lui seul, la garantie de meilleures performances : la pertinence des informations disponibles joue un rôle déterminant.

L'expérimentation aboutit à la sélection de **LightGBM**, qui présente la meilleure capacité de discrimination parmi les modèles étudiés.
