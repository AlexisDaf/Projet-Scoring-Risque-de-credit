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
8. Évaluation finale, une seule fois, sur le jeu de test mis de côté dès le départ.

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

**LightGBM — Configuration M4** (20 variables), retenu sur la base de la validation.

### Évaluation finale sur le jeu de test

Le jeu de test (20 % des clients, soit 61 503 dossiers) a été mis de côté dès le départ et n'a servi ni à l'entraînement, ni au choix du modèle, ni au réglage des hyperparamètres. Les modèles évalués sont exactement ceux sélectionnés en validation, sans réentraînement.

| Modèle (M4)           | ROC-AUC validation | ROC-AUC test | AP test | Capture@10 test | Lift@10 test |
| --------------------- | -----------------: | -----------: | ------: | --------------: | -----------: |
| Régression logistique |             0,7338 |       0,7356 |  0,2153 |          30,9 % |         3,09 |
| **LightGBM**          |         **0,7443** |   **0,7484** | **0,2340** |      **32,6 %** |     **3,26** |

* La performance sur le test est **au niveau de la validation** (écart de +0,004 en ROC-AUC pour LightGBM) : la sélection sur la validation n'a pas surestimé la performance.
* Les **10 % de clients jugés les plus risqués** par LightGBM concentrent **32,6 % des défauts** du jeu de test, soit un taux de défaut 3,3 fois supérieur à la moyenne (8,1 %).

LightGBM présente les meilleures performances sur les cinq configurations étudiées, en validation comme sur le test.

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

La comparaison des configurations et le choix du modèle ont été faits sur l'ensemble de validation ; l'évaluation finale sur un jeu de test indépendant confirme les performances (voir ci-dessus).

Le découpage entraînement / validation / test est aléatoire (stratifié sur la cible) et non temporel : il ne simule pas une mise en production où le modèle serait appliqué à des demandes postérieures.

Par ailleurs, avant toute utilisation opérationnelle dans un contexte bancaire, il serait nécessaire d'approfondir la calibration des probabilités, la stabilité des performances et la vérification de la disponibilité temporelle des informations utilisées.

## Conclusion

Ce projet met en évidence l'importance du **feature engineering** et de l'enrichissement des données dans les problèmes de classification.

Il démontre également qu'un algorithme plus complexe n'est pas, à lui seul, la garantie de meilleures performances : la pertinence des informations disponibles joue un rôle déterminant.

L'expérimentation aboutit à la sélection de **LightGBM**, qui présente la meilleure capacité de discrimination parmi les modèles étudiés.
