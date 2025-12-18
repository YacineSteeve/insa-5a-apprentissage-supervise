#######################################################################
#  Binome : NOM Prénom / NOM Prénom
# #######################################################################

## Jeu de données : pré-traitement

Donnez la liste des features et ce qu'elles représentent (préciser les éventuels changements effectués en pré-traitement ou si pas de changement)

## Expérimentation 1 : Comparaison de modèles par défaut

* Jeux de données utilisé : 
  * Taille ensemble d'entrainement (nb lignes et nb colonnes) : 
  * Taille ensemble de test (nb lignes et nb colonnes) : 

* Résultats (hyper-paramètres par défaut)

|  Evaluation en train | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            |               |          |         |
|----------------------|---------------|----------|---------|
|  Temps calcul        |               |          |         |
|----------------------|---------------|----------|---------|
|  Matrice confusion   |               |          |         |
|----------------------|---------------|----------|---------|

|   Evaluation en test | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            |               |          |         |
|----------------------|---------------|----------|---------|
|  Temps calcul        |               |          |         |
|----------------------|---------------|----------|---------|
|  Matrice confusion   |               |          |         |
|----------------------|---------------|----------|---------|

* Commentaires et Analyse : 


## Expérimentation 2 : Comparaison Modèles ML par défaut
* Jeux de données utilisé : 
  * Taille ensemble d'entrainement (nb lignes et nb colonnes) : 
  * Taille ensemble de test (nb lignes et nb colonnes) : 

### Random Forest (RF)
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
  * Nombre de plis pour la validation croisée : 
  * Nombre total d'entrainement : 
* Résultats : 
  * Meilleurs hyperparamètres : 
  * Performances en entraintement : 
   * Accuracy : 
   * Temps de calcul : 
   * Matrice de Confusion : 
  * Performance en test : 
   * Accuracy : 
   * Temps de calcul : 
   * Matrice de Confusion : 
  * Commentaires / analyses (par rapport résultat expe 1)

### ADABOOST 
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
  * Nombre de plis pour la validation croisée : 
  * Nombre total d'entrainement : 
* Résultats : 
  * Meilleurs hyperparamètres : 
  * Performances en entraintement : 
   * Accuracy : 
   * Temps de calcul : 
   * Matrice de Confusion : 
  * Performance en test : 
   * Accuracy : 
   * Temps de calcul : 
   * Matrice de Confusion : 
  * Commentaires / analyses (par rapport résultat expe 1)


### XGBOOST
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
  * Nombre de plis pour la validation croisée : 
  * Nombre total d'entrainement : 
* Résultats : 
  * Meilleurs hyperparamètres : 
  * Performances en entraintement : 
   * Accuracy : 
   * Temps de calcul : 
   * Matrice de Confusion : 
  * Performance en test : 
   * Accuracy : 
   * Temps de calcul : 
   * Matrice de Confusion : 
  * Commentaires / analyses (par rapport résultat expe 1)

## Expérimentation 3 : Comparaison des "meilleurs modèles

* Jeux de données utilisé : 
  * Taille ensemble d'entrainement (nb lignes et nb colonnes) : 
  * Taille ensemble de test (nb lignes et nb colonnes) : 

* Résultats des meilleurs modèles obtenus dans Expe 2

|  Evaluation en train | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            |               |          |         |
|----------------------|---------------|----------|---------|
|  Temps calcul        |               |          |         |
|----------------------|---------------|----------|---------|
|  Matrice confusion   |               |          |         |
|----------------------|---------------|----------|---------|

|   Evaluation en test | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            |               |          |         |
|----------------------|---------------|----------|---------|
|  Temps calcul        |               |          |         |
|----------------------|---------------|----------|---------|
|  Matrice confusion   |               |          |         |
|----------------------|---------------|----------|---------|

* Commentaires et Analyse : 

## Expérimentation 4 : inférence sur un autre jeu de données (optionnel)
Résultats / Commentaires / Analyses : 

## Expérimentation 5 : impact de la taille du jeu de données
Résultats / Commentaires / Analyses : 

## Modèle choisi pour la suite : 
* quel modèle : 
* pourquoi ? 

## Explicabilité : "permutation feature importance"

* Résultats obtenus : 
* Analyses :

## Explicabilité : avec LIME et SHAP

* Méthode LIME
  * Exemple(s) choisi(s)
  * Résultats
  * Commentaires / analyses
* Méthode SHAP
  * Exemple(s) choisi(s)
  * Résultats
  * Commentaires / analyses
* Comparaison LIME et SHAP
* Analyse summary-plot de SHAP

## Explicabilité : contrefactuelle
Résultats / Commentaires / Analyses : 

