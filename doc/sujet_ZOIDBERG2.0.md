# ZOIDBERG2.0
## Computer Aided Diagnosis

## Objectif du projet

À partir d’images de radiographies (X-ray), utiliser le **machine learning** pour aider les médecins à détecter la **pneumonie**.

Les médecins vous donnent accès à **3 datasets**.  
C’est à vous de décider **quand et comment** les utiliser :
- entraînement
- test
- évaluation des performances
- réglage des paramètres

---

## Contraintes obligatoires

Vous devez impérativement :

- utiliser une procédure **train / validation / test**
- utiliser une procédure de **cross-validation**
- comparer vos résultats avec un **simple train-test split**
- utiliser **un des datasets pour le tuning** de vos algorithmes

Vous **DEVEZ** explorer et tester **plusieurs méthodes**, et **comparer les résultats**.

Thématiques attendues :
- optimisation
- feature engineering
- métriques
- PCA

> Une présentation claire et concise des résultats doit toujours primer.

---

## Livrables (Delivery)

### Documents techniques
- un fichier de type **Jupyter Notebook**
  - contenant du code et du texte
  - éventuellement des graphiques
- un **fichier HTML**
  - permettant de prouver les résultats **sans relancer le code**

### Document de synthèse
- un **PDF**
  - résumant les résultats
  - incluant les figures pertinentes

> Il existe des méthodes pour sauvegarder un algorithme entraîné et le recharger afin d’obtenir exactement les mêmes résultats lors d’une nouvelle exécution.

---

## Bonus (optionnel)

- implémenter une **Self-Organizing Map**
- apprentissage approfondi via des **réseaux de neurones**
- prédiction sur **3 classes** :
  - pas de pneumonie
  - pneumonie virale
  - pneumonie bactérienne

---

## Recommandations

### Temps et espace

Avant de commencer l’implémentation, réfléchissez aux ressources nécessaires :
- complexité algorithmique
- temps d’exécution
- espace de stockage

---

### Mauvaises habitudes à éviter

- trouver l’équilibre entre **biais** et **variance**
- utiliser la **cross-validation**
- régler correctement les **hyper-paramètres**

---

### Bonnes métriques

- présenter les résultats de manière lisible
- choisir des métriques adaptées
- explorer le **ROC-AUC**

Vous devez être capable d’expliquer ce que mesure l’AUC et ses avantages.

---

**Version** : v2.2
