# ZOIDBERG2.0 - Détection de pneumonie sur radiographie

ZOIDBERG2.0 est un projet de diagnostic assisté par ordinateur appliqué aux radiographies thoraciques. Le dépôt regroupe plusieurs approches de machine learning et deep learning pour distinguer trois classes d’images : normale, pneumonie bactérienne et pneumonie virale.

L’application principale est une interface [Streamlit](https://streamlit.io/) qui permet d’uploader une radiographie et d’obtenir les prédictions de plusieurs modèles entraînés sur le projet.

## Contenu du projet

- Une application Streamlit dans [Streamlit/app.py](Streamlit/app.py)
- Des modèles sauvegardés dans [models](models)
- Des notebooks d’exploration et d’entraînement dans [notebooks](notebooks)
- Du code Python réutilisable dans [src](src)
- Les ressources visuelles dans [img](img)
- La documentation du sujet et des critères dans [doc](doc)
- Un Rapport Global du projet avec nos résultats et conclusions dans [doc](doc)
- Le jeu de données principal dans [data/chest_Xray](data/chest_Xray)

## Fonctionnalités

L’interface Streamlit charge une image X-ray et affiche les prédictions de plusieurs familles de modèles :

- Random Forest
- Balanced Random Forest
- SVC
- KNN
- CNN entraîné from scratch
- CNN en transfer learning avec ResNet50

## Installation

Créer un environnement Python puis installer les dépendances :

```bash
pip install -r requirements.txt
pip install streamlit
```

Le projet repose notamment sur `numpy`, `pandas`, `scikit-learn`, `opencv-python`, `tensorflow`, `joblib` et `streamlit`.

## Lancement de l’application

L’application Streamlit utilise des chemins relatifs vers `../models` et `../img`. Il faut donc la lancer depuis le dossier `Streamlit` :

```bash
cd Streamlit
streamlit run app.py
```

## Données et modèles

Le dépôt contient des artefacts volumineux versionnés avec **Git LFS** :

- modèles `*.joblib`
- modèles Keras `*.h5`

Avant de travailler sur le projet, vérifiez que Git LFS est bien installé :

```bash
git lfs install
git lfs version
```

Les règles LFS sont définies dans [.gitattributes](.gitattributes).

## Organisation du dépôt

```text
├── data/
│   └── chest_Xray/
├── doc/
├── img/
│   ├── data_viz/
│   └── zoidberg_icon.png
├── models/
├── notebooks/
│   ├── KNN.ipynb
│   ├── CNN_V2.ipynb
│   ├── CNN_TL_ResNet50.ipynb
│   └── notebooks_html/
├── src/
│   ├── KNN/
│   └── utils/
└── Streamlit/
	└── app.py
```

## Notebooks et code source

Les notebooks servent à reproduire les expérimentations du projet : exploration des données, modèles classiques, CNN et transfer learning. Le code Python réutilisable est organisé dans `src/` avec des modules dédiés aux features, aux métriques, aux chemins et aux pipelines KNN.

## Avertissement médical

Cette application est un outil pédagogique. Elle ne remplace pas un avis médical ni un diagnostic réalisé par un professionnel de santé.

## Auteurs

- Mehdi Michel NADER - @mmnader44
- Léo Guerizec
- Chloé Belard - @chloe-bel
- Sylvain Conan - @SylvainMJC

Projet EPITECH - T-DEV-810-NAN_2
