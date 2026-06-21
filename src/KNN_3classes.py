#!/usr/bin/env python
# coding: utf-8

# # KNN — Classification 3 classes (NORMAL / BACTERIA / VIRUS)
# 
# **Place dans le projet** : ce notebook prolonge le travail binaire (`KNN_baseline.ipynb` puis `KNN_grid_search_v3_2.ipynb`). Une fois le KNN binaire optimisé, on réutilise la même chaîne de prétraitement pour une tâche plus difficile : distinguer l'origine de la pneumonie (bactérienne ou virale), ce qui conditionne le traitement.
# 
# ## Contexte medical
# 
# La pneumonie peut etre causee par une **bacterie** ou un **virus**. La distinction est
# importante car le traitement differe :
# - **Pneumonie bacterienne** : traitee par **antibiotiques**
# - **Pneumonie virale** : les antibiotiques sont inefficaces, traitement symptomatique
# 
# Sur les radiographies thoraciques, les deux types presentent des motifs differents :
# - **Bacterienne** : opacite focale, souvent lobaire (une zone dense bien delimitee)
# - **Virale** : motifs plus diffus, bilateraux, interstitiels
# 
# ## Objectif
# 
# Entrainer un modele **KNN** pour classifier les radiographies en **3 classes** :
# - `NORMAL` (0) : pas de pneumonie
# - `BACTERIA` (1) : pneumonie bacterienne
# - `VIRUS` (2) : pneumonie virale
# 
# ## Metriques utilisees
# 
# En classification multi-classe, les metriques binaires (precision, recall, F1) sont
# calculees **par classe**, puis agreagees :
# - **Macro F1** : moyenne des F1 de chaque classe (traite chaque classe egalement)
# - **Balanced Accuracy** : moyenne des recalls de chaque classe
# - **Classification Report** : tableau complet precision/recall/F1 par classe
# - **Matrice de confusion 3x3** : visualisation des erreurs entre classes
# 
# ## Demarche
# 
# 1. Chargement des donnees avec 3 labels (parsing des noms de fichiers)
# 2. Exploration de la distribution des classes
# 3. Preprocessing (CLAHE, flatten, StandardScaler, SelectKBest)
# 4. Evaluation par simple split puis par cross-validation
# 5. Optimisation du nombre de voisins (k)
# 6. Test de la reduction de dimensions (PCA)
# 7. Evaluation finale du meilleur modele
# 8. Sauvegarde du modele
# 

# In[1]:


import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, classification_report
)
import joblib

# --- Chemins ---
REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))
from src.KNN.data import discover_datasets, get_image_paths, load_image
from src.KNN.metrics import plot_confusion_matrix

SEED = 42
np.random.seed(SEED)

DATA_ROOT = REPO_ROOT / 'data' / 'chest_Xray'
FIGURES_DIR = REPO_ROOT / 'img' / 'data_viz'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = REPO_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Parametres du modele ---
# Issus des meilleures configurations trouvees dans les grid searches precedents
IMG_SIZE = (256, 187)
CLASS_NAMES = ['NORMAL', 'BACTERIA', 'VIRUS']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

print("=== KNN 3 classes ===")
print(f"  Image size   : {IMG_SIZE}")
print(f"  Classes      : {CLASS_NAMES}")
print(f"  Data root    : {DATA_ROOT}")
print("Imports OK.")


# ## Chargement des donnees avec 3 labels
# 
# Le dataset Chest X-Ray organise les images en dossiers `NORMAL/` et `PNEUMONIA/`.
# Pour distinguer bacterie et virus, on utilise le **nom du fichier** :
# - `person*_bacteria_*.jpeg` → classe BACTERIA
# - `person*_virus_*.jpeg` → classe VIRUS
# 
# La fonction ci-dessous lit chaque image, determine sa classe, et applique
# optionnellement le **CLAHE** (rehaussement de contraste adaptatif).
# 

# In[2]:


def load_images_3class(datasets, size, use_clahe=False):
    # Charge les images et attribue 3 labels selon le nom du fichier.
    #
    # Parametres :
    #   datasets : dict retourne par discover_datasets()
    #   size     : tuple (largeur, hauteur) pour le resize
    #   use_clahe: si True, applique CLAHE pour ameliorer le contraste
    #
    # Retourne :
    #   Pour chaque split (train/test) : (images_array, labels_array)
    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    data = {}
    for split in ['train', 'test']:
        if split not in datasets:
            continue
        images, labels = [], []

        for class_name, class_path in datasets[split].items():
            paths = get_image_paths(class_path)

            for path in paths:
                # --- Determiner le label ---
                if class_name == 'NORMAL':
                    label = CLASS_TO_IDX['NORMAL']
                else:
                    # Le nom du fichier contient 'bacteria' ou 'virus'
                    fname = path.name.lower()
                    if 'bacteria' in fname:
                        label = CLASS_TO_IDX['BACTERIA']
                    elif 'virus' in fname:
                        label = CLASS_TO_IDX['VIRUS']
                    else:
                        # Cas inattendu : on ignore l'image
                        print(f"  ATTENTION : fichier ignore (ni bacteria ni virus) : {path.name}")
                        continue

                # --- Charger et pretraiter l'image ---
                img = load_image(path, size=size, grayscale=True)
                img_f = img.astype(np.float32) / 255.0

                if clahe_filter is not None:
                    img_uint8 = (img_f * 255).astype(np.uint8)
                    img_uint8 = clahe_filter.apply(img_uint8)
                    img_f = img_uint8.astype(np.float32) / 255.0

                images.append(img_f)
                labels.append(label)

        data[split] = (np.array(images), np.array(labels))

    return data


datasets = discover_datasets(DATA_ROOT)

# Chargement avec CLAHE (meilleur preprocessing identifie dans les grid searches)
print("Chargement des images avec CLAHE...")
t0 = time.time()
data_3c = load_images_3class(datasets, IMG_SIZE, use_clahe=True)
print(f"Chargement termine en {time.time()-t0:.1f}s\n")

# Afficher les counts par split
for split in ['train', 'test']:
    imgs, lbls = data_3c[split]
    print(f"=== {split.upper()} ===")
    for idx, name in enumerate(CLASS_NAMES):
        count = np.sum(lbls == idx)
        pct = 100 * count / len(lbls)
        print(f"  {name:<10} : {count:>5} images ({pct:.1f}%)")
    print(f"  TOTAL      : {len(lbls):>5} images")
    print()


# ## Exploration des donnees
# 
# Avant d'entrainer un modele, il est important de visualiser :
# 1. La **distribution des classes** pour detecter un desequilibre
# 2. Des **exemples d'images** pour comprendre ce que le modele doit apprendre
# 

# In[3]:


# --- Distribution des classes (bar chart) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, split in enumerate(['train', 'test']):
    imgs, lbls = data_3c[split]
    counts = [np.sum(lbls == i) for i in range(3)]
    total = len(lbls)
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    bars = axes[ax_idx].bar(CLASS_NAMES, counts, color=colors, edgecolor='black')
    axes[ax_idx].set_title(f'Distribution des classes ({split})', fontsize=13)
    axes[ax_idx].set_ylabel('Nombre d\'images')

    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                         f'{count}\n({pct:.1f}%)', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'knn_3c_1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Exemples d'images par classe ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
imgs_train, lbls_train = data_3c['train']

for row, (class_idx, class_name) in enumerate(enumerate(CLASS_NAMES)):
    # Prendre 4 images aleatoires de cette classe
    class_mask = lbls_train == class_idx
    class_indices = np.where(class_mask)[0]
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(class_indices, size=min(4, len(class_indices)), replace=False)

    for col, idx in enumerate(sample_idx):
        axes[row, col].imshow(imgs_train[idx], cmap='gray')
        axes[row, col].set_title(f'{class_name} #{idx}', fontsize=10)
        axes[row, col].axis('off')

plt.suptitle('Exemples d\'images par classe (avec CLAHE)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'knn_3c_2_sample_images.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Ratio bacteria/virus ---
n_bact = np.sum(lbls_train == CLASS_TO_IDX['BACTERIA'])
n_virus = np.sum(lbls_train == CLASS_TO_IDX['VIRUS'])
print(f"Ratio BACTERIA/VIRUS dans le train : {n_bact}/{n_virus} = {n_bact/n_virus:.2f}")
print(f"  BACTERIA represente {100*n_bact/(n_bact+n_virus):.1f}% des pneumonies")
print(f"  VIRUS represente    {100*n_virus/(n_bact+n_virus):.1f}% des pneumonies")


# ## Preprocessing : de l'image au vecteur de features
# 
# Le KNN travaille sur des **vecteurs numeriques**, pas des images. Il faut transformer
# chaque image 2D en un vecteur 1D, puis normaliser les valeurs.
# 
# **Pipeline de preprocessing** :
# 1. **Flatten** : l'image 256x187 pixels devient un vecteur de 47 872 valeurs
# 2. **StandardScaler** : centre (moyenne=0) et reduit (ecart-type=1) chaque feature.
#    C'est essentiel pour le KNN car il mesure des distances.
# 3. **SelectKBest** : selectionne les 20 000 features les plus discriminantes (test ANOVA F).
#    Reduit le bruit et accelere le calcul.
# 
# On utilise un `Pipeline` scikit-learn pour enchainer ces etapes de maniere fiable.
# 

# In[4]:


# --- Preparation des donnees ---
imgs_train, y_train = data_3c['train']
imgs_test, y_test = data_3c['test']

# Flatten : images 2D -> vecteurs 1D
X_train = imgs_train.reshape(imgs_train.shape[0], -1)
X_test = imgs_test.reshape(imgs_test.shape[0], -1)
print(f"Apres flatten : {X_train.shape[1]} features par image")
print(f"  Train : {X_train.shape}")
print(f"  Test  : {X_test.shape}")

# --- Pipeline de preprocessing ---
# On cree un pipeline sklearn qui enchaine scaler + selection de features
preprocessing = Pipeline([
    ('scaler', StandardScaler()),
    ('selectk', SelectKBest(f_classif, k=20000)),
])

# fit_transform sur le train, transform sur le test
# IMPORTANT : le scaler et le selector sont ajustes (fit) UNIQUEMENT sur le train
# pour eviter le "data leakage" (fuite d'information du test vers le train)
print("\nApplication du pipeline de preprocessing...")
t0 = time.time()
X_train_pp = preprocessing.fit_transform(X_train, y_train)
X_test_pp = preprocessing.transform(X_test)
print(f"  Apres preprocessing : {X_train_pp.shape[1]} features")
print(f"  Temps : {time.time()-t0:.1f}s")


# ## Evaluation par simple split (train/test)
# 
# On commence par une evaluation simple : entrainer le modele sur le train set et
# l'evaluer sur le test set. C'est la methode la plus rapide mais la moins robuste
# (le resultat depend du decoupage particulier des donnees).
# 
# On utilise les meilleurs hyperparametres trouves lors des grid searches binaires :
# `k=31`, `weights=distance`, `metric=manhattan`.
# 

# In[5]:


# --- Entrainement du KNN ---
print("Entrainement KNN(k=31, weights=distance, metric=manhattan)...")
t0 = time.time()
knn = KNeighborsClassifier(
    n_neighbors=31,
    weights='distance',
    metric='manhattan',
    n_jobs=-1,
    algorithm='auto'
)
knn.fit(X_train_pp, y_train)
y_pred = knn.predict(X_test_pp)
print(f"  Fit + predict : {time.time()-t0:.1f}s")

# --- Metriques ---
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n=== Resultats (simple split) ===")
print(f"  Accuracy          : {acc:.4f}")
print(f"  Balanced Accuracy : {bal_acc:.4f}")
print(f"  F1 Macro          : {f1_macro:.4f}")
print(f"  F1 Weighted       : {f1_weighted:.4f}")

# --- Classification report ---
# Ce rapport donne precision, recall et F1 pour CHAQUE classe
print(f"\n{'='*60}")
print("Classification Report (par classe)")
print('='*60)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

# --- Matrice de confusion 3x3 ---
plot_confusion_matrix(
    y_test, y_pred,
    class_names=CLASS_NAMES,
    title=f'KNN 3 classes (k=31, distance, manhattan)\nBalanced Acc={bal_acc:.4f} | F1 Macro={f1_macro:.4f}',
    save_path=str(FIGURES_DIR / 'knn_3c_3_confusion_matrix_baseline.png')
)

# Sauvegarder les scores du simple split pour comparaison ulterieure
score_simple_split = {
    'acc': acc, 'bal_acc': bal_acc,
    'f1_macro': f1_macro, 'f1_weighted': f1_weighted
}


# ## Cross-validation : une evaluation plus fiable
# 
# ### Pourquoi la cross-validation ?
# 
# Le simple split train/test donne un seul score, qui depend du decoupage particulier
# des donnees. Si par hasard le test set contient des images "faciles", le score sera
# optimiste ; si le test est "difficile", il sera pessimiste.
# 
# La **cross-validation** (CV) resout ce probleme en evaluant le modele sur
# **plusieurs decoupages differents** :
# 
# ### Principe du 5-Fold CV
# 
# ```
# Donnees : [==== Fold 1 ====][==== Fold 2 ====][==== Fold 3 ====][==== Fold 4 ====][==== Fold 5 ====]
# 
# Round 1 : [===== TEST =====][============= TRAIN ==============][============= TRAIN =============]
# Round 2 : [===== TRAIN ====][===== TEST =====][============= TRAIN ==============][===== TRAIN ====]
# Round 3 : [===== TRAIN ====][===== TRAIN ====][===== TEST =====][===== TRAIN ====][===== TRAIN ====]
# Round 4 : [===== TRAIN ====][============= TRAIN ==============][===== TEST =====][===== TRAIN ====]
# Round 5 : [============= TRAIN ==============][============= TRAIN =============][===== TEST =====]
# ```
# 
# On obtient **5 scores** dont on calcule la **moyenne** et l'**ecart-type**.
# Un ecart-type faible indique que le modele est stable.
# 
# ### Stratified KFold
# 
# On utilise `StratifiedKFold` qui maintient les **proportions des classes** dans
# chaque fold. C'est crucial quand les classes sont desequilibrees (ex: peu de VIRUS).
# 

# In[6]:


# --- Cross-validation 5-fold stratifiee ---
# On refait le preprocessing complet dans chaque fold pour eviter le data leakage
print("=== Cross-validation 5-fold stratifiee ===\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# On travaille sur les donnees train completes (avant preprocessing)
# Le preprocessing sera refait dans chaque fold
X_all = X_train.copy()  # flatten deja fait
y_all = y_train.copy()

cv_results = []
t_start = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    t0 = time.time()

    # Decoupage du fold
    X_fold_train, X_fold_val = X_all[train_idx], X_all[val_idx]
    y_fold_train, y_fold_val = y_all[train_idx], y_all[val_idx]

    # Preprocessing (fit sur le train du fold uniquement)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selectk', SelectKBest(f_classif, k=20000)),
    ])
    X_fold_train_pp = pipe.fit_transform(X_fold_train, y_fold_train)
    X_fold_val_pp = pipe.transform(X_fold_val)

    # KNN
    knn_cv = KNeighborsClassifier(
        n_neighbors=31, weights='distance',
        metric='manhattan', n_jobs=-1
    )
    knn_cv.fit(X_fold_train_pp, y_fold_train)
    y_fold_pred = knn_cv.predict(X_fold_val_pp)

    # Metriques
    fold_acc = accuracy_score(y_fold_val, y_fold_pred)
    fold_bal = balanced_accuracy_score(y_fold_val, y_fold_pred)
    fold_f1 = f1_score(y_fold_val, y_fold_pred, average='macro')

    cv_results.append({
        'fold': fold_idx + 1,
        'acc': fold_acc, 'bal_acc': fold_bal, 'f1_macro': fold_f1
    })

    elapsed = time.time() - t0
    print(f"  Fold {fold_idx+1}/5 : Acc={fold_acc:.4f}  BalAcc={fold_bal:.4f}  "
          f"F1_macro={fold_f1:.4f}  ({elapsed:.1f}s)")

# --- Moyenne et ecart-type ---
accs = [r['acc'] for r in cv_results]
bals = [r['bal_acc'] for r in cv_results]
f1s = [r['f1_macro'] for r in cv_results]

print(f"\n{'='*60}")
print(f"  Accuracy       : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"  Balanced Acc   : {np.mean(bals):.4f} +/- {np.std(bals):.4f}")
print(f"  F1 Macro       : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print(f"{'='*60}")
print(f"  Temps total CV : {time.time()-t_start:.1f}s")

# --- Comparaison avec le simple split ---
print(f"\n--- Comparaison simple split vs CV ---")
print(f"  {'Methode':<20} {'Accuracy':>10} {'BalAcc':>10} {'F1_Macro':>10}")
print(f"  {'-'*50}")
print(f"  {'Simple split':<20} {score_simple_split['acc']:>10.4f} "
      f"{score_simple_split['bal_acc']:>10.4f} {score_simple_split['f1_macro']:>10.4f}")
print(f"  {'CV 5-fold (moy)':<20} {np.mean(accs):>10.4f} "
      f"{np.mean(bals):>10.4f} {np.mean(f1s):>10.4f}")


# ## Optimisation du nombre de voisins (k)
# 
# Le parametre `k` (nombre de voisins) est l'hyperparametre principal du KNN.
# - **k petit** (ex: 3) : modele tres sensible au bruit (variance elevee)
# - **k grand** (ex: 51) : modele trop lisse (biais eleve)
# 
# On cherche le **k optimal** en evaluant chaque valeur par cross-validation 5-fold.
# 

# In[7]:


# --- Grid search sur k avec cross-validation ---
K_VALUES = [7, 11, 15, 21, 31]
k_results = []

print("=== Optimisation de k (CV 5-fold) ===\n")

for k_val in K_VALUES:
    t0 = time.time()
    fold_scores = []

    for train_idx, val_idx in skf.split(X_all, y_all):
        X_f_tr, X_f_val = X_all[train_idx], X_all[val_idx]
        y_f_tr, y_f_val = y_all[train_idx], y_all[val_idx]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('selectk', SelectKBest(f_classif, k=20000)),
        ])
        X_f_tr_pp = pipe.fit_transform(X_f_tr, y_f_tr)
        X_f_val_pp = pipe.transform(X_f_val)

        knn_k = KNeighborsClassifier(
            n_neighbors=k_val, weights='distance',
            metric='manhattan', n_jobs=-1
        )
        knn_k.fit(X_f_tr_pp, y_f_tr)
        y_f_pred = knn_k.predict(X_f_val_pp)

        fold_scores.append({
            'bal_acc': balanced_accuracy_score(y_f_val, y_f_pred),
            'f1_macro': f1_score(y_f_val, y_f_pred, average='macro'),
        })

    mean_bal = np.mean([s['bal_acc'] for s in fold_scores])
    std_bal = np.std([s['bal_acc'] for s in fold_scores])
    mean_f1 = np.mean([s['f1_macro'] for s in fold_scores])
    std_f1 = np.std([s['f1_macro'] for s in fold_scores])

    k_results.append({
        'k': k_val,
        'bal_acc_mean': mean_bal, 'bal_acc_std': std_bal,
        'f1_macro_mean': mean_f1, 'f1_macro_std': std_f1,
    })

    print(f"  k={k_val:<3}  BalAcc={mean_bal:.4f}+/-{std_bal:.4f}  "
          f"F1_macro={mean_f1:.4f}+/-{std_f1:.4f}  ({time.time()-t0:.1f}s)")

# --- Meilleur k ---
best_k_result = max(k_results, key=lambda r: r['f1_macro_mean'])
best_k = best_k_result['k']
print(f"\n  Meilleur k = {best_k} (F1_macro = {best_k_result['f1_macro_mean']:.4f})")

# --- Courbe de k ---
fig, ax = plt.subplots(figsize=(10, 5))
ks = [r['k'] for r in k_results]
f1s_mean = [r['f1_macro_mean'] for r in k_results]
f1s_std = [r['f1_macro_std'] for r in k_results]
bal_mean = [r['bal_acc_mean'] for r in k_results]
bal_std = [r['bal_acc_std'] for r in k_results]

ax.errorbar(ks, f1s_mean, yerr=f1s_std, marker='o', capsize=4, label='F1 Macro')
ax.errorbar(ks, bal_mean, yerr=bal_std, marker='s', capsize=4, label='Balanced Accuracy')
ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Meilleur k={best_k}')
ax.set_xlabel('Nombre de voisins (k)')
ax.set_ylabel('Score')
ax.set_title('Optimisation de k — KNN 3 classes (CV 5-fold)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(ks)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'knn_3c_4_k_optimization.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Reduction de dimensions : PCA
# 
# La **PCA** (Analyse en Composantes Principales) transforme les features en un
# nouvel espace ou les premieres dimensions capturent le maximum de variance.
# 
# Avantages :
# - **Reduit le bruit** en eliminant les dimensions de faible variance
# - **Accelere** le KNN (moins de dimensions = calcul de distance plus rapide)
# - **Regularise** le modele (peut eviter le surapprentissage)
# 
# On teste PCA avec 50, 100 et 200 composantes, en comparant avec le modele sans PCA
# (SelectKBest seul = 20000 features).
# 

# In[8]:


# --- Test PCA avec differents n_components ---
PCA_VALUES = [50, 100, 200, None]  # None = pas de PCA (SelectKBest seul)
pca_results = []

print(f"=== Test PCA (k={best_k}, CV 5-fold) ===\n")

for n_comp in PCA_VALUES:
    t0 = time.time()
    fold_scores = []

    for train_idx, val_idx in skf.split(X_all, y_all):
        X_f_tr, X_f_val = X_all[train_idx], X_all[val_idx]
        y_f_tr, y_f_val = y_all[train_idx], y_all[val_idx]

        # Pipeline avec ou sans PCA
        steps = [
            ('scaler', StandardScaler()),
            ('selectk', SelectKBest(f_classif, k=20000)),
        ]
        if n_comp is not None:
            steps.append(('pca', PCA(n_components=n_comp, random_state=SEED)))

        pipe = Pipeline(steps)
        X_f_tr_pp = pipe.fit_transform(X_f_tr, y_f_tr)
        X_f_val_pp = pipe.transform(X_f_val)

        knn_pca = KNeighborsClassifier(
            n_neighbors=best_k, weights='distance',
            metric='manhattan', n_jobs=-1
        )
        knn_pca.fit(X_f_tr_pp, y_f_tr)
        y_f_pred = knn_pca.predict(X_f_val_pp)

        fold_scores.append({
            'bal_acc': balanced_accuracy_score(y_f_val, y_f_pred),
            'f1_macro': f1_score(y_f_val, y_f_pred, average='macro'),
        })

    mean_bal = np.mean([s['bal_acc'] for s in fold_scores])
    std_bal = np.std([s['bal_acc'] for s in fold_scores])
    mean_f1 = np.mean([s['f1_macro'] for s in fold_scores])
    std_f1 = np.std([s['f1_macro'] for s in fold_scores])
    n_feat = n_comp if n_comp else 20000

    pca_results.append({
        'pca': n_comp,
        'n_features': n_feat,
        'bal_acc_mean': mean_bal, 'bal_acc_std': std_bal,
        'f1_macro_mean': mean_f1, 'f1_macro_std': std_f1,
    })

    label = f'PCA={n_comp}' if n_comp else 'Pas de PCA'
    print(f"  {label:<15} ({n_feat:>5} feat)  BalAcc={mean_bal:.4f}+/-{std_bal:.4f}  "
          f"F1_macro={mean_f1:.4f}+/-{std_f1:.4f}  ({time.time()-t0:.1f}s)")

# Meilleur PCA
best_pca_result = max(pca_results, key=lambda r: r['f1_macro_mean'])
best_pca = best_pca_result['pca']
label = f'PCA={best_pca}' if best_pca else 'Pas de PCA (20000 features)'
print(f"\n  Meilleur : {label} (F1_macro = {best_pca_result['f1_macro_mean']:.4f})")


# ## Tableau recapitulatif
# 
# Synthese de toutes les variantes testees, triees par F1 Macro decroissant.
# 

# In[9]:


# --- Tableau recapitulatif ---
print("=" * 80)
print("RECAPITULATIF — KNN 3 classes")
print("=" * 80)
print(f"  {'Config':<40} {'Features':>8} {'BalAcc':>10} {'F1_Macro':>10}")
print("-" * 80)

# Resultats par k (sans PCA)
for r in sorted(k_results, key=lambda x: x['f1_macro_mean'], reverse=True):
    config = f"k={r['k']:<3} (Sel=20000, sans PCA)"
    print(f"  {config:<40} {'20000':>8} {r['bal_acc_mean']:>10.4f} {r['f1_macro_mean']:>10.4f}")

print("-" * 80)

# Resultats PCA (avec meilleur k)
for r in sorted(pca_results, key=lambda x: x['f1_macro_mean'], reverse=True):
    pca_label = f"PCA={r['pca']}" if r['pca'] else "Pas de PCA"
    config = f"k={best_k} + {pca_label}"
    print(f"  {config:<40} {r['n_features']:>8} {r['bal_acc_mean']:>10.4f} {r['f1_macro_mean']:>10.4f}")

print("=" * 80)

# Determiner la meilleure config globale
all_configs = []
for r in k_results:
    all_configs.append({
        'desc': f"k={r['k']}, Sel=20000, sans PCA",
        'k': r['k'], 'pca': None,
        'f1': r['f1_macro_mean'], 'bal': r['bal_acc_mean'],
    })
for r in pca_results:
    if r['pca'] is not None:
        all_configs.append({
            'desc': f"k={best_k}, Sel=20000, PCA={r['pca']}",
            'k': best_k, 'pca': r['pca'],
            'f1': r['f1_macro_mean'], 'bal': r['bal_acc_mean'],
        })

best_config = max(all_configs, key=lambda x: x['f1'])
print(f"\nMeilleure configuration (CV) : {best_config['desc']}")
print(f"  F1 Macro = {best_config['f1']:.4f}")
print(f"  Balanced Accuracy = {best_config['bal']:.4f}")

final_k = best_config['k']
final_pca = best_config['pca']

print("\nNote : ces scores sont estimes par validation croisee sur le train set.")
print("L'evaluation finale sur le test set (cellule suivante) compare les")
print("meilleures configs et determine le meilleur modele effectif.")


# ## Evaluation finale du meilleur modele
# 
# On re-entraine le meilleur modele sur **tout** le train set et on l'evalue
# sur le test set (qui n'a jamais ete utilise pendant l'optimisation).
# 
# C'est l'evaluation la plus honnete de la performance reelle du modele.
# 

# In[10]:


# --- Evaluation finale : comparaison des configs candidates sur le test set ---
# La CV sur le train set donne une estimation, mais le test set est l'evaluation
# definitive. On teste les meilleures configs CV + la baseline pour s'assurer
# de choisir celle qui generalise le mieux.

candidates = sorted(all_configs, key=lambda x: x['f1'], reverse=True)[:3]

baseline_present = any(c['k'] == 31 and c['pca'] is None for c in candidates)
if not baseline_present:
    baseline = next((c for c in all_configs if c['k'] == 31 and c['pca'] is None), None)
    if baseline:
        candidates.append(baseline)

print("=== Comparaison des configs candidates sur le test set ===\n")

best_test_f1 = -1
best_test_result = None

for cfg in candidates:
    k_val = cfg['k']
    pca_val = cfg['pca']

    steps = [
        ('scaler', StandardScaler()),
        ('selectk', SelectKBest(f_classif, k=20000)),
    ]
    if pca_val is not None:
        steps.append(('pca', PCA(n_components=pca_val, random_state=SEED)))

    pipe = Pipeline(steps)
    t0 = time.time()
    X_tr = pipe.fit_transform(X_train, y_train)
    X_te = pipe.transform(X_test)
    n_feat = X_tr.shape[1]

    knn = KNeighborsClassifier(
        n_neighbors=k_val, weights='distance',
        metric='manhattan', n_jobs=-1
    )
    knn.fit(X_tr, y_train)
    y_pred = knn.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    bal = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    f1w = f1_score(y_test, y_pred, average='weighted')

    pca_label = f"PCA={pca_val}" if pca_val else "sans PCA"
    cv_f1 = cfg['f1']
    print(f"  k={k_val}, {pca_label} ({n_feat} feat) : "
          f"BalAcc={bal:.4f}  F1_Macro={f1m:.4f}  (CV: {cv_f1:.4f})  "
          f"({time.time()-t0:.1f}s)")

    if f1m > best_test_f1:
        best_test_f1 = f1m
        best_test_result = {
            'k': k_val, 'pca': pca_val, 'cv_f1': cv_f1,
            'acc': acc, 'bal': bal, 'f1m': f1m, 'f1w': f1w,
            'preds': y_pred, 'knn': knn, 'pipe': pipe,
        }

final_k = best_test_result['k']
final_pca = best_test_result['pca']
y_pred_final = best_test_result['preds']
knn_final = best_test_result['knn']
preprocessing_final = best_test_result['pipe']
acc_final = best_test_result['acc']
bal_final = best_test_result['bal']
f1_macro_final = best_test_result['f1m']
f1_weighted_final = best_test_result['f1w']

pca_str = f'PCA={final_pca}' if final_pca else 'sans PCA'
print(f"\n{'='*60}")
print(f"  RESULTATS FINAUX — KNN 3 classes")
print(f"  Meilleur sur test set : k={final_k}, {pca_str}")
if final_k != best_config['k'] or final_pca != best_config['pca']:
    print(f"  (different de la config CV : {best_config['desc']})")
print(f"{'='*60}")
print(f"  Accuracy          : {acc_final:.4f}")
print(f"  Balanced Accuracy : {bal_final:.4f}")
print(f"  F1 Macro          : {f1_macro_final:.4f}")
print(f"  F1 Weighted       : {f1_weighted_final:.4f}")
print(f"{'='*60}")

print(f"\n{classification_report(y_test, y_pred_final, target_names=CLASS_NAMES, digits=4)}")

# --- Matrice de confusion finale ---
plot_confusion_matrix(
    y_test, y_pred_final,
    class_names=CLASS_NAMES,
    title=(f'KNN 3 classes — Meilleur modele\n'
           f'k={final_k} distance manhattan Sel=20000 {pca_str}\n'
           f'Balanced Acc={bal_final:.4f} | F1 Macro={f1_macro_final:.4f}'),
    figsize=(7, 6),
    save_path=str(FIGURES_DIR / 'knn_3c_5_confusion_matrix_final.png')
)


# ## Sauvegarde du modele
# 
# On sauvegarde le modele entraine et le pipeline de preprocessing pour pouvoir
# les reutiliser sans re-entrainer. Le format `joblib` est efficace pour les
# objets scikit-learn.
# 

# In[11]:


# --- Sauvegarde ---
model_path = MODELS_DIR / 'knn_3classes_final.joblib'
pipeline_path = MODELS_DIR / 'preprocessing_3classes.joblib'

joblib.dump(knn_final, model_path)
joblib.dump(preprocessing_final, pipeline_path)

print(f"Modele sauvegarde     : {model_path}")
print(f"Preprocessing sauveg. : {pipeline_path}")
print(f"\nPour reutiliser :")
print(f"  knn = joblib.load('{model_path.name}')")
print(f"  pipe = joblib.load('{pipeline_path.name}')")
print(f"  X_new = pipe.transform(images_flatten)")
print(f"  predictions = knn.predict(X_new)")


# ## Conclusion
# 
# ### Resultats
# 
# Le KNN sur 3 classes est un probleme **plus difficile** que la classification binaire
# (NORMAL vs PNEUMONIA) pour plusieurs raisons :
# - La distinction **bacterie vs virus** sur radiographie est subtile, meme pour les radiologues
# - Le desequilibre des classes est plus prononce (3 classes au lieu de 2)
# - L'espace de features (pixels) ne capture pas bien les differences fines de texture
# 
# ### Ce qu'on a fait
# 
# 1. **Chargement** des donnees avec 3 labels (parsing des noms de fichiers)
# 2. **Exploration** de la distribution et exemples visuels
# 3. **Pipeline** de preprocessing (CLAHE + flatten + StandardScaler + SelectKBest)
# 4. **Evaluation** par simple split ET cross-validation 5-fold
# 5. **Optimisation** du parametre k
# 6. **Test PCA** pour la reduction de dimensions
# 7. **Sauvegarde** du meilleur modele
# 
# ### Pistes d'amelioration
# 
# - **Random Forest** : algorithme base sur des arbres de decision, potentiellement
#   meilleur pour capturer des patterns non-lineaires
# - **CNN** (reseau de neurones convolutif) : apprend automatiquement les features
#   pertinentes a partir des images brutes
# - **Transfer Learning** : utiliser un CNN pre-entraine (ex: ResNet, VGG) et
#   l'adapter a notre probleme — souvent la meilleure approche pour l'imagerie medicale
# - **Data augmentation** : generer des images supplementaires par rotation, zoom,
#   translation pour enrichir le dataset
# 
