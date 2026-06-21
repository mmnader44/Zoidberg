#!/usr/bin/env python
# coding: utf-8

# # KNN : Prétraitement, Entraînement et Prédiction
# 
# **Objectif** : Implémenter un pipeline complet KNN avec prétraitement optimisé.
# 
# **Contexte du dataset** :
# - 5856 images totales (train: 5216, val: 16, test: 624)
# - Déséquilibre : 27% NORMAL / 73% PNEUMONIA
# - Tailles variables : 796-2434 (largeur), 469-2376 (hauteur)
# - Formats : JPEG, modes L (grayscale) et RGB
# 
# **Plan** :
# 1. Chargement et prétraitement optimisé
# 2. Analyse des données prétraitées
# 3. Entraînement KNN avec différentes valeurs de k
# 4. Évaluation et sélection du meilleur modèle
# 5. Prédiction et analyse des erreurs

# ## 1. Imports et Configuration

# In[11]:


import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

# Racine du repo T-DEV-810-NAN_2 (parent du dossier notebooks)
REPO_ROOT = Path.cwd().parent
# Imports depuis src.KNN (dossier src du repo T-DEV-810-NAN_2)
sys.path.insert(0, str(REPO_ROOT))
from src.KNN.data import discover_datasets, get_image_paths, load_image
from src.KNN.metrics import plot_confusion_matrix, print_metrics, compare_models

# Reproductibilité
SEED = 42
np.random.seed(SEED)

# Configuration des chemins (structure T-DEV-810-NAN_2)
# - Données : data/
# - Modèles : models/
# - Visualisations : data_viz/
IMAGE_SIZE = (130, 95)  # (largeur, hauteur) - ratio préservé
DATA_ROOT = REPO_ROOT / 'data' / 'chest_Xray'
FIGURES_DIR = REPO_ROOT / 'img' / 'data_viz'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration :")
print(f"  - Taille images : {IMAGE_SIZE}")
print(f"  - Features après flatten : {IMAGE_SIZE[0] * IMAGE_SIZE[1]}")
print(f"  - Seed : {SEED}")


# ## 2. Prétraitement Optimisé
# 
# ### Stratégie de prétraitement :
# - **Resize** : 130x95 pixels (ratio original ~1.37:1 préservé, évite le stretching)
# - **Grayscale** : Conversion uniforme en niveaux de gris
# - **Normalisation** : [0, 255] → [0, 1] puis StandardScaler
# - **Flatten** : Image 2D → vecteur 1D (12,350 features)

# In[2]:


def load_and_preprocess_images(datasets, splits, size=IMAGE_SIZE, verbose=True):
    """
    Charge et prétraite les images avec normalisation [0,1].
    """
    class_names = ['NORMAL', 'PNEUMONIA']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    images = []
    labels = []

    for split in splits:
        if split not in datasets:
            continue

        for class_name, class_path in datasets[split].items():
            if class_name not in class_to_idx:
                continue

            label = class_to_idx[class_name]
            paths = get_image_paths(class_path)

            if verbose:
                print(f"Chargement {split}/{class_name}: {len(paths)} images...")

            for path in paths:
                # Charger en grayscale et resize
                img = load_image(path, size=size, grayscale=True)
                # Normaliser [0, 255] → [0, 1]
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    # Flatten
    X_flat = X.reshape(X.shape[0], -1)

    if verbose:
        print(f"\nDonnées chargées :")
        print(f"  - Shape avant flatten : {X.shape}")
        print(f"  - Shape après flatten : {X_flat.shape}")
        print(f"  - Labels : {np.bincount(y)} [NORMAL, PNEUMONIA]")

    return X_flat, y, class_names


# In[3]:


# Découverte des datasets
datasets = discover_datasets(DATA_ROOT)
print("Splits disponibles :", list(datasets.keys()))


# In[4]:


# Charger les données d'entraînement (train original)
print("=" * 50)
print("CHARGEMENT DES DONNÉES D'ENTRAÎNEMENT")
print("=" * 50)

start_time = time.time()
X_train_full, y_train_full, class_names = load_and_preprocess_images(
    datasets, splits=['train'], size=IMAGE_SIZE
)
print(f"\nTemps de chargement : {time.time() - start_time:.1f}s")


# In[6]:


# Charger les données de test (test original - réservé pour évaluation finale)
print("\n" + "=" * 50)
print("CHARGEMENT DES DONNÉES DE TEST (évaluation finale)")
print("=" * 50)

X_test_final, y_test_final, _ = load_and_preprocess_images(
    datasets, splits=['test'], size=IMAGE_SIZE
)


# ## 3. Split Train/Validation
# 
# On divise le train original en train/validation pour le tuning.
# Le test original est réservé pour l'évaluation finale uniquement.

# In[7]:


# Split train/validation (80/20) avec stratification
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full
)

print("=== Split Train/Validation ===")
print(f"Train : {X_train.shape[0]} images")
print(f"  - NORMAL: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  - PNEUMONIA: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
print(f"\nValidation : {X_val.shape[0]} images")
print(f"  - NORMAL: {(y_val == 0).sum()} ({(y_val == 0).mean()*100:.1f}%)")
print(f"  - PNEUMONIA: {(y_val == 1).sum()} ({(y_val == 1).mean()*100:.1f}%)")
print(f"\nTest final (réservé) : {X_test_final.shape[0]} images")


# ## 4. Normalisation avec StandardScaler
# 
# Important : fit sur train uniquement, puis transform sur val et test.

# In[8]:


# Normalisation StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

print("Normalisation StandardScaler appliquée")
print(f"Train  - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"Val    - mean: {X_val_scaled.mean():.6f}, std: {X_val_scaled.std():.6f}")
print(f"Test   - mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")


# ## 5. Entraînement KNN - Recherche du meilleur k
# 
# On teste différentes valeurs de k et on évalue sur le set de validation.

# In[9]:


# Valeurs de k à tester
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]

results = []

print("=== Recherche du meilleur k ===")
print(f"{'k':<5} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Temps':>8}")
print("-" * 60)

for k in k_values:
    start_time = time.time()

    # Entraînement
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    # Prédiction sur validation
    y_pred = knn.predict(X_val_scaled)

    # Métriques
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    elapsed = time.time() - start_time

    results.append({
        'k': k,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'time': elapsed
    })

    print(f"{k:<5} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {elapsed:>7.2f}s")

df_results = pd.DataFrame(results)


# In[16]:


# Visualisation des résultats
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Métriques en fonction de k
axes[0].plot(df_results['k'], df_results['accuracy'], 'o-', label='Accuracy', linewidth=2)
axes[0].plot(df_results['k'], df_results['precision'], 's-', label='Precision', linewidth=2)
axes[0].plot(df_results['k'], df_results['recall'], '^-', label='Recall', linewidth=2)
axes[0].plot(df_results['k'], df_results['f1'], 'd-', label='F1-Score', linewidth=2)
axes[0].set_xlabel('Nombre de voisins (k)')
axes[0].set_ylabel('Score')
axes[0].set_title('Métriques KNN en fonction de k')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(k_values)

# Graphique 2 : F1-Score avec meilleur k mis en évidence
best_idx = df_results['f1'].idxmax()
best_k = df_results.loc[best_idx, 'k']
best_f1 = df_results.loc[best_idx, 'f1']

bars = axes[1].bar(df_results['k'].astype(str), df_results['f1'], color='steelblue', alpha=0.7)
bars[best_idx].set_color('darkgreen')
bars[best_idx].set_alpha(1.0)
axes[1].axhline(y=best_f1, color='darkgreen', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Nombre de voisins (k)')
axes[1].set_ylabel('F1-Score')
axes[1].set_title(f'F1-Score par k (meilleur : k={best_k})')
axes[1].set_ylim(0.9, 1)

plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'knn_k_selection.png'), dpi=150)
plt.show()

print(f"\nMeilleur k : {best_k} (F1-Score = {best_f1:.4f})")


# ## 6. Cross-Validation avec le meilleur k
# 
# Validation de la stabilité du modèle avec StratifiedKFold.

# In[15]:


# Cross-validation avec le meilleur k
print(f"=== Cross-Validation (k={best_k}) ===")

knn_best = KNeighborsClassifier(n_neighbors=int(best_k), n_jobs=-1)

# StratifiedKFold 5-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Scores CV sur le train complet (avant split train/val)
cv_scores = cross_val_score(knn_best, scaler.fit_transform(X_train_full), y_train_full, 
                            cv=cv, scoring='f1', n_jobs=-1)

print(f"F1-Scores par fold : {cv_scores}")
print(f"F1-Score moyen : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ## 7. Entraînement du Modèle Final
# 
# Entraînement sur toutes les données train avec le meilleur k.

# In[17]:


# Entraînement final sur tout le train
print(f"=== Entraînement du modèle final (k={best_k}) ===")

# Re-fit scaler sur tout le train
scaler_final = StandardScaler()
X_train_full_scaled = scaler_final.fit_transform(X_train_full)
X_test_final_scaled = scaler_final.transform(X_test_final)

# Entraînement
knn_final = KNeighborsClassifier(n_neighbors=int(best_k), n_jobs=-1)
knn_final.fit(X_train_full_scaled, y_train_full)

print(f"Modèle entraîné sur {len(X_train_full)} images")


# ## 8. Évaluation Finale sur le Test Set
# 
# **Attention** : Cette évaluation ne doit être faite qu'une seule fois !

# In[ ]:


# Prédiction sur le test final
y_pred_final = knn_final.predict(X_test_final_scaled)
y_proba_final = knn_final.predict_proba(X_test_final_scaled)[:, 1]

# Métriques
print("=" * 60)
print("ÉVALUATION FINALE SUR LE TEST SET")
print("=" * 60)

acc = accuracy_score(y_test_final, y_pred_final)
prec = precision_score(y_test_final, y_pred_final)
rec = recall_score(y_test_final, y_pred_final)
f1 = f1_score(y_test_final, y_pred_final)
roc_auc = roc_auc_score(y_test_final, y_proba_final)

print(f"\nKNN (k={best_k}) - Résultats sur {len(y_test_final)} images de test :")
print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")


# In[19]:


# Matrice de confusion
plot_confusion_matrix(
    y_test_final, y_pred_final,
    class_names=class_names,
    title=f'KNN (k={best_k}) - Matrice de Confusion (Test Final)',
    save_path=str(FIGURES_DIR / 'knn_confusion_matrix_final.png')
)


# In[20]:


# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test_final, y_proba_final)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'KNN (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title(f'Courbe ROC - KNN (k={best_k})')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'knn_roc_curve.png'), dpi=150)
plt.show()


# ## 9. Analyse des Erreurs
# 
# Identifier les images mal classées pour comprendre les limites du modèle.

# In[21]:


# Identifier les erreurs
errors = y_test_final != y_pred_final
error_indices = np.where(errors)[0]

# Types d'erreurs
false_positives = np.where((y_test_final == 0) & (y_pred_final == 1))[0]  # NORMAL prédit PNEUMONIA
false_negatives = np.where((y_test_final == 1) & (y_pred_final == 0))[0]  # PNEUMONIA prédit NORMAL

print("=== Analyse des Erreurs ===")
print(f"Total erreurs : {len(error_indices)} / {len(y_test_final)} ({100*len(error_indices)/len(y_test_final):.1f}%)")
print(f"  - Faux Positifs (NORMAL → PNEUMONIA) : {len(false_positives)}")
print(f"  - Faux Négatifs (PNEUMONIA → NORMAL) : {len(false_negatives)}")
print(f"\nEn contexte médical, les faux négatifs sont plus graves (pneumonie non détectée).")


# ## 10. Solutions au Déséquilibre des Classes
# 
# Le modèle prédit trop souvent PNEUMONIA car :
# - 73% des données d'entraînement sont PNEUMONIA
# - KNN vote majoritaire → les voisins sont souvent PNEUMONIA
# 
# ### Solutions testées :
# 1. **KNN pondéré** : `weights='distance'` (voisins proches comptent plus)
# 2. **Ajustement du seuil** : Modifier le seuil de décision (défaut=0.5)
# 3. **SMOTE** : Sur-échantillonnage synthétique de la classe minoritaire

# ### 10.1 KNN Pondéré (weights='distance')

# In[22]:


# KNN avec pondération par distance
print("=== KNN Pondéré (weights='distance') ===")

knn_weighted = KNeighborsClassifier(n_neighbors=int(best_k), weights='distance', n_jobs=-1)
knn_weighted.fit(X_train_full_scaled, y_train_full)

y_pred_weighted = knn_weighted.predict(X_test_final_scaled)

acc_w = accuracy_score(y_test_final, y_pred_weighted)
prec_w = precision_score(y_test_final, y_pred_weighted)
rec_w = recall_score(y_test_final, y_pred_weighted)
f1_w = f1_score(y_test_final, y_pred_weighted)

print(f"Accuracy  : {acc_w:.4f}")
print(f"Precision : {prec_w:.4f}")
print(f"Recall    : {rec_w:.4f}")
print(f"F1-Score  : {f1_w:.4f}")

# Matrice de confusion
cm_weighted = confusion_matrix(y_test_final, y_pred_weighted)
print(f"\nMatrice de confusion :")
print(f"  NORMAL correct   : {cm_weighted[0,0]} ({100*cm_weighted[0,0]/cm_weighted[0].sum():.1f}%)")
print(f"  PNEUMONIA correct : {cm_weighted[1,1]} ({100*cm_weighted[1,1]/cm_weighted[1].sum():.1f}%)")


# ### 10.2 Ajustement du Seuil de Décision

# In[23]:


# Tester différents seuils
print("=== Ajustement du Seuil de Décision ===")

y_proba = knn_final.predict_proba(X_test_final_scaled)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
print(f"{'Seuil':<8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'NORMAL%':>10} {'PNEUM%':>10}")
print("-" * 68)

best_threshold = 0.5
best_f1_threshold = 0

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)

    acc_t = accuracy_score(y_test_final, y_pred_thresh)
    prec_t = precision_score(y_test_final, y_pred_thresh, zero_division=0)
    rec_t = recall_score(y_test_final, y_pred_thresh, zero_division=0)
    f1_t = f1_score(y_test_final, y_pred_thresh, zero_division=0)

    # Taux de NORMAL et PNEUMONIA correctement classés
    cm_t = confusion_matrix(y_test_final, y_pred_thresh)
    normal_correct = 100 * cm_t[0,0] / cm_t[0].sum() if cm_t[0].sum() > 0 else 0
    pneumo_correct = 100 * cm_t[1,1] / cm_t[1].sum() if cm_t[1].sum() > 0 else 0

    print(f"{thresh:<8} {acc_t:>8.4f} {prec_t:>8.4f} {rec_t:>8.4f} {f1_t:>8.4f} {normal_correct:>9.1f}% {pneumo_correct:>9.1f}%")

    # Garder le meilleur F1 avec un bon équilibre
    if f1_t > best_f1_threshold and normal_correct > 40:
        best_f1_threshold = f1_t
        best_threshold = thresh

print(f"\nMeilleur seuil équilibré : {best_threshold}")
print("\nNote: NORMAL% = vrais négatifs, PNEUM% = vrais positifs (Recall)")


# In[24]:


# Visualisation de l'effet du seuil
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Métriques vs Seuil
thresh_range = np.arange(0.1, 0.99, 0.02)
accs, precs, recs, f1s, normals, pneumos = [], [], [], [], [], []

for thresh in thresh_range:
    y_pred_t = (y_proba >= thresh).astype(int)
    accs.append(accuracy_score(y_test_final, y_pred_t))
    precs.append(precision_score(y_test_final, y_pred_t, zero_division=0))
    recs.append(recall_score(y_test_final, y_pred_t, zero_division=0))
    f1s.append(f1_score(y_test_final, y_pred_t, zero_division=0))
    cm_t = confusion_matrix(y_test_final, y_pred_t)
    normals.append(100 * cm_t[0,0] / cm_t[0].sum() if cm_t[0].sum() > 0 else 0)
    pneumos.append(100 * cm_t[1,1] / cm_t[1].sum() if cm_t[1].sum() > 0 else 0)

axes[0].plot(thresh_range, accs, '-', label='Accuracy')
axes[0].plot(thresh_range, precs, '-', label='Precision')
axes[0].plot(thresh_range, recs, '-', label='Recall')
axes[0].plot(thresh_range, f1s, '-', label='F1-Score', linewidth=2)
axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Seuil par défaut')
axes[0].set_xlabel('Seuil de décision')
axes[0].set_ylabel('Score')
axes[0].set_title('Métriques en fonction du seuil')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphique 2 : % NORMAL et PNEUMONIA correct vs Seuil
axes[1].plot(thresh_range, normals, 'g-', linewidth=2, label='NORMAL% (Spécificité)')
axes[1].plot(thresh_range, pneumos, 'r-', linewidth=2, label='PNEUMONIA% (Sensibilité)')
axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Seuil par défaut')
axes[1].axhline(y=90, color='orange', linestyle=':', alpha=0.7, label='90% (objectif médical)')
axes[1].set_xlabel('Seuil de décision')
axes[1].set_ylabel('% correctement classés')
axes[1].set_title('Trade-off NORMAL vs PNEUMONIA')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'knn_threshold_analysis.png'), dpi=150)
plt.show()

# Trouver le point d'intersection approximatif
for i, thresh in enumerate(thresh_range):
    if abs(normals[i] - pneumos[i]) < 5:  # Différence < 5%
        print(f"Point d'équilibre approximatif : seuil={thresh:.2f}, NORMAL%={normals[i]:.1f}%, PNEUM%={pneumos[i]:.1f}%")
        break


# ### 10.3 SMOTE (Sur-échantillonnage Synthétique)

# In[25]:


# Installation si nécessaire : pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("SMOTE non disponible. Installer avec : pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

if SMOTE_AVAILABLE:
    print("=== SMOTE (Sur-échantillonnage Synthétique) ===")

    # Appliquer SMOTE sur les données d'entraînement
    smote = SMOTE(random_state=SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_full_scaled, y_train_full)

    print(f"Avant SMOTE : {len(y_train_full)} images")
    print(f"  - NORMAL    : {(y_train_full == 0).sum()}")
    print(f"  - PNEUMONIA : {(y_train_full == 1).sum()}")
    print(f"\nAprès SMOTE : {len(y_train_smote)} images")
    print(f"  - NORMAL    : {(y_train_smote == 0).sum()}")
    print(f"  - PNEUMONIA : {(y_train_smote == 1).sum()}")


# In[26]:


if SMOTE_AVAILABLE:
    # Entraîner KNN sur données SMOTE
    knn_smote = KNeighborsClassifier(n_neighbors=int(best_k), n_jobs=-1)
    knn_smote.fit(X_train_smote, y_train_smote)

    y_pred_smote = knn_smote.predict(X_test_final_scaled)

    acc_s = accuracy_score(y_test_final, y_pred_smote)
    prec_s = precision_score(y_test_final, y_pred_smote)
    rec_s = recall_score(y_test_final, y_pred_smote)
    f1_s = f1_score(y_test_final, y_pred_smote)

    print(f"\n=== Résultats KNN + SMOTE ===")
    print(f"Accuracy  : {acc_s:.4f}")
    print(f"Precision : {prec_s:.4f}")
    print(f"Recall    : {rec_s:.4f}")
    print(f"F1-Score  : {f1_s:.4f}")

    # Matrice de confusion
    cm_smote = confusion_matrix(y_test_final, y_pred_smote)
    print(f"\nMatrice de confusion :")
    print(f"  NORMAL correct   : {cm_smote[0,0]} ({100*cm_smote[0,0]/cm_smote[0].sum():.1f}%)")
    print(f"  PNEUMONIA correct : {cm_smote[1,1]} ({100*cm_smote[1,1]/cm_smote[1].sum():.1f}%)")


# In[27]:


if SMOTE_AVAILABLE:
    # Matrice de confusion SMOTE
    plot_confusion_matrix(
        y_test_final, y_pred_smote,
        class_names=class_names,
        title=f'KNN + SMOTE (k={best_k}) - Matrice de Confusion',
        save_path=str(FIGURES_DIR / 'knn_smote_confusion_matrix.png')
    )


# ### 10.4 Comparaison des Méthodes

# In[28]:


# Tableau comparatif
print("=" * 85)
print("COMPARAISON DES MÉTHODES POUR GÉRER LE DÉSÉQUILIBRE")
print("=" * 85)
print(f"{'Méthode':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'NORMAL%':>10} {'PNEUM%':>10}")
print("-" * 85)

# KNN de base
cm_base = confusion_matrix(y_test_final, y_pred_final)
normal_base = 100 * cm_base[0,0] / cm_base[0].sum()
pneumo_base = 100 * cm_base[1,1] / cm_base[1].sum()
print(f"{'KNN de base':<25} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {normal_base:>9.1f}% {pneumo_base:>9.1f}%")

# KNN pondéré
normal_weighted = 100 * cm_weighted[0,0] / cm_weighted[0].sum()
pneumo_weighted = 100 * cm_weighted[1,1] / cm_weighted[1].sum()
print(f"{'KNN pondéré (distance)':<25} {acc_w:>8.4f} {prec_w:>8.4f} {rec_w:>8.4f} {f1_w:>8.4f} {normal_weighted:>9.1f}% {pneumo_weighted:>9.1f}%")

# Seuil ajusté
y_pred_best_thresh = (y_proba >= best_threshold).astype(int)
acc_bt = accuracy_score(y_test_final, y_pred_best_thresh)
prec_bt = precision_score(y_test_final, y_pred_best_thresh)
rec_bt = recall_score(y_test_final, y_pred_best_thresh)
f1_bt = f1_score(y_test_final, y_pred_best_thresh)
cm_bt = confusion_matrix(y_test_final, y_pred_best_thresh)
normal_bt = 100 * cm_bt[0,0] / cm_bt[0].sum()
pneumo_bt = 100 * cm_bt[1,1] / cm_bt[1].sum()
print(f"{'Seuil=' + str(best_threshold):<25} {acc_bt:>8.4f} {prec_bt:>8.4f} {rec_bt:>8.4f} {f1_bt:>8.4f} {normal_bt:>9.1f}% {pneumo_bt:>9.1f}%")

# SMOTE
if SMOTE_AVAILABLE:
    normal_smote = 100 * cm_smote[0,0] / cm_smote[0].sum()
    pneumo_smote = 100 * cm_smote[1,1] / cm_smote[1].sum()
    print(f"{'KNN + SMOTE':<25} {acc_s:>8.4f} {prec_s:>8.4f} {rec_s:>8.4f} {f1_s:>8.4f} {normal_smote:>9.1f}% {pneumo_smote:>9.1f}%")

print("=" * 85)
print("\nNOTE: NORMAL% = Spécificité (vrais négatifs), PNEUM% = Sensibilité/Recall (vrais positifs)")
print("En contexte médical : PNEUM% élevé = moins de pneumonies ratées (faux négatifs)")


# ## 11. Sauvegarde du Meilleur Modèle

# In[37]:


import joblib

from src.utils.paths import relative_path, format_file_size

# Sauvegarde
models_dir = REPO_ROOT / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / 'knn_final.joblib'
scaler_path = models_dir / 'scaler_knn.joblib'

joblib.dump(knn_final, model_path, compress=3)
joblib.dump(scaler_final, scaler_path, compress=3)

print(f"Modèle sauvegardé : {relative_path(model_path)}  ({format_file_size(model_path)})")
print(f"Scaler sauvegardé : {relative_path(scaler_path)} ({format_file_size(scaler_path)})")


# ## 11. Démonstration : Charger et Prédire

# In[30]:


# Charger le modèle sauvegardé
knn_loaded = joblib.load(models_dir / 'knn_final.joblib')
scaler_loaded = joblib.load(models_dir / 'scaler_knn.joblib')

# Prédiction sur quelques images de test
n_samples = 5
sample_indices = np.random.choice(len(X_test_final), n_samples, replace=False)

X_sample = X_test_final[sample_indices]
y_sample_true = y_test_final[sample_indices]

X_sample_scaled = scaler_loaded.transform(X_sample)
y_sample_pred = knn_loaded.predict(X_sample_scaled)
y_sample_proba = knn_loaded.predict_proba(X_sample_scaled)[:, 1]

print("=== Démonstration de prédiction ===")
for i in range(n_samples):
    true_label = class_names[y_sample_true[i]]
    pred_label = class_names[y_sample_pred[i]]
    proba = y_sample_proba[i]
    status = "✓" if y_sample_true[i] == y_sample_pred[i] else "✗"
    print(f"  Image {i+1}: Vrai={true_label:<10} Prédit={pred_label:<10} (P(PNEUMONIA)={proba:.2f}) {status}")


# ## 12. Résumé et Conclusions
# 
# ### Résultats obtenus
# 
# | Métrique | Valeur |
# |----------|--------|
# | Meilleur k | 11 |
# | Accuracy | 0.7340 (73.40 %) |
# | Precision | 0.7022 |
# | Recall | 0.9974 |
# | F1-Score | 0.8242 |
# | ROC-AUC | 0.8548 |
# 
# ### Points clés
# 
# 1. **Prétraitement** : Images redimensionnées en 130x95 (ratio préservé), grayscale, normalisées avec StandardScaler
# 2. **Sélection de k** : Testé k ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31} sur validation
# 3. **Cross-validation** : Validation de la stabilité avec 5-fold CV
# 4. **Déséquilibre** : Dataset déséquilibré (73% PNEUMONIA) - le Recall est important
# 
# ### Pistes d'amélioration
# 
# - Tester différentes métriques de distance (euclidienne, manhattan, minkowski)
# - Pondération des voisins (uniform vs distance)
# - Réduction de dimensionnalité avec PCA avant KNN
# - Feature engineering : HOG, histogrammes de texture
