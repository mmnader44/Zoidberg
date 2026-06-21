#!/usr/bin/env python
# coding: utf-8
Contraintes des datasets (cf. zoidberg_explo_data.ipynb):
    - images de tailles variables
    - forts déséquilibres des classes + taille des datasets :
        * dataset total : 3/4 PNEUMONIA vs 1/4 NORMAL
        * train : surreprésentation PNEUMONIA (bacteria)
        * val : 8 images NORMAL vs 8 images PNEUMONIA (uniquement bacteria)
        * test : un peu + de 10 % du dataset total
    - données médicales : 2 scénarios proposés
        -> viser le meilleur équilibre faux positifs/faux négatifs
        → limiter au maximum les faux négatifsDans ce notebook : essai de prédictions avec un support vector machines classifier précédé d'une data augmentation:
    - détermination des meilleurs hyperparamètres le jeu de validation
    - détermination d'un seuil de décision optimisé dans chaque scénario
# In[34]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


from pathlib import Path
import re

data_dir = Path("data_chest/chest_Xray")


# # Exclusion des radiographies des patients avec une pneumonie mixte (bacteria + virus)

# In[ ]:


from collections import defaultdict

patients_ambigus = set()
patient_pathologies = defaultdict(set)

for img_path in data_dir.rglob("*.jp*g"):
    filename = img_path.name.lower()

    # Extraction rapide de l'ID
    if "person" in filename:
        pid = re.match(r"(person\d+)", filename)
        pid = pid.group(1) if pid else None
    elif filename.startswith("im-"):
        pid = re.match(r"(im-\d+)", filename)
        pid = pid.group(1).upper() if pid else None
    else:
        pid = filename.split('_')[0].split('-')[0]

    if pid:
        if "bacteria" in filename:
            patient_pathologies[pid].add("BACTERIA")
        elif "virus" in filename:
            patient_pathologies[pid].add("VIRUS")

# On remplit le set des exclus
for pid, pathologies in patient_pathologies.items():
    if "BACTERIA" in pathologies and "VIRUS" in pathologies:
        patients_ambigus.add(pid)

print(f"ℹ️ {len(patients_ambigus)} patients ambigus (Bactérie + Virus) détectés de manière globale. Ils seront ignorés.")


# In[ ]:


def extract_patient_ids(split_name):
    patient_ids = set()
    split_path = data_dir / split_name

    # On parcourt récursivement toutes les images (.jpeg ou .jpg) dans les sous-dossiers
    for img_path in split_path.rglob("*.jp*g"):
        filename = img_path.name

        # Cas 1 : Format "personXXX_..." (Pneumonie)
        if "person" in filename:
            match = re.match(r"(person\d+)", filename)
            if match:
                patient_ids.add(match.group(1))

        # Cas 2 : Format "IM-XXXX-..." (cas Normaux : mais peut-être aussi ID d'images : on les identifie sans les supprimer)
        elif filename.startswith("IM-"):
            match = re.match(r"(IM-\d+)", filename)
            if match:
                patient_ids.add(match.group(1))

        # Cas 3 : Sécurité si le format varie (on prend le premier bloc avant le premier tiret/underscore)
        else:
            fallback_id = filename.split('_')[0].split('-')[0]
            patient_ids.add(fallback_id)

    return patient_ids

# Extraction des IDs pour chaque groupe
train_patients = extract_patient_ids("train")
test_patients = extract_patient_ids("test")
val_patients = extract_patient_ids("val")

print(f"Nombre de patients uniques - Train: {len(train_patients)}, Test: {len(test_patients)}, Val: {len(val_patients)}")


# In[38]:


# --- VERIFICATION DES INTERSECTIONS ---
leak_train_test = train_patients.intersection(test_patients)
leak_train_val = train_patients.intersection(val_patients)
leak_val_test = val_patients.intersection(test_patients)

print("\n--- Résultats de la vérification ---")
if leak_train_test:
    print(f"❌ FUITE DÉTECTÉE entre Train et Test ! ({len(leak_train_test)} patients en commun)")
    print(f"IDs corrompus : {list(leak_train_test)}")
else:
    print("✅ Pas de fuite entre Train et Test.")

if leak_train_val:
    print(f"❌ FUITE DÉTECTÉE entre Train et Val ! ({len(leak_train_val)} patients en commun)")
    print(f"IDs corrompus : {list(leak_train_val)}")
else:
    print("✅ Pas de fuite entre Train et Val.")

if leak_val_test:
    print(f"❌ FUITE DÉTECTÉE entre Val et Test ! ({len(leak_val_test)} patients en commun)")
    print(f"IDs corrompus : {list(leak_val_test)}")
else:
    print("✅ Pas de fuite entre Val et Test.")

=> radiographies de patients estampillées 'bacteria' et 'virus' entre les 3 jeux de train, test et valid :pneumonies mixtes
Exclusion des radiographies de ces patients du dataset pour avoir un entrainement uniquement sur des 'normal', 'bacteria' ou 'virus' bien définis
Et pour supprimer les fuites entre les jeux de données
# In[39]:


# on isole les ID des patients à pneumonies mixtes
patients_a_exclure = set(leak_train_test)
patients_a_exclure.discard("NORMAL2")

print(f"Nombre d'IDs de patients à filtrer : {len(patients_a_exclure)}")

# 2. Nettoyage des sets existants en soustrayant les exclus
# L'opérateur '-' retire du set de gauche tous les éléments présents dans le set de droite
train_patients_clean = train_patients - patients_a_exclure
test_patients_clean = test_patients - patients_a_exclure
val_patients_clean = val_patients - patients_a_exclure

print(f"\n--- Nombre de patients uniques APRÈS filtrage ---")
print(f"Train: {len(train_patients_clean)}, Test: {len(test_patients_clean)}, Val: {len(val_patients_clean)}")


# ## Dataframe de base "nettoyé"

# In[ ]:


data_records = []

clean_splits = {
    "train": train_patients_clean,
    "test": test_patients_clean,
    "val": val_patients_clean
}

for split_name, clean_patients in clean_splits.items():
    split_path = data_dir / split_name

    for img_path in split_path.rglob("*.jp*g"):
        filename = img_path.name

        # On extrait l'ID
        if "person" in filename:
            match = re.match(r"(person\d+)", filename)
            pid = match.group(1) if match else None
        elif filename.startswith("IM-"):
            match = re.match(r"(IM-\d+)", filename)
            pid = match.group(1) if match else None
        else:
            pid = filename.split('_')[0].split('-')[0]

        # On ne garde l'image que si le patient fait partie du set nettoyé
        if pid in clean_patients:
            # On détermine le label (NORMAL, BACTERIA, VIRUS) basé sur le nom du fichier
            if "bacteria" in filename.lower():
                label = "BACTERIA"
            elif "virus" in filename.lower():
                label = "VIRUS"
            else:
                label = "NORMAL" # Les fichiers IM-XXXX ou sans mention

            # On enregistre les infos de l'image
            data_records.append({
                "path": str(img_path),
                "patient_id": pid,
                "label": label,
                "split": split_name
            })

df = pd.DataFrame(data_records)
print(f"✅ DataFrame créé avec succès ! Nombre total d'images conservées : {len(df)}")
print(df["split"].value_counts())


# In[41]:


df.head()


# In[42]:


df_counts = df.pivot_table(
    index="split", 
    columns="label", 
    values="patient_id", 
    aggfunc="nunique", 
    fill_value=0
)

print("📊 Nombre de PATIENTS UNIQUES par Split et par Label :")
display(df_counts)


# # Preprocessing des images

# In[ ]:


import cv2

def preprocess_image(path, size=(130, 95), apply_clahe=False):
    """
    Charge une image, supprime les bordures noires et les lettres parasites,
    la passe en niveaux de gris, applique optionnellement CLAHE, et la redimensionne.
    """
    try:
        # Chargement de l'image en niveaux de gris directe avec OpenCV
        img_raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            raise ValueError("Impossible de charger l'image avec OpenCV.")

        # --- ÉTAPE A : SUPPRESSION DES BORDURES NOIRES ---
        # On crée un masque binaire des zones non-noires (seuil à 10 sur 255)
        _, thresh_crop = cv2.threshold(img_raw, 10, 255, cv2.THRESH_BINARY)
        # On trouve les contours de la cage thoracique
        contours, _ = cv2.findContours(thresh_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # On prend le plus grand contour qui est (normalement) la zone de la radio elle-même
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # On rogne l'image sur ce contour
            img_processed = img_raw[y:y+h, x:x+w]
        else:
            img_processed = img_raw.copy()

        # --- ÉTAPE B : SUPPRESSION DES LETTRES PARASITES (L, R) ---
        # Les lettres et annotations sont généralement d'un blanc pur (proche de 255)
        # On crée un masque pour cibler ce blanc très intense
        _, text_mask = cv2.threshold(img_processed, 250, 255, cv2.THRESH_BINARY)

        # recommandé : On dilate légèrement le masque pour bien englober les bords des lettres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        # On applique l'inpainting pour effacer le texte en le remplaçant par le voisinage direct
        img_processed = cv2.inpaint(img_processed, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # --- ÉTAPE C : CLAHE pour accentuer le contraste ---
        if apply_clahe:
            clip = 2.0
            grid = 8
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            img_processed = clahe.apply(img_processed)

        # --- ÉTAPE D : REDIMENSIONNEMENT FINALE ---
        # OpenCV utilise (largeur, hauteur), donc (130, 95)
        img_resized = cv2.resize(img_processed, size, interpolation=cv2.INTER_AREA)

        # Normalisation finale par la division par 255
        return img_resized / 255.0

    except Exception as e:
        print(f"Erreur sur l'image {path} : {e}")
        return None


# In[ ]:


# Visualisation des images avant / après prétraitement

def show_raw_and_processed(df, n=3, size=(130, 95), apply_clahe=False):
    """
    Affiche n images du dataset :
    - Ligne 1 : Image originale brute (RAW)
    - Ligne 2 : Image après preprocess_image (accepte False, True, ou "random")
    """
    # On s'assure de ne pas demander plus d'images que le DataFrame n'en contient
    n = min(n, len(df))

    plt.figure(figsize=(5 * n, 8))

    for i in range(n):
        row = df.iloc[i]
        path = row["path"]
        label = row["label"]

        # 1. Chargement de l'image originale brute
        img_raw = Image.open(path)

        # 2. Traitement
        img_proc = preprocess_image(path, size=size, apply_clahe=apply_clahe)

        if img_proc is None:
            continue

        # Ligne 1 : Image brute (RAW)
        plt.subplot(2, n, i + 1)
        plt.imshow(img_raw, cmap="gray")
        plt.axis("off")
        plt.title(f"RAW\nLabel: {label}")

        # Ligne 2 : Image traitée (S'adapte automatiquement au mode choisi)
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(img_proc, cmap="gray")
        plt.axis("off")

        # Dynamisation du titre si le traitement est appliqué
        if apply_clahe is True:
            title_type = "PREPROCESSED (STANDARD CLAHE)"
        else:
            title_type = "PREPROCESSED (STANDARD)"

        plt.title(f"{title_type}\n{size[0]}×{size[1]}")

    plt.suptitle("Comparatif Images Brutes vs Prétraitées", fontsize=16)
    plt.tight_layout()
    plt.show()


# In[45]:


show_raw_and_processed(df[df["split"] == "train"], n=3, apply_clahe=False)


# In[46]:


show_raw_and_processed((df[df["split"] == "train"]).sample(n=3), n=3, size=(130, 95), apply_clahe=True)


# # Train-test-split

# ## Fusion datasets de train + valid

# In[47]:


df_train = df[df["split"] == "train"]
df_val = df[df["split"] == "val"]

df_train_full = pd.concat([df_train, df_val], ignore_index=True)
df_train_full["split"] = "train"


# In[48]:


len(df_train_full)


# ## Train-test-split stratifié 80/20

# In[49]:


# création du dataset de validation
from sklearn.model_selection import train_test_split

df_train, df_valid = train_test_split(
    df_train_full,
    test_size=0.2,
    stratify=df_train_full["label"],
    random_state=42
)


# In[50]:


df_test = df[df["split"] == "test"]

print("Nb images dans le jeu de train :", len(df_train))
print("Nb images dans le jeu de valid:", len(df_valid))
print("Nb images dans le jeu de test :", len(df_test))


# In[51]:


print(df_train["label"].value_counts())
print(df_valid["label"].value_counts())


# # Récupération des X (images) et y (étiquettes) pour chaque jeu

# In[52]:


def build_dataset(df_subset, size=(130, 95), apply_clahe=False):
    label_mapping = {"NORMAL": 0, "BACTERIA": 1, "VIRUS": 2}
    y = (
        df_subset["label"]
        .map(label_mapping)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )

    # Liste de matrices 2D, avec le prétraitement CLAHE activé
    X_list = [
        preprocess_image(p, size, apply_clahe=apply_clahe) for p in df_subset["path"]
    ]
    return np.array(X_list), y


# In[53]:


X_test, y_test = build_dataset(df_test, apply_clahe=True)
X_train, y_train = build_dataset(df_train, apply_clahe=True)
X_valid, y_valid = build_dataset(df_valid, apply_clahe=True)


# In[54]:


print(X_train[0:1], y_train[0:1])


# In[55]:


print(X_train.shape, y_train.shape)


# In[56]:


X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test  = X_test.reshape(X_test.shape[0], -1)


# In[57]:


print(type(X_train))


# # Distribution des classes dans chaque jeu

# In[58]:


import seaborn as sns

classes = ["NORMAL", "BACTERIA", "VIRUS"]

# np.bincount compte les occurrences de 0, 1 et 2 d'un seul coup
df_counts = pd.DataFrame({
    "NORMAL":   [np.bincount(y_train)[0], np.bincount(y_valid)[0], np.bincount(y_test)[0]],
    "BACTERIA": [np.bincount(y_train)[1], np.bincount(y_valid)[1], np.bincount(y_test)[1]],
    "VIRUS":    [np.bincount(y_train)[2], np.bincount(y_valid)[2], np.bincount(y_test)[2]],
    "split":    ["train", "valid", "test"]
})

# 2. On reformate le DataFrame pour Seaborn (format "Long")
df_melted = df_counts.melt(id_vars="split", var_name="label", value_name="count")

# 3. Plot avec Seaborn
plt.figure(figsize=(9, 5))
sns.barplot(
    data=df_melted,
    x="split",
    y="count",
    hue="label",
    hue_order=classes,
    palette="viridis"
)

plt.title("Répartition des 3 classes par jeu de données", fontsize=14)
plt.xlabel("Jeu de données (Split)", fontsize=12)
plt.ylabel("Nombre d'images", fontsize=12)
plt.legend(title="Classe")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# # PCA
=> PCA pour réduire le nb de dimensions (et donc les effets d'une variance trop élevée):
peu d'images + accélère l'entrainement
-> limite le surapprentissage

S'effectue toujours sur les données standardisées
# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_valid)
X_test_sc = scaler.transform(X_test)


# In[ ]:


# on fit sur X_train_sc uniquement

from sklearn.decomposition import PCA

pca_model = PCA()
pca_model.fit(X_train_sc)


# ## Cumulative Explained Variance
# 
# https://medium.com/@megha.natarajan/understanding-cumulative-explained-variance-in-pca-with-python-653e3592a77c
# 
# "Principal Component Analysis (PCA) is a technique used to emphasize variation and capture strong patterns in a dataset. 
# But how can we measure the effectiveness of PCA? This is where the Cumulative Explained Variance plot comes into play. [...] 
# Visually, the Cumulative Explained Variance plot often shows a sharp turn or “elbow,” indicating the point at which adding more components 
# has diminishing returns in terms of explained variance."

# In[61]:


individual_variances = pca_model.explained_variance_ratio_

# cumulative explained variance
cumulative_variances = np.cumsum(individual_variances)
cumulative_variances.shape

for i in range(len(cumulative_variances)):
    if cumulative_variances[i] >= 0.95 and cumulative_variances[i] <= 0.96:
        print(f"{i}, avec no composantes {cumulative_variances[i]}")


# In[62]:


cumulative = np.cumsum(pca_model.explained_variance_ratio_)
n_comp = np.argmax(cumulative >= 0.95)

n_comp


# In[63]:


pca = PCA(n_components=n_comp, svd_solver='full', random_state=42)

X_train_pca = pca.fit_transform(X_train_sc)
X_valid_pca = pca.transform(X_val_sc)
X_test_pca = pca.transform(X_test_sc)


# In[64]:


X_train_pca.shape


# # CLASSIFICATION CLASSES 'NORMAL' VS 'BACTERIA' VS 'VIRUS'

# ## Recherche hyperparamètres Support Vector Machines Classifier

# In[65]:


from sklearn.svm import SVC


# In[66]:


from sklearn.model_selection import GridSearchCV

# on teste plusieurs hypothèses de poids à appliquer à chaque classe 
# en fonction de leur distribution dans le jeu de train
# 0 : NORMAL ; 1: BACTERIA ; 2: VIRUS
hypothèses_poids = [
    {0: 1.8, 2: 0.9, 1: 0.75},

    # Hypothèse B : On booste modérément le Virus pour l'aider face à Bacteria
    {0: 1.8, 2: 1.5, 1: 0.75},

    # Hypothèse C : On booste agressivement le Virus au niveau du Normal
    {0: 1.8, 2: 1.8, 1: 0.75},

    # Hypothèse D : On se focalise uniquement sur le duel Bacteria/Virus en égalisant Normal
    {0: 1.0, 2: 2.0, 1: 0.5}
]

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto", 0.01, 0.001],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "class_weight": hypothèses_poids
}

model = SVC(max_iter=10000,
          #class_weight="balanced", # "pénalise" les PNEUMONIA
          random_state=42,
          probability=True)

grid = GridSearchCV(
    model,
    param_grid,
    scoring="roc_auc_ovr", # on cherche à optimiser le ROC AUC et non l'accuracy
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_pca, y_train)


# In[67]:


best_model = grid.best_estimator_
print("Meilleurs hyperparamètres :", grid.best_params_)


# # Entrainement SVC avec les meilleurs hyperparamètres

# In[68]:


svc = SVC(C=best_model.C,
          gamma=best_model.gamma,
          kernel=best_model.kernel,
          class_weight=best_model.class_weight,
            probability=True,
          max_iter=10000,
          random_state=42)


svc.fit(X_train_pca, y_train)


# # Performances jeu de train

# In[69]:


class_names = ["NORMAL", "BACTERIA", "VIRUS"]


# In[70]:


y_pred_train = svc.predict(X_train_pca)


# In[71]:


y_pred_proba_train = svc.predict_proba(X_train_pca)


# In[72]:


# ROC-AUC score
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(
    y_train, 
    y_pred_proba_train,
    multi_class="ovr")

print(f"ROC-AUC (train) : {roc_auc:.3f}")


# In[73]:


from sklearn.metrics import classification_report

print(classification_report(
        y_train,
        y_pred_train,
        target_names=[
            "NORMAL",
            "BACTERIA",
            "VIRUS"
        ]
    ))


# In[74]:


from sklearn.metrics import confusion_matrix

# Matrice de confusion
cm_train = confusion_matrix(y_train, y_pred_train)

# Normalisation par ligne (classe réelle)
cm_normalized = cm_train.astype(float) / cm_train.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 6))
plt.imshow(cm_normalized, cmap="Blues")
plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

# Affichage des pourcentages dans chaque case
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]*100:.1f}%",
            ha="center",
            va="center",
            color="black"
        )

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion (%) – Random Forest Train")

plt.tight_layout()
plt.show()


# In[75]:


# Sur le jeu de train
for i, true_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        print(
            f"Réel={true_class}, Prédit={pred_class} : {cm_train[i,j]}"
        )


# In[76]:


for classe, nom in enumerate(class_names):

    TP = cm_train[classe, classe]

    FN = cm_train[classe, :].sum() - TP

    FP = cm_train[:, classe].sum() - TP

    TN = cm_train.sum() - TP - FN - FP

    print(f"\n--- {nom} vs reste ---")
    print(f"TP : {TP}")
    print(f"FP : {FP}")
    print(f"FN : {FN}")
    print(f"TN : {TN}")


# # Performances jeu de validation

# In[77]:


y_pred_valid = svc.predict(X_valid_pca)
y_pred_proba_valid = svc.predict_proba(X_valid_pca)


# In[78]:


roc_auc = roc_auc_score(
    y_valid, 
    y_pred_proba_valid,
    multi_class="ovr") # capacité globale du modèle à séparer les 3 classes, indépendamment du seuil

print(f"ROC-AUC (valid) : {roc_auc:.3f}")


# In[79]:


print(classification_report(
        y_valid,
        y_pred_valid,
        target_names=[
            "NORMAL",
            "BACTERIA",
            "VIRUS"
        ]
    ))


# In[80]:


# Matrice de confusion
cm_valid = confusion_matrix(y_valid, y_pred_valid)

# Normalisation par ligne (classe réelle)
cm_normalized = cm_valid.astype(float) / cm_valid.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 6))
plt.imshow(cm_normalized, cmap="Blues")
plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

# Affichage des pourcentages dans chaque case
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]*100:.1f}%",
            ha="center",
            va="center",
            color="black"
        )

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion (%) – Random Forest Valid")

plt.tight_layout()
plt.show()


# In[81]:


for i, true_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        print(
            f"Réel={true_class}, Prédit={pred_class} : {cm_valid[i,j]}")


# In[82]:


for classe, nom in enumerate(class_names):

    TP = cm_valid[classe, classe]

    FN = cm_valid[classe, :].sum() - TP

    FP = cm_valid[:, classe].sum() - TP

    TN = cm_valid.sum() - TP - FN - FP

    print(f"\n--- {nom} vs reste ---")
    print(f"TP : {TP}")
    print(f"FP : {FP}")
    print(f"FN : {FN}")
    print(f"TN : {TN}")


# # Essai avec un seuil de décision pour classification : sur le jeu de validation

# ## Seuils pour maximiser la détection des NORMAL vs PNEUMONIE, puis BACTERIA vs VIRUS
Sur le jeu de validation pour ne pas optimiser sur des données déjà "vues" par le modèle
2 scénarios :
    - scénario 1 : optimiser le recall pour limiter les faux négatifs
    - scénario 2 : optimiser l'équilibre entre faux positifs et faux négatifs
# ### Scénario 1 : limiter au maximum les FN pour la catégorie NORMAL

# In[83]:


proba_valid_normal = y_pred_proba_valid[:, 0] # Probabilité d'être NORMAL selon le modèle
proba_valid_bacteria = y_pred_proba_valid[:, 1] # Probabilité d'être BACTERIE selon le modèle
proba_valid_virus = y_pred_proba_valid[:, 2] # Probabilité d'être VIRUS selon le modèle

# 1. On isole la probabilité globale d'être MALADE (Bacteria ou Virus)
# C'est l'inverse de la probabilité d'être sain (Normal)
proba_valid_malade = 1 - proba_valid_normal


# In[84]:


thresholds = np.linspace(0.05, 0.95, 300)
rows = []
#la classe d'intérêt (1 / True / Positive) correspond aux PNEUMONIES.
# Donc tout ce qui n'est pas NORMAL (différent de 0) devient True.
y_valid_malade = (y_valid != 0)

for t in thresholds:
    y_pred = (proba_valid_malade >= t).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_valid_malade, y_pred).ravel()

    rows.append({
        "threshold": t,
        "FN": FN,
        "FP": FP,
        "recall": TP / (TP + FN),
        "fpr": FP / (FP + TN) # faux positifs / total des vrais négatifs
    })

df_thresh = pd.DataFrame(rows)


# In[85]:


df_thresh.head()


# In[86]:


print(df_thresh["recall"].min())
print(df_thresh["recall"].max())
print(df_thresh["fpr"].min())
print(df_thresh["fpr"].max())


# In[87]:


df_sc1 = df_thresh[
    (df_thresh["recall"] >= 0.98)
]


# In[88]:


best_row_sc1 = (
    df_sc1
    .query(f"fpr <= {df_sc1['fpr'].min()}")
    .sort_values("FN")        
    .iloc[0]
)

print(best_row_sc1)


# In[89]:


best_threshold_sc1 = best_row_sc1["threshold"]

best_threshold_sc1


# ### Scénario 2 : Limiter les FN sans trop pénaliser les FP : indice de Younden

# In[90]:


#TROUVER LES MEILLEURS SEUILS POUR CHAQUE CLASSE
from sklearn.metrics import roc_curve

best_thresholds = {}

class_indices = [0, 1, 2]

# 2. Boucler sur chaque classe pour trouver son seuil optimal
for i, class_name in zip(class_indices, class_names):
    # Créer un y_valid binaire pour la classe en cours (1 si c'est la classe, 0 sinon)
    y_valid_binary = (y_valid == i).astype(int)

    # Récupérer les probabilités pour cette classe spécifique
    scores_for_class = y_pred_proba_valid[:, i]

    # Calculer la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_valid_binary, scores_for_class)

    # Calculer l'indice de Youden pour chaque seuil
    youden_index = tpr - fpr

    # PIÈGE : On ignore le tout premier seuil (index 0) s'il vaut l'infini
    if thresholds[0] == np.inf or np.isinf(thresholds[0]):
        # On force l'indice de Youden du seuil infini à -1 pour l'éliminer
        youden_index[0] = -1

    # Trouver l'index du score maximum
    idx_optimal = np.argmax(youden_index)

    # Stocker le meilleur seuil
    best_thresholds[class_name] = thresholds[idx_optimal]

print("Seuils optimaux par classe :", best_thresholds)


# In[91]:


best_threshold_sc2 = best_thresholds['NORMAL']

best_threshold_sc2


# ## Performances jeu de test sans seuil

# In[92]:


y_pred= svc.predict(X_test_pca)
y_pred_proba_test = svc.predict_proba(X_test_pca)


# In[93]:


print(y_pred_proba_test)


# In[94]:


# Matrice de confusion
cm_test = confusion_matrix(y_test, y_pred)

# Normalisation par ligne (classe réelle)
cm_normalized = cm_test.astype(float) / cm_test.sum(axis=1, keepdims=True)

# Noms des classes
class_names = ["NORMAL", "BACTERIA", "VIRUS"]

plt.figure(figsize=(7, 6))
plt.imshow(cm_normalized, cmap="Blues")
plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

# Affichage des pourcentages dans chaque case
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]*100:.1f}%",
            ha="center",
            va="center",
            color="black"
        )

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion (%) – Random Forest Test")

plt.tight_layout()
plt.show()


# In[95]:


roc_auc = roc_auc_score(
    y_test, 
    y_pred_proba_test,
    multi_class="ovr") # capacité globale du modèle à séparer les 3 classes, indépendamment du seuil

print(f"ROC-AUC (test) : {roc_auc:.3f}")


# In[96]:


print(classification_report(
        y_test,
        y_pred,
        target_names=[
            "NORMAL",
            "BACTERIA",
            "VIRUS"
        ]
    ))


# In[97]:


class_names = ["NORMAL", "BACTERIA", "VIRUS"]

for i, true_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        print(
            f"Réel={true_class}, Prédit={pred_class} : {cm_test[i,j]}")


# ## Performances jeu de test avec seuil

# In[98]:


# classement "en cascade" des prédictions en fonction des seuils des 2 scénarios
# d'abord on applique le seuil optimisé pour la distinction NORMAL/MALADE, 
# puis on reclassifie BACTERIA et VIRUS avec les seuils optimaux respectifs

def predict_cascade(y_prob_matrix, thresh_malade, thresh_bacteria, thresh_virus):
    """
    y_prob_matrix : probabilités de predict_proba() [NORMAL, BACTERIA, VIRUS]
    thresh_malade : seuil de détection de la maladie (Scénario 1 ou Youden)
    thresh_bacteria : seuil Youden pour BACTERIA
    thresh_virus : seuil Youden pour VIRUS
    """
    final_predictions = []

    for probs in y_prob_matrix:
        # 1. On calcule le risque d'être malade
        prob_malade = 1 - probs[0] 

        # ÉTAPE 1 : Si le risque de maladie dépasse notre seuil de sécurité
        if prob_malade >= thresh_malade:

            # ÉTAPE 2 : Le patient est malade. On utilise les seuils Youden pour Bacteria vs Virus
            score_bacteria = probs[1] / (thresh_bacteria if thresh_bacteria > 0 else 1e-6)
            score_virus = probs[2] / (thresh_virus if thresh_virus > 0 else 1e-6)

            if score_bacteria > score_virus:
                final_predictions.append(1) # Prédit BACTERIA
            else:
                final_predictions.append(2) # Prédit VIRUS
        else:
            # Le risque est trop faible, le patient est sain
            final_predictions.append(0) # Prédit NORMAL

    return np.array(final_predictions)


# ### Scénario 1

# In[99]:


y_pred_sc1 = predict_cascade(
    y_pred_proba_test, 
    thresh_malade=best_threshold_sc1,
    thresh_bacteria=1,                 # On met 1 pour ne pas modifier le score brut
    thresh_virus=1)


# In[100]:


cm_sc1 = confusion_matrix(y_test, y_pred_sc1)

print(f"Seuil utilisé : {best_threshold_sc1}")

print(classification_report(
        y_test,
        y_pred_sc1,
        target_names=[
            "NORMAL",
            "BACTERIA",
            "VIRUS"
        ]
    ))


# In[101]:


class_names = ["NORMAL", "BACTERIA", "VIRUS"]

for i, true_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        print(
            f"Réel={true_class}, Prédit={pred_class} : {cm_sc1[i,j]}")


# In[102]:


# Normalisation par ligne (classe réelle)
cm_normalized = cm_sc1.astype(float) / cm_sc1.sum(axis=1, keepdims=True)

# Noms des classes
class_names = ["NORMAL", "BACTERIA", "VIRUS"]

plt.figure(figsize=(7, 6))
plt.imshow(cm_normalized, cmap="Blues")
plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

# Affichage des pourcentages dans chaque case
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]*100:.1f}%",
            ha="center",
            va="center",
            color="black"
        )

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion (%) – Random Forest Test - Seuil 1")

plt.tight_layout()
plt.show()


# ### Scénario 2

# In[103]:


seuil_maladie_youden = 1 - best_thresholds['NORMAL']

y_pred_sc2 = predict_cascade(
    y_pred_proba_test, 
    thresh_malade=seuil_maladie_youden,       # Seuil Youden converti en risque maladie
    thresh_bacteria=best_thresholds['BACTERIA'], # Seuil Youden Bacteria
    thresh_virus=best_thresholds['VIRUS']        # Seuil Youden Virus
)


# In[104]:


cm_sc2 = confusion_matrix(y_test, y_pred_sc2)

print(f"Seuil utilisé : {best_threshold_sc2}")

print(classification_report(
        y_test,
        y_pred_sc2,
        target_names=[
            "NORMAL",
            "BACTERIA",
            "VIRUS"
        ]
    ))


# In[105]:


class_names = ["NORMAL", "BACTERIA", "VIRUS"]

for i, true_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        print(
            f"Réel={true_class}, Prédit={pred_class} : {cm_sc2[i,j]}")


# In[106]:


# Normalisation par ligne (classe réelle)
cm_normalized = cm_sc2.astype(float) / cm_sc2.sum(axis=1, keepdims=True)

# Noms des classes
class_names = ["NORMAL", "BACTERIA", "VIRUS"]

plt.figure(figsize=(7, 6))
plt.imshow(cm_normalized, cmap="Blues")
plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

# Affichage des pourcentages dans chaque case
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]*100:.1f}%",
            ha="center",
            va="center",
            color="black"
        )

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion (%) – Random Forest Test - Seuil 2")

plt.tight_layout()
plt.show()


# # Quelles images sont encore des FN ?

# In[ ]:


df_test = df[df["split"] == "test"].reset_index(drop=True)

fn_idx = np.where((y_test != 0) & (y_pred_sc2 == 0))[0]


# In[108]:


def show_errors(df_test, fn_indices, max_images=10):
    n = min(len(fn_indices), max_images)

    plt.figure(figsize=(15, 4 * ((n + 3) // 4)))

    for i, idx in enumerate(fn_indices[:n]):
        img_path = df_test.loc[idx, "path"]
        label = df_test.loc[idx, "label"]
        patient_id = df_test.loc[idx, "patient_id"]

        img = Image.open(img_path).convert("L")

        plt.subplot((n + 3) // 4, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"FN – {label}\nPatient ID: {patient_id}")

    plt.suptitle("Faux négatifs (cas de PNEUMONIA prédit NORMAL)", fontsize=16)
    plt.tight_layout()
    plt.show()


# In[109]:


show_errors(df_test, fn_idx, max_images=10)

