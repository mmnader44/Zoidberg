#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning avec ResNet50
# 
# Dans ce notebook, on résout le **même problème** que dans le notebook du CNN entraîné de zéro : classer des radios des poumons en trois catégories (`NORMAL`, `PNEUMONIA_bacteria`, `PNEUMONIA_viral`). Ce qui change, c'est la méthode.
# 
# Ici on fait du **transfert learning** : au lieu de partir de zéro, on réutilise **ResNet50**, un gros réseau déjà entraîné sur ImageNet (des millions d'images variées). Il a donc déjà appris à reconnaître des formes, des textures et des contours généraux. On garde cette « base » et on ne réentraîne qu'une petite partie + une nouvelle tête adaptée à nos 3 classes.
# 
# L'intérêt par rapport au CNN from scratch : on a besoin de moins de données et de moins de temps pour obtenir de bons résultats. Les deux notebooks suivent les mêmes grandes étapes pour que la comparaison soit honnête.

# ## 1. Imports, configuration et préparation des données
# 
# Cette première cellule fait beaucoup de choses d'un coup :
# - les **imports** (TensorFlow/Keras, pandas, NumPy, matplotlib/seaborn, scikit-learn, et surtout ResNet50 avec son `preprocess_input`) ;
# - la **configuration** (taille d'image 224×224, batch de 32, les 3 classes cibles) ;
# - la fonction `build_split_dataframe` qui parcourt les dossiers, lit le nom des fichiers pour deviner si une pneumonie est virale ou bactérienne, et range tout dans un tableau pandas ;
# - `load_image` et `make_dataset` qui transforment ces chemins en `tf.data.Dataset` prêts pour le modèle ;
# - enfin le **découpage** train/val/test (80/20 sur train+val, en gardant les proportions de classes avec `stratify`).

# In[ ]:


import os
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATASET_ROOT = Path("../data/chest_Xray")
SPLITS = ["train", "val", "test"]
SOURCE_LABELS = ["NORMAL", "PNEUMONIA"]
TARGET_LABELS = ["NORMAL", "PNEUMONIA_bacteria", "PNEUMONIA_viral"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(TARGET_LABELS)}
NUM_CLASSES = len(TARGET_LABELS)
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_split_dataframe():
    rows = []
    for split in SPLITS:
        for source_label in SOURCE_LABELS:
            class_path = os.path.join(DATASET_ROOT, split, source_label)
            for filename in sorted(os.listdir(class_path)):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                if source_label == "NORMAL":
                    label = "NORMAL"
                    pneumonia_type = "Aucune"
                else:
                    lowered = filename.lower()
                    if "virus" in lowered:
                        label = "PNEUMONIA_viral"
                        pneumonia_type = "virus"
                    elif "bacteria" in lowered:
                        label = "PNEUMONIA_bacteria"
                        pneumonia_type = "bacteria"
                    else:
                        continue

                rows.append({
                    "split": split,
                    "label": label,
                    "pneumonia_type": pneumonia_type,
                    "filename": filename,
                    "path": os.path.join(class_path, filename),
                })

    frame = pd.DataFrame(rows)
    frame["class_index"] = frame["label"].map(LABEL_TO_INDEX)
    return frame


def load_image(path, label_index):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label_index, depth=NUM_CLASSES)
    return image, label


def make_dataset(frame, shuffle=False):
    paths = frame["path"].to_numpy()
    labels = frame["class_index"].to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(frame), reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


df = build_split_dataframe()
train_val_df = df[df["split"].isin(["train", "val"])].copy()
test_df = df[df["split"] == "test"].copy()
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.20,
    random_state=42,
    stratify=train_val_df["label"],
)


# ## 2. Exploration des données
# 
# On regarde comment les images sont réparties par split et par classe (tableaux + diagramme en barres), et on affiche un exemple d'image de chaque classe. Comme dans l'autre notebook, ça permet de repérer le **déséquilibre** entre classes et de vérifier visuellement à quoi ressemblent les radios.

# In[28]:


# Exploration des données en amont sur 3 classes

print(df[["split", "label", "pneumonia_type", "filename"]].head())
print()
print(df.groupby(["split", "label"]).size().reset_index(name="count"))
print()
print(df[df["label"] == "PNEUMONIA_bacteria"].groupby("split").size())
print(df[df["label"] == "PNEUMONIA_viral"].groupby("split").size())

plt.figure(figsize=(10, 4))
sns.countplot(data=df, x="split", hue="label", order=SPLITS, hue_order=TARGET_LABELS)
plt.title("Répartition des images par split et par classe")
plt.xlabel("Split")
plt.ylabel("Nombre d'images")
plt.legend(title="Classe")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, label in zip(axes, TARGET_LABELS):
    sample = df[df["label"] == label].sample(1, random_state=42).iloc[0]
    image = tf.keras.utils.load_img(sample["path"], color_mode="grayscale")
    axis.imshow(image, cmap="gray")
    axis.set_title(label)
    axis.axis("off")

plt.tight_layout()
plt.show()


# # Explication des imports supplémentaires
# - `preprocess_input` : applique le prétraitement spécifique à ResNet50 (normalisation comme sur ImageNet), ce qui aligne les radios sur ce que le backbone a vu pendant son pré-entraînement et améliore la qualité des features extraites.
# - `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` : callbacks utilisés pour contrôler l'entraînement, éviter le sur-apprentissage et adapter le learning rate. En pratique, ils aident à obtenir une meilleure généralisation sur les trois classes du problème : `NORMAL`, `PNEUMONIA_bacteria` et `PNEUMONIA_viral`.

# ## 3. Rappel de la configuration
# 
# On refixe la taille des images et des batchs (déjà définies plus haut, c'est juste un rappel explicite avant de construire les datasets).

# In[29]:


IMG_SIZE = 224
BATCH_SIZE = 32


# ## 4. Construction des datasets train / val / test
# 
# On crée les trois jeux de données avec `make_dataset` et on affiche le nombre d'images par classe dans chacun. Seul le train est mélangé (`shuffle=True`). On retrouve ici le même déséquilibre de classes que dans l'autre notebook.

# In[30]:


train_ds = make_dataset(train_df, shuffle=True)
train_class_names = TARGET_LABELS
print("Classes d'entraînement:", train_class_names)
print(train_df["label"].value_counts().reindex(TARGET_LABELS, fill_value=0))


# In[31]:


val_ds = make_dataset(val_df)
print(val_df["label"].value_counts().reindex(TARGET_LABELS, fill_value=0))


# In[32]:


test_ds = make_dataset(test_df)
print(test_df["label"].value_counts().reindex(TARGET_LABELS, fill_value=0))


# ## 5. Data augmentation
# 
# Comme pour le CNN from scratch, on applique de la **data augmentation** sur le train : retournement horizontal, petite rotation, zoom et variation de contraste. Les commentaires du code expliquent pourquoi chaque transformation reste « raisonnable » sur des radios médicales (par exemple pas de retournement vertical : une radio à l'envers n'a pas de sens). Le but n'est pas d'ajouter des images, mais de montrer plus de variété au modèle à chaque epoch pour qu'il généralise mieux.

# In[33]:


# Augmentation des données avec des transformations aléatoires pour améliorer la généralisation du modèle
# Ces transformations sont choisies pour refléter des variations réalistes dans les images médicales
# et éviter de créer des images non représentatives ou médicalement incorrectes.
# ATTENTION : La data augmentation n’est pas de l’oversampling conceptuel, c’est une augmentation de support du manifold. (=/= une duplication des données avec SMOTE, Oversampling, etc.)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),   # Les radios peuvent être prises côté gauche ou droit. Anatomiquement valide (le cœur change juste de côté visuellement). Pas de flip vertical : une radio à l'envers n'a aucun sens médical.
    tf.keras.layers.RandomRotation(0.05),       # Petite rotation pour simuler des variations dans la prise de vue Patient peut être légèrement penché lors de la prise. Pas plus de ±5-7° : au-delà, ça déforme l'anatomie de manière non réaliste.
    tf.keras.layers.RandomZoom(0.1),            # Zoom léger pour simuler des variations de distance entre le patient et la machine à rayons X. Pas plus de ±10% : au-delà, ça peut couper des parties importantes de l'image.
    tf.keras.layers.RandomContrast(0.2),        # Variation de contraste pour simuler des différences dans les réglages de la machine à rayons X ou les conditions d'éclairage. Pas plus de ±20% : au-delà, ça peut rendre l'image irréaliste.
    ])                                          # !! Les pneumonies se manifestent par des variations de contraste dans les poumons. Une augmentation excessive pourrait masquer ces signes. !!!


"""
Comment ça marche concrètement:

À chaque epoch, les transformations changent :

Epoch 1 : Image 1 → flip + rotation -3°
Epoch 2 : Image 1 → pas de flip + rotation +1°
Epoch 3 : Image 1 → flip + zoom 7%
Etc.

Pourquoi ça marche alors ?
Sur 20 epochs :

Sans augmentation : Le modèle voit 624 images identiques 20 fois
Avec augmentation : Le modèle voit ~12,480 variations des 624 images (chaque epoch = nouvelles transformations)
Donc on augmente pas le nombre d'images, mais la diversité des images vues par le modèle.
"""


# ## 6. Prétraitement ResNet50 + pipeline tf.data
# 
# Étape importante et **spécifique au transfert learning** : on applique `preprocess_input`, le prétraitement exact que ResNet50 attend (la même normalisation qu'au moment de son pré-entraînement sur ImageNet). Sans ça, les images ne seraient pas dans le bon format pour le réseau et les résultats seraient moins bons.
# 
# On enchaîne data augmentation → preprocessing sur le train (sans cache, pour garder une augmentation différente à chaque epoch), et juste preprocessing + cache sur val/test. Le `prefetch` accélère l'alimentation du modèle.

# In[34]:


# On applique l'augmentation de données uniquement sur le jeu d'entraînement,
train_ds_augmented = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Puis on applique le prétraitement adapté à ResNet50 (normalisation spécifique ImageNet)
AUTOTUNE = tf.data.AUTOTUNE

# IMPORTANT : on ne met pas de cache sur le train après data_augmentation
train_ds_prep = train_ds_augmented.map(lambda x, y: (preprocess_input(x), y)).shuffle(1000).prefetch(AUTOTUNE)
val_ds_prep = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)
test_ds_prep = test_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)


# # Explication de la pipeline TF.data + prétraitement
# - `train_ds_augmented` : contient les images d'entraînement avec data augmentation (flip, rotation, zoom, contraste), ce qui augmente la diversité des exemples vus à chaque epoch et réduit le risque d'overfitting.
# - `train_ds_prep` : applique ensuite `preprocess_input`, mélange les données avec `shuffle(1000)` et utilise `prefetch(AUTOTUNE)` pour alimenter le GPU sans attendre l'I/O. On **ne met pas de cache** ici pour garder une augmentation aléatoire à chaque epoch.
# - `val_ds_prep` et `test_ds_prep` : on applique `preprocess_input` puis `cache().prefetch(AUTOTUNE)` (pas de data augmentation sur val/test). Cela stabilise et accélère l'évaluation sans changer les données.
# Toutes ces étapes visent à améliorer la qualité et la diversité des données vues par le modèle, ce qui permet de réduire faux positifs et faux négatifs dans la matrice de confusion.

# ## 7. Chargement du backbone ResNet50 pré-entraîné
# 
# On charge **ResNet50** avec ses poids appris sur ImageNet, mais sans sa dernière couche (`include_top=False`), car celle-ci servait à classer les 1000 catégories d'ImageNet, pas nos 3 classes.
# 
# Juste après, on **gèle** ce backbone (`base_model.trainable = False`) : ses poids ne bougeront pas pendant la première phase. On garde ainsi tout ce que ResNet50 a déjà appris, et on n'entraîne au début que la partie qu'on ajoute par-dessus.

# In[35]:


# Charger un modèle préentraîné (ResNet50) sans la couche finale
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[36]:


# Geler les couches de base pour ne pas les entraîner
base_model.trainable = False


# ## 8. Tête de classification
# 
# On branche par-dessus ResNet50 nos propres couches : un `GlobalAveragePooling2D` qui résume les features de ResNet50 en un vecteur, une couche `Dense(512)` + `ReLU`, un `Dropout(0.4)` contre le surapprentissage, et enfin une couche `Dense(3)` + `softmax` pour sortir les probabilités des 3 classes. C'est cette tête, légère, qu'on entraîne en premier.

# In[37]:


# Ajouter des couches personnalisées pour ta tâche de classification multi-classes
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')  # 3 classes : Normal, Pneumonia_bacteria, Pneumonia_viral
])


# # Explication de la tête du modèle
# - `GlobalAveragePooling2D` : compresse les cartes de features spatiales de ResNet50 en un vecteur global, ce qui réduit le nombre de paramètres et le risque d'overfitting.
# - `Dense(512, activation='relu')` : couche entièrement connectée plus compacte (512 neurones au lieu de 1024), suffisante pour séparer trois classes tout en limitant la complexité.
# - `Dropout(0.4)` : désactive aléatoirement 40 % des neurones pendant l'entraînement, ce qui force le modèle à ne pas trop dépendre de quelques features spécifiques et améliore la généralisation.
# Ensemble, ces choix aident le modèle à mieux généraliser sur des radios jamais vues, et donc à se rapprocher d'une matrice de confusion idéale.

# ## 9. Compilation
# 
# On configure l'entraînement avec l'optimiseur `Adam`, la perte `categorical_crossentropy` (adaptée ici car les labels sont en one-hot) et plusieurs métriques : accuracy, AUC et AUC-PR. On suit l'AUC car c'est une mesure robuste quand les classes sont déséquilibrées.

# In[38]:


# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=3),
        tf.keras.metrics.AUC(curve="PR", name="auc_pr", multi_label=True, num_labels=3)
    ]
)


# ## 10. Validation croisée (5-fold)
# 
# Comme dans l'autre notebook, on fait une validation croisée en 5 folds pour vérifier la stabilité du modèle. À chaque fold, on recrée un modèle propre (`clone_model`), on l'entraîne sur 4/5 des données et on le valide sur le 1/5 restant, avec ses propres callbacks. Ça permet de voir si les performances sont régulières d'un découpage à l'autre.

# In[39]:


# Cross-validation stratifiée sur le train fusionné (90%)
# La CV est une boucle extérieure : pour chaque fold, on recrée et réentraîne un modèle complet.
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train_df["path"].to_numpy()
y = train_df["class_index"].to_numpy()

cv_scores = []

for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n===== Fold {fold}/5 =====")

    fold_train_df = train_df.iloc[fold_train_idx].copy()
    fold_val_df = train_df.iloc[fold_val_idx].copy()

    fold_train_ds = make_dataset(fold_train_df, shuffle=True)
    fold_val_ds = make_dataset(fold_val_df)

    fold_train_ds_augmented = fold_train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    fold_train_ds_prep = fold_train_ds_augmented.map(lambda x, y: (preprocess_input(x), y)).shuffle(1000).prefetch(AUTOTUNE)
    fold_val_ds_prep = fold_val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)

    fold_model = tf.keras.models.clone_model(model)
    fold_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=3),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr", multi_label=True, num_labels=3),
        ],
    )

    fold_callbacks = [
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    fold_model.fit(
        fold_train_ds_prep,
        epochs=5,
        validation_data=fold_val_ds_prep,
        callbacks=fold_callbacks,
        verbose=1,
    )

    fold_scores = fold_model.evaluate(fold_val_ds_prep, verbose=0)
    cv_scores.append(fold_scores)
    print(f"Fold {fold} scores: {fold_scores}")

print("CV scores:", cv_scores)


# ## 11. Entraînement en deux phases : tête gelée puis fine-tuning
# 
# C'est le cœur du transfert learning, en deux temps :
# 1. **Phase 1** — on entraîne d'abord seulement la tête, ResNet50 restant gelé. Les callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`) surveillent l'entraînement et sauvegardent le meilleur modèle.
# 2. **Phase 2 (fine-tuning)** — on « dégèle » les 30 dernières couches de ResNet50 et on continue avec un learning rate très faible (`1e-5`). Le but est d'adapter en douceur les features de haut niveau de ResNet50 à nos radios, sans casser ce qu'il avait déjà appris.

# In[40]:


# Entraîner le modèle avec callbacks + fine-tuning
initial_epochs = 10

early_stopping = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_resnet_pneumonia_3class.h5',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, reduce_lr, checkpoint]

history = model.fit(
    train_ds_prep,
    epochs=initial_epochs,
    validation_data=val_ds_prep,
    callbacks=callbacks
)

# Phase de fine-tuning : on défige les derniers blocs de ResNet50
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # on garde gelées les couches basses

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=3),
        tf.keras.metrics.AUC(curve="PR", name="auc_pr", multi_label=True, num_labels=3)
    ]
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds_prep,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    validation_data=val_ds_prep,
    callbacks=callbacks
)


# # Explication de la stratégie d'entraînement et de fine-tuning
# ## Phase 1 : entraînement avec backbone gelé
# - On entraîne d'abord uniquement la tête du réseau avec `base_model.trainable = False` pendant `initial_epochs` epochs.
# - `EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True)` arrête l'entraînement quand l'AUC validation n'augmente plus et restaure les meilleurs poids, pour éviter le sur-apprentissage.
# - `ReduceLROnPlateau(monitor='val_loss')` réduit automatiquement le learning rate quand la perte de validation stagne, ce qui permet d'affiner les poids au lieu d'osciller.
# - `ModelCheckpoint(monitor='val_auc', save_best_only=True)` sauvegarde le modèle qui sépare le mieux les classes sur la validation.
# ## Phase 2 : fine-tuning de ResNet50
# - On rend `base_model.trainable = True`, puis on ne défige que les dernières couches (`fine_tune_at = len(base_model.layers) - 30`). Les couches basses restent gelées pour conserver les features génériques.
# - On re-compile avec un learning rate plus faible (`Adam(1e-5)`) et on continue l'entraînement en repartant de `initial_epochs`.
# - Cette étape permet d'adapter les features haut niveau de ResNet50 au domaine des radios thoraciques, ce qui améliore la séparation Normal/Pneumonia, surtout sur les cas difficiles.
# Globalement, cette stratégie en deux phases vise à augmenter AUC, précision et rappel, et donc à rapprocher la matrice de confusion de [[100 %, 0 %], [0 %, 100 %]].

# ## 12. Évaluation : matrice de confusion et rapport
# 
# On teste le modèle sur le jeu de test prétraité. On compare les prédictions (`argmax` des probabilités) aux vraies classes, et on affiche le `classification_report` et la **matrice de confusion** pour voir où le modèle se trompe, exactement comme dans le notebook du CNN from scratch.

# In[41]:


# matrice de confusion et rapport de classification (jeu de test prétraité)
y_true = np.concatenate([y for _, y in test_ds_prep], axis=0)
y_pred_prob = model.predict(test_ds_prep)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true_labels = np.argmax(y_true, axis=1)

print("Classes:", train_class_names)
cm = confusion_matrix(y_true_labels, y_pred)
print("Classification Report:\n", classification_report(y_true_labels, y_pred, target_names=train_class_names))
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_class_names, yticklabels=train_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (heatmap) - Test')
plt.show()


# ## 13. Récupération des probabilités prédites
# 
# On récupère ici, pour toutes les images de test, les vraies classes et les probabilités prédites par le modèle. On en a besoin pour tracer les courbes ROC et chercher les seuils dans les cellules suivantes.

# In[42]:


y_true = []
y_proba = []

for images, labels in test_ds_prep:
    y_true.extend(labels.numpy())
    y_proba.extend(model.predict(images))

y_true = np.array(y_true)
y_proba = np.array(y_proba)


# ## 14. Courbe ROC et AUC (one-vs-rest)
# 
# Comme pour le CNN from scratch, on trace une courbe ROC par classe (one-vs-rest) et on calcule l'AUC de chacune (1 = parfait, 0.5 = au hasard), plus l'AUC moyen. Ça mesure la capacité du modèle à bien séparer chaque classe des autres.

# In[43]:


# tracer la courbe ROC et calculer l’AUC en one-vs-rest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true_bin = y_true

plt.figure(figsize=(7, 7))
roc_aucs = []
for idx, class_name in enumerate(train_class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_proba[:, idx])
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    plt.plot(fpr, tpr, label=f"{class_name} AUC = {roc_auc:.3f}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - One vs Rest")
plt.legend()
plt.grid()
plt.show()

print("AUC moyen macro:", np.mean(roc_aucs))


# ## 15. Seuils de Youden par classe
# 
# On calcule pour chaque classe le **seuil de Youden**, c'est-à-dire le seuil qui offre le meilleur compromis entre vrais positifs et faux positifs sur la courbe ROC.

# In[44]:


# Calcul du seuil optimal en utilisant l’indice de Youden pour chaque classe (one-vs-rest)
youden_thresholds = {}
for idx, class_name in enumerate(train_class_names):
    fpr, tpr, thresholds = roc_curve(y_true_bin[:, idx], y_proba[:, idx])
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    youden_thresholds[class_name] = thresholds[best_idx]
    print(f"Seuil optimal (Youden) pour {class_name}: {thresholds[best_idx]:.3f}")


# ## 16. Décision multiclasses (argmax)
# 
# En multi-classes, on ne fixe pas un seul seuil global comme en binaire : la classe prédite est simplement celle qui a la plus grande probabilité (`argmax`). Ces cellules réaffichent le rapport et la matrice de confusion avec cette règle de décision, et rappellent les seuils de Youden par classe à titre indicatif.

# In[45]:


# En multiclasses, on ne retient pas un seuil binaire global.
# La décision finale repose sur l'argmax des probabilités et sur les seuils par classe calculés plus bas.
print("Décision multiclasses active : argmax(y_proba)")
cm = confusion_matrix(y_true_labels, y_pred)
print(classification_report(y_true_labels, y_pred, target_names=train_class_names))
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_class_names, yticklabels=train_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (heatmap) - Multiclass (argmax)')
plt.show()


# In[46]:


# Variante pour forcer un rappel élevé sur la classe Pneumonia_viral si nécessaire
# En multiclasses, on préfère en général ajuster les seuils par classe ou utiliser la matrice de confusion
# plutôt qu'un seuil binaire global.
print("Seuils Youden par classe:", youden_thresholds)


# In[ ]:


# Évaluer les performances du modèle avec les prédictions multiclasses
from sklearn.metrics import classification_report, confusion_matrix

# Ici on garde la décision argmax globale pour la classification multiclasses.
cm = confusion_matrix(y_true_labels, y_pred)
print(classification_report(y_true_labels, y_pred, target_names=train_class_names))
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_class_names, yticklabels=train_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (heatmap) - Évaluation')
plt.show()


# ## 17. Seuils F1 (précision-rappel) par classe
# 
# On cherche, pour chaque classe, le seuil qui maximise le **F1-score** à partir de la courbe précision-rappel (même logique que dans l'autre notebook). On réaffiche ensuite le rapport et la matrice de confusion ; la décision finale reste l'`argmax`, les seuils servant surtout d'analyse.

# In[48]:


# Seuils PR: en multiclasses, on calcule plutôt les scores PR one-vs-rest par classe.
from sklearn.metrics import precision_recall_curve

pr_thresholds = {}
for idx, class_name in enumerate(train_class_names):
    precision, recall, thresholds = precision_recall_curve(y_true_bin[:, idx], y_proba[:, idx])
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    pr_thresholds[class_name] = thresholds[best_idx]
    print(f"Seuil optimal F1 (PR) pour {class_name} : {thresholds[best_idx]:.3f}")


# In[49]:


# Évaluer les performances du modèle avec les seuils PR par classe
from sklearn.metrics import classification_report, confusion_matrix

print("Seuils PR par classe:", pr_thresholds)
print("Pour une décision finale multiclasses, on conserve argmax sur les probabilités.")
cm = confusion_matrix(y_true_labels, y_pred)
print(classification_report(y_true_labels, y_pred, target_names=train_class_names))
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_class_names, yticklabels=train_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (heatmap) - PR thresholds')
plt.show()


# ## 18. Analyse des faux négatifs / faux positifs
# 
# On regarde classe par classe le nombre de vrais positifs (TP), vrais négatifs (TN), faux positifs (FP) et faux négatifs (FN). C'est utile en contexte médical pour repérer, par exemple, sur quelle classe le modèle rate le plus de malades (FN).

# In[50]:


# Analyse simple des faux négatifs / faux positifs par classe en multiclasses
rows = []

for idx, class_name in enumerate(train_class_names):
    y_true_class = (y_true_labels == idx).astype(int)
    y_pred_class = (y_pred == idx).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_true_class, y_pred_class).ravel()
    rows.append({"class": class_name, "FN": FN, "FP": FP, "TP": TP, "TN": TN})

df = pd.DataFrame(rows)
print(df)


# ## 19. Matrice de confusion finale
# 
# Dernier rappel de la matrice de confusion et du rapport de classification, comme résumé final des performances du modèle sur le test.

# In[51]:


from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_true_labels, y_pred)
print(classification_report(y_true_labels, y_pred, target_names=train_class_names))
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_class_names, yticklabels=train_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (heatmap) - Résumé final')
plt.show()

