import os
import tempfile

import numpy as np
import streamlit as st
import cv2
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

with st.sidebar:
    st.markdown("<h1 style='text-align: center; '>PROJET ZOIDBERG</h1>", unsafe_allow_html=True)

    st.image("https://ibb.co/nNjMLCr8")

    st.markdown("<h1 style='text-align: center; '>Welcome to our pneumonia detection project</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; '>This application uses machine learning on chest X-ray images to help detect signs of pneumonia.</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; '>Project objective: build an AI model capable of assisting medical diagnosis from radiographic images.</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; '>Disclaimer: This tool is for educational purposes only and must not be used as a substitute for professional medical diagnosis.</h4>", unsafe_allow_html=True)

    st.write("L'Equipe de choc:")
    st.markdown("[Belard Chloé](https://github.com/chloe-bel)")
    st.markdown("[CONAN Sylvain](https://github.com/SylvainMJC)")
    st.markdown("[GUERIZEC Léo](https://github.com/)")
    st.markdown("[NADER Mehdi-Michel](https://github.com/mmnader44)")



CLASSES = ["normal", "bacterien", "viral"]

# =====================================================================
# 1) PRÉTRAITEMENT ML
# =====================================================================
def preprocess_image(path, size=(130, 95), apply_clahe=True):
    img_raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise ValueError("Impossible de charger l'image avec OpenCV.")

    # A. Suppression des bordures noires
    _, thresh_crop = cv2.threshold(img_raw, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img_raw[y:y + h, x:x + w]
    else:
        img = img_raw.copy()

    # B. Suppression des lettres parasites (L / R) par inpainting
    _, text_mask = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    img = cv2.inpaint(img, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # C. CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # D. Redimensionnement + normalisation 0-1
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img / 255.0


def features_ml(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        arr = preprocess_image(tmp_path, size=(130, 95), apply_clahe=True)
    finally:
        os.remove(tmp_path)
    return arr.reshape(1, -1)  # (1, 12350)


# =====================================================================
# 2) PRÉTRAITEMENT CNN
# =====================================================================
def features_cnn_scratch(pil_img, size=(224, 224)):
    x = np.array(pil_img.convert("RGB").resize(size, Image.BILINEAR)).astype("float32")
    return np.expand_dims(x, axis=0)


def features_cnn_tl(pil_img, size=(224, 224)):
    x = np.array(pil_img.convert("RGB").resize(size, Image.BILINEAR)).astype("float32")
    x = preprocess_input(x)
    return np.expand_dims(x, axis=0)


# =====================================================================
# Chargement des modèles  
# =====================================================================
@st.cache_resource
def load_models():
    models = {}

    models["rfc"] = joblib.load(ROOT_DIR / "models" / "rfc.joblib")
    models["brfc"] = joblib.load(ROOT_DIR / "models" / "brfc.joblib")

    # SVC (pipeline complet)
    models["svc"] = joblib.load(ROOT_DIR / "models" / "scaler_pca_svc.joblib")

    models["knn"] = joblib.load(ROOT_DIR / "models" / "knn_final.joblib")
    models["knn_scaler"] = joblib.load(ROOT_DIR / "models" / "scaler_knn.joblib")

    models["cnn"] = load_model(ROOT_DIR / "models" / "cnn.h5")
    models["cnn_tl"] = load_model(ROOT_DIR / "models" / "cnn_tl.h5")

    return models


m = load_models()

st.title("Zoidscanner")

st.markdown("""
Cet outil permet d’analyser des radiographies pulmonaires afin d’aider à la détection de trois classes :

- Normal : absence de signe de maladie  
- Pneumonie virale : infection pulmonaire d’origine virale  
- Pneumonie bactérienne : infection pulmonaire d’origine bactérienne  

Cet outil est développé à des fins pédagogiques et ne remplace pas un diagnostic médical professionnel.
""")



uploaded = st.file_uploader("Uploader une image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image chargée", width=300)

    # --- Features ---
    x_ml    = features_ml(uploaded)       # (1, 12350) pour RFC / BRFC / KNN
    x_cnn   = features_cnn_scratch(img)   # CNN from scratch
    x_cnn_tl = features_cnn_tl(img)       # ResNet50

    # --- Prédictions ML ---
    pred_rfc  = CLASSES[int(m["rfc"].predict(x_ml)[0])]
    pred_brfc = CLASSES[int(m["brfc"].predict(x_ml)[0])]

    # SVC désactivé (scaler/pca manquants)
    #x_svc    = m["svc_pca"].transform(m["svc_scaler"].transform(x_ml))
    # pred_svc = CLASSES[int(m["svc"].predict(x_svc)[0])]

    # SVC : Pipeline complet (scaler + PCA + SVC encapsulés)
    pred_svc = CLASSES[int(m["svc"].predict(x_ml)[0])]

    # KNN : scaler puis prédiction
    x_knn    = m["knn_scaler"].transform(x_ml)
    pred_knn = CLASSES[int(m["knn"].predict(x_knn)[0])]

    # --- Prédictions CNN ---
    pred_cnn    = CLASSES[int(np.argmax(m["cnn"].predict(x_cnn)[0]))]
    pred_cnn_tl = CLASSES[int(np.argmax(m["cnn_tl"].predict(x_cnn_tl)[0]))]

    st.subheader("Résultats")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ML")
        st.write("Random Forest (RFC) :",              pred_rfc)
        st.write("Balanced Random Forest (BRFC) :",    pred_brfc)
        st.write("SVC :",                            pred_svc)
        st.write("KNN :",                              pred_knn)
    with col2:
        st.markdown("### CNN")
        st.write("CNN (entraîné de zéro) :",           pred_cnn)
        st.write("CNN Transfer Learning (ResNet50) :", pred_cnn_tl)
