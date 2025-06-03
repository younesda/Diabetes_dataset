import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Charger les données
diabetes = pd.read_csv("diabetes.csv")

# Nettoyage
for col in ['BloodPressure', 'Glucose', 'BMI', 'Insulin', 'SkinThickness']:
    diabetes[col] = diabetes[col].replace(0, diabetes[col].median())

X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']

# Standardisation
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Modèle
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Interface Streamlit
st.set_page_config(page_title="Prédiction Diabète", layout="centered")
st.title("🩺 Prédiction du Diabète (Modèle fiable à 78%)")

st.write("## Veuillez saisir les données médicales ci-dessous :")

# --- Saisie utilisateur ---
pregnancies = st.number_input('Grossesses', min_value=0, max_value=20, value=1)
st.progress(min(pregnancies / 20, 1.0))
glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=200, value=120)
st.progress(min(glucose / 200, 1.0))
blood_pressure = st.number_input('Pression artérielle (mmHg)', min_value=0, max_value=122, value=70)
st.progress(min(blood_pressure / 122, 1.0))
skin_thickness = st.number_input('Épaisseur de la peau (mm)', min_value=0, max_value=100, value=20)
st.progress(min(skin_thickness / 100, 1.0))
insulin = st.number_input("Insuline (µU/mL)", min_value=0, max_value=846, value=79)
st.progress(min(insulin / 846, 1.0))
bmi = st.number_input("IMC", min_value=0.0, max_value=70.0, value=25.0)
st.progress(min(bmi / 70.0, 1.0))
dpf = st.number_input("Fonction d'hérédité (DPF)", min_value=0.0, max_value=3.0, value=0.5)
st.progress(min(dpf / 3.0, 1.0))
age = st.number_input("Âge", min_value=10, max_value=100, value=30)
st.progress(min(age / 100, 1.0))

# --- Prédiction ---
if st.button("🧪 Prédire"):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age)
    input_np = np.asarray(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = classifier.predict(input_scaled)

    st.subheader("🔎 Résultat de la prédiction :")
    if prediction[0] == 0:
        st.success("✅ La personne **n'est pas diabétique**.")
    else:
        st.error("⚠️ La personne **est diabétique**.")

# --- Bouton de rafraîchissement ---
if st.button("🔄 Recommencer"):
    st.rerun()