import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# ---------- Chargement des données ----------
diabetes = pd.read_csv("diabetes.csv")

# Remplacement des zéros anormaux par la médiane pour certaines colonnes
for col in ['BloodPressure', 'Glucose', 'BMI', 'Insulin', 'SkinThickness']:
    diabetes[col] = diabetes[col].replace(0, diabetes[col].median())

# Séparation des variables
X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']

# Standardisation
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Division du jeu de données
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

# ---------- Modèle ----------
classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(X_train, Y_train)

# Évaluation
train_pred = classifier.predict(X_train)
test_pred = classifier.predict(X_test)
train_acc = accuracy_score(Y_train, train_pred)
test_acc = accuracy_score(Y_test, test_pred)

# ---------- Interface Streamlit ----------
st.set_page_config(page_title="Prédiction Diabète", layout="centered")
st.title("🩺 Prédiction du Diabète (Modèle fiable à 78%)")

st.write(f"### 🔍 Précision entraînement : {train_acc*100:.2f}%")
st.write(f"### 🧪 Précision test : {test_acc*100:.2f}%")
st.markdown("---")
st.write("## Veuillez saisir les données médicales ci-dessous :")

# --- Saisie utilisateur ---
pregnancies = st.number_input('Grossesses', 0, 20, 1)
glucose = st.number_input('Glucose (mg/dL)', 0, 200, 120)
blood_pressure = st.number_input('Pression artérielle (mmHg)', 0, 140, 70)
skin_thickness = st.number_input('Épaisseur de la peau (mm)', 0, 100, 20)
insulin = st.number_input("Insuline (µU/mL)", 0, 900, 79)
bmi = st.number_input("IMC", 0.0, 70.0, 25.0)
dpf = st.number_input("Fonction héréditaire (DPF)", 0.0, 3.0, 0.5)
age = st.number_input("Âge", 10, 120, 30)

# --- Prédiction ---
if st.button("🔎 Prédire"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = classifier.predict(input_scaled)
    proba = classifier.predict_proba(input_scaled)[0][prediction[0]]

    st.subheader("🔬 Résultat de la prédiction :")
    if prediction[0] == 0:
        st.success(f"✅ La personne **n'est pas diabétique** — confiance : **{proba:.2%}**")
    else:
        st.error(f"⚠️ La personne **est diabétique** — confiance : **{proba:.2%}**")

# --- Bouton de réinitialisation ---
if st.button("🔄 Recommencer"):
    st.rerun()
