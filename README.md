# 🩺 Application de Prédiction du Diabète avec Streamlit & XGBoost

## 📌 Description

Cette application web permet de **prédire si une personne est diabétique ou non** à partir de données médicales simples. Elle utilise **XGBoost**, un algorithme de machine learning puissant, et est déployée avec **Streamlit** pour une interface intuitive et accessible.
Au départ, j'ai utilisé un modèle SVM (Support Vector Machine) avec un noyau linéaire. Les performances étaient les suivantes :

🎯 Précision entraînement (train) : 78%

🧪 Précision test : 77%

Bien que ces résultats soient acceptables, j'ai souhaité optimiser le modèle pour obtenir de meilleures performances d'apprentissage.

J'ai donc expérimenté un algorithme plus robuste : XGBoost (Extreme Gradient Boosting), en conservant les mêmes données et la même méthode de standardisation. Résultat :

✅ Précision entraînement : 88%

🧪 Précision test : toujours 77%

👉 Cela signifie que le modèle XGBoost apprend mieux sur les données sans pour autant suradapter (overfitting), car la précision test reste stable.

Cette comparaison m’a permis de comprendre que :

XGBoost est plus expressif et puissant que SVM sur ce type de données tabulaires.

Une précision test stable indique une bonne généralisation, même si l’amélioration du score test nécessite davantage de données ou d'ingénierie de variables.

---

## 🎯 Objectif

L'objectif de ce projet est de :
- Construire un modèle de classification fiable à partir du dataset **Pima Indians Diabetes**.
- Fournir une **interface utilisateur conviviale** pour tester la prédiction.
- Permettre à toute personne (même non technique) d’utiliser un outil de détection précoce du diabète.

---

## ⚙️ Fonctionnalités

- Prétraitement des données (nettoyage et standardisation)
- Utilisation de **XGBoostClassifier** pour améliorer les performances
- Interface **Streamlit** permettant :
  - La saisie manuelle des données médicales
  - Une prédiction instantanée (avec niveau de confiance)
  - Un bouton de **réinitialisation** rapide

---

## 📊 Données utilisées

Le fichier `diabetes.csv` contient les colonnes suivantes :

| Colonne           | Description                                |
|-------------------|---------------------------------------------|
| Pregnancies       | Nombre de grossesses                        |
| Glucose           | Taux de glucose dans le sang               |
| BloodPressure     | Pression artérielle                         |
| SkinThickness     | Épaisseur du pli cutané                     |
| Insulin           | Taux d'insuline                             |
| BMI               | Indice de masse corporelle                  |
| DiabetesPedigreeFunction (DPF) | Antécédents familiaux               |
| Age               | Âge de la personne                          |
| Outcome           | 1 = diabétique, 0 = non diabétique          |

---

## 🧠 Modèle d’apprentissage automatique

- **Algorithme** : XGBoost (classification binaire)
- **Standardisation** : `StandardScaler`
- **Découpage** : 80% entraînement / 20% test
- **Performance obtenue** :
  - Précision entraînement : `~100%`
  - Précision test : `~78%`

> 🔍 La précision test peut varier selon les paramètres du modèle.

---

## 🖥️ Interface utilisateur (Streamlit)

### Champs à remplir :
- Grossesses
- Glucose
- Pression artérielle
- Épaisseur de la peau
- Insuline
- IMC
- DPF (antécédents familiaux)
- Âge

### Résultat :
- Affichage du diagnostic : `diabétique` ou `non diabétique`
- Niveau de confiance (probabilité)

---

## 🚀 Lancer l'application localement

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
