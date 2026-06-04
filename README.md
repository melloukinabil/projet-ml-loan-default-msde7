# Prediction du Defaut de Remboursement d'un Pret Immobilier

## Description du projet

Projet de Machine Learning (Module 5 - MSDE7, EHTP) portant sur la **prediction du defaut de remboursement d'un pret immobilier (mortgage)**.

Le modele est entraine sur un dataset de **148 670 dossiers de prets** contenant des variables demographiques, financieres et contractuelles. L'objectif est de classifier si un emprunteur sera en **defaut de paiement** (Status=1) ou non (Status=0).

## Dataset

- **Source** : [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
- **Taille** : 148 670 lignes, 34 colonnes
- **Target** : `Status` (1 = defaut, 0 = pas de defaut)
- **Type ML** : Classification binaire

## Structure du projet

```
app.py                    # Application Streamlit (deploiement principal)
flask_app.py              # Application Flask (bonus Render)
notebook.ipynb            # Notebook complet du pipeline ML
pipeline_loan_default.pkl # Pipeline serialise (modele + scaler + encoders)
Loan_Default.csv          # Dataset
requirements.txt          # Dependances Python
Procfile                  # Configuration Render (gunicorn)
README.md                 # Ce fichier
templates/index.html      # Template Flask
```

## Pipeline ML

1. **EDA** : Exploration des donnees, distributions, correlations, taux de defaut par variable
2. **Pre-processing** : Traitement des valeurs manquantes (mediane/mode), LabelEncoding, StandardScaler
3. **Modelisation** : 12 algorithmes testes (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, Bagging, Naive Bayes, XGBoost, MLP)
4. **Tuning** : Top 3 modeles optimises par RandomizedSearchCV
5. **Deploiement** : Pipeline serialise, Streamlit Cloud + Flask/Render

## Application deployee

- **Streamlit** : [Lien de l'application](https://votre-app.streamlit.app)
- **Flask (bonus)** : [Lien Render](https://votre-app.onrender.com)

## Installation locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Auteur

- **Nom** : MELLOUKI
- **Formation** : MSDE7 - Ecole Hassania des Travaux Publics
- **Module** : Machine Learning - Pr. Abdelhamid Fadil
- **Annee** : 2025/2026
