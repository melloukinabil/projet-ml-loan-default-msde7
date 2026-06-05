import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuration de la page
st.set_page_config(
    page_title="Prediction Defaut Pret Immobilier",
    page_icon="\U0001F3E6",
    layout="wide"
)

# Chargement du pipeline
@st.cache_resource
def load_pipeline():
    with open('pipeline_loan_default.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()
model = pipeline['model']
scaler = pipeline['scaler']
label_encoders = pipeline['label_encoders']
feature_names = pipeline['feature_names']
categorical_cols = pipeline['categorical_cols']
numerical_cols = pipeline['numerical_cols']

# Debug info sidebar
import sys
with st.sidebar:
    st.caption(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    st.caption(f"Features attendues: {len(feature_names)}")
    with st.expander("Feature names"):
        st.write(feature_names)

# Titre
st.title("\U0001F3E6 Prediction du Defaut de Remboursement - Pret Immobilier")
st.markdown("Application de prediction basee sur un modele de Machine Learning entraine sur 148 670 dossiers de prets.")
st.markdown("---")

# Choix du mode
mode = st.radio("Mode de saisie :", ["Saisie manuelle", "Upload CSV"], horizontal=True)

if mode == "Saisie manuelle":
    st.subheader("Caracteristiques du pret")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.selectbox("Annee", [2019, 2020, 2021, 2022, 2023, 2024])
        loan_limit = st.selectbox("Loan Limit", ["cf", "ncf"])
        gender = st.selectbox("Genre", ["Male", "Female", "Sex Not Available", "Joint"])
        approv_in_adv = st.selectbox("Approbation anticipee", ["pre", "nopre"])
        loan_type = st.selectbox("Type de pret", ["type1", "type2", "type3"])
        loan_purpose = st.selectbox("Objectif du pret", ["p1", "p2", "p3", "p4"])
        credit_worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
        open_credit = st.selectbox("Open Credit", ["nopc", "opc"])
        business_or_commercial = st.selectbox("Business/Commercial", ["nob/c", "b/c"])
        loan_amount = st.number_input("Montant du pret ($)", min_value=10000, max_value=2000000, value=200000, step=10000)
    
    with col2:
        term = st.number_input("Duree du pret (mois)", min_value=60, max_value=480, value=360, step=12)
        neg_ammortization = st.selectbox("Amortissement negatif", ["not_neg", "neg_amm"])
        interest_only = st.selectbox("Interet seulement", ["not_int", "int_only"])
        lump_sum_payment = st.selectbox("Paiement forfaitaire", ["not_lpsm", "lpsm"])
        property_value = st.number_input("Valeur du bien ($)", min_value=20000, max_value=5000000, value=250000, step=10000)
        construction_type = st.selectbox("Type de construction", ["sb", "mh"])
        occupancy_type = st.selectbox("Type d'occupation", ["pr", "sr", "ir"])
    
    with col3:
        secured_by = st.selectbox("Garanti par", ["home", "land"])
        total_units = st.selectbox("Nombre d'unites", ["1U", "2U", "3U", "4U"])
        income = st.number_input("Revenu annuel ($)", min_value=500, max_value=1000000, value=5000, step=500)
        credit_type = st.selectbox("Type de credit", ["EXP", "EQUI", "CRIF", "CIB"])
        credit_score = st.number_input("Score de credit", min_value=300, max_value=900, value=700, step=10)
        co_applicant_credit_type = st.selectbox("Type credit co-emprunteur", ["CIB", "EXP"])
        age = st.selectbox("Tranche d'age", ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"])
        submission_of_application = st.selectbox("Soumission", ["to_inst", "not_inst"])
        ltv = st.number_input("LTV (%)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
        region = st.selectbox("Region", ["North", "south", "central", "North-East"])
        security_type = st.selectbox("Type de securite", ["direct", "Indriect"])
        dtir1 = st.number_input("DTIR (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)

    if st.button("\U0001F50D Predire le risque de defaut", type="primary"):
        # Construction du dataframe
        input_data = {
            'year': year,
            'loan_limit': loan_limit,
            'Gender': gender,
            'approv_in_adv': approv_in_adv,
            'loan_type': loan_type,
            'loan_purpose': loan_purpose,
            'Credit_Worthiness': credit_worthiness,
            'open_credit': open_credit,
            'business_or_commercial': business_or_commercial,
            'loan_amount': loan_amount,
            'term': term,
            'Neg_ammortization': neg_ammortization,
            'interest_only': interest_only,
            'lump_sum_payment': lump_sum_payment,
            'property_value': property_value,
            'construction_type': construction_type,
            'occupancy_type': occupancy_type,
            'Secured_by': secured_by,
            'total_units': total_units,
            'income': income,
            'credit_type': credit_type,
            'Credit_Score': credit_score,
            'co-applicant_credit_type': co_applicant_credit_type,
            'age': age,
            'submission_of_application': submission_of_application,
            'LTV': ltv,
            'Region': region,
            'Security_Type': security_type,
            'dtir1': dtir1
        }
        
        df_input = pd.DataFrame([input_data])
        
        # Encodage
        for col in categorical_cols:
            if col in df_input.columns:
                try:
                    df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
                except ValueError:
                    df_input[col] = 0
        
        # Reordonner les colonnes et verifier
        missing_cols = [c for c in feature_names if c not in df_input.columns]
        extra_cols = [c for c in df_input.columns if c not in feature_names]
        if missing_cols:
            st.error(f"Colonnes manquantes dans l'input: {missing_cols}")
            st.stop()
        if extra_cols:
            df_input = df_input.drop(columns=extra_cols)
        df_input = df_input[feature_names]
        
        # Normalisation
        X_input = scaler.transform(df_input)
        
        # Prediction
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        
        st.markdown("---")
        st.subheader("Resultat de la prediction")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            if prediction == 1:
                st.error(f"\U000026A0\U0000FE0F **DEFAUT PREDIT** - Risque eleve de non-remboursement")
            else:
                st.success(f"\U00002705 **PAS DE DEFAUT** - Remboursement prevu normal")
        
        with col_r2:
            st.metric("Probabilite de defaut", f"{proba[1]*100:.1f}%")
            st.metric("Confiance", f"{max(proba)*100:.1f}%")
        
        # Barre de probabilite
        st.progress(proba[1])
        st.caption(f"Probabilites : Pas de defaut = {proba[0]*100:.1f}% | Defaut = {proba[1]*100:.1f}%")

elif mode == "Upload CSV":
    st.subheader("Upload d'un fichier CSV")
    st.info("Le fichier doit contenir les memes colonnes que le dataset d'entrainement (sans ID et Status).")
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"Fichier charge : {df_upload.shape[0]} lignes, {df_upload.shape[1]} colonnes")
        st.dataframe(df_upload.head())
        
        if st.button("\U0001F50D Predire pour tous les dossiers", type="primary"):
            # Preprocessing
            if 'ID' in df_upload.columns:
                df_upload = df_upload.drop('ID', axis=1)
            if 'Status' in df_upload.columns:
                df_upload = df_upload.drop('Status', axis=1)
            # Supprimer colonnes leaky
            for c in ['Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']:
                if c in df_upload.columns:
                    df_upload = df_upload.drop(c, axis=1)
            
            # Traitement valeurs manquantes
            for col in numerical_cols:
                if col in df_upload.columns:
                    df_upload[col].fillna(df_upload[col].median(), inplace=True)
            for col in categorical_cols:
                if col in df_upload.columns:
                    df_upload[col].fillna(df_upload[col].mode()[0], inplace=True)
            
            # Encodage
            for col in categorical_cols:
                if col in df_upload.columns:
                    try:
                        df_upload[col] = label_encoders[col].transform(df_upload[col].astype(str))
                    except ValueError:
                        df_upload[col] = 0
            
            df_upload = df_upload[feature_names]
            X_upload = scaler.transform(df_upload)
            
            predictions = model.predict(X_upload)
            probas = model.predict_proba(X_upload)[:, 1]
            
            results = pd.DataFrame({
                'Prediction': ['Defaut' if p == 1 else 'Pas de defaut' for p in predictions],
                'Probabilite_Defaut (%)': (probas * 100).round(2)
            })
            
            st.subheader("Resultats")
            st.dataframe(results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dossiers en defaut", f"{sum(predictions)}/{len(predictions)}")
            with col2:
                st.metric("Taux de defaut predit", f"{sum(predictions)/len(predictions)*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("*Projet ML MSDE7 - EHTP - Pr. Fadil - 2025/2026*")
