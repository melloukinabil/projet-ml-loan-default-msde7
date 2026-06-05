from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Configuration pour Render
app.config['JSON_AS_ASCII'] = False

# Chargement du pipeline une seule fois
try:
    with open('pipeline_loan_default.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    model = pipeline['model']
    scaler = pipeline['scaler']
    label_encoders = pipeline['label_encoders']
    feature_names = pipeline['feature_names']
    categorical_cols = pipeline['categorical_cols']
    numerical_cols = pipeline['numerical_cols']
except Exception as e:
    print(f"Erreur chargement pipeline: {e}")
    model = None

def preprocess_input(data_dict):
    df_input = pd.DataFrame([data_dict])
    
    for col in categorical_cols:
        if col in df_input.columns:
            try:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
            except:
                df_input[col] = 0
    
    # Réordonner et filtrer
    df_input = df_input.reindex(columns=feature_names, fill_value=0)
    X_input = scaler.transform(df_input)
    return X_input


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Modèle non chargé")
    
    try:
        data = {
            'year': int(request.form.get('year', 2020)),
            'loan_limit': request.form.get('loan_limit', 'cf'),
            'Gender': request.form.get('Gender', 'Male'),
            'approv_in_adv': request.form.get('approv_in_adv', 'nopre'),
            'loan_type': request.form.get('loan_type', 'type1'),
            'loan_purpose': request.form.get('loan_purpose', 'p1'),
            'Credit_Worthiness': request.form.get('Credit_Worthiness', 'l1'),
            'open_credit': request.form.get('open_credit', 'nopc'),
            'business_or_commercial': request.form.get('business_or_commercial', 'nob/c'),
            'loan_amount': float(request.form.get('loan_amount', 200000)),
            'term': float(request.form.get('term', 360)),
            'Neg_ammortization': request.form.get('Neg_ammortization', 'not_neg'),
            'interest_only': request.form.get('interest_only', 'not_int'),
            'lump_sum_payment': request.form.get('lump_sum_payment', 'not_lpsm'),
            'property_value': float(request.form.get('property_value', 250000)),
            'construction_type': request.form.get('construction_type', 'sb'),
            'occupancy_type': request.form.get('occupancy_type', 'pr'),
            'Secured_by': request.form.get('Secured_by', 'home'),
            'total_units': request.form.get('total_units', '1U'),
            'income': float(request.form.get('income', 5000)),
            'credit_type': request.form.get('credit_type', 'EXP'),
            'Credit_Score': int(request.form.get('Credit_Score', 700)),
            'co-applicant_credit_type': request.form.get('co-applicant_credit_type', 'CIB'),
            'age': request.form.get('age', '35-44'),
            'submission_of_application': request.form.get('submission_of_application', 'to_inst'),
            'LTV': float(request.form.get('LTV', 80.0)),
            'Region': request.form.get('Region', 'North'),
            'Security_Type': request.form.get('Security_Type', 'direct'),
            'dtir1': float(request.form.get('dtir1', 40.0))
        }

        X_input = preprocess_input(data)
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]

        result = {
            'prediction': int(prediction),
            'label': 'DEFAUT' if prediction == 1 else 'PAS DE DEFAUT',
            'proba_default': round(float(proba) * 100, 2),
            'risque_niveau': 'Très élevé' if proba >= 0.75 else 'Élevé' if proba >= 0.5 else 'Modéré' if proba >= 0.25 else 'Faible'
        }

        return render_template('index.html', result=result, form_data=data)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
