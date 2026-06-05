from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

app = Flask(__name__)

# Chargement du pipeline
with open('pipeline_loan_default.pkl', 'rb') as f:
    pipeline = pickle.load(f)

model = pipeline['model']
scaler = pipeline['scaler']
label_encoders = pipeline['label_encoders']
feature_names = pipeline['feature_names']
categorical_cols = pipeline['categorical_cols']
numerical_cols = pipeline['numerical_cols']


def preprocess_input(data_dict):
    """Pretraitement d'un dictionnaire de features pour la prediction."""
    df_input = pd.DataFrame([data_dict])
    
    for col in categorical_cols:
        if col in df_input.columns:
            try:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
            except (ValueError, KeyError):
                df_input[col] = 0
    
    df_input = df_input[feature_names]
    X_input = scaler.transform(df_input)
    return X_input


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'year': int(request.form.get('year', 2019)),
            'loan_limit': request.form.get('loan_limit', 'cf'),
            'Gender': request.form.get('Gender', 'Male'),
            'approv_in_adv': request.form.get('approv_in_adv', 'nopre'),
            'loan_type': request.form.get('loan_type', 'type1'),
            'loan_purpose': request.form.get('loan_purpose', 'p1'),
            'Credit_Worthiness': request.form.get('Credit_Worthiness', 'l1'),
            'open_credit': request.form.get('open_credit', 'nopc'),
            'business_or_commercial': request.form.get('business_or_commercial', 'nob/c'),
            'loan_amount': float(request.form.get('loan_amount', 200000)),
            'rate_of_interest': float(request.form.get('rate_of_interest', 4.5)),
            'Interest_rate_spread': float(request.form.get('Interest_rate_spread', 0.5)),
            'Upfront_charges': float(request.form.get('Upfront_charges', 500)),
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
            'LTV': float(request.form.get('LTV', 80)),
            'Region': request.form.get('Region', 'North'),
            'Security_Type': request.form.get('Security_Type', 'direct'),
            'dtir1': float(request.form.get('dtir1', 40))
        }
        
        X_input = preprocess_input(data)
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        
        result = {
            'prediction': int(prediction),
            'label': 'Defaut' if prediction == 1 else 'Pas de defaut',
            'proba_no_default': round(float(proba[0]) * 100, 2),
            'proba_default': round(float(proba[1]) * 100, 2),
            'confidence': round(float(max(proba)) * 100, 2)
        }
        
        return render_template('index.html', result=result, form_data=data)
        
    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API REST pour la prediction."""
    try:
        data = request.get_json()
        X_input = preprocess_input(data)
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'label': 'Defaut' if prediction == 1 else 'Pas de defaut',
            'proba_no_default': round(float(proba[0]), 4),
            'proba_default': round(float(proba[1]), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
