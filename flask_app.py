from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import traceback

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# === CHARGEMENT PIPELINE ===
pipeline = None
try:
    with open('pipeline_loan_default.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    print("✅ Pipeline chargé avec succès")
except Exception as e:
    print(f"❌ Erreur chargement pipeline: {e}")

model = pipeline['model'] if pipeline else None
scaler = pipeline['scaler'] if pipeline else None
label_encoders = pipeline['label_encoders'] if pipeline else None
feature_names = pipeline.get('feature_names', []) if pipeline else []
categorical_cols = pipeline.get('categorical_cols', []) if pipeline else []

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise Exception("Modèle non chargé")

        # Récupération des données du formulaire
        data = {
            'year': int(request.form.get('year', 2020)),
            'loan_limit': request.form.get('loan_limit'),
            'Gender': request.form.get('Gender'),
            'approv_in_adv': request.form.get('approv_in_adv'),
            'loan_type': request.form.get('loan_type'),
            'loan_purpose': request.form.get('loan_purpose'),
            'Credit_Worthiness': request.form.get('Credit_Worthiness'),
            'open_credit': request.form.get('open_credit'),
            'business_or_commercial': request.form.get('business_or_commercial'),
            'loan_amount': float(request.form.get('loan_amount', 200000)),
            'term': float(request.form.get('term', 360)),
            'Neg_ammortization': request.form.get('Neg_ammortization'),
            'interest_only': request.form.get('interest_only'),
            'lump_sum_payment': request.form.get('lump_sum_payment'),
            'property_value': float(request.form.get('property_value', 250000)),
            'construction_type': request.form.get('construction_type'),
            'occupancy_type': request.form.get('occupancy_type'),
            'Secured_by': request.form.get('Secured_by'),
            'total_units': request.form.get('total_units'),
            'income': float(request.form.get('income', 5000)),
            'credit_type': request.form.get('credit_type'),
            'Credit_Score': int(request.form.get('Credit_Score', 700)),
            'co-applicant_credit_type': request.form.get('co-applicant_credit_type'),
            'age': request.form.get('age'),
            'submission_of_application': request.form.get('submission_of_application'),
            'LTV': float(request.form.get('LTV', 80.0)),
            'Region': request.form.get('Region'),
            'Security_Type': request.form.get('Security_Type'),
            'dtir1': float(request.form.get('dtir1', 40.0))
        }

        # Prétraitement
        df_input = pd.DataFrame([data])
        for col in categorical_cols:
            if col in df_input.columns:
                try:
                    df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
                except:
                    df_input[col] = 0

        df_input = df_input.reindex(columns=feature_names, fill_value=0)
        X_input = scaler.transform(df_input)

        prediction = int(model.predict(X_input)[0])
        proba = float(model.predict_proba(X_input)[0][1])

        result = {
            'prediction': prediction,
            'label': 'DEFAUT' if prediction == 1 else 'PAS DE DEFAUT',
            'proba_default': round(proba * 100, 2),
            'risque_niveau': 'Très élevé' if proba >= 0.75 else 'Élevé' if proba >= 0.5 else 'Modéré' if proba >= 0.25 else 'Faible'
        }

        return render_template('index.html', result=result)

    except Exception as e:
        print("❌ ERREUR dans /predict :")
        print(traceback.format_exc())
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
