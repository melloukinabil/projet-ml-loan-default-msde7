# flask_app.py
import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' in request.files:  # Mode Upload CSV
                file = request.files['file']
                if file.filename == '':
                    return render_template('index.html', error="Aucun fichier sélectionné")
                
                df = pd.read_csv(file)
                
                # Preprocessing identique à Streamlit
                if 'ID' in df.columns:
                    df = df.drop('ID', axis=1)
                if 'Status' in df.columns:
                    df = df.drop('Status', axis=1)
                
                # Supprimer colonnes leaky
                for col in ['Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']:
                    if col in df.columns:
                        df = df.drop(col, axis=1)
                
                # Gestion NaN
                for col in numerical_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].median())
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].mode()[0])
                
                # Encodage
                for col in categorical_cols:
                    if col in df.columns and col in label_encoders:
                        df[col] = label_encoders[col].transform(df[col].astype(str))
                
                df = df.reindex(columns=feature_names)
                X = scaler.transform(df)
                
                predictions = model.predict(X)
                probas = model.predict_proba(X)[:, 1]
                
                results = pd.DataFrame({
                    'Prediction': ['Défaut' if p == 1 else 'Pas de défaut' for p in predictions],
                    'Risque (%)': (probas * 100).round(2)
                })
                
                return render_template('index.html', 
                                     results=results.to_html(classes='table table-striped', index=False),
                                     mode="csv")

            else:  # Mode Saisie Manuelle
                input_data = {
                    'year': int(request.form['year']),
                    'loan_limit': request.form['loan_limit'],
                    'Gender': request.form['Gender'],
                    'approv_in_adv': request.form['approv_in_adv'],
                    'loan_type': request.form['loan_type'],
                    'loan_purpose': request.form['loan_purpose'],
                    'Credit_Worthiness': request.form['Credit_Worthiness'],
                    'open_credit': request.form['open_credit'],
                    'business_or_commercial': request.form['business_or_commercial'],
                    'loan_amount': float(request.form['loan_amount']),
                    'term': int(request.form['term']),
                    'Neg_ammortization': request.form['Neg_ammortization'],
                    'interest_only': request.form['interest_only'],
                    'lump_sum_payment': request.form['lump_sum_payment'],
                    'property_value': float(request.form['property_value']),
                    'construction_type': request.form['construction_type'],
                    'occupancy_type': request.form['occupancy_type'],
                    'Secured_by': request.form['Secured_by'],
                    'total_units': request.form['total_units'],
                    'income': float(request.form['income']),
                    'credit_type': request.form['credit_type'],
                    'Credit_Score': int(request.form['Credit_Score']),
                    'co-applicant_credit_type': request.form['co-applicant_credit_type'],
                    'age': request.form['age'],
                    'submission_of_application': request.form['submission_of_application'],
                    'LTV': float(request.form['LTV']),
                    'Region': request.form['Region'],
                    'Security_Type': request.form['Security_Type'],
                    'dtir1': float(request.form['dtir1'])
                }

                df_input = pd.DataFrame([input_data])
                
                # Encodage + Scaling
                for col in categorical_cols:
                    if col in df_input.columns and col in label_encoders:
                        df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
                
                df_input = df_input[feature_names]
                X_input = scaler.transform(df_input)
                
                prediction = model.predict(X_input)[0]
                proba = model.predict_proba(X_input)[0][1]
                risque = proba * 100

                result = {
                    'prediction': prediction,
                    'risque': round(risque, 2),
                    'niveau': "Très élevé" if risque >= 75 else "Élevé" if risque >= 50 else "Modéré" if risque >= 25 else "Faible"
                }
                
                return render_template('index.html', result=result, mode="manual")

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
