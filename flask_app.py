# flask_app.py
import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Chargement du modèle
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
    mode = "manual"  # par défaut

    if request.method == 'POST':
        try:
            # ==================== MODE CSV ====================
            if 'file' in request.files and request.files['file'].filename != '':
                mode = "csv"
                file = request.files['file']
                df = pd.read_csv(file)

                # Nettoyage
                if 'ID' in df.columns: df = df.drop('ID', axis=1)
                if 'Status' in df.columns: df = df.drop('Status', axis=1)
                for col in ['Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']:
                    if col in df.columns:
                        df = df.drop(col, axis=1)

                # Gestion des valeurs manquantes
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

                df = df.reindex(columns=feature_names, fill_value=0)
                X = scaler.transform(df)

                predictions = model.predict(X)
                probas = model.predict_proba(X)[:, 1]

                results_df = pd.DataFrame({
                    'Prediction': ['Défaut' if p == 1 else 'Pas de défaut' for p in predictions],
                    'Risque (%)': (probas * 100).round(2)
                })

                return render_template('index.html', 
                                     results=results_df.to_html(classes='table table-striped', index=False),
                                     mode=mode)

            # ==================== MODE SAISIE MANUELLE ====================
            else:
                mode = "manual"
                input_data = {
                    'year': int(request.form.get('year')),
                    'loan_limit': request.form.get('loan_limit'),
                    'Gender': request.form.get('Gender'),
                    'approv_in_adv': request.form.get('approv_in_adv'),
                    'loan_type': request.form.get('loan_type'),
                    'loan_purpose': request.form.get('loan_purpose'),
                    'Credit_Worthiness': request.form.get('Credit_Worthiness'),
                    'open_credit': request.form.get('open_credit'),
                    'business_or_commercial': request.form.get('business_or_commercial'),
                    'loan_amount': float(request.form.get('loan_amount')),
                    'term': int(request.form.get('term')),
                    'Neg_ammortization': request.form.get('Neg_ammortization'),
                    'interest_only': request.form.get('interest_only'),
                    'lump_sum_payment': request.form.get('lump_sum_payment'),
                    'property_value': float(request.form.get('property_value')),
                    'construction_type': request.form.get('construction_type'),
                    'occupancy_type': request.form.get('occupancy_type'),
                    'Secured_by': request.form.get('Secured_by'),
                    'total_units': request.form.get('total_units'),
                    'income': float(request.form.get('income')),
                    'credit_type': request.form.get('credit_type'),
                    'Credit_Score': int(request.form.get('Credit_Score')),
                    'co-applicant_credit_type': request.form.get('co-applicant_credit_type'),
                    'age': request.form.get('age'),
                    'submission_of_application': request.form.get('submission_of_application'),
                    'LTV': float(request.form.get('LTV')),
                    'Region': request.form.get('Region'),
                    'Security_Type': request.form.get('Security_Type'),
                    'dtir1': float(request.form.get('dtir1'))
                }

                df_input = pd.DataFrame([input_data])

                for col in categorical_cols:
                    if col in df_input.columns and col in label_encoders:
                        df_input[col] = label_encoders[col].transform(df_input[col].astype(str))

                df_input = df_input[feature_names]
                X_input = scaler.transform(df_input)

                prediction = model.predict(X_input)[0]
                proba = model.predict_proba(X_input)[0][1]
                risque = round(proba * 100, 2)

                result = {
                    'prediction': prediction,
                    'risque': risque,
                    'niveau': "Très élevé" if risque >= 75 else "Élevé" if risque >= 50 else "Modéré" if risque >= 25 else "Faible"
                }

                return render_template('index.html', result=result, mode=mode)

        except Exception as e:
            return render_template('index.html', error=f"Erreur : {str(e)}", mode=mode)

    return render_template('index.html', mode=mode)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
