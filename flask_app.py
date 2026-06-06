from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

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


def _risk_level(prob_default):
    if prob_default >= 0.75:
        return 'Tres eleve'
    if prob_default >= 0.50:
        return 'Eleve'
    if prob_default >= 0.25:
        return 'Modere'
    return 'Faible'


def preprocess_input(data_dict):
    """Pretraitement d'un dictionnaire de features pour la prediction."""
    df_input = pd.DataFrame([data_dict])
    
    for col in categorical_cols:
        if col in df_input.columns:
            try:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
            except (ValueError, KeyError):
                df_input[col] = 0
    
    missing_cols = [c for c in feature_names if c not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans l'input: {missing_cols}")
    extra_cols = [c for c in df_input.columns if c not in feature_names]
    if extra_cols:
        df_input = df_input.drop(columns=extra_cols)

    df_input = df_input[feature_names]
    X_input = scaler.transform(df_input)
    return X_input


def preprocess_batch(df_upload):
    """Pretraitement batch conforme a la logique de l'app Streamlit."""
    df_clean = df_upload.copy()

    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop('ID', axis=1)
    if 'Status' in df_clean.columns:
        df_clean = df_clean.drop('Status', axis=1)

    for c in ['Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']:
        if c in df_clean.columns:
            df_clean = df_clean.drop(c, axis=1)

    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in categorical_cols:
        if col in df_clean.columns and not df_clean[col].mode().empty:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    for col in categorical_cols:
        if col in df_clean.columns:
            try:
                df_clean[col] = label_encoders[col].transform(df_clean[col].astype(str))
            except (ValueError, KeyError):
                df_clean[col] = 0

    missing_cols = [c for c in feature_names if c not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing_cols}")

    extra_cols = [c for c in df_clean.columns if c not in feature_names]
    if extra_cols:
        df_clean = df_clean.drop(columns=extra_cols)

    df_clean = df_clean[feature_names]
    return scaler.transform(df_clean)


def get_manual_form_data(form):
    return {
        'year': int(form.get('year', 2024)),
        'loan_limit': form.get('loan_limit', 'cf'),
        'Gender': form.get('Gender', 'Male'),
        'approv_in_adv': form.get('approv_in_adv', 'nopre'),
        'loan_type': form.get('loan_type', 'type1'),
        'loan_purpose': form.get('loan_purpose', 'p1'),
        'Credit_Worthiness': form.get('Credit_Worthiness', 'l1'),
        'open_credit': form.get('open_credit', 'nopc'),
        'business_or_commercial': form.get('business_or_commercial', 'nob/c'),
        'loan_amount': float(form.get('loan_amount', 200000)),
        'term': float(form.get('term', 360)),
        'Neg_ammortization': form.get('Neg_ammortization', 'not_neg'),
        'interest_only': form.get('interest_only', 'not_int'),
        'lump_sum_payment': form.get('lump_sum_payment', 'not_lpsm'),
        'property_value': float(form.get('property_value', 250000)),
        'construction_type': form.get('construction_type', 'sb'),
        'occupancy_type': form.get('occupancy_type', 'pr'),
        'Secured_by': form.get('Secured_by', 'home'),
        'total_units': form.get('total_units', '1U'),
        'income': float(form.get('income', 5000)),
        'credit_type': form.get('credit_type', 'EXP'),
        'Credit_Score': int(form.get('Credit_Score', 700)),
        'co-applicant_credit_type': form.get('co-applicant_credit_type', 'CIB'),
        'age': form.get('age', '35-44'),
        'submission_of_application': form.get('submission_of_application', 'to_inst'),
        'LTV': float(form.get('LTV', 80)),
        'Region': form.get('Region', 'North'),
        'Security_Type': form.get('Security_Type', 'direct'),
        'dtir1': float(form.get('dtir1', 40))
    }


@app.route('/')
def index():
    mode = request.args.get('mode', 'manual')
    if mode not in ('manual', 'upload'):
        mode = 'manual'
    return render_template('index.html', mode=mode)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = get_manual_form_data(request.form)
        
        X_input = preprocess_input(data)
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        prob_default = float(proba[1])
        
        result = {
            'prediction': int(prediction),
            'label': 'Defaut' if prediction == 1 else 'Pas de defaut',
            'proba_no_default': round(float(proba[0]) * 100, 2),
            'proba_default': round(prob_default * 100, 2),
            'risque_niveau': _risk_level(prob_default),
            'commentaire': (
                f"Le modele penche vers le defaut ({prob_default * 100:.1f}%), "
                f"marge {'faible' if prob_default < 0.6 else 'claire'}."
                if prob_default > 0.5
                else f"Le modele penche vers le remboursement normal ({(1 - prob_default) * 100:.1f}% de confiance)."
            )
        }
        
        return render_template('index.html', result=result, form_data=data, mode='manual')
        
    except Exception as e:
        return render_template('index.html', error=str(e), mode='manual')


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        if 'csv_file' not in request.files:
            raise ValueError('Aucun fichier recu. Veuillez choisir un CSV.')

        uploaded_file = request.files['csv_file']
        if uploaded_file.filename == '':
            raise ValueError('Aucun fichier selectionne.')
        if not uploaded_file.filename.lower().endswith('.csv'):
            raise ValueError('Le fichier doit etre au format CSV.')

        df_upload = pd.read_csv(uploaded_file)
        if df_upload.empty:
            raise ValueError('Le fichier CSV est vide.')

        X_upload = preprocess_batch(df_upload)
        predictions = model.predict(X_upload)
        probas = model.predict_proba(X_upload)[:, 1]

        results_df = pd.DataFrame({
            'Prediction': ['Defaut' if p == 1 else 'Pas de defaut' for p in predictions],
            'Probabilite_Defaut (%)': (probas * 100).round(2)
        })

        summary = {
            'rows': int(df_upload.shape[0]),
            'cols': int(df_upload.shape[1]),
            'default_count': int(predictions.sum()),
            'default_rate': round(float(predictions.mean()) * 100, 2),
            'avg_risk': round(float(probas.mean()) * 100, 2)
        }

        preview_rows = min(len(results_df), 25)
        batch_preview = results_df.head(preview_rows).to_dict(orient='records')

        return render_template(
            'index.html',
            mode='upload',
            batch_summary=summary,
            batch_preview=batch_preview,
            batch_preview_rows=preview_rows,
            batch_total_rows=len(results_df)
        )
    except Exception as e:
        return render_template('index.html', error=str(e), mode='upload')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API REST pour la prediction."""
    try:
        data = request.get_json()
        X_input = preprocess_input(data)
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        
        prob_default = float(proba[1])
        return jsonify({
            'prediction': int(prediction),
            'label': 'Defaut' if prediction == 1 else 'Pas de defaut',
            'proba_no_default': round(float(proba[0]), 4),
            'proba_default': round(prob_default, 4),
            'risque_niveau': _risk_level(prob_default)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
