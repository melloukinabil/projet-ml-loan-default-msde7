import pickle, pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Charger les donnees
df = pd.read_csv('Loan_Default.csv')
df = df.drop('ID', axis=1)

# SUPPRIMER les colonnes avec data leakage
# Interest_rate_spread (100% NaN si defaut), rate_of_interest (99.5%), Upfront_charges (99.6%)
leaky_cols = ['Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']
df = df.drop(leaky_cols, axis=1)
print(f"Colonnes supprimees (data leakage): {leaky_cols}")

y = df.pop('Status').values

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remplir les NaN
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Scaler
feature_names = df.columns.tolist()
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrainer XGBoost (meilleur pour ce type de donnees)
print("Entrainement XGBoost...")
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                      scale_pos_weight=3, random_state=42, eval_metric='logloss',
                      verbosity=0, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred, target_names=['Pas defaut','Defaut']))

# Sauvegarder le pipeline
pipeline = {
    'model': model, 'model_name': 'XGBoost',
    'scaler': scaler, 'label_encoders': label_encoders,
    'feature_names': feature_names,
    'categorical_cols': categorical_cols, 'numerical_cols': numerical_cols
}
pickle.dump(pipeline, open('pipeline_loan_default.pkl', 'wb'))
print("\nPipeline sauvegarde.")

# Test avec profil risque
print("\n--- Test profil risque ---")
data = {'year':2019,'loan_limit':'ncf','Gender':'Female','approv_in_adv':'nopre',
        'loan_type':'type2','loan_purpose':'p4','Credit_Worthiness':'l2',
        'open_credit':'opc','business_or_commercial':'b/c','loan_amount':500000,
        'term':360,'Neg_ammortization':'neg_amm','interest_only':'int_only',
        'lump_sum_payment':'lpsm','property_value':450000,'construction_type':'mh',
        'occupancy_type':'ir','Secured_by':'land','total_units':'4U','income':3000,
        'credit_type':'EQUI','Credit_Score':520,'co-applicant_credit_type':'EXP',
        'age':'25-34','submission_of_application':'not_inst','LTV':111.0,
        'Region':'south','Security_Type':'Indriect','dtir1':55.0}
dfi = pd.DataFrame([data])
for col in categorical_cols:
    dfi[col] = label_encoders[col].transform(dfi[col].astype(str))
dfi = dfi[feature_names]
Xi = scaler.transform(dfi)
pred = model.predict(Xi)[0]
proba = model.predict_proba(Xi)[0]
print(f"Prediction: {'DEFAUT' if pred==1 else 'PAS DEFAUT'}")
print(f"P(defaut): {proba[1]*100:.1f}%")

# Test profil sain
print("\n--- Test profil sain ---")
data2 = {'year':2019,'loan_limit':'cf','Gender':'Male','approv_in_adv':'pre',
         'loan_type':'type1','loan_purpose':'p1','Credit_Worthiness':'l1',
         'open_credit':'nopc','business_or_commercial':'nob/c','loan_amount':200000,
         'term':360,'Neg_ammortization':'not_neg','interest_only':'not_int',
         'lump_sum_payment':'not_lpsm','property_value':350000,'construction_type':'sb',
         'occupancy_type':'pr','Secured_by':'home','total_units':'1U','income':7000,
         'credit_type':'EXP','Credit_Score':780,'co-applicant_credit_type':'CIB',
         'age':'35-44','submission_of_application':'to_inst','LTV':57.0,
         'Region':'North','Security_Type':'direct','dtir1':28.0}
dfi2 = pd.DataFrame([data2])
for col in categorical_cols:
    dfi2[col] = label_encoders[col].transform(dfi2[col].astype(str))
dfi2 = dfi2[feature_names]
Xi2 = scaler.transform(dfi2)
pred2 = model.predict(Xi2)[0]
proba2 = model.predict_proba(Xi2)[0]
print(f"Prediction: {'DEFAUT' if pred2==1 else 'PAS DEFAUT'}")
print(f"P(defaut): {proba2[1]*100:.1f}%")
