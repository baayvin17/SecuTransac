# ======================================
# Projet : SécuTransac
# Entraînement du modèle avec MLflow
# ======================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# ===============================
# CONFIGURATION MLFLOW
# ===============================
mlflow.set_experiment("SecuTransac_Fraud_Detection")

# ===============================
# CHARGEMENT DES DONNÉES
# ===============================
df = pd.read_csv("transactions.csv")

# ===============================
# FEATURE ENGINEERING
# ===============================
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["day_of_week"] = df["timestamp"].dt.dayofweek

# Pays à risque
HIGH_RISK_COUNTRIES = ["RU", "CN", "NG", "UA", "BR"]
df["country_risk"] = df["country"].apply(
    lambda x: 1 if x in HIGH_RISK_COUNTRIES else 0
)

# Encodage type de transaction
transaction_type_map = {
    "paiement": 0,
    "retrait": 1,
    "virement": 2,
    "dépôt": 3
}
df["transaction_type_encoded"] = df["transaction_type"].map(transaction_type_map)

# Encodage catégorie marchand
df["merchant_category_encoded"] = (
    df["merchant_category"]
    .astype("category")
    .cat.codes
)

# ===============================
# FEATURES & TARGET
# ===============================
features = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "country_risk",
    "transaction_type_encoded",
    "merchant_category_encoded"
]

target = "is_fraud"

# ===============================
# NETTOYAGE DES DONNÉES (IMPORTANT)
# ===============================
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# ===============================
# SPLIT TRAIN / TEST
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# NORMALISATION
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# GESTION DU DÉSÉQUILIBRE (SMOTE)
# ===============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(
    X_train_scaled,
    y_train
)

# ===============================
# ENTRAÎNEMENT AVEC MLFLOW
# ===============================
with mlflow.start_run():

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    # ===============================
    # PRÉDICTIONS
    # ===============================
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ===============================
    # MÉTRIQUES
    # ===============================
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Accuracy :", accuracy)
    print("ROC AUC  :", roc_auc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ===============================
    # LOG MLFLOW
    # ===============================
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("smote", True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="SecuTransac_Fraud_Model"
)

    # ===============================
    # SAUVEGARDE LOCALE (STREAMLIT)
    # ===============================
    joblib.dump(model, "model_xgb.pkl")
    joblib.dump(scaler, "scaler.pkl")

print("\n✅ Entraînement terminé et modèle sauvegardé")
