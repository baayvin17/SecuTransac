# ======================================
# Projet : SécuTransac
# Entraînement du modèle avec MLflow
# ======================================

import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ===============================
# PATHS PROJET (ROBUSTES)
# ===============================
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
MLRUNS_DIR = PROJECT_DIR / "mlruns"

MODELS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

# ===============================
# CONFIGURATION MLFLOW
# ===============================
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
registry_uri = os.getenv("MLFLOW_REGISTRY_URI")

# Si Docker/Compose fournit un tracking URI (ex: http://mlflow:5000), on le respecte.
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())

if registry_uri:
    mlflow.set_registry_uri(registry_uri)
else:
    mlflow.set_registry_uri(MLRUNS_DIR.as_uri())

mlflow.set_experiment("SecuTransac_Fraud_Detection")

# ===============================
# CHARGEMENT DES DONNÉES
# ===============================
df = pd.read_csv(DATA_DIR / "transactions.csv")

# ===============================
# FEATURE ENGINEERING
# ===============================
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["day_of_week"] = df["timestamp"].dt.dayofweek

HIGH_RISK_COUNTRIES = ["RU", "CN", "NG", "UA", "BR"]
df["country_risk"] = df["country"].apply(lambda x: 1 if x in HIGH_RISK_COUNTRIES else 0)

transaction_type_map = {
    "paiement": 0,
    "retrait": 1,
    "virement": 2,
    "dépôt": 3
}
df["transaction_type_encoded"] = df["transaction_type"].map(transaction_type_map)

df["merchant_category_encoded"] = df["merchant_category"].astype("category").cat.codes

# ===============================
# FEATURES & TARGET
# ===============================
FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "country_risk",
    "transaction_type_encoded",
    "merchant_category_encoded",
]
TARGET = "is_fraud"

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

print("Transactions totales après nettoyage :", len(df))
print("Fraudes :", y.sum())
print("Non fraudes :", len(df) - y.sum())

# ===============================
# SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# SCALER
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# SMOTE
# ===============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# ===============================
# TRAINING + MLFLOW
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

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Accuracy :", accuracy)
    print("ROC AUC :", roc_auc)
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="SecuTransac_Fraud_Model"
    )

    run_id = mlflow.active_run().info.run_id
    (MODELS_DIR / "latest_run_id.txt").write_text(run_id, encoding="utf-8")

    joblib.dump(model, MODELS_DIR / "model_xgb.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print("✅ Entraînement terminé")
