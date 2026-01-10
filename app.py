import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow.sklearn
import joblib
import os
from pathlib import Path

# ===============================
# CONFIGURATION PAGE
# ===============================
st.set_page_config(
    page_title="SécuTransac",
    layout="wide"
)

# ===============================
# CHARGEMENT MODELE DEPUIS MLFLOW
# ===============================
@st.cache_resource
def load_model():
    run_id = Path("latest_run_id.txt").read_text(encoding="utf-8").strip()
    return mlflow.sklearn.load_model(
        model_uri=f"runs:/{run_id}/model"
    )

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")


model = load_model()

# ===============================
# LISTES METIER
# ===============================
countries = ["FR", "US", "DE", "ES", "IT", "RU", "CN", "NG", "UA", "BR"]
HIGH_RISK_COUNTRIES = ["RU", "CN", "NG", "UA", "BR"]

FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "country_risk",
    "transaction_type_encoded",
    "merchant_category_encoded",
]


transaction_types = {
    "Paiement": 0,
    "Retrait": 1,
    "Virement": 2,
    "Dépôt": 3
}

merchant_categories = {
    "Restaurant": 0,
    "Technologie": 1,
    "Divertissement": 2,
    "Voyage": 3,
    "Commerce de détail": 4
}

# ===============================
# FICHIER DE STOCKAGE
# ===============================
CSV_FILE = "tested_transactions.csv"

COLUMNS = [
    "amount", "hour_of_day", "day_of_week",
    "country", "country_risk",
    "transaction_type", "transaction_type_encoded",
    "merchant_category", "merchant_category_encoded",
    "fraud_probability"
]

if os.path.exists(CSV_FILE):
    df_transactions = pd.read_csv(CSV_FILE)
else:
    df_transactions = pd.DataFrame(columns=COLUMNS)

if "transactions" not in st.session_state:
    st.session_state.transactions = df_transactions

# ===============================
# ONGLETS
# ===============================
tabs = st.tabs(["Tester une transaction", "Dashboard dynamique"])

# ======================================================
# ONGLET 1 — TEST TRANSACTION
# ======================================================
with tabs[0]:
    st.header("Tester une transaction")

    amount = st.number_input(
        "Montant (€)",
        min_value=1.0,
        max_value=200_000.0,
        value=500.0,
        step=50.0
    )

    hour_of_day = st.slider("Heure de la transaction", 0, 23, 14)

    day_of_week = st.selectbox(
        "Jour de la semaine",
        options=[
            ("Lundi", 0), ("Mardi", 1), ("Mercredi", 2),
            ("Jeudi", 3), ("Vendredi", 4),
            ("Samedi", 5), ("Dimanche", 6)
        ],
        format_func=lambda x: x[0]
    )[1]

    country = st.selectbox("Pays", countries)
    country_risk = 1 if country in HIGH_RISK_COUNTRIES else 0

    transaction_label = st.selectbox("Type de transaction", transaction_types.keys())
    transaction_encoded = transaction_types[transaction_label]

    merchant_label = st.selectbox("Catégorie du marchand", merchant_categories.keys())
    merchant_encoded = merchant_categories[merchant_label]

    if st.button("Prédire le risque de fraude"):
        input_df = pd.DataFrame([{
            "amount": amount,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "country_risk": country_risk,
            "transaction_type_encoded": transaction_encoded,
            "merchant_category_encoded": merchant_encoded
        }])

        scaler = load_scaler()
        X_scaled = scaler.transform(input_df[FEATURES])
        fraud_probability = float(model.predict_proba(X_scaled)[:, 1][0]) * 100

        st.subheader("Résultat")

        if fraud_probability >= 70:
            st.error(f"🚨 Risque ÉLEVÉ : {fraud_probability:.2f}%")
        elif fraud_probability >= 40:
            st.warning(f"⚠️ Transaction suspecte : {fraud_probability:.2f}%")
        else:
            st.success(f"✅ Transaction sûre : {fraud_probability:.2f}%")

        new_row = pd.DataFrame([{
            "amount": amount,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "country": country,
            "country_risk": country_risk,
            "transaction_type": transaction_label,
            "transaction_type_encoded": transaction_encoded,
            "merchant_category": merchant_label,
            "merchant_category_encoded": merchant_encoded,
            "fraud_probability": fraud_probability
        }])

        st.session_state.transactions = pd.concat(
            [st.session_state.transactions, new_row],
            ignore_index=True
        )

        st.session_state.transactions.to_csv(CSV_FILE, index=False)

# ======================================================
# ONGLET 2 — DASHBOARD
# ======================================================
with tabs[1]:
    st.header("Dashboard des transactions testées")

    df = st.session_state.transactions

    if df.empty:
        st.info("Aucune transaction enregistrée.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Transactions", len(df))
        col2.metric("Risque élevé", len(df[df["fraud_probability"] >= 70]))
        col3.metric("Risque moyen", len(df[(df["fraud_probability"] >= 40) & (df["fraud_probability"] < 70)]))
        col4.metric("Fraude moyenne (%)", f"{df['fraud_probability'].mean():.2f}")

        st.divider()

        st.plotly_chart(
            px.scatter(df, x="amount", y="fraud_probability",
                       title="Montant vs Probabilité de fraude"),
            use_container_width=True
        )

        st.plotly_chart(
            px.bar(df.groupby("country")["fraud_probability"].mean().reset_index(),
                   x="country", y="fraud_probability",
                   title="Risque moyen par pays"),
            use_container_width=True
        )
