# app.py - Streamlit multi-onglets : test transaction + dashboard dynamique

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ===============================
# CONFIGURATION DE LA PAGE
# ===============================
st.set_page_config(
    page_title="SécuTransac",
    layout="wide"
)

# ===============================
# CHARGEMENT DU MODELE
# ===============================
model = joblib.load("fraud_xgboost_model.h5")
scaler = joblib.load("scaler.pkl")

# ===============================
# LISTES MÉTIER
# ===============================
countries = ["FR", "US", "DE", "ES", "IT", "RU", "CN", "NG", "UA", "BR"]
HIGH_RISK_COUNTRIES = ["RU", "CN", "NG", "UA", "BR"]

transaction_types = {"Paiement":0, "Retrait":1, "Virement":2, "Dépôt":3}
merchant_categories = {"Restaurant":0, "Technologie":1, "Divertissement":2, "Voyage":3, "Commerce de détail":4}

# ===============================
# FICHIER DE STOCKAGE DES TRANSACTIONS TESTÉES
# ===============================
csv_file = "tested_transactions.csv"

if os.path.exists(csv_file):
    try:
        tested_transactions = pd.read_csv(csv_file)
        if tested_transactions.empty:
            tested_transactions = pd.DataFrame(columns=[
                "amount", "hour_of_day", "day_of_week", "country", "country_risk",
                "transaction_type", "transaction_type_encoded",
                "merchant_category", "merchant_category_encoded", "fraud_probability"
            ])
    except pd.errors.EmptyDataError:
        tested_transactions = pd.DataFrame(columns=[
            "amount", "hour_of_day", "day_of_week", "country", "country_risk",
            "transaction_type", "transaction_type_encoded",
            "merchant_category", "merchant_category_encoded", "fraud_probability"
        ])
else:
    tested_transactions = pd.DataFrame(columns=[
        "amount", "hour_of_day", "day_of_week", "country", "country_risk",
        "transaction_type", "transaction_type_encoded",
        "merchant_category", "merchant_category_encoded", "fraud_probability"
    ])

# ===============================
# STOCKAGE DANS STREAMLIT
# ===============================
if "tested_transactions" not in st.session_state:
    st.session_state.tested_transactions = tested_transactions

# ===============================
# ONGLETS
# ===============================
tabs = st.tabs(["Tester une transaction", "Dashboard dynamique"])

# ===============================
# ONGLET 1 : TESTER UNE TRANSACTION
# ===============================
with tabs[0]:
    st.header("Tester une transaction")
    
    amount = st.number_input("Montant (€)", min_value=1.0, max_value=10000.0, value=250.0, step=10.0)
    hour_of_day = st.slider("Heure", 0, 23, 14)
    day_of_week = st.selectbox("Jour de la semaine", 
                               options=[("Lundi",0),("Mardi",1),("Mercredi",2),("Jeudi",3),
                                        ("Vendredi",4),("Samedi",5),("Dimanche",6)],
                               format_func=lambda x:x[0])[1]
    country = st.selectbox("Pays", countries)
    country_risk = 1 if country in HIGH_RISK_COUNTRIES else 0
    transaction_type_label = st.selectbox("Type de transaction", list(transaction_types.keys()))
    transaction_type_encoded = transaction_types[transaction_type_label]
    merchant_category_label = st.selectbox("Catégorie du commerçant", list(merchant_categories.keys()))
    merchant_category_encoded = merchant_categories[merchant_category_label]

    if st.button("Prédire le risque de fraude"):
        # Préparer les données pour le modèle
        input_data = pd.DataFrame([{
            "amount": amount,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "country_risk": country_risk,
            "transaction_type_encoded": transaction_type_encoded,
            "merchant_category_encoded": merchant_category_encoded
        }])
        input_scaled = scaler.transform(input_data)
        fraud_probability = model.predict_proba(input_scaled)[0][1] * 100

        # Affichage du résultat
        st.subheader("Résultat")
        if fraud_probability >= 70:
            st.error(f"Transaction à RISQUE ÉLEVÉ ({fraud_probability:.2f} %)")
        elif fraud_probability >= 40:
            st.warning(f"Transaction suspecte ({fraud_probability:.2f} %)")
        else:
            st.success(f"Transaction considérée comme sûre ({fraud_probability:.2f} %)")

        # Enregistrement de la transaction testée
        new_tx = pd.DataFrame([{
            "amount": amount,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "country": country,
            "country_risk": country_risk,
            "transaction_type": transaction_type_label,
            "transaction_type_encoded": transaction_type_encoded,
            "merchant_category": merchant_category_label,
            "merchant_category_encoded": merchant_category_encoded,
            "fraud_probability": fraud_probability
        }])
        st.session_state.tested_transactions = pd.concat([st.session_state.tested_transactions, new_tx], ignore_index=True)
        st.session_state.tested_transactions.to_csv(csv_file, index=False)  # Sauvegarde dans CSV

# ===============================
# ONGLET 2 : DASHBOARD DYNAMIQUE
# ===============================
with tabs[1]:
    st.header("Dashboard dynamique basé sur les transactions testées")

    if st.session_state.tested_transactions.empty:
        st.info("Aucune transaction testée pour l'instant. Testez une transaction dans l'onglet précédent.")
    else:
        df_dash = st.session_state.tested_transactions

        # KPI
        total_tx = len(df_dash)
        high_risk_tx = len(df_dash[df_dash["fraud_probability"] >= 70])
        medium_risk_tx = len(df_dash[(df_dash["fraud_probability"] >= 40) & (df_dash["fraud_probability"] < 70)])
        avg_prob = df_dash["fraud_probability"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions testées", total_tx)
        col2.metric("Risque élevé", high_risk_tx)
        col3.metric("Risque moyen", medium_risk_tx)
        col4.metric("Probabilité moyenne de fraude (%)", f"{avg_prob:.2f}")

        st.divider()

        # Graphiques
        st.subheader("Montants vs Probabilité de fraude")
        fig1 = px.scatter(df_dash, x="amount", y="fraud_probability", color="fraud_probability",
                          labels={"amount":"Montant (€)", "fraud_probability":"Probabilité de fraude (%)"})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Transactions par type")
        fig2 = px.histogram(df_dash, x="transaction_type", y="fraud_probability", color="fraud_probability",
                            labels={"transaction_type":"Type de transaction", "fraud_probability":"Probabilité (%)"}, barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Transactions par pays")
        fig3 = px.bar(df_dash.groupby("country")["fraud_probability"].mean().reset_index(),
                      x="country", y="fraud_probability", color="fraud_probability",
                      labels={"country":"Pays","fraud_probability":"Probabilité moyenne (%)"})
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Transactions par catégorie de marchand")
        fig4 = px.bar(df_dash.groupby("merchant_category")["fraud_probability"].mean().reset_index(),
                      x="merchant_category", y="fraud_probability", color="fraud_probability",
                      labels={"merchant_category":"Catégorie marchand","fraud_probability":"Probabilité moyenne (%)"})
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Répartition par heure de la journée")
        fig5 = px.bar(df_dash.groupby("hour_of_day")["fraud_probability"].mean().reset_index(),
                      x="hour_of_day", y="fraud_probability", labels={"hour_of_day":"Heure","fraud_probability":"Probabilité moyenne (%)"})
        st.plotly_chart(fig5, use_container_width=True)

        st.divider()
        st.markdown("### Analyse métier")
        st.markdown("""
        - Les transactions avec probabilité élevée de fraude sont priorisées pour vérification.
        - Le dashboard permet de visualiser l'impact des montants, types, catégories et pays sur le risque.
        - Chaque nouvelle transaction testée met à jour le dashboard en temps réel et est sauvegardée pour les prochaines sessions.
        """)
