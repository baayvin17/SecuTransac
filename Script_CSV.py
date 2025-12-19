import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -----------------------------
# Paramètres
# -----------------------------
N_TRANSACTIONS = 10_000
FRAUD_RATE = 0.02  # 2 % de fraude

# Pays simulés et pays à risque (basé sur rapports officiels fraude bancaire)
countries = ['FR', 'US', 'DE', 'UK', 'ES', 'IT', 'CN', 'RU', 'JP', 'NG', 'BR', 'UA']
risky_countries = ['RU', 'CN', 'NG', 'UA', 'BR']

# Types de transaction en français
transaction_types = ['paiement', 'retrait', 'virement', 'dépôt', 'prélèvement', 'remboursement']
risky_transaction_types = ['retrait', 'virement', 'prélèvement']

# Catégories de marchands en français
merchant_categories = ['commerce', 'restaurant', 'technologie', 'voyage', 'loisir', 'mode', 'santé', 'banque']
risky_categories = ['technologie', 'voyage', 'banque']

start_date = datetime(2017, 1, 1)
end_date = datetime(2024, 12, 31)

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def random_date(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def random_ip():
    return f"10.0.{random.randint(0,255)}.{random.randint(0,255)}"

# -----------------------------
# Génération des données
# -----------------------------
data = []

for i in range(N_TRANSACTIONS):
    transaction_id = f"T{i+1:07d}"
    timestamp = random_date(start_date, end_date)
    hour_of_day = timestamp.hour

    country = random.choice(countries)
    transaction_type = random.choice(transaction_types)
    merchant_category = random.choice(merchant_categories)

    merchant_id = f"M{random.randint(1,500):04d}"
    customer_id = f"C{random.randint(1,1000):04d}"
    device_id = f"D{random.randint(1,4000):04d}"
    ip_address = random_ip()

    # Montant
    if random.random() < FRAUD_RATE:
        amount = round(np.random.uniform(500, 3000), 2)
    else:
        amount = round(np.random.exponential(120), 2)

    # -----------------------------
    # Calcul du score de fraude
    # -----------------------------
    fraud_score = 0

    if amount > random.choice([800, 1000, 1200]):
        fraud_score += 1
    if hour_of_day < 6 or hour_of_day > 22:
        fraud_score += 1
    if country in risky_countries and random.random() > 0.2:
        fraud_score += 1
    if transaction_type in risky_transaction_types and random.random() > 0.2:
        fraud_score += 1
    if merchant_category in risky_categories and random.random() > 0.2:
        fraud_score += 1

    is_fraud = 1 if fraud_score >= 3 else 0

    data.append([
        transaction_id,
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        amount,
        merchant_id,
        customer_id,
        transaction_type,
        country,
        device_id,
        ip_address,
        merchant_category,
        hour_of_day,
        is_fraud
    ])

# -----------------------------
# Création du DataFrame et CSV
# -----------------------------
columns = [
    'transaction_id', 'timestamp', 'amount', 'merchant_id', 'customer_id',
    'transaction_type', 'country', 'device_id', 'ip_address',
    'merchant_category', 'hour_of_day', 'is_fraud'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("transactions.csv", index=False)

print("✅ Fichier transactions.csv généré avec succès")
print(df.head())
print(df['is_fraud'].value_counts())
