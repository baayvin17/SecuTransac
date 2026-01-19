# SecuTransac

**LES DONNEES DU DATASET ONT ETE GENEREES MANUELLEMENT VIA UN SCRIPT C EST LA RAISON POUR LAQUELLE LE DATASET EST PRESENT**

Bienvenue dans **SecuTransac**, votre assistant de détection de fraude bancaire propulsé par l'intelligence artificielle. 

## Qu'est-ce que c'est ?

SecuTransac est une application qui analyse vos transactions en temps réel pour détecter les comportements frauduleux. Imaginez avoir un expert en sécurité bancaire qui examine chaque transaction en quelques millisecondes, c'est exactement ce que fait cet outil.

### Ce qui rend SecuTransac unique

- **Intelligence artificielle** : Utilise XGBoost, un algorithme de machine learning puissant et éprouvé
- **Suivi des performances** : MLflow garde une trace de toutes les expérimentations et permet de comparer les modèles
- **Interface intuitive** : Une application Streamlit simple et agréable à utiliser, même pour les non-techniques
- **Prêt à l'emploi** : Docker rend l'installation et le déploiement vraiment simples

## Comment ça marche ?

SecuTransac fonctionne en trois grandes étapes :

### 1. Entraînement du modèle
Le modèle apprend à reconnaître les fraudes en analysant des milliers de transactions historiques. Plus il s'entraîne, plus il devient précis dans ses détections.

### 2. Gestion de la mémoire
MLflow joue le rôle de bibliothécaire : il garde une trace de tous les modèles entraînés, leurs performances, et permet de revenir en arrière si besoin.

### 3. Utilisation de l'application
L'interface Streamlit vous permet de tester des transactions et d'obtenir instantanément un verdict : est-ce une fraude ou pas ?

## Organisation des fichiers

Voici comment le projet est organisé (pas de panique, c'est plus simple qu'il n'y paraît !) :

```
SecuTransac/
├── app/
│   └── app.py                      # L'interface que vous utilisez
├── ml/
│   └── train_mlflow.py             # Le script qui entraîne l'IA
├── data/
│   ├── transactions.csv            # Les données d'apprentissage
│   └── tested_transactions.csv     # Vos tests sauvegardés
├── models/
│   └── scaler.pkl                  # Outil de normalisation des données
├── docker-compose.yml              # Configuration Docker
├── Dockerfile.app                  # Recette pour l'app Streamlit
├── Dockerfile.mlflow               # Recette pour MLflow
└── requirements.txt                # Liste des bibliothèques Python
```

## Les données qu'on utilise

Le modèle analyse plusieurs informations sur chaque transaction :

- Le montant de la transaction
- L'heure à laquelle elle a été effectuée
- Le jour de la semaine
- Le pays d'origine
- Le type de transaction
- La catégorie du marchand

### Quelques chiffres sur notre base de données

Dans notre dataset d'entraînement :
- On a analysé **6 710 transactions**
- Dont **809 fraudes** détectées
- Soit environ **12% de fraudes**

**Note importante** : Ce déséquilibre est normal et reflète la réalité. Dans la vraie vie, heureusement, la majorité des transactions sont légitimes.

## Le modèle XGBoost

On a choisi XGBoost parce que c'est un champion reconnu pour la détection de patterns complexes. Voici comment on prépare les données :

1. **Normalisation** : On standardise les données pour que toutes les variables soient comparables
2. **Équilibrage** : On utilise la technique SMOTE pour compenser le déséquilibre entre fraudes et non-fraudes
3. **Entraînement** : Le modèle apprend à reconnaître les comportements suspects

Notre objectif principal ? **Maximiser la détection des fraudes**, même si cela implique parfois quelques fausses alertes. Dans le domaine bancaire, mieux vaut prévenir que guérir.

## Comment entraîner le modèle

Quand vous lancez l'entraînement, voici ce qui se passe dans les coulisses :

1. Chargement et nettoyage des données brutes
2. Séparation en deux groupes : entraînement (80%) et test (20%)
3. Entraînement du modèle sur les données d'apprentissage
4. Calcul des performances (précision, ROC AUC, recall...)
5. Sauvegarde automatique dans MLflow
6. Publication sous le nom `SecuTransac_Fraud_Model` dans le Model Registry

## MLflow : votre journal de bord

MLflow, c'est un peu comme un carnet de notes pour data scientists. Il conserve :

- L'historique complet de toutes les expérimentations
- Les métriques de performance de chaque version du modèle
- Tous les fichiers et artefacts nécessaires à la reproduction
- La version "production" du modèle actuellement utilisée

L'application Streamlit charge toujours automatiquement la dernière version validée du modèle depuis le registry.

## L'application Streamlit

C'est votre interface de travail au quotidien. Elle vous permet de :

- Saisir manuellement les détails d'une transaction
- Obtenir un score de risque précis (de 0 à 100%)
- Visualiser la classification : transaction sûre, suspecte ou frauduleuse
- Consulter l'historique complet de vos tests précédents

Toutes vos transactions testées sont automatiquement sauvegardées pour analyse ultérieure.

## Démarrage rapide avec Docker

C'est la méthode recommandée, la plus simple et la plus fiable. Voici comment faire :

### Ce dont vous avez besoin
- Docker Desktop installé sur votre ordinateur (et lancé !)

### Lancer le projet

Ouvrez un terminal et tapez :

```bash
docker compose up --build
```

Attendez un peu (le temps que Docker prépare tout), puis ouvrez votre navigateur :

- **Interface de détection** : http://localhost:8501
- **Tableau de bord MLflow** : http://localhost:5000

### Première utilisation : entraîner le modèle

Dans un nouveau terminal (laissez le premier ouvert), lancez :

```bash
docker compose --profile train run --rm trainer
```

Cette étape prend quelques minutes. Une fois terminée, rechargez la page de l'application Streamlit et c'est parti !

### Pour arrêter tout ça

Quand vous avez fini :

```bash
docker compose down
```

Si vous voulez aussi supprimer toutes les données MLflow :

```bash
docker compose down -v
```

## Alternative : lancement manuel (pour les développeurs)

Vous préférez tout contrôler ? Pas de problème !

### Prérequis
- Python 3.10 ou plus récent
- pip installé

### Installation

```bash
pip install -r requirements.txt
```

### Entraîner le modèle

```bash
python ml/train_mlflow.py
```

### Lancer MLflow (dans un terminal)

```bash
mlflow ui
```

Rendez-vous sur http://localhost:5000

### Lancer l'application (dans un autre terminal)

```bash
streamlit run app/app.py
```

Rendez-vous sur http://localhost:8501

## Guide d'utilisation simplifié

1. **Installation** : `docker compose up --build`
2. **Entraînement** : `docker compose --profile train run --rm trainer`
3. **Utilisation** : Ouvrez http://localhost:8501
4. **Expérimentations** : Consultez http://localhost:5000

## Petites choses à savoir

Quelques points importants à garder en tête :

- Le modèle doit être entraîné au moins une fois avant de pouvoir utiliser l'application de prédiction
- Chaque transaction testée est automatiquement sauvegardée dans `data/tested_transactions.csv`
- Le scaler (outil de normalisation) est stocké dans `models/scaler.pkl` et réutilisé à chaque prédiction
- Le modèle est référencé sous le nom `SecuTransac_Fraud_Model` dans le MLflow Registry

## Besoin d'aide ?

Si vous rencontrez un problème, voici quelques vérifications de base :

1. Assurez-vous que Docker Desktop est bien lancé sur votre machine
2. Vérifiez que vous avez entraîné le modèle au moins une fois
3. Consultez les logs avec la commande `docker compose logs` pour identifier l'erreur

Bon détective de fraudes !