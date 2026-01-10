# SecuTransac – Détection de fraude transactionnelle

Application de détection de fraude basée sur le Machine Learning (XGBoost), utilisant **MLflow** pour le suivi des modèles et **Streamlit** pour l’interface utilisateur.  
Le projet est entièrement lancé via **Docker** pour garantir un environnement reproductible.

---

## Prérequis

- Docker Desktop installé et lancé  
  https://www.docker.com/products/docker-desktop/

---

## Lancement avec Docker

### Démarrer MLflow et l’application

Depuis la racine du projet :

docker compose up --build

- MLflow UI : http://localhost:5000
- Application Streamlit : http://localhost:8501


### Entraîner le modèle (obligatoire au moins une fois)

docker compose --profile train run --rm trainer
docker compose restart streamlit   

### Arrêt du projet

docker compose down

Supprimer également les données MLflow (runs, modèles) :

docker compose down -v

### Commandes essentielles (récapitulatif)

docker compose up --build
docker compose --profile train run --rm trainer
docker compose down
