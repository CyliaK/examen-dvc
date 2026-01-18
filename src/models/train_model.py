import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import os

# Chemins fichiers
PROCESSED_PATH = "data/processed_data"
X_train_path = f"{PROCESSED_PATH}/X_train_scaled.csv"
y_train_path = f"{PROCESSED_PATH}/y_train.csv"

MODEL_PATH = "models/final_model.pkl"

# Charger les données
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()  # transformer en 1D

# Meilleurs paramètres trouvés
best_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "random_state": 42
}

# Créer et entraîner le modèle
rf_model = RandomForestRegressor(**best_params)
rf_model.fit(X_train, y_train)

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle entraîné
with open(MODEL_PATH, "wb") as f:
    pickle.dump(rf_model, f)

print(f"Modèle entraîné et sauvegardé dans {MODEL_PATH}")