import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score
import os

# Chemins fichiers
PROCESSED_PATH = "data/processed_data"
X_test_path  = f"{PROCESSED_PATH}/X_test_scaled.csv"
y_test_path  = f"{PROCESSED_PATH}/y_test.csv"
MODEL_PATH   = "models/final_model.pkl"
PRED_PATH    = "data/processed_data/y_pred.csv"
METRICS_PATH = "metrics/scores.json"

# Créer le dossier metrics si nécessaire
os.makedirs("metrics", exist_ok=True)

# Charger le modèle
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Charger les données de test
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()

# Prédictions
y_pred = model.predict(X_test)

# Sauvegarder les prédictions
pd.DataFrame({"y_pred": y_pred}).to_csv(PRED_PATH, index=False)

# Calculer métriques
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

metrics = {"mse": mse, "r2": r2}

# Sauvegarder les métriques dans un fichier JSON
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Évaluation terminée. Prédictions sauvegardées dans {PRED_PATH}")
print(f"Métriques sauvegardées dans {METRICS_PATH}")
print(metrics)