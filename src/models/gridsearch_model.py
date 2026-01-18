
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Chemins fichiers
PROCESSED_PATH = "data/processed_data"
X_train_path = f"{PROCESSED_PATH}/X_train_scaled.csv"
y_train_path = f"{PROCESSED_PATH}/y_train.csv"

MODEL_PATH = "models/best_model_params.pkl"

# Charger les données
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()  # transformer en 1D

# Définir le modèle
rf = RandomForestRegressor(random_state=42)

# Définir la grille de paramètres
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

# GridSearch
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Sauvegarder les meilleurs paramètres
with open(MODEL_PATH, "wb") as f:
    pickle.dump(grid_search.best_params_, f)

print("GridSearch terminé. Meilleurs paramètres :", grid_search.best_params_)